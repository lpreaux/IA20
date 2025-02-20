import math

import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from dataset.state import DatasetState

from dataset.forms.config import dataset_config_form

dataset = DatasetState.get_dataset()


######
# UI #
######

def render_no_dataset_skeleton():
    """Affiche un squelette de la page quand aucun dataset n'est chargé."""
    with st.container():
        # En-tête
        st.markdown("# Jeu de données")

        # Skeleton pour le message d'erreur
        with st.container(border=True):
            # Utilisation des colonnes pour centrer le contenu
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("### :ghost: Aucun jeu de données")
                st.caption("Chargez un jeu de données pour commencer l'analyse")
                st.button("📤 Charger un jeu de données",
                          type="primary",
                          on_click=dataset_config_form,
                          use_container_width=True)

        # Skeleton pour le tableau
        ## TODO


def page():
    df_raw = dataset.data

    TARGET = dataset.target_columns[0]
    FEATURES = dataset.features_columns

    df_raw = (
        df_raw
        .assign(
            target_num=lambda x: pd.Categorical(x[TARGET]).codes
        )
    )

    df_target_sizes = (
        df_raw
        .groupby(TARGET)
        .size()
        .rename('count')
    )

    st.write("""
    # Analyse Exploratoire des Données - Dataset Vin
    """)

    st.write("""
    ___
    ## 1. Premier aperçu des données

    Examinons dans un premier temps les caractéristiques générales de notre dataset.py :
    - Structure des données (types de variables, valeurs manquantes)
    - Statistiques descriptives (moyenne, écart-type, quartiles, etc.)
    """)

    st.write("#### a. Informations générales")
    st.table(df_raw.dtypes)

    st.write("#### b. Statistiques descriptives")
    printed_df = st.dataframe(df_raw.describe())

    st.write("#### c. Échantillons concrets")
    st.dataframe(df_raw.head(10))

    st.write("""
    ___
    ## 2. Analyse de la Target

    Maintenant, analysons la distribution de notre variable cible (target) pour évaluer l'équilibre des classes dans notre
    dataset.py. Un déséquilibre important pourrait nécessiter des techniques de rééchantillonnage.
    """)

    st.write("#### a. Distribution des classes")
    st.dataframe(df_target_sizes)
    expand = st.expander("Barplot", icon=":material/bar_chart:")
    expand.bar_chart(df_target_sizes)

    st.write("#### b. Équilibre des classes")

    st.write("""
    ___
    ## 3. Analyse des Features

    Étudions de manière approfondie nos features :
    1. Leur distribution
    2. Leurs relations entre-elles
    """)

    st.write("### 4.1 Distribution des Variables")

    st.write("#### a. Boîtes à moustaches (Vue d'ensemble)")
    fig = px.box(df_raw[FEATURES], title="Boxplot élégant avec Plotly")
    st.plotly_chart(fig)

    # Calcul du nombre de lignes et colonnes
    n_cols = 5
    n_rows = math.ceil(len(FEATURES) / n_cols)

    # Création des subplots
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=FEATURES,
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )

    # Pour chaque feature, créer une ligne KDE
    for i, col in enumerate(FEATURES):
        row = i // n_cols + 1
        col_pos = i % n_cols + 1

        # Conversion explicite en float64 standard et suppression des valeurs NA
        data = df_raw[col]

        # Calcul du KDE
        try:
            kde = stats.gaussian_kde(data.tolist())
            x_range = np.linspace(data.min(), data.max(), 200)
            y_kde = kde(x_range)

            # Ajout de la ligne
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_kde,
                    mode='lines',
                    name=col,
                    line=dict(color='#1f77b4'),
                    showlegend=False
                ),
                row=row,
                col=col_pos
            )
        except Exception as e:
            st.warning(f"Impossible de créer le KDE pour la colonne {col}: {str(e)}")
            continue

    # Mise à jour du layout
    fig.update_layout(
        height=400 * n_rows,
        width=1000,
        showlegend=False,
        title_text="Distribution des features",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # Mise à jour des axes
    fig.update_xaxes(title_text="Valeur", showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(title_text="Densité", showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

    # Pour l'utiliser dans Streamlit
    st.plotly_chart(fig, use_container_width=True)

    #####
    # Calcul du nombre de lignes et colonnes
    n_cols = 5
    n_rows = math.ceil(len(FEATURES) / n_cols)

    # Création des subplots
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=FEATURES,
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )

    # Génération d'une palette de couleurs en fonction du nombre de classes
    unique_targets = sorted(df_raw[TARGET].unique())
    n_classes = len(unique_targets)

    # Création d'une palette de couleurs
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    # Si on a besoin de plus de couleurs, on peut répéter la palette
    while len(colors) < n_classes:
        colors.extend(colors)
    colors = colors[:n_classes]

    # Pour chaque feature, créer des lignes KDE pour chaque classe
    for i, col in enumerate(FEATURES):
        row = i // n_cols + 1
        col_pos = i % n_cols + 1

        # Pour chaque classe
        for idx, target_class in enumerate(unique_targets):
            # Filtrer les données pour la classe actuelle
            data = df_raw[df_raw[TARGET] == target_class][col]

            try:
                # Calcul du KDE
                kde = stats.gaussian_kde(data.tolist())
                x_range = np.linspace(data.min(), data.max(), 200)
                y_kde = kde(x_range)

                # Ajout de la ligne
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=y_kde,
                        mode='lines',
                        name=f'Classe {target_class}',
                        line=dict(color=colors[idx]),
                        showlegend=True if i == 0 else False  # Légende uniquement pour le premier plot
                    ),
                    row=row,
                    col=col_pos
                )
            except Exception as e:
                st.warning(f"Impossible de créer le KDE pour la colonne {col}, classe {target_class}: {str(e)}")
                continue

    # Mise à jour du layout
    fig.update_layout(
        height=400 * n_rows,
        width=1000,
        title_text="Distribution des features par classe",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1
        )
    )

    # Mise à jour des axes
    fig.update_xaxes(
        title_text="Valeur",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)'
    )
    fig.update_yaxes(
        title_text="Densité",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)'
    )

    # Affichage dans Streamlit
    st.plotly_chart(fig, use_container_width=True)


if not dataset:
    render_no_dataset_skeleton()
else:
    page()
