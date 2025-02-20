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


def create_distribution_plot(data, features, target=None):
    """Crée un plot de distribution pour chaque feature."""
    n_cols = 3
    n_rows = math.ceil(len(features) / n_cols)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=features,
        horizontal_spacing=0.1,
        vertical_spacing=0.1
    )

    for i, col in enumerate(features):
        row = i // n_cols + 1
        col_pos = i % n_cols + 1

        if target is not None:
            # Distribution par classe
            for target_class in data[target].unique():
                subset = data[data[target] == target_class][col]
                try:
                    kde = stats.gaussian_kde(subset)
                    x_range = np.linspace(subset.min(), subset.max(), 200)
                    fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=kde(x_range),
                            name=f"{target_class}",
                            showlegend=(i == 0),
                            line=dict(width=2)
                        ),
                        row=row,
                        col=col_pos
                    )
                except Exception:
                    continue
        else:
            # Distribution globale
            try:
                kde = stats.gaussian_kde(data[col])
                x_range = np.linspace(data[col].min(), data[col].max(), 200)
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=kde(x_range),
                        showlegend=False,
                        line=dict(color='#1f77b4', width=2)
                    ),
                    row=row,
                    col=col_pos
                )
            except Exception:
                continue

    fig.update_layout(
        height=300 * n_rows,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.05)',
        font=dict(family="sans-serif"),
        showlegend=target is not None,
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1
        )
    )

    return fig


def create_correlation_heatmap(data, features):
    """Crée une heatmap de corrélation."""
    corr_matrix = data[features].corr()

    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    corr_matrix_masked = corr_matrix.copy()
    corr_matrix_masked[mask] = np.nan

    heat_map_text = np.round(corr_matrix_masked, 2)
    heat_map_text[mask] = ""

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix_masked,
        x=features,
        y=features,
        text=heat_map_text,
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        colorscale='RdBu_r',
        zmid=0,
        zmin=-1,
        zmax=1
    ))

    fig.update_layout(
        title='Matrice de Corrélation',
        height=600,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.05)'
    )

    return fig


def page():
    """Rendu principal de la page d'exploration."""
    dataset = DatasetState.get_dataset()
    df = dataset.data

    # En-tête
    st.markdown("""
        # 📊 Analyse Exploratoire des Données

        Explorez et comprenez votre jeu de données à travers différentes visualisations interactives.
        Utilisez les filtres dans la barre latérale pour personnaliser votre analyse.
    """)

    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Observations",
            len(df),
            help="Nombre total de lignes dans le dataset"
        )
    with col2:
        st.metric(
            "Features",
            len(dataset.features_columns),
            help="Nombre de variables explicatives"
        )
    with col3:
        st.metric(
            "Target(s)",
            len(dataset.target_columns),
            help="Nombre de variables cibles"
        )
    with col4:
        if len(dataset.target_columns) > 0:
            st.metric(
                "Classes",
                len(df[dataset.target_columns[0]].unique()),
                help="Nombre de classes uniques dans la target principale"
            )

    # Onglets principaux
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Vue d'ensemble",
        "📊 Distributions",
        "🔗 Corrélations",
        "🎯 Pairplot"
    ])

    with tab1:
        st.markdown("### Structure des données")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Types des variables")
            st.dataframe(
                pd.DataFrame({
                    "Type": df.dtypes,
                    "Non-nulls": df.count(),
                    "Nulls (%)": (df.isna().sum() / len(df) * 100).round(2)
                }),
                use_container_width=True
            )

        with col2:
            st.markdown("#### Statistiques descriptives")
            st.dataframe(
                df.describe().round(2),
                use_container_width=True
            )

        st.markdown("#### Aperçu des données")
        st.dataframe(df.head())

    with tab2:
        # Filtres pour la distribution
        selected_features = st.multiselect(
            "Sélectionner les variables à analyser",
            options=dataset.features_columns,
            default=dataset.features_columns[:6]
        )

        if len(selected_features) > 0:
            # Distribution univariée
            st.markdown("### 📈 Distributions univariées")
            fig_dist = create_distribution_plot(df, selected_features)
            st.plotly_chart(fig_dist, use_container_width=True)

            # Distribution par target si disponible
            if len(dataset.target_columns) > 0:
                st.markdown("### 🎯 Distributions par classe")
                fig_dist_target = create_distribution_plot(
                    df,
                    selected_features,
                    dataset.target_columns[0]
                )
                st.plotly_chart(fig_dist_target, use_container_width=True)

    with tab3:
        if len(dataset.features_columns) > 1:
            st.markdown("### 🔗 Analyse des corrélations")

            # Sélection des features pour la corrélation
            corr_features = st.multiselect(
                "Sélectionner les variables pour la matrice de corrélation",
                options=dataset.features_columns,
                default=dataset.features_columns[:8]
            )

            if len(corr_features) > 1:
                fig_corr = create_correlation_heatmap(df, corr_features)
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Sélectionnez au moins 2 variables pour voir leurs corrélations")
        else:
            st.info("Il faut au moins 2 variables pour analyser les corrélations")

    with tab4:
        st.markdown("### 🎯 Scatter Matrix")

        # Sélection des features pour le pairplot
        pair_features = st.multiselect(
            "Sélectionner les variables pour la matrice de visualisation (max 6 recommandé)",
            options=dataset.features_columns,
            default=dataset.features_columns[:4],
            key="pairplot_features"
        )

        if len(pair_features) > 1:
            # Création du scatter matrix
            fig = px.scatter_matrix(
                df,
                dimensions=pair_features,
                color=dataset.target_columns[0] if dataset.target_columns else None,
                title="Matrice de visualisation des relations entre variables",
                labels={col: col.replace('_', ' ').title() for col in pair_features}
            )

            # Personnalisation du graphique
            fig.update_traces(
                diagonal_visible=False,
                showupperhalf=False,
                hovertemplate='%{xaxis.title.text}: %{x}<br>%{yaxis.title.text}: %{y}<br>'
            )

            # Mise en page
            fig.update_layout(
                height=800,
                width=800,
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.05)',
                title_x=0.5,
            )

            st.plotly_chart(fig, use_container_width=True)

            if len(pair_features) > 6:
                st.warning(
                    "⚠️ Un grand nombre de variables peut rendre le graphique difficile à interpréter. Considérez de réduire la sélection à 6 variables maximum pour une meilleure lisibilité.")
        else:
            st.info("Sélectionnez au moins 2 variables pour visualiser leurs relations")


def render_no_dataset_skeleton():
    """Affiche un squelette visuel de la page quand aucun dataset n'est chargé."""
    # En-tête
    st.markdown("# 📊 Analyse Exploratoire des Données")
    st.markdown("""
        Explorez et comprenez votre jeu de données à travers différentes visualisations interactives.
        Utilisez les filtres dans la barre latérale pour personnaliser votre analyse.
    """)

    # Message principal pour charger un dataset
    with st.container(border=True):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### :ghost: Aucun jeu de données")
            st.caption("Chargez un jeu de données pour commencer l'exploration")
            st.button(
                "📤 Charger un jeu de données",
                type="primary",
                on_click=dataset_config_form,
                use_container_width=True
            )

    # Métriques placeholder
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Observations", "---")
    with col2:
        st.metric("Features", "---")
    with col3:
        st.metric("Target(s)", "---")
    with col4:
        st.metric("Classes", "---")

    # Onglets placeholder
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Vue d'ensemble",
        "📊 Distributions",
        "🔗 Corrélations",
        "🎯 Pairplot"
    ])

    with tab1:
        st.markdown("### Structure des données")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Types des variables")
            placeholder_df = pd.DataFrame({
                "Type": ["---"],
                "Non-nulls": ["---"],
                "Nulls (%)": ["---"]
            })
            st.dataframe(placeholder_df, use_container_width=True)

        with col2:
            st.markdown("#### Statistiques descriptives")
            st.dataframe(pd.DataFrame({"": ["---"]}), use_container_width=True)

        st.markdown("#### Aperçu des données")
        st.dataframe(pd.DataFrame({"": ["---"]}), use_container_width=True)

    with tab2:
        st.markdown("### 📈 Distributions")
        st.info("Les distributions des variables seront affichées ici une fois un dataset chargé.")
        # Placeholder pour le graphique
        with st.container(border=True, height=300):
            st.markdown("#### Graphique de distribution")
            st.caption("Chargez un dataset pour visualiser les distributions")

    with tab3:
        st.markdown("### 🔗 Corrélations")
        st.info("La matrice de corrélation sera affichée ici une fois un dataset chargé.")
        # Placeholder pour la heatmap
        with st.container(border=True, height=300):
            st.markdown("#### Matrice de corrélation")
            st.caption("Chargez un dataset pour visualiser les corrélations")

    with tab4:
        st.markdown("### 🎯 Scatter Matrix")
        st.info("La matrice de visualisation sera affichée ici une fois un dataset chargé.")
        # Placeholder pour le scatter matrix
        with st.container(border=True, height=300):
            st.markdown("#### Matrice de visualisation")
            st.caption("Chargez un dataset pour visualiser les relations entre variables")


dataset = DatasetState.get_dataset()

if not dataset:
    render_no_dataset_skeleton()
else:
    page()