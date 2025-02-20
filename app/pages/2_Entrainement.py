import math

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dataset.state import DatasetState
from dataset.forms.config import dataset_config_form
import utils.training as train


def display_training_overview(dataset):
    """Affiche une vue d'ensemble des données d'entraînement."""
    st.markdown("""
    # 🤖 Entraînement des Modèles

    Cette page vous permet d'entraîner et d'évaluer différents modèles de machine learning 
    sur vos données. Vous pouvez :
    - Sélectionner et configurer différents modèles
    - Visualiser les performances des modèles
    - Comparer les résultats
    """)

    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Observations",
            f"{len(dataset.data):,}",
            help="Nombre total d'observations disponibles"
        )
    with col2:
        st.metric(
            "Variables",
            len(dataset.features_columns),
            help="Nombre de variables explicatives utilisées"
        )
    with col3:
        # Calcul de la proportion des classes
        target = dataset.target_columns[0]
        class_balance = dataset.data[target].value_counts(normalize=True)
        majority_class_pct = (class_balance.max() * 100)
        st.metric(
            "Balance des classes",
            f"{majority_class_pct:.1f}%",
            help="Pourcentage de la classe majoritaire"
        )
    with col4:
        st.metric(
            "Classes",
            len(dataset.data[target].unique()),
            help="Nombre de classes à prédire"
        )


def configure_model_parameters(model_name, model):
    """Configure les paramètres du modèle sélectionné."""
    st.markdown("### ⚙️ Configuration du modèle")

    params_changed = False
    new_params = {}

    if model_name == "Logistic Regression":
        col1, col2 = st.columns(2)
        with col1:
            new_c = st.number_input(
                "Paramètre de régularisation (C)",
                min_value=0.001,
                max_value=10.0,
                value=float(model.C),
                help="Plus C est grand, moins la régularisation est forte"
            )
            if new_c != model.C:
                params_changed = True
                new_params['C'] = new_c

        with col2:
            new_solver = st.selectbox(
                "Solveur",
                options=["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
                index=["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"].index(model.solver),
                help="Algorithme à utiliser pour l'optimisation"
            )
            if new_solver != model.solver:
                params_changed = True
                new_params['solver'] = new_solver

    elif model_name == "Decision Tree":
        col1, col2, col3 = st.columns(3)
        with col1:
            new_depth = st.number_input(
                "Profondeur maximale",
                min_value=1,
                max_value=20,
                value=model.max_depth if model.max_depth else 5,
                help="Profondeur maximale de l'arbre"
            )
            if new_depth != model.max_depth:
                params_changed = True
                new_params['max_depth'] = new_depth

        with col2:
            new_min_samples = st.number_input(
                "Échantillons minimum par feuille",
                min_value=1,
                max_value=50,
                value=model.min_samples_leaf,
                help="Nombre minimum d'échantillons requis dans une feuille"
            )
            if new_min_samples != model.min_samples_leaf:
                params_changed = True
                new_params['min_samples_leaf'] = new_min_samples

        with col3:
            new_criterion = st.selectbox(
                "Critère de division",
                options=["gini", "entropy"],
                index=["gini", "entropy"].index(model.criterion),
                help="Mesure de la qualité d'une division"
            )
            if new_criterion != model.criterion:
                params_changed = True
                new_params['criterion'] = new_criterion

    elif model_name == "Forest":
        col1, col2, col3 = st.columns(3)
        with col1:
            new_n_trees = st.number_input(
                "Nombre d'arbres",
                min_value=10,
                max_value=200,
                value=model.n_estimators,
                help="Nombre d'arbres dans la forêt"
            )
            if new_n_trees != model.n_estimators:
                params_changed = True
                new_params['n_estimators'] = new_n_trees

        with col2:
            new_depth = st.number_input(
                "Profondeur maximale",
                min_value=1,
                max_value=20,
                value=model.max_depth if model.max_depth else 5,
                help="Profondeur maximale des arbres"
            )
            if new_depth != model.max_depth:
                params_changed = True
                new_params['max_depth'] = new_depth

        with col3:
            new_features = st.selectbox(
                "Sélection des features",
                options=["sqrt", "log2", None],
                format_func=lambda x: "Auto" if x is None else x,
                index=["sqrt", "log2", None].index(model.max_features),
                help="Nombre de features à considérer pour chaque division"
            )
            if new_features != model.max_features:
                params_changed = True
                new_params['max_features'] = new_features

    return params_changed, new_params


def plot_confusion_matrix(conf_matrix, class_names=None):
    """Crée une matrice de confusion interactive et stylisée."""
    if class_names is None:
        conf = DatasetState.get_dataset()
        target_col = conf.target_columns[0]
        class_names = conf.data[target_col].unique()

    fig = go.Figure()

    # Création de la heatmap
    fig.add_trace(go.Heatmap(
        z=conf_matrix,
        x=class_names,
        y=class_names,
        text=conf_matrix,
        texttemplate="%{z}",
        textfont={"size": 16},
        colorscale=[[0, '#1f2937'], [1, '#3b82f6']],
        showscale=True,
    ))

    # Calcul des métriques par classe
    n_classes = len(conf_matrix)
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)

    for i in range(n_classes):
        precision[i] = conf_matrix[i, i] / conf_matrix[:, i].sum()
        recall[i] = conf_matrix[i, i] / conf_matrix[i, :].sum()

    # Annotations pour précision et rappel
    for i in range(n_classes):
        for j in range(n_classes):
            fig.add_annotation(
                x=class_names[j],
                y=class_names[i],
                text=f"<b>{conf_matrix[i, j]}</b>",
                showarrow=False,
                font=dict(color='white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black')
            )

    # Configuration de la mise en page
    fig.update_layout(
        title={
            'text': "Matrice de Confusion",
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20)
        },
        xaxis_title="Prédiction",
        yaxis_title="Réalité",
        xaxis={'side': 'bottom'},
        template="plotly_dark",
        height=500,
        width=700,
        margin=dict(t=100, l=70, r=70, b=70),
    )

    return fig


def plot_feature_importance(model, features):
    """Crée un graphique interactif de l'importance des features."""
    importance = None

    if hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_

    if importance is None:
        return None

    # Création du DataFrame
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    })

    # Tri et normalisation
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=True)
    feature_importance_df['Importance_Norm'] = (feature_importance_df['Importance'] -
                                                feature_importance_df['Importance'].min()) / (
                                                       feature_importance_df['Importance'].max() -
                                                       feature_importance_df['Importance'].min())

    # Création du graphique
    fig = go.Figure()

    # Ajout des barres
    fig.add_trace(go.Bar(
        x=feature_importance_df['Importance'],
        y=feature_importance_df['Feature'],
        orientation='h',
        marker=dict(
            color=feature_importance_df['Importance_Norm'],
            colorscale='Blues',
            line=dict(width=1, color='white')
        ),
        text=feature_importance_df['Importance'].round(3),
        textposition='outside',
        textfont=dict(size=12),
    ))

    # Mise en page
    fig.update_layout(
        title={
            'text': "Importance des Variables",
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20)
        },
        xaxis_title="Importance",
        yaxis_title=None,
        template="plotly_dark",
        height=max(400, len(features) * 30),
        showlegend=False,
        margin=dict(l=20, r=120, t=60, b=40),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.1)'),
    )

    return fig


def plot_cv_scores(cv_scores, report):
    """Crée une visualisation détaillée des scores de validation croisée."""
    fig = go.Figure()

    # Scores individuels
    fig.add_trace(go.Scatter(
        x=list(range(1, len(cv_scores) + 1)),
        y=cv_scores,
        mode='lines+markers',
        name='Score CV',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=10, symbol='circle'),
    ))

    # Ligne moyenne
    mean_score = cv_scores.mean()
    fig.add_hline(
        y=mean_score,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"Moyenne: {mean_score:.3f}",
        annotation_position="top right"
    )

    # Intervalle de confiance
    std_score = cv_scores.std()
    fig.add_traces([
        go.Scatter(
            name='Intervalle de confiance',
            x=list(range(1, len(cv_scores) + 1)),
            y=[mean_score + 2 * std_score] * len(cv_scores),
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Intervalle de confiance',
            x=list(range(1, len(cv_scores) + 1)),
            y=[mean_score - 2 * std_score] * len(cv_scores),
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(239, 68, 68, 0.2)',
            fill='tonexty',
            showlegend=False
        )
    ])

    # Mise en page
    fig.update_layout(
        title={
            'text': "Scores de Validation Croisée",
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20)
        },
        xaxis_title="Fold",
        yaxis_title="Score",
        template="plotly_dark",
        height=400,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)"
        ),
        margin=dict(l=50, r=50, t=80, b=50),
    )

    # Grille et axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255, 255, 255, 0.1)',
        tickmode='linear',
        tick0=1,
        dtick=1
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255, 255, 255, 0.1)',
    )

    return fig


def draw_cv_results(report, cv_scores):
    """Affiche les résultats détaillés de la validation croisée."""
    # Graphique principal des scores CV
    fig_cv = plot_cv_scores(cv_scores, report)
    st.plotly_chart(fig_cv, use_container_width=True)

    # Statistiques détaillées
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Score CV Moyen",
            f"{cv_scores.mean():.3f}",
            help="Moyenne des scores de validation croisée"
        )

    with col2:
        st.metric(
            "Écart-type",
            f"{cv_scores.std():.3f}",
            help="Écart-type des scores de validation croisée"
        )

    with col3:
        st.metric(
            "Intervalle de confiance (95%)",
            f"[{cv_scores.mean() - 2 * cv_scores.std():.3f}, {cv_scores.mean() + 2 * cv_scores.std():.3f}]",
            help="Intervalle de confiance à 95% pour le score moyen"
        )


def draw_performance_metrics(report):
    """Affiche les métriques de performance détaillées."""
    st.subheader("Métriques de Performance")

    metrics_df = pd.DataFrame({
        'Métrique': ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1-score (macro)'],
        'Score': [
            report['accuracy'],
            report['macro avg']['precision'],
            report['macro avg']['recall'],
            report['macro avg']['f1-score']
        ]
    })

    # Création d'un graphique en jauge pour chaque métrique
    fig = go.Figure()

    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']

    for idx, (metric, score) in enumerate(zip(metrics_df['Métrique'], metrics_df['Score'])):
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            domain={'row': 0, 'column': idx},
            title={'text': metric},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': colors[idx]},
                'steps': [
                    {'range': [0, 60], 'color': 'rgba(255, 255, 255, 0.1)'},
                    {'range': [60, 80], 'color': 'rgba(255, 255, 255, 0.2)'},
                    {'range': [80, 100], 'color': 'rgba(255, 255, 255, 0.3)'}
                ],
            }
        ))

    # Mise en page
    fig.update_layout(
        grid={'rows': 1, 'columns': 4, 'pattern': "independent"},
        template="plotly_dark",
        height=250,
        margin=dict(l=30, r=30, t=30, b=30),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Détails par classe
    st.subheader("Détails par classe")

    # Filtrage et formatage du rapport détaillé
    detailed_report = {k: v for k, v in report.items()
                       if k not in ['accuracy', 'macro avg', 'weighted avg']}

    detailed_df = pd.DataFrame(detailed_report).T.round(3)
    detailed_df.columns = ['Précision', 'Rappel', 'F1-Score', 'Support']

    st.dataframe(
        detailed_df.style.background_gradient(cmap='Blues', subset=['Précision', 'Rappel', 'F1-Score']),
        use_container_width=True
    )


def compute_tree_depth(children_left, children_right, node_id=0):
    """Calcule la profondeur maximale de l'arbre."""
    if children_left[node_id] == -1 and children_right[node_id] == -1:
        return 0
    left_depth = (compute_tree_depth(children_left, children_right, children_left[node_id])
                  if children_left[node_id] != -1 else -1)
    right_depth = (compute_tree_depth(children_left, children_right, children_right[node_id])
                   if children_right[node_id] != -1 else -1)
    return max(left_depth, right_depth) + 1


def convert_tree_to_plotly_data(tree_model, feature_names, class_names):
    """Convertit un arbre de décision en données pour Plotly avec espacement amélioré."""
    n_nodes = tree_model.tree_.node_count
    children_left = tree_model.tree_.children_left
    children_right = tree_model.tree_.children_right
    feature = tree_model.tree_.feature
    threshold = tree_model.tree_.threshold
    values = tree_model.tree_.value

    # Calcul de la profondeur maximale pour l'espacement dynamique
    max_depth = compute_tree_depth(children_left, children_right)
    vertical_spacing = -0.15 - (0.05 * max_depth)  # Ajustement dynamique de l'espacement

    def compute_node_depths(node_id=0, depth=0, depths=None):
        """Calcule la profondeur de chaque nœud."""
        if depths is None:
            depths = {}
        depths[node_id] = depth
        if children_left[node_id] != -1:
            compute_node_depths(children_left[node_id], depth + 1, depths)
        if children_right[node_id] != -1:
            compute_node_depths(children_right[node_id], depth + 1, depths)
        return depths

    def compute_node_positions():
        """Calcule les positions x,y de chaque nœud avec espacement amélioré."""
        positions = {}
        depths = compute_node_depths()
        max_depth = max(depths.values())

        # Pour chaque niveau, répartir les nœuds horizontalement
        level_counts = {}
        level_current = {}

        # Initialisation des compteurs
        for depth in range(max_depth + 1):
            level_counts[depth] = sum(1 for node_depth in depths.values() if node_depth == depth)
            level_current[depth] = 0

        # Calcul des positions avec espacement vertical dynamique
        def assign_positions(node_id=0):
            depth = depths[node_id]
            x_spacing = 1.0 / (level_counts[depth] + 1)
            level_current[depth] += 1
            x_pos = level_current[depth] * x_spacing
            y_pos = depth * vertical_spacing

            positions[node_id] = {
                'x': x_pos,
                'y': y_pos
            }

            if children_left[node_id] != -1:
                assign_positions(children_left[node_id])
            if children_right[node_id] != -1:
                assign_positions(children_right[node_id])

        assign_positions()
        return positions

    def get_node_color(value_array):
        """Détermine la couleur du nœud basée sur la distribution des classes."""
        total = value_array.sum()
        if total == 0:
            return 'rgba(59, 130, 246, 0.2)'
        props = value_array / total

        if props.max() > 0.75:
            return 'rgba(34, 197, 94, 0.8)'  # Vert pour forte probabilité
        elif props.max() > 0.5:
            return 'rgba(59, 130, 246, 0.8)'  # Bleu pour probabilité moyenne
        return 'rgba(249, 115, 22, 0.8)'  # Orange pour faible probabilité

    def format_node_text(node_id, feature_idx, threshold, value_array):
        """Formate le texte du nœud avec une meilleure lisibilité."""
        if feature_idx != -2:  # Nœud interne
            feature_name = feature_names[feature_idx]
            if len(feature_name) > 15:
                feature_name = feature_name[:12] + "..."
            text = f'<b>{feature_name}</b><br>≤ {threshold:.2f}'
        else:  # Feuille
            value_array = value_array.flatten()
            total = value_array.sum()
            if total > 0:
                props = value_array / total
                majority_class = class_names[props.argmax()]
                text = f'<b>Classe: {majority_class}</b><br>{props.max():.0%}'
            else:
                text = "Vide"

        # Calcul de la taille du nœud adaptatif
        text_length = len(text.replace('<b>', '').replace('</b>', ''))
        base_size = 40
        size_factor = min(max(text_length / 10, 1), 2)
        node_size = int(base_size * size_factor)

        return text, node_size

    # Calcul des positions
    node_positions = compute_node_positions()

    # Création des nœuds et des connexions
    nodes = []
    edge_traces = []

    for i in range(n_nodes):
        pos = node_positions[i]
        # Configuration du nœud
        text, node_size = format_node_text(i, feature[i], threshold[i], values[i])
        shape = 'circle' if feature[i] != -2 else 'square'

        nodes.append(dict(
            x=[pos['x']],
            y=[pos['y']],
            text=text,
            mode='markers+text',
            textposition='middle center',
            hoverinfo='text',
            marker=dict(
                symbol=shape,
                size=node_size,
                color=get_node_color(values[i].flatten()),
                line=dict(color='white', width=2)
            ),
            name=f'Node {i}'
        ))

        # Style des connexions amélioré
        line_style = dict(
            color='rgba(255, 255, 255, 0.5)',
            width=1.5,
            dash='dot'  # Ajout d'un style pointillé pour mieux visualiser les connexions
        )

        # Ajout des connexions aux enfants
        if children_left[i] != -1:
            child_pos = node_positions[children_left[i]]
            edge_traces.append(go.Scatter(
                x=[pos['x'], child_pos['x']],
                y=[pos['y'], child_pos['y']],
                mode='lines',
                line=line_style,
                hoverinfo='none',
                showlegend=False
            ))

        if children_right[i] != -1:
            child_pos = node_positions[children_right[i]]
            edge_traces.append(go.Scatter(
                x=[pos['x'], child_pos['x']],
                y=[pos['y'], child_pos['y']],
                mode='lines',
                line=line_style,
                hoverinfo='none',
                showlegend=False
            ))

    return nodes, edge_traces, max_depth


def plot_interactive_tree(model, feature_names, class_names):
    """Crée une visualisation interactive de l'arbre de décision avec hauteur adaptative."""
    nodes, edge_traces, tree_depth = convert_tree_to_plotly_data(model, feature_names, class_names)

    # Calcul de la hauteur dynamique
    base_height = 600  # Hauteur minimale
    height_per_level = 120  # Pixels additionnels par niveau
    dynamic_height = max(base_height, base_height + (tree_depth * height_per_level))

    fig = go.Figure()

    # Ajout des connexions
    for edge in edge_traces:
        fig.add_trace(edge)

    # Ajout des nœuds
    for node in nodes:
        fig.add_trace(go.Scatter(
            x=node['x'],
            y=node['y'],
            mode=node['mode'],
            text=node['text'],
            textposition=node['textposition'],
            hoverinfo=node['hoverinfo'],
            marker=node['marker'],
            name=node['name']
        ))

    # Configuration de la mise en page
    fig.update_layout(
        title={
            'text': "Visualisation Interactive de l'Arbre de Décision",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        showlegend=False,
        hovermode='closest',
        margin=dict(
            b=50,  # Augmentation des marges pour les grands arbres
            l=50,
            r=50,
            t=100
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.1, 1.1]  # Ajout d'une marge horizontale
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="x",
            scaleratio=1  # Maintient un ratio d'aspect carré
        ),
        height=dynamic_height,
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0.1)',
        paper_bgcolor='rgba(0,0,0,0)',
        annotations=[
            dict(
                text=f"Profondeur: {tree_depth} niveaux",
                xref="paper",
                yref="paper",
                x=0,
                y=1,
                showarrow=False,
                font=dict(size=12, color="gray")
            )
        ]
    )

    # Ajout de boutons de zoom prédéfinis
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Réinitialiser la vue",
                        method="relayout",
                        args=[{
                            "xaxis.range": [-0.1, 1.1],
                            "yaxis.range": [tree_depth * (-0.15 - 0.05 * tree_depth), 0.1]
                        }]
                    )
                ],
                x=0.05,
                y=0.05,
                xanchor="left",
                yanchor="bottom",
            )
        ]
    )

    return fig


def plot_interactive_forest(forest_model, feature_names, class_names, n_trees=4):
    """Crée une visualisation interactive d'une forêt aléatoire."""
    # Création des sous-figures
    fig = make_subplots(
        rows=math.ceil(n_trees / 2),
        cols=2,
        subplot_titles=[f'Arbre {i + 1}' for i in range(n_trees)]
    )

    # Calculer la profondeur maximale parmi tous les arbres sélectionnés
    max_tree_depth = 0

    # Ajout de chaque arbre
    for i in range(n_trees):
        tree_model = forest_model.estimators_[i]
        nodes, edge_traces, tree_depth = convert_tree_to_plotly_data(tree_model, feature_names, class_names)
        max_tree_depth = max(max_tree_depth, tree_depth)

        row = i // 2 + 1
        col = i % 2 + 1

        # Ajout des edges
        for edge in edge_traces:
            fig.add_trace(edge, row=row, col=col)

        # Ajout des nodes
        for node in nodes:
            fig.add_trace(
                go.Scatter(
                    x=node['x'],
                    y=node['y'],
                    mode=node['mode'],
                    text=node['text'],
                    hoverinfo=node['hoverinfo'],
                    marker=node['marker'],
                    name=node['name']
                ),
                row=row,
                col=col
            )

    # Calcul de la hauteur dynamique basée sur la profondeur maximale des arbres
    base_height = 300  # Hauteur de base par ligne
    height_per_depth = 50  # Pixels additionnels par niveau de profondeur
    dynamic_height = base_height * math.ceil(n_trees / 2) + height_per_depth * max_tree_depth

    # Mise en page
    fig.update_layout(
        height=dynamic_height,
        showlegend=False,
        template="plotly_dark",
        title=dict(
            text="Visualisation Interactive de la Forêt Aléatoire",
            x=0.5,
            y=0.99
        )
    )

    # Configuration des axes
    for i in range(1, n_trees + 1):
        row = math.ceil(i / 2)
        col = ((i - 1) % 2) + 1

        # Configuration des axes X
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.1, 1.1],  # Ajout d'une marge horizontale
            row=row,
            col=col
        )

        # Configuration des axes Y
        fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[max_tree_depth * (-0.15 - 0.05 * max_tree_depth), 0.1],  # Ajustement basé sur la profondeur
            row=row,
            col=col
        )

    return fig


def display_model_results(model, model_name):
    """Affiche les résultats du modèle."""
    # Entraînement et évaluation
    model, y_pred, cv_scores, report, conf_matrix = train.train_and_evaluate_model(model)

    # Métriques de validation croisée
    mean_cv = cv_scores.mean()
    std_cv = cv_scores.std()

    # Affichage des métriques CV
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Score CV Moyen",
            f"{mean_cv:.3f}",
            help="Score moyen de validation croisée"
        )
    with col2:
        st.metric(
            "Écart-type CV",
            f"{std_cv:.3f}",
            help="Écart-type des scores de validation croisée"
        )
    with col3:
        st.metric(
            "Intervalle de confiance",
            f"[{mean_cv - 2 * std_cv:.3f}, {mean_cv + 2 * std_cv:.3f}]",
            help="Intervalle de confiance à 95% du score CV"
        )

    # Organisation en onglets pour les résultats détaillés
    tab1, tab2, tab3 = st.tabs([
        "📊 Performances",
        "🎯 Validation croisée",
        "📈 Visualisations"
    ])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Métriques globales")
            draw_performance_metrics(report)
        with col2:
            st.markdown("### Matrice de confusion")
            conf_matrix_fig = plot_confusion_matrix(conf_matrix)
            st.plotly_chart(conf_matrix_fig, use_container_width=True)

    with tab2:
        st.markdown("### Résultats de la validation croisée")
        draw_cv_results(report, cv_scores)

        # Ajout d'un graphique de distribution des scores CV
        fig = go.Figure()
        fig.add_trace(go.Violin(
            y=cv_scores,
            box_visible=True,
            line_color='#1f77b4',
            fillcolor='#1f77b4',
            opacity=0.6,
            name='Distribution des scores'
        ))
        fig.update_layout(
            title="Distribution des scores de validation croisée",
            yaxis_title="Score",
            showlegend=False,
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Importance des variables")
        feature_importance = plot_feature_importance(model, train.get_features())
        st.plotly_chart(feature_importance, use_container_width=True)

        # Affichage des arbres pour les modèles appropriés
        if model_name in ["Decision Tree", "Forest"]:
            st.markdown("### Visualisation de la structure")
            conf = DatasetState.get_dataset()
            target_col = conf.target_columns[0]
            vals = conf.data[target_col].unique()

            if model_name == "Decision Tree":
                st.markdown("#### Visualisation interactive de l'arbre de décision")
                tree_fig = plot_interactive_tree(model, train.get_features(), vals)
                st.plotly_chart(tree_fig, use_container_width=True)

            elif model_name == "Forest":
                st.markdown("#### Visualisation interactive de la forêt aléatoire")
                col1, col2 = st.columns([2, 1])
                with col1:
                    n_trees = st.slider(
                        "Nombre d'arbres à afficher",
                        min_value=2,
                        max_value=6,
                        value=4,
                        help="Sélectionnez le nombre d'arbres à visualiser"
                    )
                with col2:
                    starting_tree = st.number_input(
                        "Premier arbre à afficher",
                        min_value=0,
                        max_value=model.n_estimators - n_trees,
                        value=0,
                        help="Index du premier arbre à afficher"
                    )
                forest_fig = plot_interactive_forest(model, train.get_features(), vals, n_trees)
                st.plotly_chart(forest_fig, use_container_width=True)



def main():
    """Point d'entrée principal de la page."""
    dataset = DatasetState.get_dataset()

    if not dataset:
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("### :ghost: Aucun jeu de données")
                st.caption("Chargez un jeu de données pour commencer l'entraînement")
                st.button(
                    "📤 Charger un jeu de données",
                    type="primary",
                    on_click=dataset_config_form,
                    use_container_width=True
                )
        return

    # Affichage de la vue d'ensemble
    display_training_overview(dataset)

    # Sélection et configuration du modèle
    with st.sidebar:
        st.markdown("### 🔧 Configuration")

        models = train.get_models()
        model_names = {
            "Logistic Regression": "Régression Logistique",
            "Decision Tree": "Arbre de Décision",
            "Forest": "Forêt Aléatoire"
        }
        model_descriptions = {
            "Logistic Regression": "Modèle linéaire pour la classification binaire ou multi-classe",
            "Decision Tree": "Arbre de décision unique, facile à interpréter",
            "Forest": "Ensemble d'arbres de décision, généralement plus performant mais moins interprétable"
        }

        selected_model_name = st.selectbox(
            "Sélection du modèle",
            options=list(models.keys()),
            format_func=lambda x: model_names.get(x, x),
            help="Choisissez le type de modèle à entraîner"
        )

        # Afficher la description après la sélection
        if selected_model_name in model_descriptions:
            st.caption(model_descriptions[selected_model_name])

        model = models[selected_model_name]

        # Configuration des paramètres
        params_changed, new_params = configure_model_parameters(selected_model_name, model)

        if params_changed:
            if st.button("🔄 Mettre à jour le modèle", type="primary"):
                for param, value in new_params.items():
                    setattr(model, param, value)
                st.success("✅ Paramètres mis à jour!")
                st.rerun()

    # Affichage des résultats
    display_model_results(model, selected_model_name)


main()
