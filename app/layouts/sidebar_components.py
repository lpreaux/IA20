"""
Composants réutilisables pour les sidebars.
"""
import streamlit as st

from dataset.state import DatasetState

from dataset.forms.config import dataset_config_form


def render_dataset_manager():
    """Affiche le gestionnaire de dataset global."""
    dataset = DatasetState.get_dataset()

    if dataset:
        st.info(f"""
        ### Dataset actuel
        
        📊 **{dataset.filename}**  
        📈 {len(dataset.data.columns)} colonnes | {len(dataset.data)} lignes
        """)

        if st.button("🔄 Changer de dataset", use_container_width=True):
            dataset_config_form()
    else:
        st.warning("""
        ### Aucun dataset
        
        Configurez un jeu de données pour commencer
        """)

        if st.button("📤 Configurer un dataset", type="primary", use_container_width=True):
            dataset_config_form()


def render_dataset_stats():
    """Affiche les statistiques du dataset dans la sidebar.
    Retourne True si des statistiques ont été affichées."""
    dataset = DatasetState.get_dataset()
    if dataset:
        st.sidebar.markdown("### Statistiques")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric(
                "Variables",
                len(dataset.features_columns),
                "features",
                delta_color="normal",
                help="Nombre de variables explicatives",
            )
        with col2:
            st.metric(
                "Cibles",
                len(dataset.target_columns),
                "targets",
                delta_color="normal",
                help="Nombre de variables à prédire",
            )


def render_quick_actions(dataset):
    """Affiche les actions rapides dans la sidebar."""
    with st.sidebar:
        st.markdown("### Actions rapides")
        col1, col2 = st.columns(2)
        with col1:
            st.button("🔍 Explorer", use_container_width=True)
        with col2:
            st.button("🧹 Nettoyer", use_container_width=True)
