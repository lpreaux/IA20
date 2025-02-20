import streamlit as st
from dataset.state import DatasetState
from dataset.forms.config import dataset_config_form
import utils.training as train

def main():
    models = train.get_models()

    cols = st.columns(3)
    i = 0
    for key, model in models.items():
        model, y_pred, cv_scores, report, conf_matrix = train.train_and_evaluate_model(model)
        with cols[i%3]:
            st.subheader(f"Matrice de confusion pour les valeurs par défaut de {key}")
            conf_matrix_fig = train.plot_confusion_matrix(conf_matrix)
            st.plotly_chart(conf_matrix_fig, use_container_width=True)

        i += 1

def render_no_dataset_skeleton():
    """Affiche un squelette de la page quand aucun dataset n'est chargé."""
    with st.container():
        # En-tête
        st.markdown("# Comparaison")

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

# Point d'entrée de la page
dataset = DatasetState.get_dataset()

if not dataset:
    render_no_dataset_skeleton()
else:
    main()