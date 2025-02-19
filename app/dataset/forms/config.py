from typing import Optional

import streamlit as st
from ..state import DatasetState
from ..services import DataFrameService, FileTypeDetector
from .state import FormState
from .components.column_selector import ColumnSelector
from .components.data_preview import DataPreview


def dataset_form_button(label: Optional[str] = None):
    if label is None:
        dataset = DatasetState.get_dataset()
        label = "Configure dataset" if not dataset else "Change dataset"
    if st.button(label=label, icon=":material/data_table:"):
        dataset_config_form()


@st.dialog("Configuration du jeu de données", width='large')
def dataset_config_form():
    """Formulaire permettant de configurer un jeu de données pour l'application.

    Le formulaire suit un processus en 5 étapes :
    1. Chargement du fichier (CSV ou Parquet)
    2. Détection et confirmation du type de fichier
    3. Aperçu des données et configuration de l'index
    4. Sélection des variables cibles et explicatives
    5. Validation de la configuration
    """
    # Récupération ou initialisation des states
    existing_dataset = DatasetState.get_dataset()  # State global
    FormState.initialize(existing_dataset)  # State du formulaire

    file = None
    df_raw = None

    # Étape 1 : Gestion du chargement du fichier
    st.markdown("## 1. Chargement du fichier")
    if existing_dataset:
        # Affichage du dataset existant avec option de suppression
        st.info(f"Dataset actuel : {existing_dataset.filename}")
        if st.button("Supprimer la configuration actuelle"):
            DatasetState.clear_dataset()
            FormState.clear()
            st.rerun(scope="fragment")
        df_raw = existing_dataset.data_raw

        if "file_type" not in st.session_state and existing_dataset.file_type:
            st.session_state.file_type = existing_dataset.file_type
    else:
        # Interface de chargement d'un nouveau fichier
        st.markdown("Commencez par charger votre fichier de données (CSV ou Parquet)")
        # On garde la référence du dernier fichier pour détecter les changements
        last_file_name = st.session_state.get("last_file_name")

        file = st.file_uploader(
            "Charger votre jeu de données",
            type=["csv", "parquet"],
            help="Formats acceptés : CSV, Parquet. Taille maximale : 200MB",
            key="file_upload"
        )

        # Détection d'un changement de fichier
        if not existing_dataset and file and file.name != last_file_name:
            FormState.clear()
            FormState.initialize(None)
            FormState.set_value("last_file_name", file.name)

    # Étape 2 : Détection et sélection du type de fichier
    st.markdown("## 2. Type de fichier")

    # Détection automatique du type pour les nouveaux fichiers
    if file and not existing_dataset:
        try:
            detected_type = FileTypeDetector.detect_file_type(file)
            st.success(f"Type de fichier détecté : {detected_type}")
            st.session_state.file_type = detected_type
        except ValueError as e:
            st.warning(str(e))
            if "file_type" in st.session_state:
                del st.session_state.file_type

    # Réinitialisation du type si aucun fichier n'est présent
    if not file and not existing_dataset:
        st.session_state.file_type = None

    # Interface de sélection/confirmation du type
    file_type = st.selectbox(
        "Format du fichier",
        FileTypeDetector.get_supported_types(),
        placeholder="Choisissez le format",
        index=None,
        key="file_type",
        help="Sélectionnez le format correspondant à votre fichier"
    )

    # Mise à jour du state du formulaire
    form_state = FormState.get_state()
    form_state["file_type"] = file_type

    # Chargement du fichier dans un DataFrame
    if file_type and df_raw is None and file:
        try:
            df_raw = DataFrameService.load_file(file, file_type)
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {str(e)}")
            return

    # Étapes suivantes uniquement si les données sont chargées
    if df_raw is not None:
        # Étape 3 : Configuration de l'index et aperçu
        id_column = DataPreview.render(df_raw)
        df = df_raw.set_index(id_column) if id_column else df_raw

        # Étape 4 : Sélection des variables du modèle
        st.markdown("## 4. Sélection des variables")
        ColumnSelector.initialize_lists(df)
        target_columns, features_columns = ColumnSelector.render()

        # Étape 5 : Validation et enregistrement
        st.markdown("## 5. Validation")
        if st.button("Valider la configuration",
                     help="Cliquez pour enregistrer votre configuration",
                     type="primary"):
            dataset = DataFrameService.create_dataset_config(
                df_raw,
                features_columns,
                target_columns,
                filename=file.name if file else existing_dataset.filename,
                file_type=file_type,
                file=file,
                id_column=id_column
            )
            DatasetState.set_dataset(dataset)
            FormState.clear()
            st.rerun()
