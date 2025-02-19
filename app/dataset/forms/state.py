import streamlit as st
from typing import Dict, Any, Optional
from ..models import DatasetConfig


class FormState:
    """Gestionnaire du state local au formulaire."""

    INITIAL_STATE = {
        "file_type": None,
        "has_id_column": False,
        "id_column": None,
        "features_col": [],
        "targets_col": [],
        "available_features": [],
        "available_targets": [],
        "_temp_df_columns": [],
        "last_file_name": None
    }

    @staticmethod
    def initialize(dataset: Optional[DatasetConfig] = None) -> None:
        """Initialise le state du formulaire."""
        if "form_initialized" not in st.session_state:
            initial_state = FormState.INITIAL_STATE.copy()

            # Mise à jour avec les valeurs du dataset si existant
            if dataset:
                initial_state.update({
                    "file_type": dataset.file_type,
                    "has_id_column": dataset.id_column is not None,
                    "id_column": dataset.id_column,
                    "features_col": dataset.features_columns,
                    "targets_col": dataset.target_columns,
                })

            st.session_state.form_state = initial_state
            st.session_state.form_initialized = True

    @staticmethod
    def get_state() -> Dict[str, Any]:
        return st.session_state.form_state

    @staticmethod
    def set_value(key: str, value: Any) -> None:
        """Met à jour une valeur spécifique dans le state."""
        st.session_state.form_state[key] = value

    @staticmethod
    def clear() -> None:
        """Réinitialise le state du formulaire."""
        if "form_state" in st.session_state:
            del st.session_state.form_state
        if "form_initialized" in st.session_state:
            del st.session_state.form_initialized