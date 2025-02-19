import streamlit as st
import pandas as pd
from typing import List, Tuple
from ..state import FormState


class ColumnSelector:
    """Composant pour la sélection des colonnes."""

    @staticmethod
    def initialize_lists(df: pd.DataFrame) -> None:
        """Initialise les listes de colonnes disponibles."""
        form_state = FormState.get_state()

        form_state["_temp_df_columns"] = list(df.columns)
        form_state["available_targets"] = [
            col for col in df.columns
            if col not in form_state["features_col"]
        ]
        form_state["available_features"] = [
            col for col in df.columns
            if col not in form_state["targets_col"]
        ]

    @staticmethod
    def render() -> Tuple[List[str], List[str]]:
        """Affiche les sélecteurs de colonnes."""
        form_state = FormState.get_state()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Variables cibles")
            target_columns = st.multiselect(
                "Sélectionnez les variables à prédire",
                form_state["available_targets"],
                default=form_state["targets_col"],
                key="targets_selector",
                on_change=ColumnSelector._update_available_features,
                help="Variables que votre modèle cherchera à prédire"
            )
            form_state["targets_col"] = target_columns

        with col2:
            st.markdown("### Variables explicatives")
            features_columns = st.multiselect(
                "Sélectionnez les variables prédictives",
                form_state["available_features"],
                default=form_state["features_col"],
                key="features_selector",
                on_change=ColumnSelector._update_available_targets,
                help="Variables utilisées pour faire les prédictions"
            )
            form_state["features_col"] = features_columns

        return target_columns, features_columns

    @staticmethod
    def _update_available_features():
        form_state = FormState.get_state()
        form_state["available_features"] = [
            col for col in form_state["_temp_df_columns"]
            if col not in st.session_state.targets_selector
        ]

    @staticmethod
    def _update_available_targets():
        form_state = FormState.get_state()
        form_state["available_targets"] = [
            col for col in form_state["_temp_df_columns"]
            if col not in st.session_state.features_selector
        ]
