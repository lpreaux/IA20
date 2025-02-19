import streamlit as st
import pandas as pd
from typing import Optional
from ..state import FormState


class DataPreview:
    """Composant pour l'aperçu des données."""

    @staticmethod
    def render(df_raw: pd.DataFrame) -> Optional[str]:
        """Affiche l'aperçu des données et les options d'ID."""
        st.markdown("## 3. Aperçu des données")
        form_state = FormState.get_state()

        col1, col2 = st.columns([1, 2])
        id_column = None

        with col1:
            has_id_column = st.checkbox(
                "Définir une colonne ID",
                value=form_state["has_id_column"],
                key="id_column_checkbox",
                help="Sélectionnez si vous souhaitez définir une colonne comme identifiant unique"
            )
            form_state["has_id_column"] = has_id_column

        if has_id_column:
            with col2:
                current_id = form_state["id_column"]
                selected_index = (
                    list(df_raw.columns).index(current_id)
                    if current_id in df_raw.columns
                    else 0
                )
                id_column = st.selectbox(
                    "Colonne ID",
                    df_raw.columns,
                    key="id_column_selector",
                    index=selected_index
                )
                form_state["id_column"] = id_column

        df_display = df_raw.set_index(id_column) if id_column else df_raw
        st.dataframe(df_display.head(), use_container_width=True)

        return id_column