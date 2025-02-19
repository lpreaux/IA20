import streamlit as st
from typing import Optional
from .models import DatasetConfig


class DatasetState:
    """Gestionnaire global du state du dataset."""

    DATASET_KEY = "dataset"

    @classmethod
    def get_dataset(cls) -> Optional[DatasetConfig]:
        """Récupère la configuration du dataset."""
        return st.session_state.get(cls.DATASET_KEY)

    @classmethod
    def set_dataset(cls, dataset: DatasetConfig) -> None:
        """Enregistre la configuration du dataset."""
        st.session_state[cls.DATASET_KEY] = dataset

    @classmethod
    def clear_dataset(cls) -> None:
        """Supprime la configuration du dataset."""
        if cls.DATASET_KEY in st.session_state:
            del st.session_state[cls.DATASET_KEY]