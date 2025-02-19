from typing import List
from dataclasses import dataclass
import pandas as pd
from streamlit.runtime.uploaded_file_manager import UploadedFile

@dataclass
class DatasetConfig:
    """Modèle de données pour la configuration d'un dataset."""
    data_raw: pd.DataFrame
    filename: str
    file_type: str
    features_columns: List[str]
    target_columns: List[str]
    file: UploadedFile = None
    id_column: str = None

    @property
    def data(self) -> pd.DataFrame:
        """Retourne le DataFrame avec l'index configuré si nécessaire."""
        if self.id_column:
            return self.data_raw.set_index(self.id_column)
        return self.data_raw