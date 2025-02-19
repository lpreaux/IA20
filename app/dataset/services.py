import pandas as pd
from streamlit.runtime.uploaded_file_manager import UploadedFile
from typing import List
from .models import DatasetConfig


class FileTypeDetector:
    """Service de détection du type de fichier."""

    SUPPORTED_TYPES = {
        "text/csv": [".csv"],
        "application/octet-stream": [".parquet", ".pq"]
    }

    @classmethod
    def detect_file_type(cls, file: UploadedFile) -> str:
        """Détecte le type de fichier à partir de son extension et de son type MIME."""
        # Vérification du type MIME
        if file.type in cls.SUPPORTED_TYPES:
            return file.type

        # Vérification de l'extension
        file_extension = cls._get_file_extension(file.name)
        for file_type, extensions in cls.SUPPORTED_TYPES.items():
            if file_extension in extensions:
                return file_type

        raise ValueError(f"Type de fichier non supporté : {file.type} ({file_extension})")

    @staticmethod
    def _get_file_extension(filename: str) -> str:
        """Extrait l'extension d'un nom de fichier."""
        return filename[filename.rfind("."):].lower() if "." in filename else ""

    @classmethod
    def get_supported_types(cls) -> List[str]:
        """Retourne la liste des types de fichiers supportés."""
        return list(cls.SUPPORTED_TYPES.keys())


class DataFrameService:
    """Service pour la manipulation des DataFrames."""

    @staticmethod
    def load_file(file: UploadedFile, file_type: str) -> pd.DataFrame:
        """Charge un fichier en DataFrame selon son type."""
        if file_type == "text/csv":
            return pd.read_csv(file)
        elif file_type == "application/octet-stream":
            return pd.read_parquet(file)
        raise ValueError("Format de fichier non supporté")

    @staticmethod
    def create_dataset_config(
            data_raw: pd.DataFrame,
            features_columns: List[str],
            target_columns: List[str],
            filename: str,
            file_type: str,
            file: UploadedFile = None,
            id_column: str = None
    ) -> DatasetConfig:
        """Crée une nouvelle configuration de dataset."""
        return DatasetConfig(
            data_raw=data_raw,
            filename=filename,
            file_type=file_type,
            features_columns=features_columns,
            target_columns=target_columns,
            file=file,
            id_column=id_column
        )
