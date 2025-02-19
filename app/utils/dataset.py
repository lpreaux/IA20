from typing import List, Union

import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile


class DatasetConfig:
    def __init__(self, data: pd.DataFrame, filename: str, file_type: str, features_columns: List[str],
                 target_columns: List[str], id_column: str = None) -> None:
        self.data = data
        self.filename = filename
        self.file_type = file_type
        self.features_columns = features_columns
        self.target_columns = target_columns


class DatasetUtils:
    @staticmethod
    def load() -> DatasetConfig | None:
        if "dataset" not in st.session_state:
            st.session_state["dataset"] = None
        return st.session_state["dataset"]

    @staticmethod
    def store(dataset: DatasetConfig) -> None:
        st.session_state["dataset"] = dataset

    @staticmethod
    def create_dataset(
            data: pd.DataFrame,
            features_columns: List[str],
            target_columns: List[str],
            filename: str,
            file_type: str = None,
            id_column: str = None,
    ) -> DatasetConfig:
        return DatasetConfig(data, filename, file_type, features_columns, target_columns, id_column)

    @staticmethod
    def create_from_file(
            file: UploadedFile,
            features_columns: List[str],
            target_columns: List[str],
            id_column: str = None,
    ) -> DatasetConfig:
        if file.type == "text/csv":
            data = pd.read_csv(file, index_col=id_column)
        elif file.type == "application/octet-stream":
            data = pd.read_parquet(file)
        else:
            raise NotImplementedError("Unsupported file type")
        filename = file.name
        file_type = file.type
        return DatasetConfig(data, filename, file_type, features_columns, target_columns, id_column)
