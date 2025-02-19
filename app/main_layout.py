import pandas as pd
import streamlit as st

from utils.dataset import DatasetUtils


@st.dialog(f"Configure dataset")
def dataset_form():
    file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "parquet"],
    )

    if file:
        if "file_type" not in st.session_state or st.session_state['file_type'] != file.type:
            st.session_state['file_type'] = file.type

    file_type = st.selectbox(
        "File type",
        [
            "text/csv",
            "application/octet-stream",
        ],
        index=None,
        placeholder="Choose an option",
        key="file_type",
    )

    if file and file_type:
        st.markdown("""
            ___
            ## Data
            """)

        if file_type == "text/csv":
            df_raw = pd.read_csv(file)
        elif file_type == "application/octet-stream":
            df_raw = pd.read_parquet(file)
        else:
            raise ValueError("Invalid file type")

        if "df_id_col" not in st.session_state:
            st.session_state["df_id_col"] = None

        df = (
            df_raw
            .set_index(st.session_state["df_id_col"]) if st.session_state["df_id_col"] is not None else df_raw
        )

        st.dataframe(df.head())
        st.markdown("""
        ## Available Columns
        """)
        st.markdown(", ".join(df_raw.columns))

        if st.checkbox("Id column available"):
            st.selectbox("Id column", df_raw.columns, key="df_id_col")

        st.markdown("""
        ___
        ## Targets & Features selection
        """)
        if "features_col" not in st.session_state:
            st.session_state["features_col"] = []
        if "targets_col" not in st.session_state:
            st.session_state["targets_col"] = []

        available_target_columns = [col for col in df.columns if col not in st.session_state["features_col"]]
        available_features_columns = [col for col in df.columns if col not in st.session_state["targets_col"]]

        target_columns = st.multiselect("Target", available_target_columns, key="targets_col")
        features_columns = st.multiselect("Features", available_features_columns, key="features_col")

        if st.button("Submit"):
            dataset = DatasetUtils.create_dataset(df, target_columns, features_columns, filename=file.name, id_column=st.session_state["df_id_col"])
            DatasetUtils.store(dataset)
            st.rerun()


def sidebar():
    dataset = DatasetUtils.load()
    with st.sidebar:
        st.markdown(f"""
    ## Jeu de donnée
    Dataset : {dataset.filename if dataset else "No dataset.py configured yet"}
    """)
        configure_dataset_btn_label = "Configure dataset.py" if not dataset else "Change dataset.py"
        if st.button(label=configure_dataset_btn_label, icon=":material/data_table:"):
            dataset_form()
        st.markdown("___")
