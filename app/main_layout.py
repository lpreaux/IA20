import pandas as pd
import streamlit as st

from dataset.forms.config import dataset_config_form
from dataset.state import DatasetState


# from utils.dataset import DatasetUtils, dataset_form

def sidebar():
    dataset = DatasetState.get_dataset()
    with st.sidebar:
        st.markdown(f"""
    ## Jeu de donnée
    Dataset : {dataset.filename if dataset else "No dataset configured yet"}
    """)
        configure_dataset_btn_label = "Configure dataset" if not dataset else "Change dataset"
        if st.button(label=configure_dataset_btn_label, icon=":material/data_table:"):
            dataset_config_form()
        st.markdown("___")
