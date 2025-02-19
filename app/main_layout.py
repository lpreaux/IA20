import streamlit as st
from layouts.sidebar_components import render_dataset_manager, render_dataset_stats


def sidebar():
    """Configure la sidebar principale."""
    with st.sidebar:
        render_dataset_manager()
        render_dataset_stats()

        st.markdown("---")
