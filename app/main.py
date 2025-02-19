import streamlit as st

import main_layout
from routes import routes

pg = st.navigation(routes)

st.set_page_config(page_title="IA20", page_icon=":material/robot:", layout="wide")

main_layout.sidebar()

pg.run()
