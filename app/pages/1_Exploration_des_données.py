import streamlit as st
import pandas as pd

df = pd.read_csv("/home/hijokaidan/PC/cours/ia/data/vin.csv")

df_data = (
    df
    .groupby("target")
    .size()
) 

######
# UI #
######

st.write()

st.write("## Charte de la répartition des types de vin dans les données : ", unsafe_allow_html=True)

st.bar_chart(df_data)

st.write("## Tables d'informations sur les colonnes dans les données : ")
st.table(df.describe())
st.sidebar.markdown(""" # Exploration des données 🍷""")
