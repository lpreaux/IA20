import streamlit as st
import pandas as pd

df = pd.read_csv("/home/hijokaidan/PC/cours/ia/data/vin.csv")
df['target_numeric'] = (
    df['target']
    .apply(
        lambda x: 0 if x == "Vin amer" else 1 if x == "Vin éuilibré" else 2
    )
)

df_data = (
    df
    .groupby("target")
    .size()
) 


TARGET = "target"
FEATURES = "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"



######
# UI #
######

st.write()

st.write("## Charte de la répartition des types de vin dans les données : ", unsafe_allow_html=True)

st.bar_chart(df_data)

st.write("## Tables d'informations sur les colonnes dans les données : ")
st.table(df.describe())
st.sidebar.markdown(""" # Exploration des données 🍷""")
