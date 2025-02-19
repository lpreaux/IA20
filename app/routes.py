from streamlit import Page

routes = [
    Page("pages/home.py", title="Home", icon=":material/home:"),
    Page("pages/data_info.py", title="Jeu de données", icon=":material/data_table:"),
    Page("pages/1_Exploration_des_données.py", icon=":material/search_insights:"),
    Page("pages/2_Entrainement.py", icon=":material/model_training:"),
]