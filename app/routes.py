from streamlit import Page

routes = [
    Page("pages/home.py", title="Home", icon=":material/home:"),
    Page("pages/data_info.py", title="Jeu de données", icon=":material/data_table:"),
    Page("pages/data_cleaning.py", title="Nettoyage des données", icon=":material/cleaning_services:"),
    Page("pages/1_Exploration_des_données.py", icon=":material/search_insights:"),
    Page("pages/pipeline_config.py", title="Configuration du pipeline", icon=":material/timeline:"),
    Page("pages/2_Entrainement.py", icon=":material/model_training:"),
    Page("pages/3_Comparaisons.py", icon=":material/model_training:"),
]