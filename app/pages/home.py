import streamlit as st
from dataset.state import DatasetState
from layouts.sidebar_components import render_dataset_stats, render_quick_actions

# Sidebar de la page d'accueil
dataset = DatasetState.get_dataset()
# Action rapide
render_quick_actions(dataset)

# Guide rapide (spécifique à la home)
st.sidebar.markdown("""
### Guide rapide
1. Chargez vos données
2. Explorez et visualisez
3. Nettoyez si nécessaire
4. Configurez votre pipeline
5. Entraînez vos modèles
""")

# Contenu principal de la page
st.markdown("""
# Bienvenue sur IA20 

---

## Une plateforme d'analyse de données et d'apprentissage automatique 🚀

IA20 est votre compagnon intelligent pour l'analyse de données et l'entraînement de modèles prédictifs. Que vous soyez novice ou expert, notre application vous guide à travers chaque étape du processus d'analyse de données et de création de modèles d'IA.

### 🎯 Ce que vous pouvez faire avec IA20

- **Charger vos données** : Importez facilement vos jeux de données au format CSV ou Parquet
- **Explorer et comprendre** : Visualisez et analysez vos données à travers des graphiques interactifs
- **Nettoyer et préparer** : Préparez vos données pour l'apprentissage automatique
- **Entraîner des modèles** : Testez différents algorithmes d'apprentissage automatique
- **Évaluer les performances** : Analysez et comparez les résultats de vos modèles

### 🚀 Pour commencer

1. Cliquez sur "Configure dataset" dans la barre latérale
2. Chargez votre jeu de données (CSV ou Parquet)
3. Laissez-vous guider par notre interface intuitive

### 💡 Pourquoi IA20 ?

Initialement conçu pour l'analyse des vins (d'où son nom faisant un clin d'œil au monde viticole), IA20 a évolué pour devenir une plateforme polyvalente d'analyse de données et d'IA. Comme un bon vin, votre analyse de données mérite les meilleurs outils pour révéler tout son potentiel !

**Prêt à explorer vos données ? Commencez dès maintenant ! 🎯**
""")

# Message conditionnel basé sur l'état du dataset
if dataset:
    st.success(f"""
    ✨ Un jeu de données est déjà configuré : **{dataset.filename}**

    Rendez-vous dans la section "Exploration des données" pour commencer votre analyse !
    """)
else:
    st.info("""
    💡 Aucun jeu de données n'est configuré pour le moment.

    Cliquez sur "Configure dataset" dans la barre latérale pour commencer !
    """)
