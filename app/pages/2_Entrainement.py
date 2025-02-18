import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(layout="wide", page_title="Entrainement des modèles de classification du vin")

# Load and prepare data
df = pd.read_csv("data/vin.csv")
df['target_numeric'] = (
    df['target']
    .apply(
        lambda x: 0 if x == "Vin amer" else 1 if x == "Vin éuilibré" else 2
    )
)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": tree.DecisionTreeClassifier()
}

# Define features and target
TARGET = "target_numeric"
FEATURES = ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", 
            "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins", 
            "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES],
    df[TARGET],
    test_size=0.2,
    random_state=42
)

def train_and_evaluate_model(model):
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return model, y_pred, cv_scores, report, conf_matrix

def plot_confusion_matrix(conf_matrix):
    fig = px.imshow(conf_matrix,
                    labels=dict(x="Predicted", y="Actual"),
                    x=['Amer', 'Équilibré', 'Sucré'],
                    y=['Amer', 'Équilibré', 'Sucré'],
                    color_continuous_scale='RdBu')
    fig.update_layout(title='Confusion Matrix')
    return fig

def set_new_params(model):
    if isinstance(model, LogisticRegression):
        new_c = float(st.session_state.get('C', model.C))
        new_solver = st.session_state.get('solver', model.solver)
        
        new_model = LogisticRegression(
            C=new_c,
            solver=new_solver,
            max_iter=1000
        )
        
        models["Logistic Regression"] = new_model
    
    elif isinstance(model, tree.DecisionTreeClassifier):
        pass


@st.dialog(f"Gérer les paramètres du modèle")
def open_modale(smn:str):
    model = models[smn]
    if isinstance(model, LogisticRegression):
        st.text_input("C", value=model.C, key="C")
        st.selectbox("Solver", 
        ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
            key="solver"
        )
        st.button("apply", on_click=lambda : set_new_params(model))


def main():
    st.title("🍷 Entrainement des modèles de classification du vin")
    
    st.markdown("""
    Cette page a pour but de vous permettre de choisir un modèle pour l'entrainer à prédire le type du vin en fonction de ses caractéristiques.

    Vous pouvez choisir différents modèles dans la barre de sélection à gauche
    """)
    
    # Sidebar
    st.sidebar.header("Sélection des modèles")
    selected_model_name = st.sidebar.selectbox(
        "Choisissez un modèle",
        list(models.keys())
    )
    st.sidebar.button("Personnaliser", on_click=lambda : open_modale(selected_model_name))
    
    model = models[selected_model_name]
    model, y_pred, cv_scores, report, conf_matrix = train_and_evaluate_model(model)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1-score (macro)'],
            'Score': [
                report['accuracy'],
                report['macro avg']['precision'],
                report['macro avg']['recall'],
                report['macro avg']['f1-score']
            ]
        })
        st.dataframe(metrics_df.round(3))
        
        st.subheader("Cross-validation Results")
        st.write(f"Mean CV Score: {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
        
    with col2:
        st.subheader("Matrice de confusion")
        conf_matrix_fig = plot_confusion_matrix(conf_matrix)
        st.plotly_chart(conf_matrix_fig, use_container_width=True)
    
    st.subheader("Détails des performances du modèle")
    report_df = pd.DataFrame(report).round(3)
    st.dataframe(report_df.transpose())

if __name__ == "__main__":
    main()
