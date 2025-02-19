import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
import plotly.express as px
import matplotlib.pyplot as plt
import math

# Load and prepare data
df = pd.read_csv("../data/vin.csv")
df['target_numeric'] = (
    df['target']
    .apply(
        lambda x: 0 if x == "Vin amer" else 1 if x == "Vin éuilibré" else 2
    )
)

def get_models():
    if not st.session_state.get("Models"):
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": tree.DecisionTreeClassifier(),
            "Forest": RandomForestClassifier(),
        }
        st.session_state["Models"] = models
    else:
        models = st.session_state["Models"]
    return models

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
    random_state=51
)

def plot_decision_tree(model, feature_names, class_names):
    plt.figure(figsize=(20, 10))
    tree.plot_tree(model, 
                   feature_names=feature_names,
                   class_names=class_names,
                   filled=True, 
                   rounded=True, 
                   fontsize=10)
    return plt

def plot_random_forest_trees(forest_model, feature_names, class_names, n_trees=4):
    """Plot a sample of trees from the random forest"""
    # Get a sample of trees from the forest
    n_cols = 2
    n_rows = math.ceil(n_trees / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10 * n_rows))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < n_trees:
            estimator = forest_model.estimators_[i]
            tree.plot_tree(estimator,
                          feature_names=feature_names,
                          class_names=class_names,
                          filled=True,
                          rounded=True,
                          ax=ax,
                          fontsize=8)
            ax.set_title(f"Tree {i+1} from Random Forest")
        else:
            ax.axis('off')
    
    plt.tight_layout()
    return plt

def plot_feature_importance(model, features):
    importance = 0
    if isinstance(model, LogisticRegression):
        importance = abs(model.coef_[0])
    else:  # Decision Tree or Random Forest
        importance = model.feature_importances_
    
    fig = px.bar(x=features, y=importance,
                 labels={'x': 'Features', 'y': 'Importance'},
                 title='Feature Importance')
    return fig

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
    models = get_models()
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
        new_p_max = int(st.session_state.get('p_max', model.max_depth))
        new_f_max = int(st.session_state.get('f_max', model.max_features))
        
        new_model = tree.DecisionTreeClassifier(
            max_depth=new_p_max,
            max_features=new_f_max,
        )
        
        models["Decision Tree"] = new_model

    elif isinstance(model, RandomForestClassifier):
        new_n_trees = int(st.session_state.get('n_trees', model.n_estimators))
        new_max_depth = int(st.session_state.get('rf_max_depth', model.max_depth))
        
        new_model = RandomForestClassifier(
            n_estimators=new_n_trees,
            max_depth=new_max_depth,
        )
        
        models["Forest"] = new_model
    
    run_model(new_model)


@st.dialog(f"Gérer les paramètres du modèle")
def open_modale(smn:str):
    models = get_models()
    model = models[smn]

    if isinstance(model, LogisticRegression):
        st.text_input("C", value=model.C, key="C")
        st.selectbox("Solver", 
        ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
            key="solver"
        )
        st.button("Changer", on_click=lambda : set_new_params(model))

    elif isinstance(model, tree.DecisionTreeClassifier):
        st.number_input("Profondeur Max", value=model.max_depth, key="p_max")
        st.number_input("Max Features", value=model.max_features, key="f_max")
        st.button("Changer", on_click=lambda : set_new_params(model))

    elif isinstance(model, RandomForestClassifier):
        st.number_input("Nombre d'arbres", value=model.n_estimators, key="n_trees")
        st.number_input("Profondeur Max", value=model.max_depth, key="rf_max_depth")
        st.button("Changer", on_click=lambda : set_new_params(model))

def run_model(model):
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

    if isinstance(model, tree.DecisionTreeClassifier):
        st.subheader("Visualisation de l'arbre de décision")
        class_names = ['Amer', 'Équilibré', 'Sucré']
        fig = plot_decision_tree(model, FEATURES, class_names)
        st.pyplot(fig)
    
    elif isinstance(model, RandomForestClassifier):
        st.subheader("Visualisation d'échantillon d'arbres de la forêt")
        class_names = ['Amer', 'Équilibré', 'Sucré']
        
        # Add a slider to select how many trees to display
        num_trees_to_show = st.slider("Nombre d'arbres à afficher", 1, 
                                     min(6, model.n_estimators), 4)
        
        forest_fig = plot_random_forest_trees(model, FEATURES, class_names, num_trees_to_show)
        st.pyplot(forest_fig)
    
    st.subheader("Feature Importance")
    feature_importance_fig = plot_feature_importance(model, FEATURES)
    st.plotly_chart(feature_importance_fig, use_container_width=True)
    
    st.subheader("Détails des performances du modèle")
    report_df = pd.DataFrame(report).round(3)
    st.dataframe(report_df.transpose())


def main():
    models = get_models()
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
    run_model(model)


main()