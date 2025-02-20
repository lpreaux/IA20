import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
import plotly.express as px
import dataset.state as state 
import matplotlib.pyplot as plt
import math

def get_data_frame():
    config = state.DatasetState.get_dataset()
    df = config.data
    target = config.target_columns
    df = (
        df 
        .assign(
            target_num=lambda x: pd.Categorical(x[target[0]]).codes
        )
    )
    return df

def get_target():
    return "target_num"

def get_features():
    config = state.DatasetState.get_dataset()
    return config.features_columns

def train_and_evaluate_model(model):

    df = get_data_frame()

    TARGET = get_target()
    FEATURES = get_features()

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES],
        df[TARGET],
        test_size=0.2,
        random_state=51
    )

    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return model, y_pred, cv_scores, report, conf_matrix

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

def plot_confusion_matrix(conf_matrix):
    conf = state.DatasetState.get_dataset()
    target_col = conf.target_columns[0]
    vals = conf.data[target_col].unique()
    fig = px.imshow(conf_matrix,
                    labels=dict(x="Predicted", y="Actual"),
                    x=vals,
                    y=vals,
                    color_continuous_scale='RdBu')
    fig.update_layout(title='Confusion Matrix')
    return fig


def run_model(model):
    FEATURES = get_features()
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