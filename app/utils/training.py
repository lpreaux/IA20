import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
import seaborn as sns
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
    """Plot a sample of trees from the random forest with enhanced styling"""
    n_cols = 2
    n_rows = math.ceil(n_trees / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10 * n_rows), facecolor='white')
    fig.suptitle('Sample Trees from Random Forest', fontsize=22, y=0.98)
    
    axes = axes.flatten()
    
    cmap = plt.cm.Blues
    
    for i, ax in enumerate(axes):
        if i < n_trees:
            estimator = forest_model.estimators_[i]
            tree.plot_tree(estimator,
                          feature_names=feature_names,
                          class_names=class_names,
                          filled=True,
                          rounded=True,
                          ax=ax,
                          fontsize=8,
                          proportion=True,  # Show proportions
                          precision=2)      # Decimal precision
            
            ax.set_title(f"Tree {i+1} from Random Forest", fontsize=14, pad=12)
            ax.grid(axis='both', linestyle='--', alpha=0.3)
        else:
            ax.axis('off')
    
    plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.97])
    return plt


def plot_feature_importance(model, features):
    importance = 0
    if isinstance(model, LogisticRegression):
        importance = abs(model.coef_[0])
    else:
        importance = model.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    })
    
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    
    fig = px.bar(feature_importance_df, 
                 y='Feature', 
                 x='Importance',
                 orientation='h',
                 color='Importance',
                 color_continuous_scale='Viridis',
                 labels={'x': 'Importance', 'y': 'Features'},
                 title='Feature Importance',
                 height=max(400, len(features) * 30)  # Dynamic height based on feature count
                )
    
    fig.update_traces(texttemplate='%{x:.3f}', textposition='outside')
    
    fig.update_layout(
        title={
            'text': 'Feature Importance',
            'font': {'size': 20, 'family': 'Arial'},
            'y': 0.98
        },
        yaxis={'categoryorder': 'total ascending'},
        margin=dict(l=20, r=120, t=60, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#EEEEEE')
    )
    
    return fig

def plot_confusion_matrix(conf_matrix):
    conf = state.DatasetState.get_dataset()
    target_col = conf.target_columns[0]
    vals = conf.data[target_col].unique()
    
    fig = px.imshow(conf_matrix,
                    labels=dict(x="Predicted", y="Actual"),
                    x=vals,
                    y=vals,
                    color_continuous_scale='gray',
                    text_auto=True)
    
    fig.update_traces(texttemplate="%{z}", textfont={"size": 14})
    
    fig.update_layout(title='Confusion Matrix')
    return fig

def draw_cv(report, cv_scores):
    st.subheader("Cross-validation Results")
    
    cv_df = pd.DataFrame({
        'Fold': range(1, len(cv_scores) + 1),
        'CV Score': cv_scores
    })
    
    mean_score = cv_scores.mean()
    std_score = cv_scores.std() * 2
    st.markdown(f"<h4>Mean CV Score: <span style='color:#1E88E5'>{mean_score:.3f}</span> (±{std_score:.3f})</h4>", unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    bar_color = '#1E88E5'
    mean_line_color = '#E53935'
    std_area_color = '#FFE0E0'
    
    bars = sns.barplot(data=cv_df, x='Fold', y='CV Score', ax=ax, color=bar_color)
    
    for i, bar in enumerate(bars.patches):
        ax.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 0.005, 
                f'{cv_scores[i]:.3f}', 
                ha='center', fontsize=10, color='#333333')
    
    ax.axhline(y=mean_score, color=mean_line_color, linestyle='-', 
               linewidth=2, label=f'Mean: {mean_score:.3f}')
    ax.fill_between(x=range(-1, len(cv_scores) + 1), 
                y1=mean_score - cv_scores.std(),
                y2=mean_score + cv_scores.std(),
                alpha=0.3, color=std_area_color, 
                label=f'Std Dev: {cv_scores.std():.3f}')
    
    ax.set_ylim([max(0, cv_scores.min() - 0.1), min(1, cv_scores.max() + 0.1)])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    ax.set_title("Cross-validation Scores by Fold", fontsize=16, pad=15)
    
    ax.set_xlabel("Fold Number", fontsize=12)
    ax.set_ylabel("CV Score", fontsize=12)
    
    plt.tight_layout()
    st.pyplot(fig)

def draw_perf(report, key=None):
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

    st.dataframe(metrics_df.style.format({'Score': '{:.3f}'})
                .background_gradient(cmap='Blues', subset=['Score']), 
                hide_index=True)


def run_model(model):
    FEATURES = get_features()
    model, y_pred, cv_scores, report, conf_matrix = train_and_evaluate_model(model)
    
    col1, col2 = st.columns(2)
    
    with col1:
        draw_perf(report)
        draw_cv(report, cv_scores)
        
    with col2:
        st.subheader("Matrice de confusion")
        conf_matrix_fig = plot_confusion_matrix(conf_matrix)
        st.plotly_chart(conf_matrix_fig, use_container_width=True)

    if isinstance(model, tree.DecisionTreeClassifier):
        st.subheader("Visualisation de l'arbre de décision")
        conf = state.DatasetState.get_dataset()
        target_col = conf.target_columns[0]
        vals = conf.data[target_col].unique()
        fig = plot_decision_tree(model, FEATURES, vals)
        st.pyplot(fig)
    
    elif isinstance(model, RandomForestClassifier):
        st.subheader("Visualisation d'échantillon d'arbres de la forêt")
        conf = state.DatasetState.get_dataset()
        target_col = conf.target_columns[0]
        vals = conf.data[target_col].unique()
        
        num_trees_to_show = st.slider("Nombre d'arbres à afficher", 1, 
                                     min(6, model.n_estimators), 4)
        
        forest_fig = plot_random_forest_trees(model, FEATURES, vals, num_trees_to_show)
        st.pyplot(forest_fig)
    
    st.subheader("Détails des performances du modèle")
    report_df = pd.DataFrame(report).round(3)
    st.dataframe(report_df.transpose())

    st.subheader("Feature Importance")
    feature_importance_fig = plot_feature_importance(model, FEATURES)
    st.plotly_chart(feature_importance_fig, use_container_width=True)