import streamlit as st
import utils.training as train

models = train.get_models()

cols = st.columns(3)
i = 0
for key, model in models.items():
    model, y_pred, cv_scores, report, conf_matrix = train.train_and_evaluate_model(model)
    with cols[i%3]:
        st.subheader(f"Matrice de confusion pour les valeurs par défaut de {key}")
        conf_matrix_fig = train.plot_confusion_matrix(conf_matrix)
        st.plotly_chart(conf_matrix_fig, use_container_width=True)

    i += 1