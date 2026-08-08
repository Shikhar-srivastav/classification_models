import streamlit as st

from src.model_eval import evaluate_model

st.set_page_config(
    page_title="Classification Models",
    page_icon="",
    layout="centered",
)

model_map = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "K Nearest Neighbors": "knn",
    "Naive Bayes": "naive_bayes",
    "Random Forest": "random_forest",
}

st.title("ML Classification Models Showcase")
st.space("small")

selected_model = st.selectbox("Select Classifier Model", model_map.keys())
st.space("small")

st.header(f"Calculated Metrics for {selected_model}")

stats = evaluate_model(model_map[selected_model])
stats = [i for i in stats.items()]

if (len(stats)):
    stats = stats[1::]

    with st.container(horizontal=False, gap="medium"):
        for i in range(0, 6, 2):
            metric1, metric2 = stats[i], stats[i+1]

            cols = st.columns(2, width="stretch", gap="medium")

            with cols[0]:
                st.metric(metric1[0], f"{metric1[1]:.4f}", width="content")
            with cols[1]:
                st.metric(metric2[0], f"{metric2[1]:.4f}", width="content")
