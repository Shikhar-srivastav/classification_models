import streamlit as st

from src.model_eval import evaluate_model

st.set_page_config(
    page_title="Classification Models",
    page_icon="",
    layout="centered",
)

selected_model = "logistic_regression"
stats = evaluate_model(selected_model)

stats = [i for i in stats.items()]
stats = stats[1::]

with st.container(horizontal=False, gap="medium"):
    for i in range(0, 6, 2):
        metric1, metric2 = stats[i], stats[i+1]

        cols = st.columns(2, gap="medium", width=400)

        with cols[0]:
            st.metric(
                metric1[0],
                metric1[1],
                width="content",
            )

        with cols[1]:
            st.metric(
                metric2[0],
                metric2[1],
                width="content",
            )
