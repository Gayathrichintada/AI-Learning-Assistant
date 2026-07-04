import streamlit as st
import pandas as pd
import plotly.express as px

from models.bilstm_predict import predict_emotion as bilstm_predict
from models.bert_model import predict_emotion as bert_predict
from models.gemini_api import generate_guidance

from utils.logger import save_history

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI Learning Assistant")
st.markdown("### Emotion-Aware Personalized Learning Support")

# --------------------------------------------------
# User Input
# --------------------------------------------------

problem = st.text_area(
    "Enter your learning problem",
    placeholder="Example: I don't understand recursion..."
)

# --------------------------------------------------
# Analyze Button
# --------------------------------------------------

if st.button("Analyze"):

    if problem.strip() == "":
        st.warning("Please enter your learning problem.")
    else:

        with st.spinner("Analyzing your problem..."):

            # BiLSTM Prediction
            bilstm_label, bilstm_score = bilstm_predict(problem)

            # BERT Prediction
            bert_label, bert_score = bert_predict(problem)

            # Final Emotion (Higher Confidence)
            if bilstm_score >= bert_score:
                final_emotion = bilstm_label
            else:
                final_emotion = bert_label

            # Gemini Response
            response = generate_guidance(
                problem,
                final_emotion
            )

            # Save to CSV
            save_history(
                problem,
                bilstm_label,
                bert_label,
                final_emotion,
                response
            )

        st.success("Analysis Complete")

        st.divider()

        # --------------------------------------------
        # Model Comparison
        # --------------------------------------------

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("BiLSTM")
            st.write(f"**Emotion:** {bilstm_label}")
            st.write(f"**Confidence:** {bilstm_score:.2f}")

        with col2:
            st.subheader("BERT")
            st.write(f"**Emotion:** {bert_label}")
            st.write(f"**Confidence:** {bert_score:.2f}")

        st.divider()

        st.subheader("Final Emotion")

        st.success(final_emotion)

        st.divider()

        st.subheader("AI Guidance")

        st.write(response)

# --------------------------------------------------
# History
# --------------------------------------------------

st.divider()

st.header("History")

try:

    df = pd.read_csv("data/history.csv")

    st.dataframe(df, use_container_width=True)

    # --------------------------------------------------
    # Analytics Dashboard
    # --------------------------------------------------

    st.divider()

    st.header("Analytics Dashboard")

    # Emotion Distribution
    emotion_counts = (
        df["Final Emotion"]
        .value_counts()
        .reset_index()
    )

    emotion_counts.columns = ["Emotion", "Count"]

    fig1 = px.bar(
        emotion_counts,
        x="Emotion",
        y="Count",
        color="Emotion",
        title="Emotion Distribution"
    )

    st.plotly_chart(fig1, use_container_width=True)

    # Pie Chart

    fig2 = px.pie(
        emotion_counts,
        names="Emotion",
        values="Count",
        title="Emotion Distribution (Pie Chart)"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Confidence Comparison

    confidence_df = pd.DataFrame({
        "Model": ["BiLSTM", "BERT"],
        "Confidence": [
            df.iloc[-1]["BiLSTM"] if "BiLSTM" in df.columns else 0,
            df.iloc[-1]["BERT"] if "BERT" in df.columns else 0
        ]
    })

except Exception:

    st.info("No history available yet.")