# app.py

import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("breast_cancer_random_forest.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(
    page_title="Breast Cancer AI Predictor",
    page_icon="🧬",
    layout="centered"
)

# Custom CSS for advanced UI
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    }
    .main {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0px 0px 30px rgba(0,0,0,0.4);
    }
    h1, h2, h3 {
        text-align: center;
        color: #ffffff;
    }
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        font-size: 18px;
        font-weight: bold;
        background: linear-gradient(90deg, #ff416c, #ff4b2b);
        color: white;
        border: none;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #ff4b2b, #ff416c);
    }
    .result-box {
        text-align: center;
        font-size: 26px;
        font-weight: bold;
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1>🧬 Breast Cancer Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<h3>AI-Powered Medical Risk Analysis</h3>", unsafe_allow_html=True)

st.divider()

# Input Section
st.subheader("🔢 Input Medical Features")

col1, col2 = st.columns(2)

with col1:
    f1 = st.slider("Feature 1", 0.0, 50.0, 10.0)
    f2 = st.slider("Feature 2", 0.0, 50.0, 10.0)

with col2:
    f3 = st.slider("Feature 3", 0.0, 50.0, 10.0)
    f4 = st.slider("Feature 4", 0.0, 50.0, 10.0)

st.divider()

# Prediction
if st.button("🔍 Predict Cancer Type"):
    input_data = np.array([[f1, f2, f3, f4]])
    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.markdown(
            "<div class='result-box' style='background:linear-gradient(90deg,#11998e,#38ef7d);color:white;'>🟢 Benign Tumor</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='result-box' style='background:linear-gradient(90deg,#ff512f,#dd2476);color:white;'>🔴 Malignant Tumor</div>",
            unsafe_allow_html=True
        )

st.divider()

st.markdown(
    "<p style='text-align:center;color:#cccccc;'>Powered by Machine Learning & Streamlit</p>",
    unsafe_allow_html=True
)
