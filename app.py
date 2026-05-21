# ==============================
# app.py
# Customer Churn Prediction UI
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import sys
import numpy.core

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ==============================
# CUSTOM CSS
# ==============================

st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#0f172a,#111827);
    color:white;
}

.title{
    text-align:center;
    font-size:50px;
    font-weight:800;
    color:#38bdf8;
}

.subtitle{
    text-align:center;
    color:#cbd5e1;
    margin-bottom:30px;
}

.card{
    background:rgba(255,255,255,0.05);
    padding:20px;
    border-radius:20px;
    box-shadow:0px 8px 25px rgba(0,0,0,0.3);
    backdrop-filter:blur(10px);
}

.result-success{
    background:linear-gradient(135deg,#16a34a,#22c55e);
    padding:25px;
    border-radius:20px;
    text-align:center;
    font-size:28px;
    font-weight:bold;
    color:white;
}

.result-danger{
    background:linear-gradient(135deg,#dc2626,#ef4444);
    padding:25px;
    border-radius:20px;
    text-align:center;
    font-size:28px;
    font-weight:bold;
    color:white;
}

.metric{
    background:linear-gradient(135deg,#06b6d4,#3b82f6);
    padding:20px;
    border-radius:20px;
    text-align:center;
    color:white;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv("customer_churn_prediction_dataset.csv")

# ==============================
# LOAD MODEL
# ==============================

@st.cache_resource
def load_model():

    try:
        sys.modules['numpy._core'] = numpy.core
        model = joblib.load("logistic.pkl")
    except:
        with open("logistic.pkl", "rb") as file:
            model = pickle.load(file)

    return model

model = load_model()

# ==============================
# HEADER
# ==============================

st.markdown('<div class="title">📊 Customer Churn Predictor</div>', unsafe_allow_html=True)

st.markdown('<div class="subtitle">AI Based Customer Churn Analysis Dashboard</div>', unsafe_allow_html=True)

# ==============================
# SIDEBAR
# ==============================

st.sidebar.title("⚡ Navigation")

page = st.sidebar.radio(
    "Select Section",
    ["Prediction", "Dataset Overview", "Model Information"]
)

# ==============================
# PREDICTION PAGE
# ==============================

if page == "Prediction":

    st.markdown("## 🔮 Predict Customer Churn")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            df["gender"].unique()
        )

        senior = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            df["Partner"].unique()
        )

        dependents = st.selectbox(
            "Dependents",
            df["Dependents"].unique()
        )

        tenure = st.slider(
            "Tenure",
            0,
            100,
            12
        )

        phone = st.selectbox(
            "Phone Service",
            df["PhoneService"].unique()
        )

        multiline = st.selectbox(
            "Multiple Lines",
            df["MultipleLines"].unique()
        )

        internet = st.selectbox(
            "Internet Service",
            df["InternetService"].unique()
        )

        security = st.selectbox(
            "Online Security",
            df["OnlineSecurity"].unique()
        )

        backup = st.selectbox(
            "Online Backup",
            df["OnlineBackup"].unique()
        )

    with col2:

        protection = st.selectbox(
            "Device Protection",
            df["DeviceProtection"].unique()
        )

        tech = st.selectbox(
            "Tech Support",
            df["TechSupport"].unique()
        )

        tv = st.selectbox(
            "Streaming TV",
            df["StreamingTV"].unique()
        )

        movies = st.selectbox(
            "Streaming Movies",
            df["StreamingMovies"].unique()
        )

        contract = st.selectbox(
            "Contract",
            df["Contract"].unique()
        )

        paperless = st.selectbox(
            "Paperless Billing",
            df["PaperlessBilling"].unique()
        )

        payment = st.selectbox(
            "Payment Method",
            df["PaymentMethod"].unique()
        )

        monthly = st.number_input(
            "Monthly Charges",
            value=70.0
        )

        total = st.number_input(
            "Total Charges",
            value=1500.0
        )

    # ==============================
    # PREDICT BUTTON
    # ==============================

    if st.button("🚀 Predict Churn", width='stretch'):

        # ==============================
        # INPUT DATAFRAME
        # ==============================

        input_data = pd.DataFrame({
            "gender": [gender],
            "SeniorCitizen": [senior],
            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [tenure],
            "PhoneService": [phone],
            "MultipleLines": [multiline],
            "InternetService": [internet],
            "OnlineSecurity": [security],
            "OnlineBackup": [backup],
            "DeviceProtection": [protection],
            "TechSupport": [tech],
            "StreamingTV": [tv],
            "StreamingMovies": [movies],
            "Contract": [contract],
            "PaperlessBilling": [paperless],
            "PaymentMethod": [payment],
            "MonthlyCharges": [monthly],
            "TotalCharges": [total]
        })

        # ==============================
        # ONE HOT ENCODING
        # ==============================

        input_encoded = pd.get_dummies(input_data)

        # ==============================
        # MATCH MODEL FEATURES
        # ==============================

        try:
            model_features = model.feature_names_in_
        except:
            model_features = input_encoded.columns

        # Add missing columns
        for col in model_features:

            if col not in input_encoded.columns:
                input_encoded[col] = 0

        # Remove extra columns
        input_encoded = input_encoded[model_features]

        # ==============================
        # PREDICTION
        # ==============================

        try:

            prediction = model.predict(input_encoded)[0]

            probability = model.predict_proba(input_encoded)[0]

            confidence = round(max(probability) * 100, 2)

            st.markdown("<br>", unsafe_allow_html=True)

            if prediction == 1 or prediction == "Yes":

                st.markdown(
                    f"""
                    <div class="result-danger">
                    ⚠️ Customer Likely To Churn
                    <br><br>
                    Confidence: {confidence}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="result-success">
                    ✅ Customer Likely To Stay
                    <br><br>
                    Confidence: {confidence}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        except Exception as e:

            st.error(f"Prediction Error: {e}")

# ==============================
# DATASET OVERVIEW PAGE
# ==============================

elif page == "Dataset Overview":

    st.markdown("## 📁 Dataset Overview")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="metric">
            Rows
            <h2>{df.shape[0]}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric">
            Columns
            <h2>{df.shape[1]}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric">
            Missing Values
            <h2>{df.isnull().sum().sum()}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.dataframe(df.head(20), width='stretch')

    st.markdown("### 📌 Churn Distribution")

    st.bar_chart(df["Churn"].value_counts())

# ==============================
# MODEL INFORMATION PAGE
# ==============================

else:

    st.markdown("## 🤖 Model Information")

    st.markdown("""
    <div class="card">

    <h3>Model Used</h3>
    <p>Logistic Regression</p>

    <h3>Dataset</h3>
    <p>Customer Churn Prediction Dataset</p>

    <h3>Purpose</h3>
    <p>
    Predict whether customer will stay or churn
    using demographic and billing information.
    </p>

    <h3>Features</h3>

    <ul>
        <li>Customer Demographics</li>
        <li>Internet Services</li>
        <li>Billing Information</li>
        <li>Streaming Services</li>
        <li>Support Services</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

# ==============================
# FOOTER
# ==============================

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
    "<center>Made with ❤️ using Streamlit</center>",
    unsafe_allow_html=True
)