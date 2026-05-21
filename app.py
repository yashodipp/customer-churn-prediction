

# =========================================================
# app.py
# Premium Customer Churn Prediction Dashboard
# Fully Fixed Version
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Churn AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("customer_churn_prediction_dataset.csv")

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load("logistic.pkl")

    return model

model = load_model()

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family:'Poppins',sans-serif;
}

/* Main Background */

.stApp{
    background:
    radial-gradient(circle at top left,#2563eb,#0f172a 40%),
    linear-gradient(135deg,#020617,#0f172a);
    color:white;
}

/* Hide Streamlit Default */

#MainMenu{
    visibility:hidden;
}

footer{
    visibility:hidden;
}

header{
    visibility:hidden;
}

/* Hero Section */

.hero{
    padding:60px;
    border-radius:35px;
    background:rgba(255,255,255,0.06);
    backdrop-filter:blur(18px);
    border:1px solid rgba(255,255,255,0.08);
    box-shadow:0 10px 40px rgba(0,0,0,0.4);
    margin-bottom:30px;
}

.hero-title{
    font-size:65px;
    font-weight:700;
    line-height:1.1;
}

.hero-title span{
    color:#38bdf8;
}

.hero-sub{
    color:#cbd5e1;
    font-size:20px;
    margin-top:15px;
}

/* Glass Container */

.glass{
    background:rgba(255,255,255,0.05);
    padding:30px;
    border-radius:30px;
    backdrop-filter:blur(14px);
    border:1px solid rgba(255,255,255,0.08);
    box-shadow:0 10px 30px rgba(0,0,0,0.35);
}

/* Input Styling */

.stSelectbox label,
.stSlider label,
.stNumberInput label{
    color:white !important;
    font-weight:600 !important;
}

div[data-baseweb="select"]{
    background:#111827 !important;
    border-radius:15px !important;
}

.stNumberInput input{
    background:#111827 !important;
    color:white !important;
}

/* Button */

.stButton button{
    width:100%;
    background:linear-gradient(135deg,#0ea5e9,#2563eb);
    color:white;
    border:none;
    border-radius:18px;
    padding:16px;
    font-size:18px;
    font-weight:700;
    transition:0.3s;
}

.stButton button:hover{
    transform:translateY(-3px);
    box-shadow:0 12px 25px rgba(37,99,235,0.5);
}

/* Result Cards */

.success{
    background:linear-gradient(135deg,#15803d,#22c55e);
    padding:35px;
    border-radius:25px;
    text-align:center;
    font-size:32px;
    font-weight:700;
    margin-top:25px;
    color:white;
}

.danger{
    background:linear-gradient(135deg,#b91c1c,#ef4444);
    padding:35px;
    border-radius:25px;
    text-align:center;
    font-size:32px;
    font-weight:700;
    margin-top:25px;
    color:white;
}

/* Metrics */

.metric{
    background:rgba(255,255,255,0.05);
    border-radius:25px;
    padding:30px;
    text-align:center;
    border:1px solid rgba(255,255,255,0.08);
}

.metric h1{
    color:#38bdf8;
    font-size:50px;
}

.metric p{
    color:#cbd5e1;
    font-size:18px;
}

/* Dataframe */

[data-testid="stDataFrame"]{
    border-radius:20px;
    overflow:hidden;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
AI Powered <span>Customer Churn</span><br>
Prediction Dashboard
</div>

<div class="hero-sub">
Modern futuristic Machine Learning dashboard with
premium UI and advanced analytics.
</div>

</div>
""", unsafe_allow_html=True)

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "🔮 Prediction",
    "📊 Analytics",
    "🤖 Model Info"
])

# =========================================================
# TAB 1 - PREDICTION
# =========================================================

with tab1:

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            df["gender"].unique()
        )

        senior = st.selectbox(
            "Senior Citizen",
            [0,1]
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

        internet = st.selectbox(
            "Internet Service",
            df["InternetService"].unique()
        )

    with col2:

        security = st.selectbox(
            "Online Security",
            df["OnlineSecurity"].unique()
        )

        tech = st.selectbox(
            "Tech Support",
            df["TechSupport"].unique()
        )

        streaming = st.selectbox(
            "Streaming TV",
            df["StreamingTV"].unique()
        )

        contract = st.selectbox(
            "Contract",
            df["Contract"].unique()
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

    # =========================================================
    # PREDICT BUTTON
    # =========================================================

    if st.button("🚀 Predict Customer Churn"):

        input_data = pd.DataFrame({

            "gender":[gender],
            "SeniorCitizen":[senior],
            "Partner":[partner],
            "Dependents":[dependents],
            "tenure":[tenure],
            "PhoneService":[phone],
            "InternetService":[internet],
            "OnlineSecurity":[security],
            "TechSupport":[tech],
            "StreamingTV":[streaming],
            "Contract":[contract],
            "PaymentMethod":[payment],
            "MonthlyCharges":[monthly],
            "TotalCharges":[total]

        })

        # =====================================================
        # ONE HOT ENCODING
        # =====================================================

        input_encoded = pd.get_dummies(input_data)

        # =====================================================
        # MATCH MODEL FEATURES
        # =====================================================

        try:

            model_features = model.feature_names_in_

            for col in model_features:

                if col not in input_encoded.columns:
                    input_encoded[col] = 0

            input_encoded = input_encoded[model_features]

        except:
            pass

        # =====================================================
        # PREDICTION
        # =====================================================

        try:

            prediction = model.predict(input_encoded)[0]

            probability = model.predict_proba(input_encoded)[0]

            confidence = round(max(probability) * 100,2)

            if prediction == 1 or prediction == "Yes":

                st.markdown(f"""
                <div class="danger">
                ⚠️ Customer Likely To Churn
                <br><br>
                Confidence : {confidence}%
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown(f"""
                <div class="success">
                ✅ Customer Likely To Stay
                <br><br>
                Confidence : {confidence}%
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:

            st.error(f"Prediction Error : {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 2 - ANALYTICS
# =========================================================

with tab2:

    c1,c2,c3 = st.columns(3)

    with c1:

        st.markdown(f"""
        <div class="metric">
        <h1>{df.shape[0]}</h1>
        <p>Total Rows</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown(f"""
        <div class="metric">
        <h1>{df.shape[1]}</h1>
        <p>Total Columns</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown(f"""
        <div class="metric">
        <h1>{df.isnull().sum().sum()}</h1>
        <p>Missing Values</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.subheader("📈 Churn Distribution")

    st.bar_chart(
        df["Churn"].value_counts()
    )

# =========================================================
# TAB 3 - MODEL INFO
# =========================================================

with tab3:

    st.markdown("""
    <div class="glass">

    <h1 style="color:#38bdf8;">
    🤖 Model Information
    </h1>

    <hr>

    <h3>Algorithm</h3>
    <p>Logistic Regression</p>

    <h3>Dataset</h3>
    <p>Customer Churn Prediction Dataset</p>

    <h3>Goal</h3>

    <p>
    Predict whether customer will churn
    or continue service using customer
    demographic and billing data.
    </p>

    <h3>Technologies Used</h3>

    <ul>
    <li>Python</li>
    <li>Pandas</li>
    <li>NumPy</li>
    <li>Scikit-learn</li>
    <li>Streamlit</li>
    <li>Machine Learning</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<center style="color:#94a3b8;">
Made with ❤️ Yashodip
</center>
""", unsafe_allow_html=True)