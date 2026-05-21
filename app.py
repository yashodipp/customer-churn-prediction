

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSense · Telecom AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("logistic.pkl")

model = load_model()

FEATURE_NAMES = list(model.feature_names_in_)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;600&display=swap');

  /* ---------- GLOBAL RESET ---------- */
  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background: #060612;
    color: #e8e8f0;
  }

  .stApp {
    background: #060612;
  }

  /* Remove default streamlit padding */
  .block-container {
    padding: 0 !important;
    max-width: 100% !important;
  }

  /* ---------- HERO HEADER ---------- */
  .hero {
    background: linear-gradient(135deg, #0d0d2b 0%, #060612 40%, #0a0a1e 100%);
    border-bottom: 1px solid rgba(100, 80, 255, 0.2);
    padding: 48px 64px 40px;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(100,80,255,0.12) 0%, transparent 70%);
    top: -200px; right: -100px;
    pointer-events: none;
  }
  .hero::after {
    content: '';
    position: absolute;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(0,210,180,0.08) 0%, transparent 70%);
    bottom: -150px; left: 100px;
    pointer-events: none;
  }
  .badge {
    display: inline-block;
    background: rgba(100,80,255,0.15);
    border: 1px solid rgba(100,80,255,0.4);
    color: #a090ff;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 16px;
    font-family: 'JetBrains Mono', monospace;
  }
  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 52px;
    font-weight: 800;
    line-height: 1.05;
    margin: 0 0 12px 0;
    background: linear-gradient(135deg, #ffffff 0%, #c8c0ff 50%, #00d2b4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1.5px;
  }
  .hero-sub {
    color: #7070a0;
    font-size: 15px;
    font-weight: 400;
    max-width: 500px;
    line-height: 1.6;
    margin: 0;
  }
  .hero-stats {
    display: flex;
    gap: 40px;
    margin-top: 36px;
  }
  .stat-pill {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 22px;
    font-weight: 600;
    color: #a090ff;
  }
  .stat-lbl {
    font-size: 11px;
    color: #50507a;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 500;
  }

  /* ---------- CONTENT AREA ---------- */
  .content-wrap {
    padding: 40px 64px;
  }

  /* ---------- SECTION HEADER ---------- */
  .section-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #4444aa;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 6px;
  }
  .section-title {
    font-family: 'Syne', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: #d0d0f0;
    margin: 0 0 24px 0;
    letter-spacing: -0.5px;
  }

  /* ---------- CARD ---------- */
  .card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(100,80,255,0.12);
    border-radius: 16px;
    padding: 28px 32px;
    backdrop-filter: blur(10px);
    margin-bottom: 16px;
  }
  .card-title {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6060a0;
    margin-bottom: 20px;
    font-family: 'JetBrains Mono', monospace;
  }

  /* ---------- RESULT CARD ---------- */
  .result-safe {
    background: linear-gradient(135deg, rgba(0,210,100,0.08), rgba(0,150,80,0.04));
    border: 1px solid rgba(0,210,100,0.25);
    border-radius: 20px;
    padding: 36px 40px;
    text-align: center;
    position: relative;
    overflow: hidden;
  }
  .result-risk {
    background: linear-gradient(135deg, rgba(255,60,100,0.10), rgba(200,20,60,0.05));
    border: 1px solid rgba(255,60,100,0.30);
    border-radius: 20px;
    padding: 36px 40px;
    text-align: center;
    position: relative;
    overflow: hidden;
  }
  .result-neutral {
    background: linear-gradient(135deg, rgba(100,80,255,0.08), rgba(60,50,200,0.04));
    border: 1px solid rgba(100,80,255,0.20);
    border-radius: 20px;
    padding: 36px 40px;
    text-align: center;
  }
  .result-icon { font-size: 48px; margin-bottom: 12px; }
  .result-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #5050a0;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 8px;
  }
  .result-prob {
    font-family: 'Syne', sans-serif;
    font-size: 64px;
    font-weight: 800;
    letter-spacing: -3px;
    line-height: 1;
    margin-bottom: 8px;
  }
  .result-verdict {
    font-size: 17px;
    font-weight: 500;
    opacity: 0.75;
    margin-top: 6px;
  }

  /* ---------- PROGRESS BAR ---------- */
  .prob-bar-wrap { margin: 24px 0 8px; }
  .prob-bar-bg {
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 3px;
    overflow: hidden;
  }
  .prob-bar-fill-safe { height: 100%; background: linear-gradient(90deg, #00d264, #00ffa0); border-radius: 3px; }
  .prob-bar-fill-risk { height: 100%; background: linear-gradient(90deg, #ff3c64, #ff7090); border-radius: 3px; }

  /* ---------- FACTOR ROW ---------- */
  .factor-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 13px;
    color: #c0c0e0;
  }
  .factor-dot-pos { width: 8px; height: 8px; border-radius: 50%; background: #ff3c64; flex-shrink: 0; }
  .factor-dot-neg { width: 8px; height: 8px; border-radius: 50%; background: #00d264; flex-shrink: 0; }
  .factor-bar-bg {
    flex: 1;
    height: 4px;
    background: rgba(255,255,255,0.04);
    border-radius: 2px;
    overflow: hidden;
  }
  .factor-bar-pos { height: 100%; background: rgba(255,60,100,0.5); border-radius: 2px; }
  .factor-bar-neg { height: 100%; background: rgba(0,210,100,0.5); border-radius: 2px; }
  .factor-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #5050a0;
    width: 48px;
    text-align: right;
    flex-shrink: 0;
  }

  /* Streamlit widget overrides */
  .stSelectbox > div > div,
  .stNumberInput > div > div > input,
  .stSlider {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(100,80,255,0.2) !important;
    color: #e8e8f0 !important;
    border-radius: 10px !important;
  }

  label { color: #8080b0 !important; font-size: 12px !important; font-weight: 500 !important; }

  div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #6450ff, #4a3acc) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 14px 32px !important;
    width: 100% !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 4px 24px rgba(100,80,255,0.3) !important;
    transition: all 0.2s ease !important;
  }
  div[data-testid="stButton"] button:hover {
    box-shadow: 0 6px 32px rgba(100,80,255,0.5) !important;
    transform: translateY(-1px) !important;
  }

  /* Divider */
  hr { border-color: rgba(100,80,255,0.1) !important; }

  /* Hide streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }

  /* Custom scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #060612; }
  ::-webkit-scrollbar-thumb { background: rgba(100,80,255,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="badge">📡 Telecom Intelligence</div>
  <h1 class="hero-title">ChurnSense<br>Predict. Retain. Grow.</h1>
  <p class="hero-sub">Logistic Regression model trained on telecom subscriber behavior. Get real-time churn probability with feature-level explainability.</p>
  <div class="hero-stats">
    <div class="stat-pill"><span class="stat-val">45</span><span class="stat-lbl">Feature Model</span></div>
    <div class="stat-pill"><span class="stat-val">L2</span><span class="stat-lbl">Regularization</span></div>
    <div class="stat-pill"><span class="stat-val">lbfgs</span><span class="stat-lbl">Solver</span></div>
    <div class="stat-pill"><span class="stat-val">2-class</span><span class="stat-lbl">Binary Output</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="content-wrap">', unsafe_allow_html=True)

# ── LAYOUT ────────────────────────────────────────────────────────────────────
left_col, gap, right_col = st.columns([5, 0.3, 3])

with left_col:
    st.markdown('<div class="section-label">Input Parameters</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Customer Profile</div>', unsafe_allow_html=True)

    # ── ROW 1: Demographics ──────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">👤 Demographics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with c2:
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    with c3:
        partner = st.selectbox("Has Partner", ["Yes", "No"])
    with c4:
        dependents = st.selectbox("Dependents", ["No", "Yes"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── ROW 2: Account ───────────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">💳 Account & Billing</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
    with c2:
        monthly = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)
    with c3:
        total = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(tenure * monthly), step=1.0)

    c4, c5, c6 = st.columns(3)
    with c4:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    with c5:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    with c6:
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── ROW 3: Phone Services ────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">📞 Phone Services</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        phone = st.selectbox("Phone Service", ["Yes", "No"])
    with c2:
        multi = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── ROW 4: Internet Services ─────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">🌐 Internet & Add-ons</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    with c2:
        online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    with c3:
        online_bk = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

    c4, c5, c6 = st.columns(3)
    with c4:
        device = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    with c5:
        tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    with c6:
        tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])

    c7, _, _ = st.columns(3)
    with c7:
        movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── PREDICT BUTTON ───────────────────────────────────────────────────────
    predict_btn = st.button("⚡  Run Churn Analysis")

# ── RIGHT PANEL: Results ──────────────────────────────────────────────────────
with right_col:
    st.markdown('<div class="section-label">Analysis Output</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)

    if predict_btn:
        # ── Build feature vector ─────────────────────────────────────────────
        feat = {f: 0.0 for f in FEATURE_NAMES}

        # Numeric
        feat['SeniorCitizen'] = 1.0 if senior == "Yes" else 0.0
        feat['tenure'] = float(tenure)
        feat['MonthlyCharges'] = float(monthly)
        feat['TotalCharges'] = float(total)

        # One-hot: gender
        feat[f'gender_{gender}'] = 1.0

        # One-hot: partner
        feat[f'Partner_{partner}'] = 1.0

        # One-hot: dependents
        feat[f'Dependents_{dependents}'] = 1.0

        # One-hot: phone
        feat[f'PhoneService_{phone}'] = 1.0

        # One-hot: multi lines
        feat[f'MultipleLines_{multi}'] = 1.0

        # One-hot: internet
        feat[f'InternetService_{internet}'] = 1.0

        # One-hot: online security
        feat[f'OnlineSecurity_{online_sec}'] = 1.0

        # One-hot: online backup
        feat[f'OnlineBackup_{online_bk}'] = 1.0

        # One-hot: device protection
        feat[f'DeviceProtection_{device}'] = 1.0

        # One-hot: tech support
        feat[f'TechSupport_{tech}'] = 1.0

        # One-hot: streaming TV
        feat[f'StreamingTV_{tv}'] = 1.0

        # One-hot: streaming movies
        feat[f'StreamingMovies_{movies}'] = 1.0

        # One-hot: contract
        feat[f'Contract_{contract}'] = 1.0

        # One-hot: paperless billing
        feat[f'PaperlessBilling_{paperless}'] = 1.0

        # Payment method mapping
        pm_map = {
            "Electronic check": "Electronic check",
            "Mailed check": "Mailed check",
            "Bank transfer": "Bank transfer",
            "Credit card": "Credit card",
        }
        feat[f'PaymentMethod_{pm_map[payment]}'] = 1.0

        X = np.array([[feat[f] for f in FEATURE_NAMES]])
        prob = model.predict_proba(X)[0][1]
        pct = int(prob * 100)
        is_risk = prob >= 0.5

        # Result card
        if is_risk:
            icon, color_class, verdict = "⚠️", "result-risk", "High Churn Risk Detected"
            prob_color = "#ff3c64"
            bar_class = "prob-bar-fill-risk"
        else:
            icon, color_class, verdict = "✅", "result-safe", "Customer Likely to Stay"
            prob_color = "#00d264"
            bar_class = "prob-bar-fill-safe"

        st.markdown(f"""
        <div class="{color_class}">
          <div class="result-icon">{icon}</div>
          <div class="result-label">Churn Probability</div>
          <div class="result-prob" style="color:{prob_color}">{pct}%</div>
          <div class="result-verdict">{verdict}</div>
          <div class="prob-bar-wrap">
            <div class="prob-bar-bg">
              <div class="{bar_class}" style="width:{pct}%"></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Feature contributions ─────────────────────────────────────────────
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="card"><div class="card-title">🔍 Top Churn Drivers</div>', unsafe_allow_html=True)

        coefs = model.coef_[0]
        contributions = [(FEATURE_NAMES[i], coefs[i] * X[0][i]) for i in range(len(FEATURE_NAMES)) if X[0][i] != 0]
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        top = contributions[:8]
        max_abs = max(abs(c[1]) for c in top) if top else 1

        for name, val in top:
            is_pos = val > 0
            dot_cls = "factor-dot-pos" if is_pos else "factor-dot-neg"
            bar_cls = "factor-bar-pos" if is_pos else "factor-bar-neg"
            width_pct = int(abs(val) / max_abs * 100)
            sign = "+" if is_pos else ""
            clean = name.replace("_", " ").replace("  ", " ")
            st.markdown(f"""
            <div class="factor-item">
              <div class="{dot_cls}"></div>
              <span style="flex:1;font-size:12px;color:#9090c0">{clean}</span>
              <div class="factor-bar-bg"><div class="{bar_cls}" style="width:{width_pct}%"></div></div>
              <span class="factor-val">{sign}{val:.3f}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Model info ───────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="card">
          <div class="card-title">🧠 Model Metadata</div>
          <div class="factor-item"><span>Algorithm</span><span style="color:#a090ff;font-family:'JetBrains Mono',monospace;font-size:12px">Logistic Regression</span></div>
          <div class="factor-item"><span>Penalty</span><span style="color:#a090ff;font-family:'JetBrains Mono',monospace;font-size:12px">L2 (Ridge)</span></div>
          <div class="factor-item"><span>Solver</span><span style="color:#a090ff;font-family:'JetBrains Mono',monospace;font-size:12px">lbfgs</span></div>
          <div class="factor-item"><span>C (Regularization)</span><span style="color:#a090ff;font-family:'JetBrains Mono',monospace;font-size:12px">1.0</span></div>
          <div class="factor-item"><span>Raw Score</span><span style="color:#a090ff;font-family:'JetBrains Mono',monospace;font-size:12px">{prob:.6f}</span></div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="result-neutral">
          <div class="result-icon">📡</div>
          <div class="result-label">Awaiting Input</div>
          <div style="font-size:14px;color:#5050a0;margin-top:8px;line-height:1.6">
            Fill in the customer profile on the left, then click<br>
            <strong style="color:#8070d0">Run Churn Analysis</strong> to get predictions.
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card" style="margin-top:16px">
          <div class="card-title">📘 How It Works</div>
          <div style="font-size:13px;color:#5050a0;line-height:1.8">
            1. Enter customer demographics<br>
            2. Set billing & contract details<br>
            3. Configure service add-ons<br>
            4. Get probability score + drivers
          </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # .content-wrap