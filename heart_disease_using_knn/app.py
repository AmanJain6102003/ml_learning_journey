
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# 1. Page Config with Dark/Clean Medical Persona
st.set_page_config(
    page_title="CardioPulse AI | Diagnostic Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling injectors for polished UI elements
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 15px;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.05em;
        background: linear-gradient(90deg, #ff4b4b, #ff7676);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# 2. Optimized Artifact Caching Loader with Explicit Path Matching
@st.cache_resource
def load_artifacts():
    # Resolve the directory where this script runs from dynamically
    current_dir = Path(__file__).parent.resolve()
    
    model_path = current_dir / "detect_heart_disease.pkl"
    scaler_path = current_dir / "scaler.pkl"
    
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error(f"⚠️ **Artifact Error:** Workspace search failed at location: `{current_dir}`")
        st.info("💡 Make sure both `detect_heart_disease.pkl` and `scaler.pkl` are pushed directly into the same GitHub directory as this script.")
        return None, None

model, scaler = load_artifacts()

# 3. Sidebar Brand Navigation panel with st.sidebar:
with st.sidebar:
    st.markdown("## 📊 System Overview")
    st.info("💡 Change the parameters on the right grid panel, then click **Run Diagnostic Pipeline** below to see updating real-time predictions.")
    st.markdown("---")
    st.markdown("### 🧬 Core Framework Pipeline")
    st.write("- **Engine:** K-Nearest Neighbors Classifier")
    st.write("- **Preprocessing:** Standard Scaler Matrix")
    st.write("- **Compliance:** General Research Model Only")
    st.markdown("---")
    st.caption("CardioPulse AI Dashboard Framework")

# 4. Main Body Header Layout
st.markdown('<p class="main-title">⚡ CardioPulse AI Diagnostic Dashboard</p>', unsafe_allow_html=True)
st.write("A highly precise machine learning interface tracking interactive biometric signals for automated patient risk indicators.")
st.markdown("---")

# 5. Core Feature Input Structure (Grouped into clean visual columns)
col1, col2, col3 = st.columns([1, 1, 1], gap="large")

with col1:
    st.markdown("### 👤 1. Patient Demographics")
    age = st.slider("Patient Age", min_value=1, max_value=120, value=50, help="Slide or use arrows to input structural age index.")
    sex = st.radio(
        "Biological Sex",
        options=[0, 1],
        format_func=lambda x: "Female 👩‍⚕️" if x == 0 else "Male 👨‍⚕️",
        horizontal=True
    )
    st.markdown("---")
    st.markdown("### 🩺 2. Resting Clinical Vitals")
    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=80,
        max_value=200,
        value=120,
        help="Resting system arterial pressure measured on admission."
    )
    chol = st.number_input(
        "Serum Cholesterol (mg/dl)",
        min_value=100,
        max_value=600,
        value=200,
        help="Serum cholesterol index captured via fluoroscopy baseline."
    )

with col2:
    st.markdown("### 🫀 3. Cardiac Symptoms")
    cp = st.selectbox(
        "Chest Pain Classification (cp)",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0: Typical Angina",
            1: "1: Atypical Angina",
            2: "2: Non-anginal Pain",
            3: "3: Asymptomatic"
        }[x],
        help="Subjective pain type thresholds registered by patient."
    )
    exang = st.selectbox(
        "Exercise Induced Angina (exang)",
        options=[0, 1],
        format_func=lambda x: "No Chest Pain during Strain" if x == 0 else "Yes - Pain Induced by Strain"
    )
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl (fbs)",
        options=[0, 1],
        format_func=lambda x: "Normal range (< 120 mg/dl)" if x == 0 else "Elevated range (> 120 mg/dl)"
    )

with col3:
    st.markdown("### 📈 4. Electrocardiogram Metrics")
    thalach = st.slider("Maximum Heart Rate (thalach)", min_value=60, max_value=220, value=150)
    oldpeak = st.number_input("ST Depression Value (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox(
        "Slope of Peak ST Segment",
        options=[0, 1, 2],
        format_func=lambda x: f"Type {x}: Segment Category Descriptor"
    )
    restecg = st.selectbox("Resting ECG Result Configuration", options=[0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-4)", options=[0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal index)", options=[0, 1, 2, 3])

st.markdown("<br>", unsafe_allow_html=True)

# 6. Full-Width Clean Call-To-Action Button Row
predict_btn = st.button("🚀 Run Diagnostic Evaluation Pipeline", use_container_width=True, type="primary")

# 7. Processing Pipeline Inference Block
if predict_btn and model is not None and scaler is not None:
    # Pack input variables in exact dataframe structural ordering matching initial scaler parameters
    feature_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    
    input_df = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak, slope, ca, thal
    ]], columns=feature_names).astype(float)
    
    # Execute structural standardizing transformation matrix safely
    scaled_features = scaler.transform(input_df)
    
    # Compute classifier prediction array output
    prediction = model.predict(scaled_features)
    
    st.markdown("---")
    st.markdown("## 🔍 Diagnostic Output Analytics")
    
    # Create beautiful callout columns showing dynamically updating summary alerts
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        if prediction[0] == 1:
            st.error("🚨 **CRITICAL ALERT DETECTED**")
            st.markdown(
                "<div style='background-color:#ffebeb; border-left:6px solid #ff4b4b; padding:15px; border-radius:6px; color:#5c0000; font-weight:700;'>"
                "The system reports a HIGH probability index for cardiovascular risk patterns. Immediate verification is advised."
                "</div>",
                unsafe_allow_html=True
            )
        else:
            st.success("✅ **STABLE MATRIX DETECTED**")
            st.markdown(
                "<div style='background-color:#ebeffe; border-left:6px solid #2ea44f; padding:15px; border-radius:6px; color:#003d12; font-weight:700;'>"
                "The patient metric vectors fit well within lower risk bounds. All standard vital configurations are tracking nominal parameters."
                "</div>",
                unsafe_allow_html=True
            )
            
    with res_col2:
        # Mini diagnostic feature log grid view
        st.markdown("**Diagnostic Vector Summary Logs:**")
        st.json({
            "Age Input Tracker": age,
            "Target Pulse Ceiling": thalach,
            "ST Depress Threshold": oldpeak,
            "Calculated Decision Integer": int(prediction[0])
        })
