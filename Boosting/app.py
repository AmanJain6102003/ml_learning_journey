import streamlit as st
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

# Page config
st.set_page_config(page_title="Exam Score Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Student Exam Score Predictor")
st.write("Enter student details below to predict their final exam score using the robust JSON-serialized model.")

# Dynamically find the absolute path where app.py resides
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xgb_model.json"
ENCODER_PATH = BASE_DIR / "label_encoders.json"

# Load JSON artifacts safely
@st.cache_resource
def load_artifacts():
    # Verify files exist before loading to catch structural errors early
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing XGBoost model at: {MODEL_PATH}")
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"Missing Label Encoders configuration at: {ENCODER_PATH}")

    # Load XGBoost JSON model using absolute path string
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    
    # Load Label Encoders configuration using absolute path
    with open(ENCODER_PATH, 'r') as json_file:
        encoders_config = json.load(json_file)
        
    return model, encoders_config

try:
    model, encoders_config = load_artifacts()
    st.success("Model & Configuration loaded successfully from native JSON!")
except Exception as e:
    st.error(f"Error loading configuration files: {e}")
    st.stop()

# Setup inputs
st.subheader("Student Details Input")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=20, step=1)
    gender = st.selectbox("Gender", options=encoders_config['gender'])
    course = st.selectbox("Course", options=encoders_config['course'])
    study_method = st.selectbox("Study Method", options=encoders_config['study_method'])
    study_hours = st.number_input("Weekly Study Hours", min_value=0.0, max_value=168.0, value=5.0, step=0.5)

with col2:
    class_attendance = st.slider("Class Attendance (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.5)
    internet_access = st.selectbox("Internet Access", options=encoders_config['internet_access'])
    sleep_hours = st.number_input("Average Daily Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
    sleep_quality = st.selectbox("Sleep Quality", options=encoders_config['sleep_quality'])
    facility_rating = st.selectbox("Facility Rating", options=encoders_config['facility_rating'])
    exam_difficulty = st.selectbox("Exam Difficulty", options=encoders_config['exam_difficulty'])

if st.button("🎯 Predict Exam Score"):
    try:
        # Map selected categories to integer tokens using index values from classes arrays
        input_data = {
            'course': encoders_config['course'].index(course),
            'study_hours': study_hours,
            'class_attendance': class_attendance,
            'sleep_hours': sleep_hours,
            'sleep_quality': encoders_config['sleep_quality'].index(sleep_quality),
            'study_method': encoders_config['study_method'].index(study_method),
            'facility_rating': encoders_config['facility_rating'].index(facility_rating)
        }
        
        # Convert mapped data to DataFrame aligned with X_train feature layout
        input_df = pd.DataFrame([input_data])
        
        # Predict
        prediction = model.predict(input_df)[0]
        
        st.balloons()
        st.markdown(f"### 📝 Predicted Exam Score: **{prediction:.2f}**")
    except Exception as e:
        st.error(f"Prediction error: {e}")
