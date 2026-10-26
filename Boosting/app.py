import os
import pickle
import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Student Exam Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Helper function to find artifacts in the current or parent directory
def find_artifact(filename):
    if os.path.exists(filename):
        return filename
    # Check parent directory just in case
    parent_path = os.path.join("..", filename)
    if os.path.exists(parent_path):
        return parent_path
    return None

# Cache the resources so they only load once into memory
@st.cache_resource
def load_artifacts():
    model_path = find_artifact('xgb_model.json')
    encoders_path = find_artifact('label_encoders.pkl')
    
    if not model_path or not encoders_path:
        raise FileNotFoundError(
            "Required deployment artifacts 'xgb_model.json' or 'label_encoders.pkl' not found. "
            "Please ensure they are in the same folder as app.py."
        )
        
    # Load XGBoost natively
    model = xgb.Booster()
    model.load_model(model_path)
    
    # Load label encoders
    with open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
        
    return model, encoders

# ----------------- MAIN APP APPLICATION -----------------

st.title("🎓 Student Exam Score Predictor")
st.markdown("Enter the student's profile details below to estimate their final exam score.")

try:
    # Load model and encoders
    booster, encoders = load_artifacts()
    
    st.subheader("Student Profiles & Academic Habits")
    
    # Visual Layout Split
    col1, col2 = st.columns(2)
    
    with col1:
        # Categorical choices driven safely by loaded LabelEncoder meta-arrays
        course = st.selectbox("Course Enrolled", options=encoders['course'].classes_)
        study_method = st.selectbox("Primary Study Method", options=encoders['study_method'].classes_)
        sleep_quality = st.selectbox("Sleep Quality", options=encoders['sleep_quality'].classes_)
        facility_rating = st.selectbox("School Facility Rating", options=encoders['facility_rating'].classes_)
        
    with col2:
        # Fixed inputs matching your exact feature footprint
        study_hours = st.number_input("Study Hours per Week", min_value=0.0, max_value=168.0, value=15.0, step=0.5)
        class_attendance = st.slider("Class Attendance Rate (%)", min_value=0, max_value=100, value=85)
        sleep_hours = st.number_input("Sleep Hours per Night", min_value=0.0, max_value=24.0, value=7.0, step=0.5)

    st.markdown("---")
    
    # Prediction trigger button
    if st.button("Predict Exam Score", type="primary"):
        with st.spinner("Calculating prediction..."):
            
            # 1. Transform categorical variables using the loaded encoders
            encoded_course = encoders['course'].transform([course])[0]
            encoded_study_method = encoders['study_method'].transform([study_method])[0]
            encoded_sleep_quality = encoders['sleep_quality'].transform([sleep_quality])[0]
            encoded_facility_rating = encoders['facility_rating'].transform([facility_rating])[0]

            # 2. Rebuilt array map matches strict model sorting rules:
            # ['course', 'study_hours', 'class_attendance', 'sleep_hours', 'sleep_quality', 'study_method', 'facility_rating']
            input_data = {
                'course': [encoded_course],
                'study_hours': [study_hours],
                'class_attendance': [class_attendance],
                'sleep_hours': [sleep_hours],
                'sleep_quality': [encoded_sleep_quality],
                'study_method': [encoded_study_method],
                'facility_rating': [encoded_facility_rating]
            }
            
            input_df = pd.DataFrame(input_data)
            
            # 3. Native booster engine compilation
            dmatrix_input = xgb.DMatrix(input_df)
            
            # 4. Predict and safely extract array scalar output
            prediction = booster.predict(dmatrix_input)[0]
            
            # Bounding predictions to realistic percentage values (0 - 100)
            final_score = float(np.clip(prediction, 0, 100))
            
            # 5. Display result nicely
            st.success(f"### Predicted Exam Score: **{final_score:.2f}%**")
            
            # Add visual feedback cues based on performance brackets
            if final_score >= 75:
                st.balloons()
                st.info("🎯 This profile indicates an excellent performance trajectory!")
            elif final_score < 50:
                st.warning("⚠️ This profile shows risk metrics. Consider increasing study hours or support systems.")

except Exception as e:
    st.error(" An error occurred while running the app.")
    st.exception(e)
