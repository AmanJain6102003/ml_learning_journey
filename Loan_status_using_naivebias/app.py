import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set Page configuration
st.set_page_config(page_title="Loan Default Risk Predictor", layout="centered")

# Load saved artifacts using a reliable absolute path
@st.cache_resource
def load_artifacts():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'loan_nb_model.pkl')
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)

try:
    artifacts = load_artifacts()
    model = artifacts['model']
    label_encoders = artifacts['label_encoders']
    categorical_cols = artifacts['categorical_cols']
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

st.title("🏦 Loan Default Risk Predictor")
st.write("Enter customer details below to estimate the likelihood of loan default.")

# Create Form for Input Fields
with st.form("prediction_form"):
    st.subheader("Personal Demographics")
    col1, col2 = st.columns(2)
    with col1:
        person_age = st.number_input("Age", min_value=18, max_value=120, value=25)
        person_gender = st.selectbox("Gender", options=list(label_encoders['person_gender'].classes_))
        person_education = st.selectbox("Education level", options=list(label_encoders['person_education'].classes_))
    with col2:
        person_income = st.number_input("Annual Income ($)", min_value=1000, max_value=10000000, value=50000)
        person_emp_exp = st.number_input("Employment Experience (years)", min_value=0, max_value=60, value=2)
        person_home_ownership = st.selectbox("Home Ownership", options=list(label_encoders['person_home_ownership'].classes_))

    st.subheader("Loan Details")
    col3, col4 = st.columns(2)
    with col3:
        loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=100000, value=10000)
        loan_intent = st.selectbox("Loan Intent", options=list(label_encoders['loan_intent'].classes_))
        loan_int_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=35.0, value=11.0)
    with col4:
        loan_percent_income = st.number_input("Loan % of Income", min_value=0.0, max_value=1.0, value=0.2)
        cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=40, value=5)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)

    previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults on File?", options=list(label_encoders['previous_loan_defaults_on_file'].classes_))
    
    submit_button = st.form_submit_button(label="Predict Risk")

if submit_button:
    # Map features exactly in the training order
    input_data = {
        'person_age': float(person_age),
        'person_gender': person_gender,
        'person_education': person_education,
        'person_income': float(person_income),
        'person_emp_exp': int(person_emp_exp),
        'person_home_ownership': person_home_ownership,
        'loan_amnt': float(loan_amnt),
        'loan_intent': loan_intent,
        'loan_int_rate': float(loan_int_rate),
        'loan_percent_income': float(loan_percent_income),
        'cb_person_cred_hist_length': float(cb_person_cred_hist_length),
        'credit_score': int(credit_score),
        'previous_loan_defaults_on_file': previous_loan_defaults_on_file
    }
    
    # Process & transform input data
    input_df = pd.DataFrame([input_data])
    for col in categorical_cols:
        le = label_encoders[col]
        input_df[col] = le.transform(input_df[col].astype(str))
        
    # Run prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    
    st.markdown("--- ")
    if prediction == 1:
        st.error(f"🚨 Prediction: **High Risk of Default** (Probability: {prediction_proba[1]:.2%})")
    else:
        st.success(f"✅ Prediction: **Low Risk of Default** (Probability of Default: {prediction_proba[1]:.2%})")
