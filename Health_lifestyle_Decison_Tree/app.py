import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(page_title="Disease Risk Predictor", page_icon="🩺")

st.title("🩺 Disease Risk Predictor")
st.write("Enter your health indicators to predict disease risk.")

# Load model using an absolute path relative to this script
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "best_decision_tree_model.pkl")
    
    with open(model_path, "rb") as file:
        return pickle.load(file)

try:
    model = load_model()

    st.subheader("Personal Health Inputs")

    col1, col2 = st.columns(2)

    with col1:
        cholesterol = st.number_input(
            "Cholesterol Level", 50, 500, 200
        )
        daily_steps = st.number_input(
            "Daily Steps", 0, 50000, 7500, step=100
        )

    with col2:
        systolic_bp = st.number_input(
            "Systolic Blood Pressure", 80, 220, 120
        )
        resting_hr = st.number_input(
            "Resting Heart Rate", 40, 150, 72
        )

    if st.button("Predict Disease Risk", type="primary"):

        # Feature order must match the training data
        features = np.array([[
            daily_steps, resting_hr, systolic_bp, cholesterol
        ]])

        prediction = model.predict(features)[0]

        st.divider()

        if prediction == 1:
            st.error("⚠️ High Disease Risk Predicted")
        else:
            st.success("✅ Low Disease Risk Predicted")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            classes = list(model.classes_)

            if prediction in classes:
                index = classes.index(prediction)
                st.write(
                    f"Prediction confidence: {probabilities[index]:.2%}"
                )

except FileNotFoundError:
    st.error("Model file not found. Check best_decision_tree_model.pkl.")
except Exception as e:
    st.error(f"Error: {e}")
