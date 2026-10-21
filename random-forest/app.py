import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os  # Added to resolve working directory path issues

st.set_page_config(page_title="Delivery Time Predictor", layout="centered")

st.title("🚚 Food Delivery Time Prediction App")
st.write("Enter the delivery details below to predict the estimated delivery time in minutes.")

@st.cache_resource
def load_assets():
    # Automatically determines the full directory path where app.py lives
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Creates reliable, absolute system paths to your pickle files
    model_path = os.path.join(BASE_DIR, 'best_rf_model.pkl')
    encoder_path = os.path.join(BASE_DIR, 'label_encoder.pkl')
    
    with open(model_path, 'rb') as m_file:
        model = pickle.load(m_file)
    with open(encoder_path, 'rb') as e_file:
        encoder = pickle.load(e_file)
        
    return model, encoder

try:
    model, encoder = load_assets()
except Exception as e:
    st.error(f"Error loading pickle files. Ensure 'best_rf_model.pkl' and 'label_encoder.pkl' are in the same folder as this script. Error: {e}")
    st.stop()

# Layout inputs
col1, col2 = st.columns(2)

with col1:
    distance = st.number_input("Distance (km)", min_value=0.1, max_value=50.0, value=5.0, step=0.1)
    preparation_time = st.number_input("Preparation Time (min)", min_value=1, max_value=120, value=15, step=1)

with col2:
    weather_selected = st.selectbox("Weather Condition", ['Clear', 'Foggy', 'Rainy', 'Snowy', 'Windy'])
    traffic_selected = st.selectbox("Traffic Level", ['Low', 'Medium', 'High'])

if st.button("Predict Delivery Time", use_container_width=True):
    try:
        # Ensure the string encoding maps to what the model expects
        weather_mapping = {val: idx for idx, val in enumerate(['Clear', 'Foggy', 'Rainy', 'Snowy', 'Windy'])}
        traffic_mapping = {val: idx for idx, val in enumerate(['High', 'Low', 'Medium'])}

        weather_encoded = weather_mapping.get(weather_selected, 0)
        traffic_encoded = traffic_mapping.get(traffic_selected, 0)

        # Features: ['Distance_km', 'Weather', 'Traffic_Level', 'Preparation_Time_min']
        features = np.array([[distance, weather_encoded, traffic_encoded, preparation_time]])
        
        prediction = model.predict(features)[0]
        
        st.success(f"✨ Estimated Delivery Time: **{prediction:.2f} minutes**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
