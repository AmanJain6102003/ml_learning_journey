import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Load all saved components
@st.cache_resource
def load_models():
    with open("taxi_price_prediction_ridge_Regression.pkl", "rb") as f:
        ridge_model = pickle.load(f)
    with open("taxi_price_prediction_lasso_Regression.pkl", "rb") as f:
        lasso_model = pickle.load(f)
    with open("scale.pkl", "rb") as f:
        scaler = pickle.load(f)
    return ridge_model, lasso_model, scaler

try:
    ridge_model, lasso_model, scaler = load_models()
except FileNotFoundError as e:
    st.error(f"Error loading files: {e}. Please ensure both models and scale.pkl are in the same folder.")
    st.stop()

# 2. App Layout
st.title("🚖 Taxi Trip Price Prediction Comparison")
st.write("Enter the trip details below to get predictions from both **Ridge** and **Lasso** regression models.")

# 3. Input Panels
st.header("1. Trip Parameters")
col1, col2 = st.columns(2)

with col1:
    trip_distance = st.number_input("Trip Distance (km)", min_value=0.0, max_value=200.0, value=25.0, step=0.1)
    passenger_count = st.number_input("Passenger Count", min_value=1, max_value=10, value=2, step=1)
    base_fare = st.number_input("Base Fare ($)", min_value=0.0, max_value=50.0, value=3.50, step=0.1)

with col2:
    per_km_rate = st.number_input("Per Km Rate ($)", min_value=0.0, max_value=10.0, value=1.20, step=0.05)
    per_minute_rate = st.number_input("Per Minute Rate ($)", min_value=0.0, max_value=5.0, value=0.30, step=0.05)
    trip_duration = st.number_input("Trip Duration (Minutes)", min_value=0.0, max_value=500.0, value=30.0, step=1.0)

st.header("2. Environment Conditions")
col3, col4 = st.columns(2)

with col3:
    time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
    day_of_week = st.selectbox("Day of Week", ["Weekday", "Weekend"])

with col4:
    traffic_conditions = st.selectbox("Traffic Conditions", ["Low", "Medium", "High"])
    weather = st.selectbox("Weather", ["Clear", "Rain", "Snow", "Cloudy"])

# --- NEW DATA PROCESSING CORRECTION ---

# A. Create a DataFrame for ONLY the numerical columns that the scaler expects
numerical_cols = ['Trip_Distance_km', 'Passenger_Count', 'Base_Fare', 'Per_Km_Rate', 'Per_Minute_Rate', 'Trip_Duration_Minutes']
num_data = {
    'Trip_Distance_km': trip_distance,
    'Passenger_Count': passenger_count,
    'Base_Fare': base_fare,
    'Per_Km_Rate': per_km_rate,
    'Per_Minute_Rate': per_minute_rate,
    'Trip_Duration_Minutes': trip_duration
}
num_df = pd.DataFrame([num_data])[numerical_cols]

# B. Define all expected dummy features exactly as they look after pd.get_dummies()
dummy_cols = [
    'Time_of_Day_Afternoon', 'Time_of_Day_Evening', 'Time_of_Day_Morning', 'Time_of_Day_Night',
    'Day_of_Week_Weekday', 'Day_of_Week_Weekend',
    'Traffic_Conditions_High', 'Traffic_Conditions_Low', 'Traffic_Conditions_Medium',
    'Weather_Clear', 'Weather_Cloudy', 'Weather_Rain', 'Weather_Snow'
]

# Create a DataFrame for categorical items initialized to 0
cat_df = pd.DataFrame(0, index=[0], columns=dummy_cols)

# Set the active dropdown fields to 1
if f"Time_of_Day_{time_of_day}" in cat_df.columns:
    cat_df[f"Time_of_Day_{time_of_day}"] = 1
if f"Day_of_Week_{day_of_week}" in cat_df.columns:
    cat_df[f"Day_of_Week_{day_of_week}"] = 1
if f"Traffic_Conditions_{traffic_conditions}" in cat_df.columns:
    cat_df[f"Traffic_Conditions_{traffic_conditions}"] = 1
if f"Weather_{weather}" in cat_df.columns:
    cat_df[f"Weather_{weather}"] = 1

# 5. Prediction Execution
st.markdown("---")
if st.button("Generate Predictions", type="primary"):
    # 1. Scale ONLY the numerical features to satisfy scale.pkl
    scaled_num_array = scaler.transform(num_df)
    scaled_num_df = pd.DataFrame(scaled_num_array, columns=numerical_cols)
    
    # 2. Combine the scaled numerical features with the binary dummy features
    # This matches the final X_train shape used during model.fit()
    final_input_df = pd.concat([scaled_num_df, cat_df], axis=1)
    
    # Run predictions on both loaded estimator models
    ridge_pred = ridge_model.predict(final_input_df)
    lasso_pred = lasso_model.predict(final_input_df)
    
    # Render outputs side-by-side
    out_col1, out_col2 = st.columns(2)
    
    with out_col1:
        st.metric(
            label="📈 Ridge Regression Estimate", 
            value=f"${round(float(ridge_pred[0]), 2)}"
        )
        
    with out_col2:
        st.metric(
            label="📉 Lasso Regression Estimate", 
            value=f"${round(float(lasso_pred[0]), 2)}"
        )
