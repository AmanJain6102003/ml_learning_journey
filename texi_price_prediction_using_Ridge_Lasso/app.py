import streamlit as st
import pandas as pd
import pickle

BASE_DIR = Path(__file__).parent 

# 1. Load saved components
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
    st.error(f"Error loading files: {e}. Make sure both models and scale.pkl are in the same folder.")
    st.stop()

# 2. Layout
st.title("🚖 Taxi Trip Price Prediction Comparison")
st.write("Enter the trip details to get predictions from both **Ridge** and **Lasso** regression models.")

# 3. Inputs (only the features the models were trained on)
col1, col2 = st.columns(2)

with col1:
    trip_distance = st.number_input("Trip Distance (km)", min_value=0.0, max_value=200.0, value=25.0, step=0.1)
    per_km_rate = st.number_input("Per Km Rate ($)", min_value=0.0, max_value=10.0, value=1.20, step=0.05)

with col2:
    trip_duration = st.number_input("Trip Duration (Minutes)", min_value=0.0, max_value=500.0, value=30.0, step=1.0)
    per_minute_rate = st.number_input("Per Minute Rate ($)", min_value=0.0, max_value=5.0, value=0.30, step=0.05)

traffic_conditions = st.selectbox("Traffic Conditions", ["Low", "Medium", "High"])

# 4. Prediction
st.markdown("---")
if st.button("Generate Predictions", type="primary"):
    # Scale the 4 numeric columns, in the order the scaler was fitted on
    scaler_cols = list(scaler.feature_names_in_)
    raw = {
        "Trip_Distance_km": trip_distance,
        "Per_Km_Rate": per_km_rate,
        "Trip_Duration_Minutes": trip_duration,
        "Per_Minute_Rate": per_minute_rate,
    }
    num_df = pd.DataFrame([[raw[c] for c in scaler_cols]], columns=scaler_cols)
    scaled = pd.DataFrame(scaler.transform(num_df), columns=scaler_cols)

    # Traffic dummies
    traffic = {
        "Traffic_Conditions_High": 0,
        "Traffic_Conditions_Low": 0,
        "Traffic_Conditions_Medium": 0,
    }
    traffic[f"Traffic_Conditions_{traffic_conditions}"] = 1

    # Combine and order exactly like the training data
    full = pd.concat([scaled, pd.DataFrame([traffic])], axis=1)
    final_input = full[list(ridge_model.feature_names_in_)]

    ridge_pred = ridge_model.predict(final_input)
    lasso_pred = lasso_model.predict(final_input)

    out1, out2 = st.columns(2)
    with out1:
        st.metric("📈 Ridge Regression Estimate", f"${float(ridge_pred[0]):.2f}")
    with out2:
        st.metric("📉 Lasso Regression Estimate", f"${float(lasso_pred[0]):.2f}")