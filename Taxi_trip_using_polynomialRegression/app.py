import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).parent 

st.set_page_config(page_title="Taxi Trip Price Predictor", page_icon="🚕", layout="centered")

# ---------- Load saved objects ----------
@st.cache_resource
def load_artifacts():
    with open(BASE_DIR / "Scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(BASE_DIR / "poly_transformer.pkl", "rb") as f:
        poly = pickle.load(f)
    with open(BASE_DIR / "taxi_trip_polynomial.pkl", "rb") as f:
        model = pickle.load(f)
    return scaler, poly, model

scaler, poly, model = load_artifacts()

# Column order the scaler was trained on
feature_cols = list(scaler.feature_names_in_)

# ---------- Categorical options ----------
CATEGORICALS = {
    "Time_of_Day": ["Afternoon", "Evening", "Morning", "Night"],
    "Day_of_Week": ["Weekday", "Weekend"],
    "Traffic_Conditions": ["High", "Low", "Medium"],
    "Weather": ["Clear", "Rain", "Snow"],
}

def encode_input(inputs: dict) -> pd.DataFrame:
    """Build a one-row DataFrame matching the training columns.
    Handles both one-hot (get_dummies) and label-encoded training data."""
    row = pd.DataFrame(0.0, index=[0], columns=feature_cols)

    for key, value in inputs.items():
        if key in CATEGORICALS:
            if key in feature_cols:
                # Label encoding (alphabetical order, same as LabelEncoder)
                row.at[0, key] = sorted(CATEGORICALS[key]).index(value)
            else:
                # One-hot encoding: e.g. Weather_Rain
                dummy_col = f"{key}_{value}"
                if dummy_col in feature_cols:
                    row.at[0, dummy_col] = 1.0
        else:
            if key in feature_cols:
                row.at[0, key] = value
    return row

# ---------- UI ----------
st.title("🚕 Taxi Trip Price Predictor")
st.write("Enter the trip details below to estimate the fare.")

col1, col2 = st.columns(2)

with col1:
    trip_distance = st.number_input("Trip Distance (km)", min_value=0.1, max_value=200.0, value=10.0, step=0.5)
    trip_duration = st.number_input("Trip Duration (minutes)", min_value=1.0, max_value=300.0, value=30.0, step=1.0)
    passengers = st.selectbox("Passenger Count", [1, 2, 3, 4], index=0)
    time_of_day = st.selectbox("Time of Day", CATEGORICALS["Time_of_Day"])
    day_of_week = st.selectbox("Day of Week", CATEGORICALS["Day_of_Week"])

with col2:
    traffic = st.selectbox("Traffic Conditions", CATEGORICALS["Traffic_Conditions"])
    weather = st.selectbox("Weather", CATEGORICALS["Weather"])
    base_fare = st.number_input("Base Fare", min_value=0.0, max_value=50.0, value=3.5, step=0.1)
    per_km_rate = st.number_input("Per Km Rate", min_value=0.0, max_value=10.0, value=1.2, step=0.1)
    per_min_rate = st.number_input("Per Minute Rate", min_value=0.0, max_value=5.0, value=0.3, step=0.05)

if st.button("Predict Price", type="primary", use_container_width=True):
    user_input = {
        "Trip_Distance_km": trip_distance,
        "Time_of_Day": time_of_day,
        "Day_of_Week": day_of_week,
        "Passenger_Count": passengers,
        "Traffic_Conditions": traffic,
        "Weather": weather,
        "Base_Fare": base_fare,
        "Per_Km_Rate": per_km_rate,
        "Per_Minute_Rate": per_min_rate,
        "Trip_Duration_Minutes": trip_duration,
    }

    input_df = encode_input(user_input)

    # Same order as training: scale -> polynomial features -> model
    scaled = scaler.transform(input_df)
    poly_features = poly.transform(scaled)
    prediction = model.predict(poly_features)[0]

    st.success(f"Estimated Trip Price: **{max(prediction, 0):.2f}**")

    with st.expander("Show input data"):
        st.dataframe(pd.DataFrame([user_input]))