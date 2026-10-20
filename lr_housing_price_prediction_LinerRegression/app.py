
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="USA House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path(__file__).parent / "usa_house_price_model.pkl"

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Place "
        "'usa_house_price_model.pkl' in the same folder as app.py."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


# -----------------------------
# Header
# -----------------------------
st.title("🏡 USA House Price Predictor")
st.markdown(
    "### Estimate a property's market price using Machine Learning"
)
st.caption(
    "Powered by your trained Ridge Regression model"
)

st.divider()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("About this project")
    st.write(
        """
        This application predicts house prices using
        five numerical features from the USA housing dataset.
        """
    )

    st.subheader("Model information")
    st.write("**Algorithm:** Ridge Regression")
    st.write("**Model file:** `usa_house_price_model.pkl`")
    st.write("**Input features:** 5")

    st.info(
        "Enter property details and click Predict House Price."
    )

# -----------------------------
# Input form
# -----------------------------
st.subheader("Enter Property Details")

with st.form("house_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        income = st.number_input(
            "Average Area Income ($)",
            min_value=0.0,
            value=68583.0,
            step=1000.0,
            help="Average income of residents in the area."
        )

        house_age = st.number_input(
            "Average Area House Age (years)",
            min_value=0.0,
            value=5.98,
            step=0.1
        )

        rooms = st.number_input(
            "Average Number of Rooms",
            min_value=0.1,
            value=6.99,
            step=0.1
        )

    with col2:
        bedrooms = st.number_input(
            "Average Number of Bedrooms",
            min_value=0.1,
            value=3.98,
            step=0.1
        )

        population = st.number_input(
            "Area Population",
            min_value=0.0,
            value=36163.0,
            step=1000.0
        )

    submitted = st.form_submit_button(
        "🔍 Predict House Price",
        use_container_width=True
    )

# -----------------------------
# Prediction
# -----------------------------
if submitted:
    # IMPORTANT:
    # Feature names and order must match model training.
    input_data = pd.DataFrame(
        [[income, house_age, rooms, bedrooms, population]],
        columns=[
            "Avg. Area Income",
            "Avg. Area House Age",
            "Avg. Area Number of Rooms",
            "Avg. Area Number of Bedrooms",
            "Area Population"
        ]
    )

    try:
        prediction = float(model.predict(input_data)[0])

        st.divider()
        st.subheader("Prediction Result")

        if prediction < 0:
            st.warning(
                "The model returned a negative price. "
                "Review the inputs and model performance."
            )
        else:
            st.success("Prediction generated successfully!")

            st.metric(
                label="Estimated House Price",
                value=f"${prediction:,.2f}"
            )

            st.caption(
                "This is a model estimate, not a guaranteed "
                "property valuation."
            )

            with st.expander("View submitted input data"):
                st.dataframe(
                    input_data,
                    use_container_width=True,
                    hide_index=True
                )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

else:
    st.info(
        "Fill in the property details above to generate a prediction."
    )
