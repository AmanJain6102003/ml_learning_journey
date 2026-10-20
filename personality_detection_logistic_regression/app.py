import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Personality Predictor", page_icon="🧭", layout="wide")

BASE = Path(__file__).parent

# ---------- Feature definitions (ORDER MUST MATCH TRAINING) ----------
# Every feature is assumed to be a 1-10 rating. Change MIN/MAX if yours differ.
MIN_VAL, MAX_VAL, DEFAULT = 1, 10, 5

GROUPS = {
    "Social life": [
        ("social_energy", "Energy I get from being around people"),
        ("alone_time_preference", "Preference for spending time alone"),
        ("talkativeness", "How talkative I am"),
        ("group_comfort", "Comfort in group settings"),
        ("party_liking", "Enjoyment of parties and gatherings"),
        ("friendliness", "How friendly I am with new people"),
        ("online_social_usage", "Time spent on social media"),
        ("public_speaking_comfort", "Comfort speaking in public"),
    ],
    "How I think and feel": [
        ("deep_reflection", "Tendency to reflect deeply"),
        ("listening_skill", "Listening skill"),
        ("empathy", "Empathy toward others"),
        ("creativity", "Creativity"),
        ("curiosity", "Curiosity"),
        ("emotional_stability", "Emotional stability"),
        ("stress_handling", "Ability to handle stress"),
        ("decision_speed", "Speed of making decisions"),
    ],
    "How I work": [
        ("organization", "Organization"),
        ("planning", "Planning ahead"),
        ("routine_preference", "Preference for routine"),
        ("spontaneity", "Spontaneity"),
        ("leadership", "Leadership"),
        ("work_style_collaborative", "Preference for collaborative work"),
    ],
    "Interests and risk": [
        ("risk_taking", "Willingness to take risks"),
        ("excitement_seeking", "Excitement seeking"),
        ("adventurousness", "Adventurousness"),
        ("travel_desire", "Desire to travel"),
        ("reading_habit", "Reading habit"),
        ("sports_interest", "Interest in sports"),
        ("gadget_usage", "Gadget and tech usage"),
    ],
}

FEATURE_ORDER = [
    "social_energy", "alone_time_preference", "talkativeness", "deep_reflection",
    "group_comfort", "party_liking", "listening_skill", "empathy", "creativity",
    "organization", "leadership", "risk_taking", "public_speaking_comfort",
    "curiosity", "routine_preference", "excitement_seeking", "friendliness",
    "emotional_stability", "planning", "spontaneity", "adventurousness",
    "reading_habit", "sports_interest", "online_social_usage", "travel_desire",
    "gadget_usage", "work_style_collaborative", "decision_speed", "stress_handling",
]

# Short descriptions shown with the result. Keys are matched case-insensitively
# against the class names stored in your label encoder.
DESCRIPTIONS = {
    "introvert": "You recharge in quiet, think before you speak, and tend to prefer depth over breadth in your relationships.",
    "extrovert": "You gain energy from people, speak up easily, and thrive in lively, social settings.",
    "ambivert": "You move comfortably between social and solitary modes, adapting to what the moment calls for.",
}


# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    with open(BASE / "personality_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(BASE / "label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, scaler, encoder


# ---------- Styling ----------
st.markdown(
    """
    <style>
    .block-container {max-width: 1100px; padding-top: 2rem;}
    h1 {font-weight: 800; letter-spacing: -0.5px;}
    .hero {
        padding: 1.6rem 1.8rem; border-radius: 16px; margin-bottom: 1.2rem;
        background: linear-gradient(120deg, #1f3b73 0%, #4b2e83 100%); color: #fff;
    }
    .hero h1 {color: #fff; margin: 0 0 .3rem 0;}
    .hero p {margin: 0; opacity: .85;}
    .result {
        padding: 1.6rem; border-radius: 16px; text-align: center;
        border: 1px solid rgba(128,128,128,.3);
    }
    .result .label {font-size: 2.4rem; font-weight: 800; margin: .2rem 0;}
    .result .desc {opacity: .8; max-width: 640px; margin: .4rem auto 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Personality Predictor</h1>
      <p>Rate yourself from 1 (not at all like me) to 10 (very much like me). Answer honestly for the best result.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model, scaler, encoder = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Put the three .pkl files in the same folder as app.py.")
    st.stop()

# Features the model was actually trained on (read from the scaler, in training order)
FEATURES = list(scaler.feature_names_in_)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Options")
    if st.button("Reset all answers", use_container_width=True):
        for key in FEATURES:
            st.session_state[key] = DEFAULT
        st.rerun()
    st.caption("Tip: use the tabs to answer all four sections, then press Predict.")

# ---------- Inputs ----------
values = {}
tabs = st.tabs(list(GROUPS.keys()))
for tab, (group, items) in zip(tabs, GROUPS.items()):
    items = [(k, l) for k, l in items if k in FEATURES]
    with tab:
        cols = st.columns(2)
        for i, (key, label) in enumerate(items):
            with cols[i % 2]:
                values[key] = st.slider(label, MIN_VAL, MAX_VAL, DEFAULT, key=key)

st.write("")
predict = st.button("Predict my personality", type="primary", use_container_width=True)

# ---------- Prediction ----------
if predict:
    X = pd.DataFrame([[values[c] for c in FEATURES]], columns=FEATURES)
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)
    label = encoder.inverse_transform(np.asarray(pred).ravel().astype(int))[0]

    st.write("")
    st.markdown(
        f"""
        <div class="result">
          <div>Your personality type</div>
          <div class="label">{str(label).title()}</div>
          <div class="desc">{DESCRIPTIONS.get(str(label).lower(), "")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        names = [str(c).title() for c in encoder.inverse_transform(model.classes_.astype(int))]
        prob_df = pd.DataFrame({"Type": names, "Probability": proba}).set_index("Type")

        st.subheader("Confidence")
        c1, c2 = st.columns([1, 1])
        with c1:
            for n, p in zip(names, proba):
                st.write(f"**{n}**")
                st.progress(float(p), text=f"{p * 100:.1f}%")
        with c2:
            st.bar_chart(prob_df)

    with st.expander("See the answers used"):
        st.dataframe(X.T.rename(columns={0: "Your rating"}), use_container_width=True)