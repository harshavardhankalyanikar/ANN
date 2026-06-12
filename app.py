import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# ==========================
# Load Model & Preprocessing Files
# ==========================

model = load_model("models/student_ann_model.keras")

scaler = joblib.load("models/scaler.pkl")

feature_columns = joblib.load(
    "models/feature_columns.pkl"
)

# ==========================
# Streamlit UI
# ==========================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓"
)

st.title("🎓 Student Performance Predictor")

st.markdown(
    "Predict student performance using an Artificial Neural Network"
)

# ==========================
# Input Fields
# ==========================

gender = st.selectbox(
    "Gender",
    ["female", "male"]
)

race = st.selectbox(
    "Race/Ethnicity",
    [
        "group A",
        "group B",
        "group C",
        "group D",
        "group E"
    ]
)

parent_education = st.selectbox(
    "Parental Level of Education",
    [
        "associate's degree",
        "bachelor's degree",
        "high school",
        "master's degree",
        "some college",
        "some high school"
    ]
)

lunch = st.selectbox(
    "Lunch Type",
    [
        "free/reduced",
        "standard"
    ]
)

test_prep = st.selectbox(
    "Test Preparation Course",
    [
        "completed",
        "none"
    ]
)

math_score = st.slider(
    "Math Score",
    0,
    100,
    50
)

reading_score = st.slider(
    "Reading Score",
    0,
    100,
    50
)

writing_score = st.slider(
    "Writing Score",
    0,
    100,
    50
)

# ==========================
# Predict Button
# ==========================

if st.button("Predict Performance"):

    input_df = pd.DataFrame({
        "gender": [gender],
        "race/ethnicity": [race],
        "parental level of education": [parent_education],
        "lunch": [lunch],
        "test preparation course": [test_prep],
        "math score": [math_score],
        "reading score": [reading_score],
        "writing score": [writing_score]
    })

    # One-hot encoding
    input_df = pd.get_dummies(input_df)

    # Match training columns exactly
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)

    predicted_class = prediction.argmax(axis=1)[0]

    labels = {
        0: "Poor",
        1: "Good",
        2: "Excellent"
    }

    st.success(
        f"Predicted Performance: {labels[predicted_class]}"
    )

    st.subheader("Prediction Probabilities")

    prob_df = pd.DataFrame(
        prediction,
        columns=["Poor", "Good", "Excellent"]
    )

    st.dataframe(prob_df)