
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("anc_late_prediction_model.pkl")

# Page configuration
st.set_page_config(
    page_title="ANC Late Attendance Prediction",
    page_icon="🤰",
    layout="centered"
)

# Header
st.title("🤰 ANC Late Attendance Risk Prediction")

st.markdown(
    """
    ### Kabarak Health Centre
    **Machine Learning–Based Public Health Decision-Support Tool**
    """
)

st.write(
    "This application estimates the probability that a pregnant woman "
    "may initiate antenatal care (ANC) late, based on selected "
    "sociodemographic, obstetric and access-related factors."
)

st.info(
    "📌 **Project definition:** Late ANC is defined as first ANC "
    "attendance at ≥13 weeks of gestation."
)

st.warning(
    "⚠️ **Important:** This is a screening and decision-support tool. "
    "It does not replace clinical assessment, counselling, or "
    "professional judgment."
)

# Client information
st.header("👩🏽‍⚕️ Enter Client Information")

age = st.number_input(
    "Age (years)",
    min_value=10,
    max_value=60,
    value=25,
    step=1
)

parity = st.number_input(
    "Parity",
    min_value=0,
    max_value=15,
    value=0,
    step=1
)

gravidity = st.number_input(
    "Gravidity",
    min_value=1,
    max_value=15,
    value=1,
    step=1
)

previous_anc_attendance = st.number_input(
    "Previous ANC Attendance",
    min_value=0,
    max_value=20,
    value=0,
    step=1
)

distance = st.number_input(
    "Distance to health facility (km)",
    min_value=0.0,
    max_value=200.0,
    value=5.0,
    step=0.5
)

marital_status = st.selectbox(
    "Marital Status",
    ["married", "single"]
)

pregnancy_intention = st.selectbox(
    "Pregnancy Intention",
    ["planned", "unplanned"]
)

social_support = st.selectbox(
    "Social Support",
    ["yes", "no"]
)

transport_difficulty = st.selectbox(
    "Transport Difficulty",
    ["yes", "no"]
)

previous_anc_experience = st.selectbox(
    "Previous ANC Experience",
    ["Good", "poor"]
)

education_level = st.selectbox(
    "Education Level",
    ["Primary", "Secondary", "College"]
)

income_level = st.selectbox(
    "Income Level",
    ["low", "medium"]
)

# Prediction
if st.button("🔍 Predict ANC Attendance Risk"):

    input_data = pd.DataFrame({
        "AGE": [age],
        "PARITY": [parity],
        "GRAVIDITY": [gravidity],
        "MARITAL STATUS": [marital_status],
        "PRVIOUS ANC ATTENDANCE": [previous_anc_attendance],
        "PREGNANCY INTENTION": [pregnancy_intention],
        "DISTANCE(KM)": [distance],
        "SOCIAL SUPPORT": [social_support],
        "TRANSPORT DIFFICULTY": [transport_difficulty],
        "PREVIOUS ANC EXPERIENCE": [previous_anc_experience],
        "EDUCATIONLEVEL": [education_level],
        "INCOME LEVEL": [income_level]
    })

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    early_probability = probabilities[0]
    late_probability = probabilities[1]

    # Results
    st.divider()
    st.header("📊 Prediction Result")

    if prediction == 1:

        st.error("⚠️ Predicted Late ANC")

        st.metric(
            "Probability of Late ANC",
            f"{late_probability * 100:.1f}%"
        )

        st.metric(
            "Probability of Early/Timely ANC",
            f"{early_probability * 100:.1f}%"
        )

        st.warning(
            "The model estimates a higher probability of late ANC "
            "initiation."
        )

    else:

        st.success("✅ Predicted Early/Timely ANC")

        st.metric(
            "Probability of Early/Timely ANC",
            f"{early_probability * 100:.1f}%"
        )

        st.metric(
            "Probability of Late ANC",
            f"{late_probability * 100:.1f}%"
        )

        st.success(
            "The model estimates a higher probability of early/timely "
            "ANC initiation."
        )

    # Interpretation
    st.subheader("ℹ️ Interpretation")

    st.write(
        "The probabilities represent the model's estimated likelihood "
        "for each outcome based on the information entered. They should "
        "be interpreted together with community assessment, counselling "
        "and professional judgment."
    )

    # Recommended community action
    st.subheader("🩺 Recommended Community Action")

    if prediction == 1:

        st.warning(
            "The client has a higher predicted probability of late ANC "
            "initiation. The CHP should consider early counselling, "
            "addressing barriers to ANC attendance, and encouraging "
            "prompt linkage to the health facility."
        )

    else:

        st.info(
            "The client has a higher predicted probability of early/timely "
            "ANC initiation. The CHP should continue reinforcing the "
            "importance of early ANC attendance and provide appropriate "
            "health education."
        )

    st.caption(
        "Model used: Logistic Regression | "
        "Prediction threshold: 0.50"
    )

# Footer
st.divider()

st.caption(
    "ANC Late Attendance Prediction Project | "
    "Kabarak Health Centre | Machine Learning for Public Health"
)

st.caption(
    "For research and decision-support purposes only. "
    "This application is not a diagnostic tool."
)
