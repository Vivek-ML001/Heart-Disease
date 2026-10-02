import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Load model and metadata
# --------------------------------------------------

model = joblib.load("lightgbm_model.pkl")
feature_names = joblib.load("feature_names.pkl")
cat_cols = joblib.load("categorical_features.pkl")


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction")
st.write(
    "Enter the patient's information below to estimate the model's "
    "predicted heart-disease risk."
)

st.info(
    "⚠️ This application is for educational/demo purposes only and "
    "is not a medical diagnosis."
)


# --------------------------------------------------
# Input section
# --------------------------------------------------

st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=50
    )

    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x: "Female (0)" if x == 0 else "Male (1)"
    )

    chest_pain = st.selectbox(
        "Chest pain type",
        options=[1, 2, 3, 4]
    )

    bp = st.number_input(
        "Blood Pressure (BP)",
        min_value=50,
        max_value=250,
        value=120
    )

    cholesterol = st.number_input(
        "Cholesterol",
        min_value=50,
        max_value=700,
        value=200
    )

    fbs = st.selectbox(
        "FBS over 120",
        options=[0, 1],
        format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
    )

    ekg = st.selectbox(
        "EKG results",
        options=[0, 1, 2]
    )


with col2:
    max_hr = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150
    )

    exercise_angina = st.selectbox(
        "Exercise angina",
        options=[0, 1],
        format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
    )

    st_depression = st.number_input(
        "ST depression",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    slope = st.selectbox(
        "Slope of ST",
        options=[1, 2, 3]
    )

    vessels = st.selectbox(
        "Number of vessels fluro",
        options=[0, 1, 2, 3]
    )

    thallium = st.selectbox(
        "Thallium",
        options=[3, 6, 7]
    )


# --------------------------------------------------
# Feature engineering
# --------------------------------------------------

# EXACTLY the same as training
age_group = pd.cut(
    pd.Series([age]),
    bins=[0, 40, 50, 60, 70, 100],
    labels=[0, 1, 2, 3, 4]
).iloc[0]

age_chol_ratio = age / (cholesterol + 1)

bp_hr_product = bp * max_hr


# --------------------------------------------------
# Create input dataframe
# --------------------------------------------------

input_data = pd.DataFrame({
    "Age": [age],
    "Sex": [sex],
    "Chest pain type": [chest_pain],
    "BP": [bp],
    "Cholesterol": [cholesterol],
    "FBS over 120": [fbs],
    "EKG results": [ekg],
    "Max HR": [max_hr],
    "Exercise angina": [exercise_angina],
    "ST depression": [st_depression],
    "Slope of ST": [slope],
    "Number of vessels fluro": [vessels],
    "Thallium": [thallium],
    "Age Group": [age_group],
    "age_chol_ratio": [age_chol_ratio],
    "bp_hr_product": [bp_hr_product]
})


# --------------------------------------------------
# Convert categorical columns exactly like training
# --------------------------------------------------

for col in cat_cols:
    input_data[col] = input_data[col].astype("category")


# Ensure exact feature order
input_data = input_data[feature_names]


# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.divider()

if st.button("🔍 Predict Heart Disease", use_container_width=True):

    try:

        prediction = model.predict(input_data)[0]

        # Probability
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1]
        else:
            probability = None


        st.subheader("Prediction Result")

        if prediction == 1:

            st.error("⚠️ Model predicts: Heart Disease")

        else:

            st.success("✅ Model predicts: No Heart Disease")


        if probability is not None:

            st.metric(
                "Predicted Probability",
                f"{probability * 100:.2f}%"
            )

            st.progress(float(probability))


        with st.expander("View processed input"):

            st.dataframe(input_data)


    except Exception as e:

        st.error("Prediction failed.")

        st.exception(e)