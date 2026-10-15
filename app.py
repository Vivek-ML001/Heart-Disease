import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# --------------------------------------------------
# Load Model and Metadata
# --------------------------------------------------

@st.cache_resource
def load_artifacts():
    model = joblib.load("lightgbm_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    cat_cols = joblib.load("categorical_features.pkl")

    return model, feature_names, cat_cols


try:
    model, feature_names, cat_cols = load_artifacts()
except Exception as e:
    st.error("Unable to load the model files. Check that all required .pkl files exist.")
    st.exception(e)
    st.stop()

# --------------------------------------------------
# Page Header
# --------------------------------------------------

st.title("❤️ Heart Disease Prediction")

st.write(
    "Enter patient information to obtain a prediction "
    "from the trained machine learning model."
)

st.info(
    "⚠️ Educational demonstration only. This application "
    "is not a medical diagnosis or a substitute for professional care."
)

# --------------------------------------------------
# Patient Input Section
# --------------------------------------------------

st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=50,
        step=1
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
        value=120,
        step=1
    )

    cholesterol = st.number_input(
        "Cholesterol",
        min_value=50,
        max_value=700,
        value=200,
        step=1
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
        value=150,
        step=1
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
# Feature Engineering
# Must match the training pipeline exactly
# --------------------------------------------------

age_group = pd.cut(
    pd.Series([age]),
    bins=[0, 40, 50, 60, 70, 100],
    labels=[0, 1, 2, 3, 4]
).iloc[0]

if pd.isna(age_group):
    st.error("Unable to calculate the age group for this input.")
    st.stop()

age_group = int(age_group)

age_chol_ratio = age / (cholesterol + 1)
bp_hr_product = bp * max_hr

# --------------------------------------------------
# Create Input DataFrame
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
# Categorical Conversion
# --------------------------------------------------

for col in cat_cols:
    if col in input_data.columns:
        input_data[col] = input_data[col].astype("category")

# Ensure the exact training feature order
try:
    input_data = input_data[feature_names]
except KeyError as e:
    st.error(
        "The input features do not match the features expected by "
        "the trained model. Check feature_names.pkl and the training code."
    )
    st.exception(e)
    st.stop()

# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.divider()

if st.button("🔍 Predict Heart Disease", use_container_width=True):

    try:
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)

            # Find the probability belonging to the predicted class
            predicted_index = classes.index(prediction)
            predicted_probability = float(
                probabilities[predicted_index]
            )

            # Display predicted class
            if prediction == 1:
                st.error("⚠️ Model predicts: Heart Disease")
            elif prediction == 0:
                st.success("✅ Model predicts: No Heart Disease")
            else:
                st.warning(f"Model predicts class: {prediction}")

            st.metric(
                "Model Confidence in Predicted Class",
                f"{predicted_probability * 100:.2f}%"
            )

            st.progress(
                max(0.0, min(1.0, predicted_probability))
            )

            # Display probability for each class
            st.subheader("Class Probabilities")

            for class_label, class_probability in zip(
                classes, probabilities
            ):
                if class_label == 1:
                    class_name = "Heart Disease"
                elif class_label == 0:
                    class_name = "No Heart Disease"
                else:
                    class_name = f"Class {class_label}"

                st.write(
                    f"**{class_name} ({class_label}):** "
                    f"{float(class_probability) * 100:.2f}%"
                )

            st.caption(
                "Probabilities are estimates from the model, "
                "not a clinical diagnosis."
            )

        else:
            if prediction == 1:
                st.error("⚠️ Model predicts: Heart Disease")
            elif prediction == 0:
                st.success("✅ Model predicts: No Heart Disease")
            else:
                st.warning(f"Model predicts class: {prediction}")

            st.warning(
                "This model does not provide probability estimates."
            )

        # Show processed model input
        with st.expander("View processed input"):
            st.dataframe(input_data)

            with st.expander("Model metadata"):
                st.write("Expected feature names:", feature_names)
                if hasattr(model, "classes_"):
                    st.write("Model class labels:", list(model.classes_))

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
