
import streamlit as st

st.set_page_config(
    page_title="About the Model | Heart Disease Prediction",
    page_icon="ℹ️",
    layout="wide"
)

st.title(" About the Model")
st.caption("Understanding the machine learning system behind the predictions")

st.markdown("---")

# Overview
st.header("1. Model Overview")

st.markdown("""
This application uses **LightGBM (Light Gradient Boosting Machine)**,
a gradient-boosting framework designed for efficient learning on structured
and tabular datasets.

The model analyzes patient-related input features and predicts one of two
classes: **Absence** or **Presence**. The exact meaning of these labels
should match the target-label mapping used during model training.
""")

# Architecture
st.header("2. How the Model Works")

steps = [
    ("01", "Patient Input", "The user enters clinical measurements and coded health-related attributes."),
    ("02", "Feature Engineering", "The application derives additional features, including age group, age-to-cholesterol ratio, and blood-pressure × maximum-heart-rate product."),
    ("03", "Preprocessing", "Categorical features are converted to the expected data types, and columns are arranged in the model's required order."),
    ("04", "LightGBM Prediction", "The trained gradient-boosted decision trees analyze the input features."),
    ("05", "Prediction Output", "The application displays a predicted class and model-generated class probabilities.")
]

for number, title, description in steps:
    with st.container(border=True):
        st.markdown(f"**{number} · {title}**")
        st.write(description)

st.info(
    "The displayed probability is the model's estimated class probability. "
    "It should not be interpreted as a clinically validated estimate of an "
    "individual's actual probability of having heart disease."
)

# Features
st.header("3. Input Features")

features = {
    "Age": "Patient age in years.",
    "Sex": "Numerically encoded sex attribute; interpretation depends on the dataset's coding.",
    "Chest Pain Type": "Coded category describing chest-pain type.",
    "Blood Pressure (BP)": "Resting blood-pressure measurement.",
    "Cholesterol": "Serum cholesterol measurement.",
    "Fasting Blood Sugar (FBS)": "Coded indicator for fasting blood sugar above the dataset's threshold.",
    "EKG Results": "Coded electrocardiogram result.",
    "Maximum Heart Rate": "Maximum heart rate recorded during testing.",
    "Exercise-Induced Angina": "Coded indicator for angina associated with exercise.",
    "ST Depression": "ST-segment depression measurement associated with exercise testing.",
    "Slope of ST Segment": "Coded slope category of the peak exercise ST segment.",
    "Number of Major Vessels": "Number of major vessels represented by the dataset's fluoroscopy feature.",
    "Thallium": "Coded thallium test result."
}

with st.expander("View input feature descriptions", expanded=True):
    for name, description in features.items():
        st.markdown(f"**{name}:** {description}")

# Engineered features
st.header("4. Engineered Features")

st.markdown("""
The application also creates additional features from the original inputs.
These calculations must remain consistent with the feature engineering used
during training.
""")

with st.container(border=True):
    st.markdown("**Age Group**")
    st.code("pd.cut(age, bins=[0, 40, 50, 60, 70, 100], labels=[0, 1, 2, 3, 4])", language="python")
    st.write("Groups age into predefined intervals.")

with st.container(border=True):
    st.markdown("**Age-to-Cholesterol Ratio**")
    st.code("age_chol_ratio = age / (cholesterol + 1)", language="python")
    st.write("Combines age and cholesterol into a derived numerical feature.")

with st.container(border=True):
    st.markdown("**Blood Pressure × Maximum Heart Rate**")
    st.code("bp_hr_product = bp * max_hr", language="python")
    st.write("Creates an interaction feature from blood pressure and maximum heart rate.")

# Model strengths
st.header("5. Why LightGBM?")

st.markdown("""
- **Captures nonlinear relationships:** Can learn patterns that are difficult to represent with a simple linear model.
- **Models feature interactions:** Tree-based splits can capture interactions between input variables.
- **Efficient on tabular data:** Uses gradient boosting with histogram-based learning.
- **Supports class probabilities:** Provides probability estimates for the model's output classes.
""")

# Evaluation

st.header("6. Model Evaluation")

st.markdown("""
The model was evaluated using **5-fold stratified cross-validation**.
Stratification helps preserve the approximate class distribution across
the folds.
""")

col1, col2 = st.columns(2)

with col1:
    st.metric("Cross-Validation ROC-AUC", "0.9551")

with col2:
    st.metric("Evaluation Method", "5-Fold Stratified CV")

st.caption(
    "Random state: 42. This result represents cross-validation performance, "
    "not independent held-out test-set performance."
)

st.info(
    "ROC-AUC measures how well the model ranks examples from the two classes "
    "across different classification thresholds. It is not the same as accuracy."
)

# Limitations
st.header("7. Limitations and Responsible Use")

st.markdown("""
- This is an educational machine learning project, not a certified medical device.
- Predictions depend on the quality, representativeness, and preprocessing of the training data.
- Dataset-specific numeric encodings must be interpreted using the original dataset documentation.
- Model probabilities may not be calibrated and may not generalize to other populations.
- A prediction must not replace clinical assessment, professional medical advice, or diagnostic testing.
""")

st.error(
    "Medical disclaimer: This application is for educational and demonstration "
    "purposes only. It must not be used to diagnose heart disease or make "
    "treatment decisions."
)

st.markdown("---")
st.caption("Built with Python, LightGBM, pandas, scikit-learn, and Streamlit.")
