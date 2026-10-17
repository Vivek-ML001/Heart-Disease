import streamlit as st

st.set_page_config(
    page_title="Heart Disease Feature Guide",
    page_icon="📘",
    layout="wide"
)

st.title("📘 Patient Input — Feature Guide")

st.write(
    "Learn what each field means before entering information "
    "into the Heart Disease Prediction application."
)

st.info(
    "This guide provides general educational information. "
    "It does not provide medical advice or diagnose heart disease."
)

st.subheader("1. General Patient Information")

with st.expander("Age", expanded=True):
    st.write("**What it means:** The patient's age in years.")
    st.write("**Example:** 50 means the patient is 50 years old.")
    st.write(
        "Age is associated with cardiovascular risk, but age alone "
        "cannot determine whether someone has heart disease."
    )

with st.expander("Sex"):
    st.write("**What it means:** The sex category encoded for the model.")
    st.write("**Options:** Female (0), Male (1).")
    st.write(
        "These codes must match the encoding used in the training dataset."
    )

st.subheader("2. Symptoms and Clinical Measurements")

with st.expander("Chest Pain Type"):
    st.write("**What it means:** A category describing the type of chest pain.")
    st.write("**Options in this app:** 1, 2, 3, 4.")
    st.write(
        "The exact meaning of each number depends on your dataset's "
        "codebook. Do not assume the categories have a universal meaning."
    )

with st.expander("Blood Pressure (BP)"):
    st.write("**What it means:** The blood pressure measurement.")
    st.write("**Unit:** Usually mmHg.")
    st.write(
        "Confirm whether your training dataset records systolic blood "
        "pressure or another measurement."
    )

with st.expander("Cholesterol"):
    st.write("**What it means:** The cholesterol measurement recorded in the dataset.")
    st.write("**Unit:** Often mg/dL in commonly used heart disease datasets.")
    st.write(
        "Check your dataset documentation to confirm the unit and "
        "whether the measurement represents total serum cholesterol."
    )

with st.expander("FBS over 120"):
    st.write(
        "**FBS** means fasting blood sugar. This field indicates whether "
        "the recorded fasting blood sugar exceeds the dataset's threshold."
    )
    st.write("**Options:** No (0), Yes (1).")
    st.write("Confirm the threshold and units in your dataset documentation.")

with st.expander("EKG Results"):
    st.write(
        "**What it means:** A coded category for electrocardiogram (ECG/EKG) "
        "results."
    )
    st.write("**Options in this app:** 0, 1, 2.")
    st.write(
        "The exact interpretation of these codes depends on your dataset."
    )

st.subheader("3. Exercise and Heart Measurements")

with st.expander("Maximum Heart Rate"):
    st.write("**What it means:** The maximum heart rate recorded for the patient.")
    st.write("**Unit:** Beats per minute (bpm).")
    st.write(
        "Interpretation depends on age, activity, medication, and the "
        "conditions under which the measurement was taken."
    )

with st.expander("Exercise Angina"):
    st.write(
        "**What it means:** Whether angina (chest discomfort associated "
        "with reduced blood supply to the heart) occurred during exercise."
    )
    st.write("**Options:** No (0), Yes (1).")
    st.write(
        "Chest pain or discomfort during exercise warrants appropriate "
        "medical assessment."
    )

with st.expander("ST Depression"):
    st.write(
        "**What it means:** A measurement of ST-segment depression "
        "recorded during a cardiac test."
    )
    st.write(
        "Its clinical interpretation depends on the ECG, test protocol, "
        "baseline findings, and other clinical information."
    )

with st.expander("Slope of ST"):
    st.write(
        "**What it means:** A coded description of the ST-segment slope "
        "in the source dataset."
    )
    st.write("**Options in this app:** 1, 2, 3.")
    st.write(
        "Check the original dataset's codebook for the exact meaning "
        "of each category."
    )

st.subheader("4. Additional Cardiac Test Fields")

with st.expander("Number of Vessels Fluoro"):
    st.write(
        "**What it means:** A coded count of major vessels visualized "
        "using fluoroscopy in the original dataset."
    )
    st.write("**Options in this app:** 0, 1, 2, 3.")
    st.write(
        "The field's precise meaning and measurement procedure should "
        "be verified against the training dataset documentation."
    )

with st.expander("Thallium"):
    st.write(
        "**What it means:** A coded result associated with a thallium "
        "cardiac imaging test in commonly used heart disease datasets."
    )
    st.write("**Options in this app:** 3, 6, 7.")
    st.write(
        "These are dataset-specific codes, not numerical measurements "
        "of disease severity. Verify their definitions in the codebook."
    )

st.subheader("5. Features Calculated Automatically")

st.write(
    "The application also creates three additional features from the "
    "entered values. You do not need to enter them manually."
)

with st.expander("Age Group"):
    st.write(
        "The application groups age into five intervals: "
        "up to 40, 41–50, 51–60, 61–70, and 71–100 years."
    )
    st.write("The intervals are encoded from 0 to 4.")

with st.expander("Age–Cholesterol Ratio"):
    st.code("Age / (Cholesterol + 1)")
    st.write(
        "This is a custom feature engineered during model development. "
        "Its usefulness depends on the training data and validation."
    )

with st.expander("Blood Pressure × Maximum Heart Rate"):
    st.code("BP * Maximum Heart Rate")
    st.write(
        "This is another custom feature created for the model. "
        "Its interpretation depends on the units and definitions of "
        "the original measurements."
    )

st.divider()

st.subheader("Important Notes")

st.markdown(
    """
    - Enter measurements using the units and definitions expected by
      the training dataset.
    - Numeric codes are categories; they do not necessarily indicate
      increasing severity.
    - The model output is an estimate, not a confirmed diagnosis.
    - If you have concerning symptoms, seek medical assessment rather
      than relying on this application.
    """
)

st.caption(
    "Heart Disease Prediction | Educational Machine Learning Project"
)
