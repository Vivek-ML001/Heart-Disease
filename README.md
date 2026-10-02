#  Heart Disease Prediction 

A machine learning web application that predicts the likelihood of heart disease using a **LightGBM classification model** with categorical feature handling and custom feature engineering.

The trained model achieved a **5-Fold Cross-Validation ROC-AUC of 0.9551** on the training data.

>  **Medical Disclaimer:** This project is intended for educational and demonstration purposes only. It is not a medical diagnostic system and should not be used to make clinical decisions.

---

##  Live Demo

The application is built using **Streamlit** and can be deployed as an interactive web application.

**Live App:**  
_https://heart-disease01.streamlit.app/_

---

##  Project Overview

Heart disease is influenced by multiple physiological and clinical factors such as age, blood pressure, cholesterol, maximum heart rate, exercise-induced angina, and ECG-related measurements.

The objective of this project is to build a machine learning classification system that learns patterns from patient-level features and predicts whether the patient belongs to the heart-disease or no-heart-disease class.

The project follows a complete machine learning workflow:

---

##  System Architecture

The overall architecture of the deployed application is:

```text
                    ┌───────────────────────┐
                    │      User Input       │
                    │                       │
                    │ Age                   │
                    │ Sex                   │
                    │ Chest Pain Type       │
                    │ Blood Pressure        │
                    │ Cholesterol           │
                    │ ECG / EKG             │
                    │ Maximum Heart Rate    │
                    │ Exercise Angina       │
                    │ etc.                  │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Data Preprocessing  │
                    │                       │
                    │ Type conversion       │
                    │ Categorical encoding  │
                    │ Feature ordering      │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Feature Engineering  │
                    │                       │
                    │ Age Group             │
                    │ Age/Cholesterol       │
                    │ BP × Max HR           │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │       LightGBM        │
                    │     Classifier        │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │      Prediction       │
                    │                       │
                    │ Heart Disease         │
                    │ or                    │
                    │ No Heart Disease      │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Probability Estimate │
                    └───────────────────────┘
```


---

##  Model Architecture

### LightGBM

The primary model used in this project is **LightGBM**, a gradient boosting framework based on decision trees.

Conceptually, the model works as:

```text
Input Features
      │
      ▼
Decision Tree 1
      │
      ▼
Prediction Error
      │
      ▼
Decision Tree 2
      │
      ▼
Prediction Error
      │
      ▼
Decision Tree 3
      │
      ▼
      ...
      │
      ▼
Combined Gradient Boosting Model
      │
      ▼
Final Prediction
```

Instead of building one very large decision tree, LightGBM builds many trees sequentially, with each tree attempting to improve the errors made by the previous trees.

---

##  Cross Validation

To obtain a more reliable estimate of model performance, **Stratified K-Fold Cross Validation** was used.

Configuration:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Architecture:

```text
                 Training Dataset
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
       Fold 1        Fold 2       Fold 3
          │            │            │
          └────────────┼────────────┘
                       │
                  Fold 4 + Fold 5
                       │
                       ▼
                Model Evaluation
                       │
                       ▼
                 Mean CV Score
```

Stratification helps maintain a similar class distribution across the folds.

---

##  Model Performance

### Cross-Validation Result

| Metric           |             Score |
| ---------------- | ----------------: |
| ROC-AUC          |       **0.95507** |
| Cross Validation | 5-Fold Stratified |
| Random State     |                42 |

### ROC-AUC

The ROC-AUC score of:

```text
0.955072504113791
```

indicates strong discrimination between the two target classes on the cross-validation evaluation.

> Note: A high ROC-AUC does not mean the model is clinically validated or suitable for medical diagnosis.

---

##  Model Artifacts

The trained model and metadata are saved separately so that the Streamlit application can load them without retraining.

```text
lightgbm_model.pkl
```

Contains the trained LightGBM model.

```text
feature_names.pkl
```

Contains the exact feature ordering expected by the model.

```text
categorical_features.pkl
```

Contains the list of categorical features used during training.

This separation helps ensure that the deployed application reproduces the same feature structure used during model training.

---

##  Project Structure

```text
Heart-Disease/
│
├── app.py
│
├── requirements.txt
│
├── lightgbm_model.pkl
├── feature_names.pkl
├── categorical_features.pkl
│
├── heart-disease-with-lightgbm (1) (1).ipynb
│
├── README.md
│
└── .gitignore
```

### File Description

| File                       | Purpose                                        |
| -------------------------- | ---------------------------------------------- |
| `app.py`                   | Streamlit deployment application               |
| `requirements.txt`         | Python dependencies                            |
| `lightgbm_model.pkl`       | Trained LightGBM model                         |
| `feature_names.pkl`        | Model feature ordering                         |
| `categorical_features.pkl` | Categorical feature metadata                   |
| `.ipynb`                   | Complete experimentation and training notebook |
| `README.md`                | Project documentation                          |

The original training and test datasets are intentionally excluded from the deployment repository because they are not required by the Streamlit application.

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/Vivek-ML001/Heart-Disease.git
```

Move into the project directory:

```bash
cd Heart-Disease
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---


---

##  Why LightGBM?

LightGBM was selected because it is particularly effective for structured/tabular datasets.

Advantages include:

* Strong performance on tabular data
* Efficient gradient boosting implementation
* Native support for categorical features
* Good computational efficiency
* Ability to model nonlinear relationships
* Ability to capture feature interactions
* Suitable for relatively small and medium-sized tabular datasets

---

##  Reproducibility

The project uses a fixed random state for cross-validation:

```python
random_state=42
```

and:

```python
n_splits=5
```

This helps make the evaluation procedure reproducible.

The exact feature ordering and categorical feature definitions are also stored as model artifacts.

---

##  Limitations

This project has several limitations.

### 1. Dataset limitations
The quality and representativeness of the training dataset directly affect model performance.

### 2. No clinical validation
The model has not undergone clinical validation.

### 3. No medical deployment
The application should not be used by patients or healthcare professionals for actual diagnosis or treatment decisions.

### 4. Probability is not medical risk
The probability returned by the classifier represents the model's estimated probability for the target class. It should not be interpreted as an individual's actual medical risk.

### 5. Potential dataset bias
If the underlying dataset contains demographic or sampling biases, the model may reproduce those biases.

---

##  Author

**Vivek Kumar**

B.Tech Computer Science & Engineering  
Machine Learning Specialization

GitHub: [https://github.com/Vivek-ML001](https://github.com/Vivek-ML001)

---
