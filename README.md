# ❤️ Heart Disease Prediction using LightGBM

A machine learning web application that predicts the likelihood of heart disease using a **LightGBM classification model** with categorical feature handling and custom feature engineering.

The trained model achieved a **5-Fold Cross-Validation ROC-AUC of 0.9551** on the training data.

> ⚠️ **Medical Disclaimer:** This project is intended for educational and demonstration purposes only. It is not a medical diagnostic system and should not be used to make clinical decisions.

---

## 🚀 Live Demo

The application is built using **Streamlit** and can be deployed as an interactive web application.

**Live App:**  
_Add your Streamlit URL here after deployment._

---

## 📌 Project Overview

Heart disease is influenced by multiple physiological and clinical factors such as age, blood pressure, cholesterol, maximum heart rate, exercise-induced angina, and ECG-related measurements.

The objective of this project is to build a machine learning classification system that learns patterns from patient-level features and predicts whether the patient belongs to the heart-disease or no-heart-disease class.

The project follows a complete machine learning workflow:

```text
Raw Dataset
     │
     ▼
Data Exploration
     │
     ▼
Data Cleaning & Feature Selection
     │
     ▼
Categorical Feature Identification
     │
     ▼
Feature Engineering
     │
     ├── Age Group
     ├── Age/Cholesterol Ratio
     └── BP × Maximum Heart Rate
     │
     ▼
LightGBM Classifier
     │
     ▼
5-Fold Stratified Cross Validation
     │
     ▼
Model Evaluation
     │
     ▼
Save Trained Model
     │
     ▼
Streamlit Application
     │
     ▼
User Input → Prediction
```

---

## 🏗️ System Architecture

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

## 🧠 Machine Learning Pipeline

### 1. Data Loading

The dataset is loaded using Pandas.

The training data contains the target variable:

```text
Heart Disease
```

The identifier and target-related columns that are not used as model features are removed.

---

### 2. Feature Selection

The model uses the following features:

#### Original Features

| Feature                   | Description                             |
| ------------------------- | --------------------------------------- |
| `Age`                     | Patient age                             |
| `Sex`                     | Patient sex                             |
| `Chest pain type`         | Type of chest pain                      |
| `BP`                      | Blood pressure                          |
| `Cholesterol`             | Cholesterol level                       |
| `FBS over 120`            | Whether fasting blood sugar is over 120 |
| `EKG results`             | Electrocardiogram result                |
| `Max HR`                  | Maximum heart rate                      |
| `Exercise angina`         | Exercise-induced angina                 |
| `ST depression`           | ST depression measurement               |
| `Slope of ST`             | Slope of the ST segment                 |
| `Number of vessels fluro` | Number of major vessels observed        |
| `Thallium`                | Thallium stress-test result             |

#### Engineered Features

| Feature          | Formula                                     |
| ---------------- | ------------------------------------------- |
| `Age Group`      | Age converted into predefined age intervals |
| `age_chol_ratio` | `Age / (Cholesterol + 1)`                   |
| `bp_hr_product`  | `BP × Max HR`                               |

---

## 🔧 Feature Engineering

Feature engineering was used to provide the tree-based model with additional relationships between the original variables.

### Age Group

Age is transformed into five groups:

```python
bins = [0, 40, 50, 60, 70, 100]
labels = [0, 1, 2, 3, 4]
```

Therefore:

```text
0 → Age ≤ 40
1 → 40 < Age ≤ 50
2 → 50 < Age ≤ 60
3 → 60 < Age ≤ 70
4 → 70 < Age ≤ 100
```

The implementation uses:

```python
pd.cut(
    df["Age"],
    bins=[0, 40, 50, 60, 70, 100],
    labels=[0, 1, 2, 3, 4]
)
```

---

### Age-Cholesterol Ratio

A ratio feature is created using:

```python
X["age_chol_ratio"] = X["Age"] / (X["Cholesterol"] + 1)
```

The `+1` prevents division by zero.

---

### Blood Pressure × Maximum Heart Rate

An interaction feature is created:

```python
X["bp_hr_product"] = X["BP"] * X["Max HR"]
```

This allows the model to consider the interaction between blood pressure and maximum heart rate.

---

## 🏷️ Categorical Features

LightGBM is provided with categorical features directly using Pandas categorical data types.

The categorical features are:

```text
Sex
Chest pain type
FBS over 120
EKG results
Exercise angina
Slope of ST
Number of vessels fluro
Thallium
Age Group
```

The categorical values used by the model are:

```text
Sex
    [0, 1]

Chest pain type
    [1, 2, 3, 4]

FBS over 120
    [0, 1]

EKG results
    [0, 1, 2]

Exercise angina
    [0, 1]

Slope of ST
    [1, 2, 3]

Number of vessels fluro
    [0, 1, 2, 3]

Thallium
    [3, 6, 7]

Age Group
    [0, 1, 2, 3, 4]
```

The categorical columns are converted using:

```python
for col in cat_cols:
    X[col] = X[col].astype("category")
```

---

## 🌳 Model Architecture

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

## 🔄 Cross Validation

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

## 📊 Model Performance

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

## 💾 Model Artifacts

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

## 📁 Project Structure

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

## ⚙️ Technologies Used

### Programming
* Python

### Data Processing
* Pandas
* NumPy

### Machine Learning
* LightGBM
* Scikit-learn

### Visualization / Analysis
* Matplotlib
* Seaborn

### Deployment
* Streamlit
* GitHub
* Streamlit Community Cloud

### Model Serialization
* Joblib

---

## 📦 Installation

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

## ▶️ Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## ☁️ Deployment

The application can be deployed using Streamlit Community Cloud.

Deployment workflow:

```text
Local Project
     │
     ▼
Git Repository
     │
     ▼
GitHub
     │
     ▼
Streamlit Community Cloud
     │
     ▼
Public Web Application
```

### Deployment Steps

1. Push the project to GitHub.
2. Open [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Connect your GitHub account.
4. Select:

```text
Vivek-ML001/Heart-Disease
```

5. Select the branch:

```text
main
```

6. Select:

```text
app.py
```

7. Click **Deploy**.

After deployment, Streamlit will provide a public URL.

---

## 🔬 Example Prediction Workflow

A typical prediction request follows this process:

```text
Patient Information
       │
       ├── Age = 55
       ├── BP = 140
       ├── Cholesterol = 240
       ├── Max HR = 150
       ├── Chest Pain Type
       ├── EKG Result
       └── Other clinical features
              │
              ▼
       Feature Engineering
              │
              ├── Age Group
              ├── Age/Cholesterol Ratio
              └── BP × Max HR
              │
              ▼
          LightGBM
              │
              ▼
       Binary Prediction
              │
              ▼
       Probability Estimate
```

The example values above are illustrative and should not be interpreted as medically meaningful recommendations.

---

## 📈 Why LightGBM?

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

## 🧪 Reproducibility

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

## 🔍 Key Machine Learning Concepts Demonstrated

This project demonstrates several practical machine learning concepts:

### Data preprocessing
Selecting relevant features and removing unnecessary columns.

### Feature engineering
Creating domain-inspired interaction and ratio features.

### Categorical feature handling
Using Pandas categorical types with LightGBM.

### Gradient boosting
Training an ensemble of decision trees sequentially.

### Stratified cross-validation
Evaluating the model across multiple stratified folds.

### ROC-AUC
Measuring binary classification discrimination.

### Model serialization
Saving the trained model using Joblib.

### ML deployment
Connecting a trained machine learning model to an interactive Streamlit application.

---

## ⚠️ Limitations

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

## 🚀 Future Improvements

Possible improvements include:

* Hyperparameter optimization using Optuna
* SHAP-based model explainability
* Feature importance visualization
* Calibration of predicted probabilities
* Precision-Recall analysis
* ROC curve visualization
* Confusion matrix dashboard
* Model monitoring
* Data drift detection
* Better input validation
* Automated ML pipeline
* Docker deployment
* REST API using FastAPI
* CI/CD using GitHub Actions
* Model versioning
* Explainable AI dashboard

---

## 🧠 Future Architecture

A more production-oriented version could use:

```text
                    ┌──────────────────┐
                    │   Streamlit UI   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Input Schema   │
                    │    Validation    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Feature Pipeline │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     LightGBM     │
                    │      Model       │
                    └────────┬─────────┘
                             │
                    ��────────┴─────────┐
                    ▼                  ▼
             ┌────────────┐    ┌─────────────┐
             │ Prediction │    │ Probability │
             └────────────┘    └─────────────┘
```

For a production system, an API layer such as FastAPI could be introduced between the frontend and model.

---

## 👨‍💻 Author

**Vivek Kumar**

B.Tech Computer Science & Engineering  
Machine Learning Specialization

GitHub: [https://github.com/Vivek-ML001](https://github.com/Vivek-ML001)

---

## ⭐ Acknowledgement

This project was developed as a practical machine learning project to explore:

* Tabular machine learning
* Gradient boosting
* Feature engineering
* Cross-validation
* Model serialization
* ML deployment

---

## 📜 License

This project is intended for educational and research purposes.

If you reuse this project or its components, please verify the licensing terms of the underlying dataset and third-party libraries.
