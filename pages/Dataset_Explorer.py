
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(
    page_title="Dataset Explorer | Heart Disease Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dataset Explorer")
st.caption("Explore the data used in the Heart Disease Prediction project")

# Locate dataset relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "train (10).csv"

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH.name}")
    st.info(
        "Place your training CSV in the project root folder, "
        "or update DATA_PATH to match your actual filename."
    )
    st.stop()

df = load_data(str(DATA_PATH))

# Sidebar controls
st.sidebar.header("Explorer Settings")

st.sidebar.write(f"Rows: {df.shape[0]:,}")
st.sidebar.write(f"Columns: {df.shape[1]:,}")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
all_cols = df.columns.tolist()

# Summary metrics
st.header("Dataset Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", f"{df.shape[0]:,}")
c2.metric("Total Features", len(df.columns))
c3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
c4.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")

st.divider()

# Raw data
with st.expander("🔎 Preview Dataset"):
    st.dataframe(df.head(100), use_container_width=True)
    st.download_button(
        "Download dataset preview",
        data=df.head(100).to_csv(index=False).encode("utf-8"),
        file_name="dataset_preview.csv",
        mime="text/csv"
    )

# Missing values
st.header("1. Missing Value Analysis")

missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if missing.empty:
    st.success("No missing values were found in the loaded dataset.")
else:
    missing_df = missing.rename("Missing Count").to_frame()
    missing_df["Missing Percentage (%)"] = (
        missing_df["Missing Count"] / len(df) * 100
    ).round(2)

    st.dataframe(missing_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    missing_df["Missing Percentage (%)"].plot(
        kind="bar", ax=ax
    )
    ax.set_ylabel("Missing (%)")
    ax.set_xlabel("Feature")
    ax.set_title("Missing Values by Feature")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.divider()

# Target class distribution
st.header("2. Target Class Distribution")

target_col = st.selectbox(
    "Select the target column",
    options=all_cols,
    index=len(all_cols) - 1
)

target_counts = df[target_col].fillna("Missing").astype(str).value_counts()

left, right = st.columns(2)

with left:
    st.dataframe(
        target_counts.rename_axis("Class").reset_index(name="Count"),
        use_container_width=True
    )

with right:
    fig, ax = plt.subplots(figsize=(7, 4))
    target_counts.plot(kind="bar", ax=ax)
    ax.set_title(f"Distribution of {target_col}")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of records")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.divider()

# Feature distribution
st.header("3. Feature Distribution")

if numeric_cols:
    selected_feature = st.selectbox(
        "Choose a numerical feature",
        options=numeric_cols
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    values = pd.to_numeric(
        df[selected_feature], errors="coerce"
    ).dropna()

    ax.hist(values, bins=30, edgecolor="black")
    ax.set_title(f"Distribution of {selected_feature}")
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Descriptive Statistics")
    st.dataframe(
        df[selected_feature].describe().to_frame().T,
        use_container_width=True
    )
else:
    st.info("No numerical columns were detected.")

st.divider()

# Correlation matrix
st.header("4. Feature Correlation")

if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        annot=len(numeric_cols) <= 12,
        fmt=".2f",
        ax=ax
    )
    ax.set_title("Correlation Matrix of Numerical Features")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Correlation measures statistical association, not causation. "
        "Numeric codes for categorical variables may not have a meaningful "
        "linear interpretation."
    )
else:
    st.info("At least two numerical columns are required for correlation analysis.")

st.divider()

# Feature comparison
st.header("5. Compare Features")

if len(numeric_cols) >= 2:
    compare_cols = st.multiselect(
        "Select up to four numerical features",
        options=numeric_cols,
        default=numeric_cols[:min(2, len(numeric_cols))]
    )

    if compare_cols:
        st.dataframe(
            df[compare_cols].describe().T,
            use_container_width=True
        )

st.divider()

st.caption(
    "Dataset Explorer | LightGBM Heart Disease Prediction | "
    "For educational and exploratory use only."
)
