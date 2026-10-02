"""
common.py - shared helpers for the Customer 360 workshop solutions.

Everything that more than one solution script needs lives here, so the
preprocessing, the segment names and the model settings are defined exactly
once and every part of the workshop stays consistent.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: figures are saved to disk, never shown

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "customer_360_ml_workshop.csv"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
PREDICTIONS_PATH = OUTPUT_DIR / "customer_360_predictions.csv"

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.2

ID_COLUMN = "CustomerID"
CLASSIFICATION_TARGET = "Churn"
REGRESSION_TARGET = "Future12MRevenueEGP"

NUMERIC_COLUMNS = [
    "Age", "TenureMonths", "MonthlyUsageGB", "MonthlyChargeEGP", "NumServices",
    "SupportCalls6M", "LatePayments12M", "SatisfactionScore", "Future12MRevenueEGP",
]
CATEGORICAL_COLUMNS = ["Region", "ContractType", "InternetType", "PaperlessBilling", "AutoPay", "Churn"]

# Behavioural / value variables used for segmentation, PCA and anomaly detection
CLUSTER_FEATURES = [
    "TenureMonths", "MonthlyUsageGB", "MonthlyChargeEGP", "NumServices",
    "SupportCalls6M", "LatePayments12M", "SatisfactionScore",
]
N_CLUSTERS = 4
CLUSTER_BUSINESS_NAMES = {
    0: "At-Risk / Late-Payment Customers",
    1: "Loyal Long-Tenured Customers",
    2: "Low-Value Basic Users",
    3: "High-Value Power Users",
}

# Tuned Random Forest found with GridSearchCV in Part C
BEST_RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "min_samples_split": 2,
    "class_weight": "balanced",
}

ANOMALY_CONTAMINATION = 0.05
CHURN_PROBABILITY_THRESHOLD = 0.5


# --------------------------------------------------------------------------- #
# Setup / IO helpers
# --------------------------------------------------------------------------- #
def setup() -> None:
    """Common display + warning settings, and make sure output folders exist."""
    warnings.filterwarnings("ignore")
    sns.set_theme(style="whitegrid")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    """Load the workshop dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def section(title: str) -> None:
    """Print a clearly visible section header."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def save_fig(name: str) -> None:
    """Save the current matplotlib figure into outputs/figures and close it."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    path = FIGURE_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
# Features / targets / preprocessing
# --------------------------------------------------------------------------- #
def churn_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Classification features and binary target (1 = churn).

    CustomerID is an identifier and Future12MRevenueEGP is a leakage risk
    (it is only known after the churn decision), so both are excluded.
    """
    y = (df[CLASSIFICATION_TARGET] == "Yes").astype(int)
    X = df.drop(columns=[ID_COLUMN, REGRESSION_TARGET, CLASSIFICATION_TARGET])
    return X, y


def revenue_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Regression features and target (Future12MRevenueEGP)."""
    y = df[REGRESSION_TARGET]
    X = df.drop(columns=[ID_COLUMN, REGRESSION_TARGET, CLASSIFICATION_TARGET])
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Median-impute + scale numeric columns, mode-impute + one-hot categoricals."""
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])


# --------------------------------------------------------------------------- #
# Unsupervised helpers (shared by Part E, Part F and Part G)
# --------------------------------------------------------------------------- #
def scale_cluster_features(df: pd.DataFrame) -> np.ndarray:
    """Median-impute and standardize the 7 segmentation variables."""
    prep = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return prep.fit_transform(df[CLUSTER_FEATURES])


def assign_segments(df: pd.DataFrame, X_scaled: np.ndarray) -> pd.DataFrame:
    """Fit K-Means (k=4) and add Cluster_KMeans + CustomerSegment columns."""
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    df = df.copy()
    df["Cluster_KMeans"] = kmeans.fit_predict(X_scaled)
    df["CustomerSegment"] = df["Cluster_KMeans"].map(CLUSTER_BUSINESS_NAMES)
    return df


def flag_anomalies(df: pd.DataFrame, X_scaled: np.ndarray) -> pd.DataFrame:
    """Fit Isolation Forest and add AnomalyLabel / AnomalyScore / AnomalyFlag."""
    model = IsolationForest(
        n_estimators=200, contamination=ANOMALY_CONTAMINATION, random_state=RANDOM_STATE
    )
    labels = model.fit_predict(X_scaled)
    df = df.copy()
    df["AnomalyLabel"] = labels
    df["AnomalyScore"] = model.decision_function(X_scaled)
    df["AnomalyFlag"] = np.where(labels == -1, "Yes", "No")
    return df
