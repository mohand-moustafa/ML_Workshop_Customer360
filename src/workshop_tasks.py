import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "customer_360_ml_workshop.csv"
OUTPUT_DIR = ROOT / "outputs"
RANDOM_STATE = 42


def load_data():
    return pd.read_csv(DATA_PATH)


def part_a_eda(df):
    print("Shape:", df.shape)
    print(df.head())
    print(df.dtypes)
    print(df.isnull().sum())


def part_b_preprocessing(df):
    print("TODO: build the preprocessing pipeline")


def part_c_classification(df):
    print("TODO: churn classification")


def part_d_regression(df):
    print("TODO: revenue regression")


def part_e_segmentation(df):
    print("TODO: customer segmentation")


def part_f_pca_anomaly(df):
    print("TODO: PCA and anomaly detection")


def part_g_integration(df):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("TODO: build outputs/customer_360_predictions.csv")


def main():
    df = load_data()
    part_a_eda(df)
    part_b_preprocessing(df)
    part_c_classification(df)
    part_d_regression(df)
    part_e_segmentation(df)
    part_f_pca_anomaly(df)
    part_g_integration(df)


if __name__ == "__main__":
    main()
