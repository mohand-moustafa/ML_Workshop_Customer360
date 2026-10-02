"""
05_customer360_integration.py - Part G: Customer 360 Integration.

Run:  python solutions/05_customer360_integration.py
Writes: outputs/customer_360_predictions.csv

Combines the outputs of Parts C-F into one row per customer:
    CustomerID | ChurnProbability | Predicted12MRevenueEGP |
    CustomerSegment | AnomalyFlag | RetentionPriorityFlag

Retention-priority rule (both conditions must hold):
    1. ChurnProbability >= 0.5
    2. Predicted12MRevenueEGP >= median predicted revenue
Requiring both targets customers who are at real risk AND worth keeping.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline

import common as c


def run() -> None:
    c.setup()
    df = c.load_data()
    # Keep an untouched copy for building model features. Prediction columns are
    # added to `df` as we go, and they must never leak into later models'
    # features (e.g. ChurnProbability must not become a revenue predictor).
    base = df.copy()

    # ------------------------------------------------------------------ #
    c.section("G1. Churn probability (tuned Random Forest from Part C)")
    X_churn, y_churn = c.churn_xy(base)
    churn_pipeline = Pipeline(steps=[
        ("preprocessor", c.build_preprocessor(X_churn)),
        ("model", RandomForestClassifier(random_state=c.RANDOM_STATE, **c.BEST_RF_PARAMS)),
    ])
    churn_pipeline.fit(X_churn, y_churn)
    df["ChurnProbability"] = churn_pipeline.predict_proba(X_churn)[:, 1]
    print(df["ChurnProbability"].describe().round(4))
    print(
        "\nNote: class_weight='balanced' shifts probabilities upward, so read them as a "
        "relative risk ranking rather than a literal chance of churning."
    )

    # ------------------------------------------------------------------ #
    c.section("G2. Predicted 12-month revenue (Gradient Boosting from Part D)")
    X_rev, y_rev = c.revenue_xy(base)
    revenue_pipeline = Pipeline(steps=[
        ("preprocessor", c.build_preprocessor(X_rev)),
        ("model", GradientBoostingRegressor(random_state=c.RANDOM_STATE)),
    ])
    revenue_pipeline.fit(X_rev, y_rev)
    df["Predicted12MRevenueEGP"] = revenue_pipeline.predict(X_rev)
    print(df["Predicted12MRevenueEGP"].describe().round(2))

    # ------------------------------------------------------------------ #
    c.section("G3. Customer segment (K-Means from Part E)")
    X_scaled = c.scale_cluster_features(base)
    df = c.assign_segments(df, X_scaled)
    print(df["CustomerSegment"].value_counts())

    # ------------------------------------------------------------------ #
    c.section("G4. Anomaly flag (Isolation Forest from Part F)")
    df = c.flag_anomalies(df, X_scaled)
    print(df["AnomalyFlag"].value_counts())

    # ------------------------------------------------------------------ #
    c.section("G5. Retention-priority flag")
    median_revenue = df["Predicted12MRevenueEGP"].median()
    print("Median predicted 12-month revenue:", round(median_revenue, 2))

    df["RetentionPriorityFlag"] = np.where(
        (df["ChurnProbability"] >= c.CHURN_PROBABILITY_THRESHOLD)
        & (df["Predicted12MRevenueEGP"] >= median_revenue),
        "Yes", "No",
    )
    print(df["RetentionPriorityFlag"].value_counts())

    print("\nAverages by retention flag:")
    print(df.groupby("RetentionPriorityFlag")[["ChurnProbability", "Predicted12MRevenueEGP"]].mean().round(2))
    print("\nRetention priority by segment:\n", pd.crosstab(df["CustomerSegment"], df["RetentionPriorityFlag"]))
    print("\nRetention priority by anomaly flag:\n", pd.crosstab(df["AnomalyFlag"], df["RetentionPriorityFlag"]))

    # ------------------------------------------------------------------ #
    c.section("G6. Final Customer 360 table")
    final_columns = [c.ID_COLUMN, "ChurnProbability", "Predicted12MRevenueEGP",
                     "CustomerSegment", "AnomalyFlag", "RetentionPriorityFlag"]
    table = df[final_columns].copy()
    table["ChurnProbability"] = table["ChurnProbability"].round(4)
    table["Predicted12MRevenueEGP"] = table["Predicted12MRevenueEGP"].round(2)

    c.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(c.PREDICTIONS_PATH, index=False)
    print("Final table shape:", table.shape)
    print(table.head(10).to_string(index=False))
    print(f"\n[saved] {c.PREDICTIONS_PATH.relative_to(c.ROOT)}")


if __name__ == "__main__":
    run()
