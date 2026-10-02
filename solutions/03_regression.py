"""
03_regression.py - Part B (Preprocessing) + Part D (Revenue Regression).

Run:  python solutions/03_regression.py
Saves figures to outputs/figures/03_*.png

Predicts Future12MRevenueEGP. The best model is chosen with cross-validation
on the TRAINING set only; the test set is kept for the final evaluation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

import common as c


def run() -> None:
    c.setup()
    df = c.load_data()

    # ------------------------------------------------------------------ #
    c.section("B1. Regression features, target and 80/20 split")
    X, y = c.revenue_xy(df)
    print("X shape:", X.shape, "| y shape:", y.shape)
    print("Features:", list(X.columns))
    for leaked in (c.ID_COLUMN, c.REGRESSION_TARGET, c.CLASSIFICATION_TARGET):
        print(f"{leaked} in features:", leaked in X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=c.TEST_SIZE, random_state=c.RANDOM_STATE
    )
    print("X_train:", X_train.shape, "| X_test:", X_test.shape)
    print("Mean revenue train/test:", round(y_train.mean(), 2), "/", round(y_test.mean(), 2))

    # ------------------------------------------------------------------ #
    c.section("B2. Preprocessing pipeline (fit on training data only)")
    preprocessor = c.build_preprocessor(X_train)
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)
    print("Processed X_train:", X_train_p.shape, "| X_test:", X_test_p.shape)
    print("Missing values after preprocessing (train/test):",
          int(np.isnan(X_train_p).sum()), "/", int(np.isnan(X_test_p).sum()))

    # ------------------------------------------------------------------ #
    c.section("D1. Cross-validating seven regressors on the training set")
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=c.RANDOM_STATE),
        "Lasso": Lasso(alpha=1.0, random_state=c.RANDOM_STATE),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=5, random_state=c.RANDOM_STATE),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=c.RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=c.RANDOM_STATE),
    }
    kfold = KFold(n_splits=5, shuffle=True, random_state=c.RANDOM_STATE)
    scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}

    cv_rows = []
    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        scores = cross_validate(pipe, X_train, y_train, cv=kfold, scoring=scoring)
        cv_rows.append({
            "Model": name,
            "CV_MAE": -scores["test_mae"].mean(),
            "CV_RMSE": -scores["test_rmse"].mean(),
            "CV_R2": scores["test_r2"].mean(),
        })
    cv_results = pd.DataFrame(cv_rows).sort_values("CV_RMSE").reset_index(drop=True)
    print(cv_results.round(4).to_string(index=False))

    final_name = cv_results.loc[0, "Model"]
    print("\nModel selected by lowest CV RMSE (training data only):", final_name)

    # ------------------------------------------------------------------ #
    c.section("D2. Final evaluation on the held-out test set")
    rows, predictions, fitted = [], {}, {}
    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        fitted[name], predictions[name] = pipe, pred
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred),
        })
    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    results[["MAE", "RMSE", "R2"]] = results[["MAE", "RMSE", "R2"]].round(4)

    baseline = np.full(len(y_test), y_train.mean())
    print("Baseline MAE (always predict mean revenue):", round(mean_absolute_error(y_test, baseline), 2))
    print("Average actual revenue in test set:", round(y_test.mean(), 2))
    print("\nRegression comparison table:\n", results.to_string(index=False))

    # ------------------------------------------------------------------ #
    c.section("D3. Actual vs predicted")
    final_pred = predictions[final_name]
    final_row = results[results["Model"] == final_name].iloc[0]

    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, final_pred, alpha=0.5, label="Customers")
    lo, hi = min(y_test.min(), final_pred.min()), max(y_test.max(), final_pred.max())
    plt.plot([lo, hi], [lo, hi], color="red", linestyle="--", label="Perfect prediction")
    plt.xlabel("Actual Future12MRevenueEGP")
    plt.ylabel("Predicted Future12MRevenueEGP")
    plt.title("Actual vs Predicted Revenue - " + final_name)
    plt.legend()
    c.save_fig("03_actual_vs_predicted.png")
    print("MAE :", round(final_row["MAE"], 2))
    print("RMSE:", round(final_row["RMSE"], 2))
    print("R2  :", round(final_row["R2"], 4))

    # ------------------------------------------------------------------ #
    c.section("D4. Residual analysis")
    residuals = y_test - final_pred
    rmse = final_row["RMSE"]
    print(pd.Series({
        "Mean residual (bias)": residuals.mean(),
        "Std of residuals": residuals.std(),
        "Smallest residual": residuals.min(),
        "Largest residual": residuals.max(),
        "Errors larger than 2 x RMSE": int((residuals.abs() > 2 * rmse).sum()),
        "Share of errors larger than 2 x RMSE (%)": (residuals.abs() > 2 * rmse).mean() * 100,
    }).round(2))

    band = pd.qcut(y_test, 4, labels=["Q1 (lowest revenue)", "Q2", "Q3", "Q4 (highest revenue)"])
    print("\nError by actual revenue group")
    print(pd.DataFrame({"abs_error": residuals.abs(), "residual": residuals})
          .groupby(band, observed=True)
          .agg(customers=("abs_error", "size"), MAE=("abs_error", "mean"), mean_residual=("residual", "mean"))
          .round(2))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].scatter(final_pred, residuals, alpha=0.5)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted Future12MRevenueEGP")
    axes[0].set_ylabel("Residual (Actual - Predicted)")
    axes[0].set_title("Residuals vs Predicted Revenue")
    sns.histplot(residuals, bins=30, kde=True, ax=axes[1])
    axes[1].set_title("Distribution of Residuals")
    axes[1].set_xlabel("Residual (Actual - Predicted)")
    c.save_fig("03_residual_analysis.png")

    # ------------------------------------------------------------------ #
    c.section("D5. What drives the final model?")
    pipe = fitted[final_name]
    estimator = pipe.named_steps["model"]
    names = pipe.named_steps["preprocessor"].get_feature_names_out()
    values = estimator.feature_importances_ if hasattr(estimator, "feature_importances_") else np.abs(estimator.coef_)
    importance = pd.Series(values, index=names).sort_values(ascending=False)
    print(f"Top 8 most important features for {final_name}")
    print(importance.head(8).round(4))


if __name__ == "__main__":
    run()
