"""
01_eda.py - Part A: Business Understanding & Exploratory Data Analysis.

Run:  python solutions/01_eda.py
Saves figures to outputs/figures/01_*.png
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import common as c


def run() -> None:
    c.setup()
    df = c.load_data()
    numeric_columns = c.NUMERIC_COLUMNS
    contract_order = ["Month-to-month", "One year", "Two year"]

    # ------------------------------------------------------------------ #
    c.section("A1. Dataset structure")
    print("Shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nData types:\n", df.dtypes)

    # ------------------------------------------------------------------ #
    c.section("A2. Missing values")
    missing = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percent": (df.isnull().mean() * 100).round(2),
    })
    print("Total missing cells:", int(df.isnull().sum().sum()))
    print(missing[missing["missing_count"] > 0])

    # ------------------------------------------------------------------ #
    c.section("A3. Descriptive statistics (numerical variables)")
    print(df[numeric_columns].describe().T.round(2))

    # ------------------------------------------------------------------ #
    c.section("A4. Categorical variables")
    print("CustomerID unique values:", df[c.ID_COLUMN].nunique(), "out of", len(df))
    for col in c.CATEGORICAL_COLUMNS:
        counts = df[col].value_counts(dropna=False)
        percents = (df[col].value_counts(dropna=False, normalize=True) * 100).round(2)
        print(f"\n{col}\n", pd.DataFrame({"count": counts, "percent": percents}))

    # ------------------------------------------------------------------ #
    c.section("A5. Churn target distribution")
    churn_counts = df["Churn"].value_counts()
    churn_percent = (df["Churn"].value_counts(normalize=True) * 100).round(2)
    print(pd.DataFrame({"count": churn_counts, "percent": churn_percent}))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.countplot(data=df, x="Churn", order=["No", "Yes"], ax=axes[0])
    axes[0].set_title("Churn Count")
    for container in axes[0].containers:
        axes[0].bar_label(container)
    axes[1].pie(churn_counts[["No", "Yes"]], labels=["No", "Yes"], autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Churn Percentage")
    c.save_fig("01_churn_distribution.png")

    # ------------------------------------------------------------------ #
    c.section("A6. Histograms and boxplots")
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, col in zip(axes.ravel(), numeric_columns):
        sns.histplot(df[col], bins=30, kde=True, ax=ax)
        ax.set_title(col)
    c.save_fig("01_histograms.png")

    fig, axes = plt.subplots(3, 3, figsize=(15, 9))
    for ax, col in zip(axes.ravel(), numeric_columns):
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(col)
    c.save_fig("01_boxplots.png")

    # ------------------------------------------------------------------ #
    c.section("A7. Churn by ContractType and AutoPay")
    churn_flag = (df["Churn"] == "Yes").astype(int)
    contract_summary = pd.DataFrame({
        "customers": df["ContractType"].value_counts(),
        "churn_rate_percent": (churn_flag.groupby(df["ContractType"]).mean() * 100).round(2),
    })
    autopay_summary = pd.DataFrame({
        "customers": df["AutoPay"].value_counts(),
        "churn_rate_percent": (churn_flag.groupby(df["AutoPay"]).mean() * 100).round(2),
    })
    combined = (churn_flag.groupby([df["ContractType"], df["AutoPay"]]).mean() * 100).round(2).unstack()
    print("Churn rate by ContractType\n", contract_summary)
    print("\nChurn rate by AutoPay\n", autopay_summary)
    print("\nChurn rate (%) by ContractType x AutoPay\n", combined)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    sns.barplot(x=contract_summary.index, y=contract_summary["churn_rate_percent"],
                order=contract_order, ax=axes[0])
    axes[0].set_title("Churn Rate by ContractType")
    sns.barplot(x=autopay_summary.index, y=autopay_summary["churn_rate_percent"],
                order=["No", "Yes"], ax=axes[1])
    axes[1].set_title("Churn Rate by AutoPay")
    for ax in axes[:2]:
        ax.set_ylabel("Churn rate (%)")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f")
    sns.heatmap(combined, annot=True, fmt=".1f", cmap="Reds", ax=axes[2])
    axes[2].set_title("Churn Rate (%) by ContractType x AutoPay")
    c.save_fig("01_churn_by_contract_autopay.png")

    # ------------------------------------------------------------------ #
    c.section("A8. Numerical variables vs Churn")
    print(df.groupby("Churn")[numeric_columns].mean().round(2).T)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for ax, col in zip(axes, ["TenureMonths", "SatisfactionScore", "SupportCalls6M", "LatePayments12M"]):
        sns.boxplot(data=df, x="Churn", y=col, order=["No", "Yes"], ax=ax)
        ax.set_title(col + " by Churn")
    c.save_fig("01_numeric_vs_churn.png")

    # ------------------------------------------------------------------ #
    c.section("A9. Correlations and the revenue target")
    corr = df[numeric_columns].corr()
    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Matrix of Numerical Variables")
    c.save_fig("01_correlation_matrix.png")

    print("Correlation with Future12MRevenueEGP")
    print(corr[c.REGRESSION_TARGET].drop(c.REGRESSION_TARGET).sort_values(ascending=False).round(3))

    # ------------------------------------------------------------------ #
    c.section("A10. Main drivers of future revenue")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    sns.scatterplot(data=df, x="MonthlyChargeEGP", y=c.REGRESSION_TARGET, alpha=0.4, ax=axes[0])
    axes[0].set_title("Revenue vs MonthlyChargeEGP")
    sns.boxplot(data=df, x="ContractType", y=c.REGRESSION_TARGET, order=contract_order, ax=axes[1])
    axes[1].set_title("Revenue by ContractType")
    sns.boxplot(data=df, x="NumServices", y=c.REGRESSION_TARGET, ax=axes[2])
    axes[2].set_title("Revenue by NumServices")
    c.save_fig("01_revenue_drivers.png")

    print("Average Future12MRevenueEGP by ContractType")
    print(df.groupby("ContractType")[c.REGRESSION_TARGET].mean().round(2))

    # ------------------------------------------------------------------ #
    c.section("A11. Data-quality check")
    print("Duplicate rows:", df.duplicated().sum())
    print("Unique CustomerID values:", df[c.ID_COLUMN].nunique(), "out of", len(df))
    print("Non-positive Future12MRevenueEGP values:", (df[c.REGRESSION_TARGET] <= 0).sum())
    print("Non-positive MonthlyChargeEGP values:", (df["MonthlyChargeEGP"] <= 0).sum())

    rows = []
    for col in numeric_columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append({
            "Column": col,
            "Missing": df[col].isnull().sum(),
            "Skewness": round(df[col].skew(), 2),
            "IQR_Outliers": int(((df[col] < lower) | (df[col] > upper)).sum()),
            "Min": df[col].min(),
            "Max": df[col].max(),
        })
    print()
    print(pd.DataFrame(rows).to_string(index=False))

    # ------------------------------------------------------------------ #
    c.section("A12. Column roles")
    roles = []
    for col in df.columns:
        if col == c.ID_COLUMN:
            role = "Identifier - exclude from all models"
        elif col == c.CLASSIFICATION_TARGET:
            role = "Classification target (Part C) - not a predictor for regression"
        elif col == c.REGRESSION_TARGET:
            role = "Regression target (Part D) - never a predictor for churn (leakage)"
        else:
            role = "Feature"
        roles.append({"Column": col, "Role": role})
    print(pd.DataFrame(roles).to_string(index=False))


if __name__ == "__main__":
    run()
