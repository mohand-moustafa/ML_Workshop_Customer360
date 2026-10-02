# Customer 360 Intelligence
## Practical Machine Learning Workshop

This repository contains the practical capstone workshop for applying the
supervised and unsupervised Machine Learning techniques covered during the course.

Students will work on a realistic **Customer 360 Intelligence** business case
for a fictional telecom/digital-services company.

The objective is not only to train ML models, but to complete an end-to-end
Machine Learning workflow:

Business Problem
→ Data Understanding
→ EDA
→ Data Preprocessing
→ Supervised Learning
→ Unsupervised Learning
→ Model Evaluation
→ Business Interpretation
→ Customer 360 Output



## Business Scenario

You are a Data Scientist at **NileConnect**, a fictional telecom/digital-services company.
Management wants a Customer 360 analytics solution that answers:

1. **Which customers are likely to churn?** — Classification
2. **How much revenue may each customer generate during the next 12 months?** — Regression
3. **What natural customer groups exist?** — Clustering
4. **Can customer behavior be represented in fewer dimensions?** — PCA
5. **Which customers have unusual behavior?** — Anomaly Detection

The supplied dataset is synthetic and created for education. It contains 3,000 realistic customer
records, categorical and numerical features, and controlled missing values.

## Learning Outcomes

By completing the project, you should be able to:

- perform EDA and identify data-quality issues;
- avoid target leakage and remove identifier columns from modeling;
- impute missing values, encode categorical variables, and scale numeric variables;
- build reusable `scikit-learn` pipelines;
- compare classification and regression algorithms;
- choose appropriate evaluation metrics;
- tune a model with cross-validation;
- interpret Random Forest feature importance;
- apply K-Means, Agglomerative Clustering, and DBSCAN;
- use PCA for dimensionality reduction and visualization;
- detect unusual observations with Isolation Forest;
- combine multiple ML outputs into a business-oriented Customer 360 table.

## Environment

The workshop is designed for:

- **Python 3.7.16**
- pandas 1.3.5
- numpy 1.21.6
- matplotlib 3.5.3
- scipy 1.7.3
- scikit-learn 1.0.2
- jupyter 1.0.0

The code intentionally avoids newer APIs such as `sparse_output=` and does not depend on
`ColumnTransformer.get_feature_names_out()`.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/mhemaly/ML_Workshop_Customer360.git
cd ML_Workshop_Customer360
```

### 2. Create a Python 3.7 environment

Using Conda:

```bash
conda create -n ml-workshop python=3.7.16 -y
conda activate ml-workshop
pip install -r requirements.txt
```

Or using a Python 3.7 virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Verify the environment

```bash
python check_environment.py
```

### 4. Start working

You can use the supplied starter script:

```bash
python src/workshop_tasks.py
```

or create your own Jupyter notebook:

```bash
jupyter notebook
```

## Repository Structure

```text
ML_Workshop_Customer360/
│
├── README.md
├── requirements.txt
├── check_environment.py
│
├── data/
│   └── customer_360_ml_workshop.csv
│
├── src/
│   └── workshop_tasks.py
│
├── notebooks/
│
├── outputs/
│
└── solutions/                 ← Will be Added AFTER workshop
    ├── README.md
    ├── common.py
    ├── 01_eda.py
    ├── 02_classification.py
    ├── 03_regression.py
    ├── 04_unsupervised.py
    └── 05_customer360_integration.py
    └── main.py
```

## Dataset

Important columns include:

| Column | Description |
|---|---|
| CustomerID | Unique customer identifier; do not use as a model feature |
| Age | Customer age |
| Region | Customer region |
| TenureMonths | Months with the company |
| ContractType | Month-to-month, one-year, or two-year |
| InternetType | Customer internet technology |
| MonthlyUsageGB | Monthly usage |
| MonthlyChargeEGP | Monthly bill |
| NumServices | Number of subscribed services |
| SupportCalls6M | Support calls during last six months |
| LatePayments12M | Late payments during last year |
| AutoPay | Whether automatic payment is enabled |
| SatisfactionScore | Satisfaction score |
| Future12MRevenueEGP | Regression target |
| Churn | Classification target |

## Workshop Tasks

### Part A — Business Understanding & EDA

Load the dataset and investigate:

- shape and data types;
- missing values;
- descriptive statistics;
- churn distribution;
- numerical distributions;
- churn patterns by contract type and AutoPay.

Answer:

1. Which column must be excluded because it is an identifier?
2. Why should `Future12MRevenueEGP` not be used to predict current churn?
3. Is accuracy sufficient for churn prediction?
4. Which algorithms are sensitive to feature scaling?

### Part B — Data Preprocessing

Create a reusable preprocessing pipeline.

Requirements:

- median imputation for numerical variables;
- most-frequent imputation for categorical variables;
- standardization for numerical variables;
- one-hot encoding for categorical variables;
- `handle_unknown="ignore"`.

Do not fit preprocessing separately to the complete dataset before train/test splitting.

### Part C — Churn Classification

Target:

```text
Churn
```

Compare at least:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Support Vector Machine
- Naive Bayes
- Gradient Boosting

Use a stratified train/test split.

Report:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

Then:

- plot a confusion matrix;
- plot the ROC curve;
- tune Random Forest using `GridSearchCV`;
- identify important churn predictors.

### Part D — Revenue Regression

Target:

```text
Future12MRevenueEGP
```

Compare at least:

- Linear Regression
- Ridge
- Lasso
- KNN Regressor
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

Report:

- MAE
- RMSE
- R²

Plot Actual Revenue vs Predicted Revenue.

### Part E — Customer Segmentation

Use behavioral/value variables such as:

```text
TenureMonths
MonthlyUsageGB
MonthlyChargeEGP
NumServices
SupportCalls6M
LatePayments12M
SatisfactionScore
```

Apply:

1. K-Means
2. Agglomerative Clustering
3. DBSCAN

For K-Means:

- test several values of `k`;
- calculate inertia;
- calculate silhouette score;
- justify the selected `k`;
- profile every cluster;
- assign a meaningful business name to each cluster.

For DBSCAN, explain the meaning of label `-1`.

### Part F — PCA & Anomaly Detection

Apply PCA with two components to the standardized segmentation data.

Report:

- explained variance;
- a PC1/PC2 scatter plot colored by K-Means segment.

Then apply Isolation Forest and investigate unusual customers.

Do **not** automatically describe anomalies as fraud.

### Part G — Customer 360 Integration

Create a final table containing:

- CustomerID
- churn probability
- predicted 12-month revenue
- customer segment
- anomaly flag
- retention-priority flag

Define and explain a reasonable retention-priority rule.

Save the final output to:

```text
outputs/customer_360_predictions.csv
```

## Required Student Submission

Submit:

1. Source code or notebook.
2. EDA summary with at least five observations.
3. Classification model-comparison table.
4. Confusion matrix and ROC curve.
5. Tuned model and important features.
6. Regression model-comparison table.
7. Actual-vs-predicted revenue visualization.
8. K-Means selection evidence and cluster profiles.
9. Comparison of K-Means, Agglomerative Clustering, and DBSCAN.
10. PCA visualization and explained variance.
11. At least ten investigated anomalies.
12. Customer 360 output CSV.
13. Three to five business recommendations.
14. Short answers to the challenge questions below.

## Challenge Questions

1. What changes when the churn threshold moves from 0.50 to 0.35 or 0.70?
2. When is recall more important than precision for churn?
3. Why must `CustomerID` be excluded from model features?
4. Why is one-hot encoding required?
5. Which algorithms are most sensitive to scaling?
6. Can a cluster be considered useful only because its silhouette score is high?
7. How would you monitor model drift after deployment?
8. What risks exist if retention offers are automatically assigned by a model?

## Rules

- Use `random_state=42` whenever the algorithm supports it.
- Keep the test set untouched during model selection.
- Do not use `CustomerID` as a predictor.
- Avoid target leakage.
- Explain business meaning, not only metrics.
- Exact metric values may differ slightly across environments.

## Solutions

The solution files are intentionally not included during the workshop.

Students are expected to complete the tasks independently or with their teams.
Reference solutions will be released after the practical session.
