# Reference Solutions

Reference solutions for the Customer 360 Intelligence workshop. Released after the practical session.

## Files

| File | Workshop part |
|---|---|
| `common.py` | Shared paths, constants, preprocessing and helpers |
| `01_eda.py` | Part A - EDA |
| `02_classification.py` | Part C - Churn classification |
| `03_regression.py` | Part B + D - Preprocessing and revenue regression |
| `04_unsupervised.py` | Part E + F - Segmentation, PCA, anomaly detection |
| `05_customer360_integration.py` | Part G - Customer 360 integration |
| `main.py` | Runs all steps end to end |

## Run

From the repository root:

```bash
python solutions/main.py
```

Run selected steps only:

```bash
python solutions/main.py eda regression
```

Available steps: `eda`, `classification`, `regression`, `unsupervised`, `integration`.

## Outputs

Figures are saved to `outputs/figures/` and the final table to `outputs/customer_360_predictions.csv`.
