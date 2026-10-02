"""
02_classification.py - Part C: Churn Classification.

Run:  python solutions/02_classification.py
Saves figures to outputs/figures/02_*.png

Seven models are compared with the same preprocessing pipeline, then the
Random Forest is tuned with GridSearchCV (optimising F1, because churn is
imbalanced and accuracy alone is misleading).
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import common as c


def run() -> None:
    c.setup()
    df = c.load_data()

    # ------------------------------------------------------------------ #
    c.section("C1. Features, target and stratified 80/20 split")
    X, y = c.churn_xy(df)
    print("X shape:", X.shape, "| y shape:", y.shape)
    print("Features:", list(X.columns))
    print("Churn rate (y=1 means churn):", round(y.mean() * 100, 2), "%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=c.TEST_SIZE, random_state=c.RANDOM_STATE, stratify=y
    )
    print("X_train:", X_train.shape, "| X_test:", X_test.shape)
    print("Churn rate train/test:", round(y_train.mean() * 100, 2), "% /", round(y_test.mean() * 100, 2), "%")

    preprocessor = c.build_preprocessor(X_train)

    # ------------------------------------------------------------------ #
    c.section("C2. Comparing seven classifiers")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=c.RANDOM_STATE),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=c.RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=c.RANDOM_STATE),
        "SVM": SVC(probability=True, random_state=c.RANDOM_STATE),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=c.RANDOM_STATE),
    }
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=c.RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
               "f1": "f1", "roc_auc": "roc_auc"}

    fitted, probabilities, rows = {}, {}, []
    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        cv = cross_validate(pipe, X_train, y_train, cv=skfold, scoring=scoring)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        fitted[name], probabilities[name] = pipe, proba
        rows.append({
            "Model": name,
            "CV_ROC_AUC": cv["test_roc_auc"].mean(),
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "ROC_AUC": roc_auc_score(y_test, proba),
        })
        print(f"  finished {name}")

    results = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    metric_cols = ["CV_ROC_AUC", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    results[metric_cols] = results[metric_cols].round(4)
    print("\nModel comparison (test set):\n", results.to_string(index=False))

    best_name = results.loc[0, "Model"]
    print("\nModel selected by test ROC-AUC:", best_name)

    # ------------------------------------------------------------------ #
    c.section("C3. Confusion matrix of the best baseline model")
    cm = confusion_matrix(y_test, fitted[best_name].predict(X_test))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix - " + best_name)
    c.save_fig("02_confusion_matrix_best_baseline.png")
    print(cm)

    # ------------------------------------------------------------------ #
    c.section("C4. ROC curves for all models")
    plt.figure(figsize=(7, 6))
    for name, proba in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, proba):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - All Models")
    plt.legend(loc="lower right", fontsize=8)
    c.save_fig("02_roc_curves.png")

    # ------------------------------------------------------------------ #
    c.section("C5. Tuning Random Forest with GridSearchCV (optimising F1)")
    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__class_weight": [None, "balanced"],
    }
    rf_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(random_state=c.RANDOM_STATE)),
    ])
    grid = GridSearchCV(rf_pipe, param_grid, cv=skfold, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best parameters found:", grid.best_params_)
    print("Best cross-validated F1 score:", round(grid.best_score_, 4))

    baseline_rf, tuned_rf = fitted["Random Forest"], grid.best_estimator_
    comparison = []
    for label, pipe in [("Random Forest (baseline)", baseline_rf), ("Random Forest (tuned)", tuned_rf)]:
        pred, proba = pipe.predict(X_test), pipe.predict_proba(X_test)[:, 1]
        comparison.append({
            "Model": label,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "ROC_AUC": roc_auc_score(y_test, proba),
        })
    print("\nBaseline vs tuned:\n", pd.DataFrame(comparison).set_index("Model").round(4))

    cm_tuned = confusion_matrix(y_test, tuned_rf.predict(X_test))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(cm_tuned, display_labels=["No Churn", "Churn"]).plot(ax=ax, cmap="Greens", colorbar=False)
    ax.set_title("Confusion Matrix - Tuned Random Forest")
    c.save_fig("02_confusion_matrix_tuned_rf.png")
    print(cm_tuned)

    # ------------------------------------------------------------------ #
    c.section("C6. Important churn predictors (tuned Random Forest)")
    names = tuned_rf.named_steps["preprocessor"].get_feature_names_out()
    importance = pd.Series(tuned_rf.named_steps["model"].feature_importances_, index=names)
    importance = importance.sort_values(ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=importance.head(10).values, y=importance.head(10).index, color="steelblue")
    plt.title("Top 10 Churn Predictors - Tuned Random Forest")
    plt.xlabel("Feature Importance")
    c.save_fig("02_churn_feature_importance.png")
    print(importance.head(10).round(4))

    print(
        "\nTakeaway: class_weight='balanced' trades accuracy for recall, which is the right "
        "trade for a retention campaign (a missed churner costs more than an extra offer)."
    )


if __name__ == "__main__":
    run()
