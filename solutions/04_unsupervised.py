"""
04_unsupervised.py - Part E (K-Means segmentation) + Part F (PCA & Anomaly Detection).

Run:  python solutions/04_unsupervised.py
Saves figures to outputs/figures/04_*.png

NOTE: Part E's notebook was not part of the original uploads, so the model
selection step below (elbow + silhouette for k = 2..8) is a standard
reconstruction. The final choice, k = 4 with random_state = 42, is the one
used by Part F and Part G of the workshop.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import common as c


def run() -> None:
    c.setup()
    df = c.load_data()

    # ------------------------------------------------------------------ #
    c.section("E1. Standardized segmentation data")
    X_scaled = c.scale_cluster_features(df)
    print("Features used:", c.CLUSTER_FEATURES)
    print("Standardized data shape:", X_scaled.shape)

    # ------------------------------------------------------------------ #
    c.section("E2. Choosing k (elbow method + silhouette score)")
    ks = list(range(2, 9))
    inertias, silhouettes = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=c.RANDOM_STATE, n_init=10).fit(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))
    print(pd.DataFrame({"k": ks, "inertia": np.round(inertias, 1), "silhouette": np.round(silhouettes, 4)})
          .to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(ks, inertias, marker="o")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[1].plot(ks, silhouettes, marker="o", color="darkorange")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette")
    c.save_fig("04_kmeans_k_selection.png")
    print(f"\nWorkshop choice: k = {c.N_CLUSTERS} (interpretable business segments).")

    # ------------------------------------------------------------------ #
    c.section("E3. Final K-Means segments and their business profile")
    df = c.assign_segments(df, X_scaled)
    print(df["CustomerSegment"].value_counts())

    profile = df.groupby("CustomerSegment")[c.CLUSTER_FEATURES].mean().round(2)
    profile["Customers"] = df["CustomerSegment"].value_counts()
    profile["ChurnRate_%"] = df.groupby("CustomerSegment")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).round(2)
    profile["AvgFutureRevenue"] = df.groupby("CustomerSegment")[c.REGRESSION_TARGET].mean().round(2)
    print("\nSegment profile:\n", profile.T)

    # ------------------------------------------------------------------ #
    c.section("F1. PCA with two components")
    pca = PCA(n_components=2, random_state=c.RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    evr = pd.Series(pca.explained_variance_ratio_, index=["PC1", "PC2"])
    print("Explained variance ratio:\n", evr.round(4))
    print("Total variance explained by PC1 + PC2:", round(evr.sum() * 100, 2), "%")

    plt.figure(figsize=(8, 6))
    for cid, name in sorted(c.CLUSTER_BUSINESS_NAMES.items()):
        mask = df["Cluster_KMeans"] == cid
        plt.scatter(coords[mask, 0], coords[mask, 1], s=15, alpha=0.6, label=name)
    plt.xlabel(f"PC1 ({evr['PC1'] * 100:.1f}% variance)")
    plt.ylabel(f"PC2 ({evr['PC2'] * 100:.1f}% variance)")
    plt.title("PCA Projection Colored by K-Means Segment")
    plt.legend(loc="best", fontsize=8)
    c.save_fig("04_pca_segments.png")

    # ------------------------------------------------------------------ #
    c.section("F2. Isolation Forest anomaly detection")
    df = c.flag_anomalies(df, X_scaled)
    print(df["AnomalyFlag"].value_counts())
    print("Share flagged:", round((df["AnomalyFlag"] == "Yes").mean() * 100, 2), "%")

    prof = df.groupby("AnomalyFlag")[c.CLUSTER_FEATURES].mean().round(2)
    prof["Count"] = df["AnomalyFlag"].value_counts()
    prof["ChurnRate_%"] = df.groupby("AnomalyFlag")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).round(2)
    prof["AvgFutureRevenue"] = df.groupby("AnomalyFlag")[c.REGRESSION_TARGET].mean().round(2)
    print("\nAnomalies vs normal customers:\n", prof.T)

    print("\nAnomalies across segments:\n", pd.crosstab(df["CustomerSegment"], df["AnomalyFlag"]))

    plt.figure(figsize=(8, 6))
    is_anom = (df["AnomalyFlag"] == "Yes").to_numpy()
    plt.scatter(coords[~is_anom, 0], coords[~is_anom, 1], c="steelblue", alpha=0.4, s=15, label="Normal")
    plt.scatter(coords[is_anom, 0], coords[is_anom, 1], c="red", marker="x", s=30, label="Anomaly")
    plt.xlabel(f"PC1 ({evr['PC1'] * 100:.1f}% variance)")
    plt.ylabel(f"PC2 ({evr['PC2'] * 100:.1f}% variance)")
    plt.title("Isolation Forest Anomalies - PCA Projection")
    plt.legend()
    c.save_fig("04_pca_anomalies.png")

    # ------------------------------------------------------------------ #
    c.section("F3. The 10 most unusual customers")
    cols = [c.ID_COLUMN] + c.CLUSTER_FEATURES + ["Churn", c.REGRESSION_TARGET, "CustomerSegment", "AnomalyScore"]
    print(df.sort_values("AnomalyScore").head(10)[cols].reset_index(drop=True).to_string())

    print(
        "\nReminder: Isolation Forest flags statistical unusualness, NOT fraud. "
        "Flagged accounts need a human review before anyone draws a conclusion."
    )


if __name__ == "__main__":
    run()
