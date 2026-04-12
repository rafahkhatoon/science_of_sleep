"""
utils/clustering.py
K-Means clustering on sleep-related features.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


CLUSTER_FEATURES = [
    "Sleep Duration",
    "Quality of Sleep",
    "Stress Level",
    "Physical Activity Level",
    "Daily Steps",
]


def run_kmeans(df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42):
    """
    Fit K-Means on the sleep feature set.

    Returns
    -------
    df_clustered : DataFrame with added 'Cluster' column (int)
    model        : fitted KMeans object
    scaler       : fitted StandardScaler
    """
    available = [c for c in CLUSTER_FEATURES if c in df.columns]
    X = df[available].copy()
    X.fillna(X.median(), inplace=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X_scaled)

    df_out = df.copy()
    df_out["Cluster"] = labels

    # Re-label clusters by descending average sleep quality
    # (Cluster 0 = best sleepers)
    quality_means = df_out.groupby("Cluster")["Quality of Sleep"].mean()
    rank_map = {old: new for new, old in enumerate(quality_means.sort_values(ascending=False).index)}
    df_out["Cluster"] = df_out["Cluster"].map(rank_map)

    return df_out, model, scaler


def get_cluster_profiles(df_clustered: pd.DataFrame, n_clusters: int) -> list[dict]:
    """Return a list of summary dicts, one per cluster, sorted by cluster index."""
    profiles = []
    total = len(df_clustered)
    for i in range(n_clusters):
        sub = df_clustered[df_clustered["Cluster"] == i]
        profiles.append({
            "cluster": i,
            "n": len(sub),
            "pct": len(sub) / total * 100,
            "sleep":    sub["Sleep Duration"].mean(),
            "quality":  sub["Quality of Sleep"].mean(),
            "stress":   sub["Stress Level"].mean(),
            "activity": sub["Physical Activity Level"].mean(),
            "steps":    sub["Daily Steps"].mean(),
        })
    return profiles
