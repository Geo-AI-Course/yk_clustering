import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).parent))
from load_and_split import DATA_PATH, RANDOM_STATE, load_data, split_data

MODEL_PATH = "models/kmeans.joblib"
K_MIN = 2
K_MAX = 8
FEATURE_COLS = [f"x{i}" for i in range(1, 12)]


def select_k(X: np.ndarray, k_min: int, k_max: int, random_state: int) -> tuple[int, dict]:
    k_max = min(k_max, len(X) - 1)
    if k_max < k_min:
        print(f"Warning: only {len(X)} training samples — silhouette selection skipped, using k={k_min}.")
        return k_min, {}
    scores = {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        scores[k] = silhouette_score(X, labels)
    if not scores:
        return k_min, {}
    return max(scores, key=scores.get), scores


def train(train_df: pd.DataFrame, k: int, random_state: int) -> KMeans:
    X = train_df[FEATURE_COLS].values
    model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    model.fit(X)
    return model


def cluster_profiles(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    df = df[FEATURE_COLS].copy()
    df["cluster"] = labels
    return df.groupby("cluster")[FEATURE_COLS].mean().round(3)


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    train_df, test_df = split_data(df)

    X_train = train_df[FEATURE_COLS].values
    X_test = test_df[FEATURE_COLS].values

    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows\n")

    best_k, scores = select_k(X_train, K_MIN, K_MAX, RANDOM_STATE)
    print("Silhouette scores by k:")
    for k, s in sorted(scores.items()):
        marker = " <-- best" if k == best_k else ""
        print(f"  k={k}: {s:.4f}{marker}")
    print()

    model = train(train_df, best_k, RANDOM_STATE)

    train_labels = model.predict(X_train)
    test_labels = model.predict(X_test)

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["cluster"] = train_labels
    test_df["cluster"] = test_labels

    print("Cluster profiles (mean feature % per cluster) — train set:")
    print(cluster_profiles(train_df, train_labels).to_string())
    print()

    print("Test set cluster assignments:")
    print(test_df[["ms_ezor", "cluster"]].to_string(index=False))

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
