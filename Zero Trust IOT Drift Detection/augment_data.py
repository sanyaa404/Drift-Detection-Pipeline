"""
augment_data.py
---------------
Expands the original 79-row dataset to ~1000 rows using SMOTE + Gaussian noise.
Outputs:
  data/reference.csv      — "healthy" baseline data for model training
  data/production_clean.csv — clean production batch (no drift yet)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from pyspark.sql import SparkSession

FEATURES = ["mean_intensity", "contrast", "sharpness", "noise_level"]
SEED = 42
np.random.seed(SEED)


# def load_data(path: str) -> pd.DataFrame:
#     df = pd.read_csv(path)
#     df["label"] = df["label"].str.lower().str.strip()
#     return df

def load_data(path: str) -> pd.DataFrame:
    print("Using Spark for scalable data ingestion...")

    spark = SparkSession.builder \
        .appName("IoT Drift Pipeline") \
        .getOrCreate()

    spark_df = spark.read.csv(
        path,
        header=True,
        inferSchema=True
    )

    print("  Spark Data Preview:")
    spark_df.show(5)

    print("  Schema:")
    spark_df.printSchema()

    # Convert to Pandas for rest of pipeline
    df = spark_df.toPandas()

    # Clean labels (same as before)
    df["label"] = df["label"].str.lower().str.strip()

    print(f"  Converted to Pandas: {df.shape}")

    return df


def augment_with_smote(df: pd.DataFrame, target_total: int = 900) -> pd.DataFrame:
    """Use SMOTE to oversample minority classes, then add Gaussian jitter."""
    le = LabelEncoder()
    X = df[FEATURES].values
    y = le.fit_transform(df["label"].values)

    # Each class gets roughly equal representation
    n_classes = len(np.unique(y))
    per_class = target_total // n_classes
    sampling_strategy = {i: per_class for i in range(n_classes)}

    sm = SMOTE(sampling_strategy=sampling_strategy, random_state=SEED, k_neighbors=min(3, min(np.bincount(y)) - 1))
    X_res, y_res = sm.fit_resample(X, y)

    # Add small Gaussian noise so samples aren't exact SMOTE interpolations
    noise_scale = X_res.std(axis=0) * 0.03
    X_res = X_res + np.random.normal(0, noise_scale, X_res.shape)
    X_res = np.clip(X_res, 0, None)  # no negative values

    labels = le.inverse_transform(y_res)
    augmented = pd.DataFrame(X_res, columns=FEATURES)
    augmented["label"] = labels
    return augmented


def split_reference_production(df: pd.DataFrame, ref_ratio: float = 0.7):
    """Stratified split into reference (train) and production (deploy) sets."""
    ref_parts, prod_parts = [], []
    for label in df["label"].unique():
        subset = df[df["label"] == label].sample(frac=1, random_state=SEED)
        cut = int(len(subset) * ref_ratio)
        ref_parts.append(subset.iloc[:cut])
        prod_parts.append(subset.iloc[cut:])
    return pd.concat(ref_parts).reset_index(drop=True), pd.concat(prod_parts).reset_index(drop=True)


def main():
    os.makedirs("data", exist_ok=True)

    raw_path = "data/image_quality_features.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Place your CSV at: {raw_path}")

    print("Loading original data...")
    df = load_data(raw_path)
    print(f"  Original shape: {df.shape}")
    print(f"  Class distribution:\n{df['label'].value_counts().to_string()}")

    print("\nAugmenting with SMOTE + Gaussian noise...")
    augmented = augment_with_smote(df, target_total=900)
    print(f"  Augmented shape: {augmented.shape}")
    print(f"  Class distribution:\n{augmented['label'].value_counts().to_string()}")

    print("\nSplitting into reference / production sets...")
    reference, production = split_reference_production(augmented)
    print(f"  Reference rows : {len(reference)}")
    print(f"  Production rows: {len(production)}")

    reference.to_csv("data/reference.csv", index=False)
    production.to_csv("data/production_clean.csv", index=False)
    print("\n✅ Saved: data/reference.csv")
    print("✅ Saved: data/production_clean.csv")


if __name__ == "__main__":
    main()
