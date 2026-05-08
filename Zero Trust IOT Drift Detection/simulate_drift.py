"""
simulate_drift.py
-----------------
Simulates realistic IoT sensor drift on the production dataset.

Drift scenarios modelled (mimicking real Zero-Trust IoT degradation):
  1. COVARIATE DRIFT  — feature distributions shift (e.g., sensor noise spikes,
                        sharpness degrades due to lens fouling or vibration)
  2. LABEL DRIFT      — class proportions shift (e.g., more blurry/dark frames
                        due to environmental change or adversarial tampering)
  3. SUDDEN DRIFT     — abrupt shift in a subset of rows (simulates sensor attack
                        or physical fault injection)

Output: data/production_drifted.csv
"""

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

FEATURES = ["mean_intensity", "contrast", "sharpness", "noise_level"]


def covariate_drift(df: pd.DataFrame) -> pd.DataFrame:
    drifted = df.copy()

    # All classes: noise_level spikes (simulates network/EMI interference)
    drifted["noise_level"] = drifted["noise_level"] * 1.6 + np.random.normal(0, 200, len(drifted))

    # blur & clear: sharpness degrades (camera/sensor fouling)
    mask_blur_clear = drifted["label"].isin(["blur", "clear"])
    drifted.loc[mask_blur_clear, "sharpness"] *= 0.5

    # dark: mean_intensity drops further (power or lighting anomaly)
    mask_dark = drifted["label"] == "dark"
    drifted.loc[mask_dark, "mean_intensity"] *= 0.8

    # noisy: contrast increases with added variance (signal amplification drift)
    mask_noisy = drifted["label"] == "noisy"
    drifted.loc[mask_noisy, "contrast"] += np.random.normal(15, 5, mask_noisy.sum())

    drifted[FEATURES] = drifted[FEATURES].clip(lower=0)
    return drifted


def label_drift(df: pd.DataFrame, extra_blur_frac: float = 0.2) -> pd.DataFrame:
    """
    Increase proportion of 'blur' and 'dark' labels by duplicating + perturbing
    them — simulates environmental change making degraded frames more frequent.
    """
    drifted = df.copy()
    degraded = drifted[drifted["label"].isin(["blur", "dark"])].sample(
        frac=extra_blur_frac, random_state=SEED, replace=True
    ).copy()

    # Perturb slightly so they're not exact duplicates
    for col in FEATURES:
        degraded[col] += np.random.normal(0, degraded[col].std() * 0.05, len(degraded))
    degraded[FEATURES] = degraded[FEATURES].clip(lower=0)

    return pd.concat([drifted, degraded]).reset_index(drop=True)


def sudden_drift(df: pd.DataFrame, attack_frac: float = 0.1) -> pd.DataFrame:
    """
    Inject sudden extreme anomalies in a random subset — simulates a Zero-Trust
    scenario where a compromised IoT node sends adversarially corrupted readings.
    """
    drifted = df.copy()
    attack_idx = drifted.sample(frac=attack_frac, random_state=SEED).index

    # Extreme values: saturate sharpness and max-out noise (spoofed sensor data)
    drifted.loc[attack_idx, "sharpness"] *= np.random.uniform(8, 15, len(attack_idx))
    drifted.loc[attack_idx, "noise_level"] *= np.random.uniform(3, 6, len(attack_idx))
    drifted.loc[attack_idx, "mean_intensity"] = np.random.uniform(200, 255, len(attack_idx))

    return drifted


def main():
    prod_path = "data/production_clean.csv"
    df = pd.read_csv(prod_path)
    print(f"Loaded production data: {df.shape}")
    print(f"Clean class distribution:\n{df['label'].value_counts().to_string()}\n")

    print("Applying covariate drift  (noise spike, sharpness/intensity shift)...")
    drifted = covariate_drift(df)

    print("Applying label drift      (more degraded frames)...")
    drifted = label_drift(drifted)

    print("Applying sudden drift     (adversarial sensor injection)...")
    drifted = sudden_drift(drifted)

    drifted.to_csv("data/production_drifted.csv", index=False)
    print(f"\nDrifted shape: {drifted.shape}")
    print(f"Drifted class distribution:\n{drifted['label'].value_counts().to_string()}")
    print("\n✅ Saved: data/production_drifted.csv")

    # Print summary of how much features shifted
    print("\n--- Feature Mean Comparison (Clean vs Drifted) ---")
    clean = pd.read_csv(prod_path)
    for col in FEATURES:
        delta = ((drifted[col].mean() - clean[col].mean()) / (clean[col].mean() + 1e-9)) * 100
        print(f"  {col:20s}: clean={clean[col].mean():8.2f}  drifted={drifted[col].mean():8.2f}  Δ={delta:+.1f}%")


if __name__ == "__main__":
    main()
