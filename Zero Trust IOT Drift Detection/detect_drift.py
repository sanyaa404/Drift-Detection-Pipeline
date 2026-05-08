"""
detect_drift.py
---------------
Drift detection engine for the Zero Trust IoT pipeline.

Two-layer detection:
  Layer 1 — Statistical: Kolmogorov-Smirnov test + Population Stability Index (PSI)
  Layer 2 — ML-based:    Evidently DataDriftPreset report

If drift is detected, automatically triggers model retraining and logs
the new model + all metrics to MLflow.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from evidently.report import Report
from evidently.metrics import DataDriftTable, DatasetDriftMetric

warnings.filterwarnings("ignore")

FEATURES = ["mean_intensity", "contrast", "sharpness", "noise_level"]
DRIFT_EXPERIMENT = "ZeroTrust_IoT_DriftDetection"

# ── Thresholds ─────────────────────────────────────────────────────────────────
KS_P_VALUE_THRESHOLD = 0.05      # p < 0.05 → statistically significant drift
PSI_WARNING_THRESHOLD = 0.1      # PSI 0.1–0.2 → slight drift
PSI_CRITICAL_THRESHOLD = 0.2     # PSI > 0.2 → significant drift (retrain!)
DRIFT_FEATURE_FRACTION = 0.5     # retrain if >50% of features show drift


# ── PSI Calculation ────────────────────────────────────────────────────────────
def compute_psi(reference: np.ndarray, production: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index:
      PSI < 0.1  → no significant change
      PSI 0.1–0.2 → moderate change (monitor)
      PSI > 0.2  → significant change (retrain)
    """
    min_val = min(reference.min(), production.min())
    max_val = max(reference.max(), production.max())
    breakpoints = np.linspace(min_val, max_val, bins + 1)

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    prod_counts, _ = np.histogram(production, bins=breakpoints)

    ref_pct = (ref_counts + 1e-6) / len(reference)
    prod_pct = (prod_counts + 1e-6) / len(production)

    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return float(psi)


# ── KS Test ────────────────────────────────────────────────────────────────────
def run_ks_tests(reference: pd.DataFrame, production: pd.DataFrame) -> dict:
    results = {}
    for feat in FEATURES:
        ks_stat, p_val = stats.ks_2samp(reference[feat].values, production[feat].values)
        drifted = p_val < KS_P_VALUE_THRESHOLD
        results[feat] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_val), 4),
            "drift_detected": drifted,
        }
    return results


# ── PSI for all features ───────────────────────────────────────────────────────
def run_psi_tests(reference: pd.DataFrame, production: pd.DataFrame) -> dict:
    results = {}
    for feat in FEATURES:
        psi = compute_psi(reference[feat].values, production[feat].values)
        severity = "none"
        if psi > PSI_CRITICAL_THRESHOLD:
            severity = "critical"
        elif psi > PSI_WARNING_THRESHOLD:
            severity = "warning"
        results[feat] = {
            "psi": round(psi, 4),
            "severity": severity,
            "drift_detected": psi > PSI_CRITICAL_THRESHOLD,
        }
    return results


# ── Evidently Drift Report ─────────────────────────────────────────────────────
def run_evidently_report(reference: pd.DataFrame, production: pd.DataFrame, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    report = Report(metrics=[DataDriftTable(), DatasetDriftMetric()])
    report.run(reference_data=reference[FEATURES + ["label"]],
               current_data=production[FEATURES + ["label"]])

    html_path = os.path.join(output_dir, "evidently_drift_report.html")
    report.save_html(html_path)

    result_dict = report.as_dict()
    dataset_drift = result_dict["metrics"][1]["result"]
    return {
        "dataset_drift_detected": dataset_drift.get("dataset_drift", False),
        "drift_share": round(dataset_drift.get("share_of_drifted_columns", 0.0), 4),
        "n_drifted_features": dataset_drift.get("number_of_drifted_columns", 0),
        "report_path": html_path,
    }


# ── Model Training / Retraining ────────────────────────────────────────────────
def train_model(train_df: pd.DataFrame, run_name: str = "baseline") -> tuple:
    le = LabelEncoder()
    X = train_df[FEATURES].values
    y = le.fit_transform(train_df["label"].values)

    model = RandomForestClassifier(n_estimators=150, max_depth=10,
                                   random_state=42, class_weight="balanced")
    model.fit(X, y)
    return model, le


def evaluate_model(model, le, eval_df: pd.DataFrame) -> dict:
    X = eval_df[FEATURES].values
    y_true = le.transform(eval_df["label"].values)
    y_pred = model.predict(X)
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "report": classification_report(y_true, y_pred, target_names=le.classes_),
    }


# ── Drift Decision Logic ───────────────────────────────────────────────────────
def should_retrain(ks_results: dict, psi_results: dict, evidently_results: dict) -> tuple[bool, str]:
    ks_drifted = sum(1 for v in ks_results.values() if v["drift_detected"])
    psi_drifted = sum(1 for v in psi_results.values() if v["drift_detected"])
    n_features = len(FEATURES)

    reasons = []
    if ks_drifted / n_features >= DRIFT_FEATURE_FRACTION:
        reasons.append(f"KS: {ks_drifted}/{n_features} features drifted")
    if psi_drifted / n_features >= DRIFT_FEATURE_FRACTION:
        reasons.append(f"PSI: {psi_drifted}/{n_features} features critical")
    if evidently_results["dataset_drift_detected"]:
        reasons.append(f"Evidently: dataset drift detected ({evidently_results['drift_share']*100:.0f}% features)")

    retrain = len(reasons) >= 2  # require at least 2 detection layers to agree
    return retrain, "; ".join(reasons) if reasons else "No significant drift detected"


# ── Main Pipeline ──────────────────────────────────────────────────────────────
def main():
    mlflow.set_experiment(DRIFT_EXPERIMENT)

    reference = pd.read_csv("data/reference.csv")
    production_drifted = pd.read_csv("data/production_drifted.csv")
    production_clean = pd.read_csv("data/production_clean.csv")

    # ── STEP 1: Train baseline model ──────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Training baseline model on reference data...")
    print("=" * 60)
    with mlflow.start_run(run_name="baseline_training") as run:
        baseline_model, le = train_model(reference, run_name="baseline")
        baseline_metrics = evaluate_model(baseline_model, le, production_clean)

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 150)
        mlflow.log_param("training_samples", len(reference))
        mlflow.log_metric("baseline_accuracy", baseline_metrics["accuracy"])
        mlflow.log_metric("baseline_f1_weighted", baseline_metrics["f1_weighted"])
        mlflow.sklearn.log_model(baseline_model, "baseline_model")

        print(f"  Baseline Accuracy : {baseline_metrics['accuracy']}")
        print(f"  Baseline F1 Score : {baseline_metrics['f1_weighted']}")
        baseline_run_id = run.info.run_id

    # ── STEP 2: Detect drift on drifted production data ───────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Running drift detection on production data...")
    print("=" * 60)
    with mlflow.start_run(run_name="drift_detection") as drift_run:
        print("\n  [Layer 1a] Kolmogorov-Smirnov Tests...")
        ks_results = run_ks_tests(reference, production_drifted)
        for feat, res in ks_results.items():
            status = "⚠️  DRIFT" if res["drift_detected"] else "✅ OK"
            print(f"    {feat:20s}: KS={res['ks_statistic']:.4f}  p={res['p_value']:.4f}  {status}")
            mlflow.log_metric(f"ks_stat_{feat}", res["ks_statistic"])
            mlflow.log_metric(f"ks_pval_{feat}", res["p_value"])
            mlflow.log_metric(f"ks_drift_{feat}", int(res["drift_detected"]))

        print("\n  [Layer 1b] Population Stability Index (PSI)...")
        psi_results = run_psi_tests(reference, production_drifted)
        for feat, res in psi_results.items():
            icon = {"none": "✅", "warning": "🟡", "critical": "🔴"}.get(res["severity"], "")
            print(f"    {feat:20s}: PSI={res['psi']:.4f}  [{res['severity'].upper()}]  {icon}")
            mlflow.log_metric(f"psi_{feat}", res["psi"])
            mlflow.log_metric(f"psi_drift_{feat}", int(res["drift_detected"]))

        print("\n  [Layer 2]  Evidently ML Drift Report...")
        evidently_results = run_evidently_report(reference, production_drifted, "reports")
        print(f"    Dataset drift detected : {evidently_results['dataset_drift_detected']}")
        print(f"    Drifted features       : {evidently_results['n_drifted_features']}/{len(FEATURES)}")
        print(f"    Drift share            : {evidently_results['drift_share']*100:.1f}%")
        mlflow.log_metric("evidently_drift_detected", int(evidently_results["dataset_drift_detected"]))
        mlflow.log_metric("evidently_drift_share", evidently_results["drift_share"])
        mlflow.log_artifact(evidently_results["report_path"])

        retrain_needed, reason = should_retrain(ks_results, psi_results, evidently_results)
        mlflow.log_param("retrain_triggered", retrain_needed)
        mlflow.log_param("drift_reason", reason)

        print(f"\n  {'🚨 DRIFT CONFIRMED' if retrain_needed else '✅ NO DRIFT'}: {reason}")
        drift_run_id = drift_run.info.run_id

    # ── STEP 3: Auto-retrain if drift detected ────────────────────────────────
    if retrain_needed:
        print("\n" + "=" * 60)
        print("STEP 3: Drift detected → Auto-retraining model...")
        print("=" * 60)

        # Retrain on reference + a portion of drifted production (drift adaptation)
        adaptation_sample = production_drifted.sample(frac=0.3, random_state=42)
        combined_train = pd.concat([reference, adaptation_sample]).reset_index(drop=True)

        with mlflow.start_run(run_name="retrained_model") as retrain_run:
            retrained_model, le_new = train_model(combined_train, run_name="retrained")
            retrained_metrics = evaluate_model(retrained_model, le_new, production_drifted)

            mlflow.log_param("trigger", "auto_drift_detection")
            mlflow.log_param("parent_drift_run_id", drift_run_id)
            mlflow.log_param("parent_baseline_run_id", baseline_run_id)
            mlflow.log_param("training_samples", len(combined_train))
            mlflow.log_param("drift_reason", reason)
            mlflow.log_metric("retrained_accuracy", retrained_metrics["accuracy"])
            mlflow.log_metric("retrained_f1_weighted", retrained_metrics["f1_weighted"])
            mlflow.log_metric("accuracy_delta",
                              round(retrained_metrics["accuracy"] - baseline_metrics["accuracy"], 4))

            mlflow.sklearn.log_model(retrained_model, "retrained_model",
                                     registered_model_name="ZeroTrust_IoT_Classifier")

            print(f"  Retrained Accuracy : {retrained_metrics['accuracy']}")
            print(f"  Retrained F1 Score : {retrained_metrics['f1_weighted']}")
            print(f"  Δ Accuracy         : {retrained_metrics['accuracy'] - baseline_metrics['accuracy']:+.4f}")
            print(f"\n  Model registered as: ZeroTrust_IoT_Classifier")
            print(f"  MLflow Run ID      : {retrain_run.info.run_id}")

    print("\n" + "=" * 60)
    print("✅ Pipeline complete. View results:")
    print("   mlflow ui   →   http://127.0.0.1:5000")
    print("   Evidently   →   reports/evidently_drift_report.html")
    print("=" * 60)


if __name__ == "__main__":
    main()
