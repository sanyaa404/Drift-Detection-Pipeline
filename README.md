# Zero Trust IoT System with DevOps-based Drift Detection for Secure Sensor Data Monitoring

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![MLflow](https://img.shields.io/badge/MLflow-3.x-orange?logo=mlflow)
![Evidently](https://img.shields.io/badge/Evidently-0.7.x-purple)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-green?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A machine learning pipeline that simulates, detects, and responds to **data drift** in IoT sensor streams — built around Zero Trust security principles. When drift is detected, the system **automatically retrains** the model and logs everything to MLflow for full observability.

---

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Pipeline Stages](#pipeline-stages)
- [Drift Simulation](#drift-simulation)
- [Drift Detection](#drift-detection)
- [Auto-Retraining](#auto-retraining)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [MLflow Dashboard](#mlflow-dashboard)

---

## Overview

In real-world IoT deployments, sensor data distributions shift over time due to environmental changes, hardware degradation, or adversarial tampering. This project implements a DevOps-style MLOps pipeline that:

1. Augments a small sensor dataset using SMOTE
2. Simulates 3 types of realistic IoT drift
3. Detects drift using a **3-layer detection system** (statistical + ML-based)
4. Automatically retrains the model when drift is confirmed
5. Tracks all experiments, metrics, and models in **MLflow**

The Zero Trust principle — *"never trust, always verify"* — is enforced through a voting mechanism: retraining only triggers when at least 2 independent detection layers agree that drift has occurred.

---

## Dataset

| Property | Value |
|---|---|
| Original rows | 79 |
| After augmentation | ~900 |
| Reference set | ~630 rows (70%) |
| Production set | ~270 rows (30%) |
| Features | mean_intensity, contrast, sharpness, noise_level |
| Classes | clear, blur, dark, noisy |

---

## Pipeline Stages
Raw CSV (79 rows)

↓

[Stage 1] augment_data.py       → SMOTE + Gaussian noise → ~900 rows

↓

[Stage 2] simulate_drift.py     → Injects 3 drift types → production_drifted.csv

↓

[Stage 3] detect_drift.py       → 3-layer detection → vote → auto-retrain

↓

MLflow logs everything

---

## Drift Simulation (3 Types)

**1. Covariate Drift** — feature distributions shift:
- noise_level ×1.6 → EMI / network interference
- sharpness ×0.5 → lens fouling or vibration
- mean_intensity ×0.8 → power fluctuation
- contrast +15 → signal amplification drift

**2. Label Drift** — 20% of blur/dark rows duplicated and perturbed → degraded frames become more frequent over time

**3. Sudden Drift (Adversarial)** — 10% of rows get extreme injected values → simulates a compromised IoT node sending spoofed readings

---

## Drift Detection (3 Layers)

**Layer 1a — Kolmogorov-Smirnov Test**
- Compares reference vs production distributions per feature
- Threshold: p-value < 0.05 → drifted
- Flags if ≥2/4 features drift

**Layer 1b — Population Stability Index (PSI)**
- PSI < 0.1 → no drift | 0.1–0.2 → warning | >0.2 → critical
- Flags if ≥2/4 features exceed 0.2

**Layer 2 — Evidently DataDrift Report**
- Wasserstein distance + chi-squared under the hood
- Outputs interactive HTML report + dataset drift flag

**Voting Logic:**
Retrain triggers only if ≥2 layers agree → Zero Trust "never trust, always verify"

---

## Auto-Retraining

When drift is confirmed the system:
1. Combines reference data + 30% of drifted production (drift adaptation)
2. Retrains Random Forest (150 trees, max_depth=10, balanced weights)
3. Logs new model to MLflow registry as `ZeroTrust_IoT_Classifier`
4. Records accuracy delta vs baseline

---

## Technologies Used

| Technology | Role |
|---|---|
| Python 3.12 | Core language |
| MLflow 3.x | Experiment tracking, model registry, UI dashboard |
| Evidently 0.7.x | ML drift reports and dataset monitoring |
| scikit-learn | Random Forest, metrics, label encoding |
| imbalanced-learn | SMOTE oversampling |
| SciPy | KS test (ks_2samp) |
| pandas / NumPy | Data manipulation |
| matplotlib / seaborn | Visualization |

---

## Usage

```bash
# Full pipeline (one command)
python pipeline.py

# Individual stages
python augment_data.py
python simulate_drift.py
python detect_drift.py
```

---

## MLflow Dashboard

```bash
mlflow ui --backend-store-uri mlruns
# Open http://127.0.0.1:5000
```

| Run | Logged |
|---|---|
| baseline_training | accuracy, F1, model artifact |
| drift_detection | KS stats, PSI scores, Evidently report, drift flags |
| retrained_model | new accuracy, F1, delta, registered model |

<img width="1259" height="691" alt="Screenshot 2026-05-09 at 3 11 49 AM" src="https://github.com/user-attachments/assets/b46422f2-9c1a-4e71-9481-f7f8c8700458" />

---

## Author
Built as a college project demonstrating Zero Trust security principles applied to IoT sensor monitoring using a DevOps/MLOps pipeline.
