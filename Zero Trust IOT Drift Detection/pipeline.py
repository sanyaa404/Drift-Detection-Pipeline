"""
pipeline.py
-----------
Zero Trust IoT Drift Detection Pipeline — Master Orchestrator

Run this single file to execute the full pipeline:
  python pipeline.py

Stages:
  1. Augment data       (augment_data.py)
  2. Simulate drift     (simulate_drift.py)
  3. Detect & retrain   (detect_drift.py)
  4. Launch MLflow UI   (optional)

Usage:
  python pipeline.py              # full run
  python pipeline.py --no-ui      # skip launching MLflow UI
  python pipeline.py --skip-aug   # skip augmentation (if already done)
"""

import argparse
import subprocess
import sys
import os
import time

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║     Zero Trust IoT System — DevOps Drift Detection           ║
║     MLflow Pipeline  |  Sensor Data Monitoring               ║
╚══════════════════════════════════════════════════════════════╝
"""

STAGES = [
    ("augment_data.py",    "Stage 1: Data Augmentation    (79 → ~1000 rows)"),
    ("simulate_drift.py",  "Stage 2: Drift Simulation     (IoT sensor corruption)"),
    ("detect_drift.py",    "Stage 3: Drift Detection +    (Auto-retrain if needed)"),
]


def run_stage(script: str, label: str) -> bool:
    print(f"\n{'─'*62}")
    print(f"  ▶  {label}")
    print(f"{'─'*62}")
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"\n❌ Stage failed: {script} (exit code {result.returncode})")
        return False
    return True


def check_requirements():
    """Verify all required packages are installed."""
    # Map: display name → actual importable module name
    required = {
        "mlflow":        "mlflow",
        "scikit-learn":  "sklearn",
        "imbalanced-learn": "imblearn",
        "pandas":        "pandas",
        "numpy":         "numpy",
        "evidently":     "evidently",
        "scipy":         "scipy",
        "matplotlib":    "matplotlib",
    }
    missing = []
    for display, module in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(display)
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print(f"   Run: pip install -r requirements.txt")
        sys.exit(1)


def setup_mlflow():
    """Set MLflow tracking URI to local mlruns folder."""
    os.makedirs("mlruns", exist_ok=True)
    os.environ["MLFLOW_TRACKING_URI"] = "mlruns"


def main():
    parser = argparse.ArgumentParser(description="Zero Trust IoT Drift Detection Pipeline")
    parser.add_argument("--no-ui",    action="store_true", help="Do not launch MLflow UI after run")
    parser.add_argument("--skip-aug", action="store_true", help="Skip data augmentation stage")
    args = parser.parse_args()

    print(BANNER)
    print("  Checking requirements...")
    check_requirements()
    setup_mlflow()

    stages_to_run = STAGES if not args.skip_aug else STAGES[1:]

    # Verify input file exists
    if not os.path.exists("data/image_quality_features.csv"):
        print("\n❌ ERROR: data/image_quality_features.csv not found!")
        print("   Place your dataset in the data/ folder and re-run.")
        sys.exit(1)

    start = time.time()
    for script, label in stages_to_run:
        success = run_stage(script, label)
        if not success:
            print("\nPipeline aborted due to stage failure.")
            sys.exit(1)

    elapsed = time.time() - start
    print(f"\n{'═'*62}")
    print(f"  ✅  All stages complete in {elapsed:.1f}s")
    print(f"{'═'*62}")
    print(f"""
  📊  Results Summary:
      • MLflow experiments  → mlruns/
      • Evidently report    → reports/evidently_drift_report.html
      • Augmented data      → data/reference.csv
      • Drifted data        → data/production_drifted.csv

  🚀  To view MLflow dashboard:
        mlflow ui --backend-store-uri mlruns
      Then open: http://127.0.0.1:5000

  📂  Folder structure:
        zero_trust_iot_drift/
        ├── data/
        │   ├── image_quality_features.csv  (original)
        │   ├── reference.csv               (augmented baseline)
        │   ├── production_clean.csv        (clean production)
        │   └── production_drifted.csv      (drifted production)
        ├── mlruns/                         (MLflow tracking)
        ├── reports/
        │   └── evidently_drift_report.html
        ├── augment_data.py
        ├── simulate_drift.py
        ├── detect_drift.py
        ├── pipeline.py
        └── requirements.txt
""")

    if not args.no_ui:
        print("  Launching MLflow UI... (Ctrl+C to stop)")
        try:
            subprocess.run(["mlflow", "ui", "--backend-store-uri", "mlruns"])
        except KeyboardInterrupt:
            print("\n  MLflow UI stopped.")


if __name__ == "__main__":
    main()
