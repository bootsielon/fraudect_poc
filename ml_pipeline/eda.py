# === ml_pipeline/eda.py ==============================================
"""
Step 0 – Exploratory Data Analysis (EDA)

• Uses the global `self.global_hash` set in base.py.
• Artefacts live in  artifacts/run_<hash>/eda/
• `self.train_mode`
      True  → build summary, metadata, class plot, and a raw head sample.
      False → re‑use training artefacts; create `raw_sample.csv` locally
               only if it is missing.
• Heavy computation occurs once per training run (skip‑guard – spec §14).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns

from ml_pipeline.utils import log_registry


# ──────────────────────────────────────────────────────────────────────
def eda(self) -> None:  # type: ignore[override]
    """
    Perform EDA for the current pipeline run.

    Side-effects
    ------------
    • Updates self.dataframes["eda"]["raw"] with the loaded raw dataset.
    • Populates self.paths["eda"] with the directory holding EDA artefacts.
    """
    step = "eda"
    param_hash = self.global_hash
    run_step_dir = os.path.join("artifacts", f"run_{param_hash}", step)
    run_manifest_dir = os.path.join(run_step_dir, "manifest.json")
    raw_csv = os.path.join(run_step_dir, "raw_sample.csv")

    # train_step_dir = os.path.join("artifacts", f"run_{self.global_train_hash}", step)
    # train_manifest_fp = os.path.join(train_step_dir, "manifest.json")

    os.makedirs(run_step_dir, exist_ok=True)
    self.dataframes[step] = {}
    # ------------------------------------------------------------------- #
    # 0️⃣  Skip‑guard – artefacts already in *current* run                #
    # ------------------------------------------------------------------- #
    if os.path.exists(run_manifest_dir):
        print(f"[{step.upper()}] Skipping — checkpoint exists at {run_step_dir}")
        manifest = json.load(open(run_manifest_dir, "r"))
        self.paths[step] = run_step_dir
        self.dataframes[step]["raw"] =  self.load_data()  # pd.read_csv(os.path.join(run_step_dir, manifest["artifacts"]["raw_sample_csv"]))
        #self.dataframes[step].update(_load_existing_numeric(self, run_step_dir))
        self.artifacts[step] = manifest.get("artifacts", {})
        self.transformations[step] = manifest.get("transformations", {})
        self.config[step] = manifest.get("config", {})
        self.metadata[step] = manifest.get("metadata", {}) 
        # self.train_paths[step] = manifest.get("train_dir")
        self.train_artifacts[step] = manifest.get("train_artifacts", {})
        self.train_models[step] = manifest.get("train_models", {})
        self.train_transformations[step] = manifest.get("train_transformations", {})
        log_registry(step, self.global_hash, manifest["config"], run_step_dir)
        return


    # ──────────────────────────────────────────────────────────────────
    # Skip‑guard (spec §14)
    # ──────────────────────────────────────────────────────────────────


    # ──────────────────────────────────────────────────────────────────
    # Training mode – full EDA
    # ──────────────────────────────────────────────────────────────────
    os.makedirs(run_step_dir, exist_ok=True)

    df = self.load_data()
    self.dataframes[step] = {}
    self.dataframes[step]["raw"] = df

    df.head(500).to_csv(raw_csv, index=False)

    summary_csv = os.path.join(run_step_dir, "summary_stats.csv")
    meta_json = os.path.join(run_step_dir, "column_metadata.json")
    class_png = os.path.join(run_step_dir, "class_distribution.png")

    df.describe(include="all").to_csv(summary_csv)

    metadata = {
        col: {
            "dtype": str(df[col].dtype),
            "cardinality": int(df[col].nunique(dropna=False)),
            "nulls": int(df[col].isnull().sum()),
        }
        for col in df.columns
    }
    with open(meta_json, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    target_col = self.config["init"].get("target_col")
    if target_col and target_col in df.columns:
        plt.figure()
        sns.countplot(x=target_col, data=df)
        plt.title("Class Distribution")
        plt.savefig(class_png, bbox_inches="tight")
        plt.close()
        class_plot = os.path.basename(class_png)
    else:
        print(f"[{step.upper()}] target_col '{target_col}' not found – skipping class plot.")
        class_plot = None

    manifest = {
        "step": step,
        "param_hash": self.global_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "target_col": target_col,
            "n_rows": len(df),
            "n_cols": df.shape[1],
        },
        "output_dir": run_step_dir,
        "outputs": {
            "raw_sample_csv": os.path.basename(raw_csv),
            "summary_stats_csv": os.path.basename(summary_csv),
            "column_metadata_json": os.path.basename(meta_json),
            "class_distribution_plot": class_plot,
        },
    }

    with open(run_manifest_dir, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    if self.config["init"].get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{self.global_hash}"):
            mlflow.set_tags({"step": step, "param_hash": self.global_hash})
            mlflow.log_artifacts(run_step_dir, artifact_path=step)

    self.paths[step] = run_step_dir
    log_registry(step, self.global_hash, manifest["config"], run_step_dir)

    print(f"[{step.upper()}] Done – artefacts at {run_step_dir}")

    """    if not self.train_mode:
        if not os.path.exists(train_manifest_fp):
            raise AssertionError(
                f"[{step.upper()}] Missing training artefacts at {train_step_dir}"
            )
        os.makedirs(run_step_dir, exist_ok=True)
        train_raw_csv = os.path.join(train_step_dir, "raw_sample.csv")
        if os.path.exists(train_raw_csv) and not os.path.exists(raw_csv):
            import shutil
            shutil.copy2(train_raw_csv, raw_csv)

        with open(train_manifest_fp, "r", encoding="utf-8") as fp:
            mdata = json.load(fp)
        self.paths[step] = train_step_dir
        log_registry(step, self.global_hash, mdata["config"], train_step_dir)
        print(f"[{step.upper()}] Re‑used training artefacts at {train_step_dir}")
        return
"""