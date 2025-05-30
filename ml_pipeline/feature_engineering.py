# === ml_pipeline/feature_engineering.py ===============================
"""
Step 1 - Feature Engineering

• Adds derived timestamp features, computes account age deltas.
• Drops zero-variance or constant features (training only).
• artifacts stored in: artifacts/run_<hash>/feature_engineering/
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone
import pandas as pd
import mlflow

from ml_pipeline.utils import make_param_hash, log_registry, convert_numpy_types


def feature_engineering(self) -> None:  # type: ignore[override]
    """
    Perform feature engineering on the raw dataset.

    Adds timestamp-derived features and drops constant columns in training.
    Stores artifacts in self.paths, self.artifacts, and self.dataframes[step].
    """
    step = "feature_engineering"
    train_mode = self.train_mode
    param_hash = self.global_hash
    train_hash = self.global_train_hash

    run_step_dir = os.path.join("artifacts", f"run_{param_hash}", step)
    run_manifest_dir = os.path.join(run_step_dir, "manifest.json")
    train_step_dir = os.path.join("artifacts", f"run_{train_hash}", step)
    train_manifest_dir = os.path.join(train_step_dir, "manifest.json")

    os.makedirs(run_step_dir, exist_ok=True)
    self.dataframes[step] = {}
    # ------------------------------------------------------------------- #
    # 0️⃣  Skip‑guard – artefacts already in *current* run                #
    # ------------------------------------------------------------------- #
    if os.path.exists(run_manifest_dir):
        print(f"[{step.upper()}] Skipping — checkpoint exists at {run_step_dir}")
        manifest = json.load(open(run_manifest_dir, "r"))
        self.paths[step] = run_step_dir
        self.dataframes[step]["feature_engineered"] = pd.read_csv(os.path.join(run_step_dir, manifest["artifacts"]["engineered_csv"]))
        self.artifacts[step] = manifest.get("artifacts", {})
        self.transformations[step] = manifest.get("transformations", {})
        self.config[step] = manifest.get("config", {})
        self.metadata[step] = manifest.get("metadata", {}) 
        self.train_paths[step] = manifest.get("train_dir")
        self.train_artifacts[step] = manifest.get("train_artifacts", {})
        self.train_models[step] = manifest.get("train_models", {})
        self.train_transformations[step] = manifest.get("train_transformations", {})
        log_registry(step, param_hash, manifest["config"], run_step_dir)
        print(f"[{step.upper()}] Skipped - artifacts already exist at {run_step_dir}")
        return


    # ──────────────────────────────────────────────────────────────────
    # Training mode - full feature engineering
    # ──────────────────────────────────────────────────────────────────
    os.makedirs(run_step_dir, exist_ok=True)
    self.dataframes[step] = {}
    df = self.dataframes["eda"]["raw"]
    df_fe = df.copy()

    # Timestamp-based features
    if "timestamp" in df_fe.columns:
        ts = pd.to_datetime(df_fe["timestamp"], errors="coerce")
        df_fe["transaction_hour"] = ts.dt.hour
        df_fe["transaction_dayofweek"] = ts.dt.dayofweek

    for col in ["account_creation_date_client", "account_creation_date_merchant"]:
        if col in df_fe.columns:
            dt = pd.to_datetime(df_fe[col], errors="coerce")
            df_fe[f"{col}_year"] = dt.dt.year
            df_fe[f"{col}_month"] = dt.dt.month
            df_fe[f"{col}_day"] = dt.dt.day
            df_fe[f"{col}_hour"] = dt.dt.hour
            df_fe[f"{col}_dayofweek"] = dt.dt.dayofweek
            if "timestamp" in df_fe.columns:
                df_fe[f"{col}_age_days"] = (pd.to_datetime(df_fe["timestamp"]) - dt).dt.days
                df_fe[f"{col}_age_years"] = df_fe[f"{col}_age_days"] / 365.25

    dropped = [col for col in df_fe.columns if df_fe[col].nunique(dropna=False) <= 1]
    df_fe.drop(columns=dropped, inplace=True)

    output_csv = os.path.join(run_step_dir, f"{step}_{param_hash}.csv")
    drop_log = os.path.join(run_step_dir, f"dropped_features_{param_hash}.json")

    df_fe.to_csv(output_csv, index=False)
    with open(drop_log, "w", encoding="utf-8") as f:
        json.dump({"dropped_features": dropped}, f, indent=2)

    config_summary = {
        "target_col": self.config["init"].get("target_col"),
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "initial_columns": list(df.columns),
        "final_columns": list(df_fe.columns),
        "n_dropped": len(dropped),
        "train_mode": train_mode,
    }
    self.paths[step] = run_step_dir
    self.config[step] = config_summary
    self.metadata[step] = {
        "dropped_features": dropped,
        "dropped_features_count": len(dropped),
        "initial_shape": df.shape,
        "final_shape": df_fe.shape,
        "initial_columns": list(df.columns),
        "final_columns": list(df_fe.columns),
    }
    artifacts = {
        "engineered_csv": os.path.basename(output_csv),
        "dropped_features": dropped,
        "dropped_features_count": len(dropped),
        "dropped_features_log": os.path.basename(drop_log),
        "retained_features": list(df_fe.columns),
        "initial_features": list(df.columns),
        "initial_shape": df.shape,
        "final_shape": df_fe.shape,
        "final_features": list(df_fe.columns),
        "new_features": list(set(df_fe.columns) - set(df.columns)),
    }
    manifest = {
        "step": step,
        "param_hash": param_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": convert_numpy_types(config_summary),
        "output_dir": run_step_dir,
        "artifacts": artifacts,
        "metadata": self.metadata[step],
        "paths": self.paths[step],
    }

    with open(run_manifest_dir, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if self.config["init"].get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{param_hash}"):
            mlflow.set_tags({"step": step, "param_hash": param_hash})
            mlflow.log_artifacts(run_step_dir, artifact_path=step)

    self.dataframes[step]["feature_engineered"] = df_fe
    
    log_registry(step, param_hash, config_summary, run_step_dir)

    print(f"[{step.upper()}] Done - artefacts at {run_step_dir}")

    """
        if not train_mode:
        if not os.path.exists(train_manifest_dir):
            raise AssertionError(f"[{step.upper()}] Missing training artefacts at {train_step_dir}")
        os.makedirs(run_step_dir, exist_ok=True)

        with open(train_manifest_dir, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        drop_list = manifest["artifacts"]["dropped_features"]
        drop_log = manifest["artifacts"]["dropped_features_log"]
        train_csv = manifest["artifacts"]["engineered_csv"]

        df_fe = self.dataframes[step]["raw"].copy()
        for col in drop_list:
            if col in df_fe.columns:
                df_fe.drop(columns=col, inplace=True)

        output_csv = os.path.join(train_step_dir, f"{step}_{param_hash}.csv")
        df_fe.to_csv(output_csv, index=False)

        copied_manifest = {
            "step": step,
            "param_hash": param_hash,
            "timestamp": datetime.utcnow().isoformat(),
            "config": manifest["config"],
            "output_dir": run_step_dir,
            "artifacts": {
                "engineered_csv": os.path.basename(output_csv),
                "dropped_features": drop_list,
                "dropped_features_count": len(drop_list),
                "dropped_features_log": os.path.basename(drop_log),
                "retained_features": list(df_fe.columns),
            },
        }

        with open(manifest_fp, "w", encoding="utf-8") as f:
            json.dump(copied_manifest, f, indent=2)

        if self.config["init"].get("use_mlflow", False):
            with mlflow.start_run(run_name=f"{step}_{param_hash}"):
                mlflow.set_tags({"step": step, "param_hash": param_hash})
                mlflow.log_artifacts(run_step_dir, artifact_path=step)

        self.dataframes[step]["feature_engineered"] = df_fe
        self.paths[step] = run_step_dir
        log_registry(step, param_hash, manifest["config"], run_step_dir)
        print(f"[{step.upper()}] Re‑used training artefacts at {train_step_dir}")
        return

    
    """