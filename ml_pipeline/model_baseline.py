"""
Step 5 - Baseline Model
=======================

Trains an XGBoost baseline classifier (or re-uses an existing one) and
stores artifacts in

    artifacts/run_<self.global_hash>/model_baseline/

Spec compliance highlights
──────────────────────────
• No per-step hashes; folder already embeds the run-hash (SPEC §1-§2, §25).  
• Skip-guard is the first runtime check (SPEC §14).  
• Inference logic: reuse → load-from-train → raise (SPEC §5).  
• `self.hashes` removed; `log_registry` invoked (SPEC §3, §7).  
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    auc,
)

from ml_pipeline.utils import log_registry, save_plot_as_artifact


# --------------------------------------------------------------------------- #
# Helper                                                                      #
# --------------------------------------------------------------------------- #
def _load_existing(self, step_dir: str) -> Dict[str, pd.DataFrame]:
    """Load numeric CSVs that are present in *step_dir*."""
    dfs: Dict[str, pd.DataFrame] = {}
    splits = ("train", "val", "test", "excluded") if self.train_mode else ("test",)
    for split in splits:
        fp = os.path.join(step_dir, f"{split}_base.csv")
        if os.path.exists(fp):
            dfs[f"{split}_base"] = pd.read_csv(fp)
    return dfs


def _load_existing_model(step_dir: str) -> Dict[str, Any]:
    """Load model + metrics previously saved in *step_dir*."""
    artifacts = {}
    model_fp = os.path.join(step_dir, "model.json")
    if os.path.exists(model_fp):
        model = XGBClassifier()
        model.load_model(model_fp)
        artifacts["model"] = model

    metrics_fp = os.path.join(step_dir, "metrics.json")
    if os.path.exists(metrics_fp):
        with open(metrics_fp) as fh:
            artifacts["metrics"] = json.load(fh)

    features_fp = os.path.join(step_dir, "feature_names.json")
    if os.path.exists(features_fp):
        with open(features_fp) as fh:
            meta = json.load(fh)
            if "model" in artifacts:
                artifacts["model"].feature_names = meta["feature_names"]
            artifacts["feature_names"] = meta["feature_names"]

    return artifacts


# --------------------------------------------------------------------------- #
# Main pipeline step                                                          #
# --------------------------------------------------------------------------- #
def model_baseline(self) -> None:  # type: ignore[override]
    """
    Train or load the baseline XGBoost classifier.
    """
    step = "model_baseline"
    run_step_dir = os.path.join("artifacts", f"run_{self.global_hash}", step)
    run_manifest_dir = os.path.join(run_step_dir, "manifest.json")
    self.models[step] = {}
    self.metrics[step] = {}
    self.train_models[step] = {}
    # self.train_metrics[step] = {}
    # ------------------------------------------------------------------- #
    # 0️⃣  Skip-guard - artifacts already in *current* run                #
    # ------------------------------------------------------------------- #
    if os.path.exists(run_manifest_dir):
        print(f"[{step.upper()}] Skipping — checkpoint exists at {run_step_dir}")
        manifest = json.load(open(run_manifest_dir, "r"))
        self.artifacts[step] = _load_existing_model(run_step_dir)
        self.models[step]["baseline"] = self.artifacts[step]["model"]
        self.metrics[step]["baseline"] = self.artifacts[step]["metrics"]
        self.paths[step] = run_step_dir
        self.artifacts[step] = {
            "model_file": os.path.join(run_step_dir, "model.json"),
            "metrics_file": os.path.join(run_step_dir, "metrics.json"),
            "feature_names_file": os.path.join(run_step_dir, "feature_names.json"),
        }
        # self.dataframes[step].update(_load_existing_model(self, run_step_dir))
        self.config[step] = manifest.get("config", {})
        self.metadata[step] = manifest.get("metadata", {})
        # self.artifacts[step] = manifest.get("artifacts", {})
        self.transformations[step] = manifest.get("transformations", {})
        self.train_paths[step] = manifest.get("train_dir")
        self.train_artifacts[step] = manifest.get("train_artifacts", {})
        self.train_models[step] = manifest.get("train_models", {})
        self.train_transformations[step] = manifest.get("train_transformations", {})

        return

    # ------------------------------------------------------------------- #
    # 1️⃣  Inference → load artifacts from the training run               #
    # ------------------------------------------------------------------- #
    if not self.train_mode:
        train_step_dir = os.path.join("artifacts", f"run_{self.global_train_hash}", step)
        train_manifest_dir = os.path.join(train_step_dir, "manifest.json")
        if os.path.exists(train_manifest_dir):
            print(f"[{step.upper()}] Reusing training artifacts from {train_step_dir}")
            train_manifest = json.load(open(train_manifest_dir))
            train_artifacts = _load_existing_model(train_step_dir)
            self.train_artifacts[step] = train_artifacts
            self.train_models[step]["baseline"] = self.train_artifacts[step]["model"]
            # self.train_metrics[step]["baseline"] = self.train_artifacts[step]["metrics"]
            self.train_paths[step] = train_step_dir
            self.train_artifacts[step] = {
                "model_file": os.path.join(train_step_dir, "model.json"),
                "metrics_file": os.path.join(train_step_dir, "metrics.json"),
                "feature_names_file": os.path.join(train_step_dir, "feature_names.json"),
            }
            return
        raise FileNotFoundError(
            f"[{step.upper()}] Expected training artifacts at {train_step_dir} but none found."
        )

    # ------------------------------------------------------------------- #
    # 2️⃣  Training mode - compute and persist                            #
    # ------------------------------------------------------------------- #
    cfg = self.config["init"] if self.train_mode else train_manifest.get("config", {})
    if cfg is None:
        raise ValueError(f"[{step.upper()}] No config found in {train_step_dir}")
    
    seed = cfg.get("random_state", 42)
    np.random.seed(seed)

    targ = cfg["target_col"]

    # -------------------- fetch scaled data generated earlier ----------
    previous_step = "scaling"

    train_df = self.dataframes[previous_step]["train_sca"] if self.train_mode else self.dataframes[step]["test_sca"]
    val_df = self.dataframes[previous_step]["val_sca"] if self.train_mode else None
    test_df = self.dataframes[previous_step]["test_sca"]
    excl_df = self.dataframes[previous_step].get("excluded_sca") if self.train_mode else None

    # drop potential non-numeric leftovers
    drop_obj = [c for c in train_df.columns if train_df[c].dtype == "object"]
    if drop_obj:
        test_df = test_df.drop(columns=drop_obj)
        if self.train_mode:
            print(f"[{step.upper()}] Dropping object columns: {drop_obj}")
            train_df = train_df.drop(columns=drop_obj)
            val_df = val_df.drop(columns=drop_obj)
            
            if excl_df is not None:
                excl_df = excl_df.drop(columns=drop_obj)

    feature_cols = [c for c in train_df.columns if c != targ]

    X_test, y_test = test_df[feature_cols], test_df[targ]

    if self.train_mode:
        X_train, y_train = train_df[feature_cols], train_df[targ]
        X_val, y_val = val_df[feature_cols], val_df[targ]
        if excl_df is not None:
            X_excl, y_excl = excl_df[feature_cols], excl_df[targ]

    # ------------------------------- train model -----------------------
    model = None
    if self.train_mode:
        model = XGBClassifier(
            n_estimators=cfg.get("n_estimators", 400),
            max_depth=cfg.get("max_depth", 4),
            learning_rate=cfg.get("learning_rate", 0.01),
            subsample=cfg.get("subsample", 0.8),
            colsample_bytree=cfg.get("colsample_bytree", 0.8),
            random_state=seed,
            use_label_encoder=False,
            eval_metric="auc",
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        model.feature_names = feature_cols  # keep explicit list
    else:
        # load model from training run
        if "model" in self.train_artifacts[step]:
            model = self.train_artifacts[step]["model"]
            model.feature_names = feature_cols

    # ------------------------------- metrics ---------------------------
    def _roc(y_true, y_prob):
        return roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5

    metrics = {
        "test": {
            "accuracy": accuracy_score(y_test, model.predict(X_test)),
            "roc_auc": _roc(y_test, model.predict_proba(X_test)[:, 1]),
        }
    }

    if self.train_mode:
        metrics.update({
            # metrics for train/val sets
            "train": {
                "accuracy": accuracy_score(y_train, model.predict(X_train)),
                "roc_auc": _roc(y_train, model.predict_proba(X_train)[:, 1]),
            },
            "val": {
                "accuracy": accuracy_score(y_val, model.predict(X_val)),
                "roc_auc": _roc(y_val, model.predict_proba(X_val)[:, 1]),
            },
        })

        # metrics for excluded set
        if excl_df is not None:
            metrics["excluded"] = {
                "accuracy": accuracy_score(y_excl, model.predict(X_excl)),
                "roc_auc": _roc(y_excl, model.predict_proba(X_excl)[:, 1]),
            }

    # ------------------------------- persist artifacts -----------------
    os.makedirs(run_step_dir, exist_ok=True)
    model_fp = os.path.join(run_step_dir, "model.json")
    metrics_fp = os.path.join(run_step_dir, "metrics.json")
    features_fp = os.path.join(run_step_dir, "feature_names.json")

    model.save_model(model_fp)
    with open(metrics_fp, "w") as fh:
        json.dump(metrics, fh, indent=2)
    with open(features_fp, "w") as fh:
        json.dump({"feature_names": feature_cols}, fh, indent=2)

    # feature importance plot
    fig, ax = plt.subplots(figsize=(10, 6))
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    ax.bar(range(len(importances)), importances[order])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(np.array(feature_cols)[order], rotation=45, ha="right")
    ax.set_title("Feature Importances")
    fig.tight_layout()
    fi_png = os.path.join(run_step_dir, "feature_importance.png")
    save_plot_as_artifact(fig, fi_png, {}, "dummy")  # dict not used further

    # ROC curve (test set)
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.2f}")
    ax_roc.plot([0, 1], [0, 1], "--")
    ax_roc.set_xlabel("FPR"), ax_roc.set_ylabel("TPR")
    ax_roc.set_title("ROC - Test")
    ax_roc.legend()
    roc_png = os.path.join(run_step_dir, "roc_curve.png")
    save_plot_as_artifact(fig_roc, roc_png, {}, "dummy2")

    manifest = {
        "step": step,
        "param_hash": self.global_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            k: cfg.get(k)
            for k in (
                "n_estimators",
                "max_depth",
                "learning_rate",
                "subsample",
                "colsample_bytree",
                "random_state",
            )
        },
        "output_dir": run_step_dir,
        "artifacts": {
            "model_file": os.path.basename(model_fp),
            "metrics_file": os.path.basename(metrics_fp),
            "feature_names_file": os.path.basename(features_fp),
            "feature_importance_png": os.path.basename(fi_png),
            "roc_curve_png": os.path.basename(roc_png),
        },
        "metrics": metrics,
    }
    with open(run_manifest_dir, "w") as fh:
        json.dump(manifest, fh, indent=2)

    # ------------------------------- MLflow / registry -----------------
    if cfg.get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{self.global_hash}"):
            mlflow.set_tags({"step": step, "global_hash": self.global_hash})
            mlflow.log_artifacts(run_step_dir, artifact_path=step)
            flat_metrics = {
                f"{ds}_{m}": v
                for ds, mset in metrics.items()
                for m, v in mset.items()
                if v is not None
            }
            mlflow.log_metrics(flat_metrics)

    log_registry(step, self.global_hash, manifest["config"], run_step_dir)

    # ------------------------------- update pipeline -------------------
    
    self.models[step]["baseline"] = model
    self.metrics[step]["baseline"] = metrics
    self.paths[step] = run_step_dir
    self.artifacts[step] = manifest["artifacts"]

    print(
        f"[{step.upper()}] Done - artifacts at {run_step_dir}  "
        f"(test AUC {metrics['test']['roc_auc']:.3f})"
    )