"""
Step 4 - Scaling / Standardisation
==================================

Applies centring and scaling to *numeric* columns using **train-only**
statistics and writes the results to
    param_hash = self.global_hash
    artifacts/run_<param_hash>/scaling/

Spec compliance highlights
──────────────────────────
• One global hash per run - no per-step hashes (SPEC-§1, §2).  
• Skip-guard is the very first runtime check (SPEC-§14).  
• Inference: reuse → load-from-train → raise (SPEC-§5).  
• Filenames carry **no hashes** because the folder already embeds it
  (SPEC-§25).  
• `self.hashes` removed; `log_registry` called (SPEC-§3, §7).  
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import mlflow

from ml_pipeline.utils import convert_numpy_types, log_registry


# --------------------------------------------------------------------------- #
# Helper                                                                      #
# --------------------------------------------------------------------------- #
def _standardise(
    df: pd.DataFrame,
    numeric_cols: List[str],
    centre: pd.Series,
    scale: pd.Series,
) -> pd.DataFrame:
    """
    Return a copy of *df* where *numeric_cols* are centred/scaled.

    Non-numeric or excluded columns are left untouched.
    """
    df_out = df.copy()
    if numeric_cols:
        df_out[numeric_cols] = (df[numeric_cols] - centre[numeric_cols]) / scale[numeric_cols]
    return df_out


def _load_existing(self, step_dir: str) -> Dict[str, pd.DataFrame]:
    """Load numeric CSVs that are present in *step_dir*."""
    dfs: Dict[str, pd.DataFrame] = {}
    splits = ("train", "val", "test", "excluded") if self.train_mode else ("test",)
    for split in splits:
        fp = os.path.join(step_dir, f"{split}_scaled.csv")
        if os.path.exists(fp):
            dfs[f"{split}_sca"] = pd.read_csv(fp)
    return dfs


# --------------------------------------------------------------------------- #
# Main pipeline step                                                          #
# --------------------------------------------------------------------------- #
def scaling(self) -> None:  # type: ignore[override]
    """
    Standardise numeric variables using train‑only statistics.

    Behaviour
    ---------
    • Training:
        - Compute centre/scale stats.
        - Apply to train/val/test(/excluded).
        - Persist artefacts + manifest.
    • Inference:
        - Reuse artefacts in *current* run if present.
        - Else load from training run.
        - Else raise FileNotFoundError.
    """
    step = "scaling"

    # ------------------------------------------------------------------- #
    # Paths / filenames                                                   #
    # ------------------------------------------------------------------- #
    param_hash = self.global_hash
    run_step_dir = os.path.join("artifacts", f"run_{param_hash}", step)
    run_manifest_dir = os.path.join(run_step_dir, "manifest.json")
    os.makedirs(run_step_dir, exist_ok=True)
    self.dataframes[step] = {}
    # ------------------------------------------------------------------- #
    # 0️⃣  Skip‑guard – artefacts already in *current* run                #
    # ------------------------------------------------------------------- #
    if os.path.exists(run_manifest_dir):
        # Check if the manifest is up to date
        print(f"[{step.upper()}] Skipping — checkpoint exists at {run_step_dir}")
        manifest = json.load(open(run_manifest_dir, "r"))
        self.paths[step] = run_step_dir
        self.dataframes[step].update(_load_existing(self, run_step_dir))
        self.config[step] = manifest.get("config", {})
        self.metadata[step] = manifest.get("metadata", {})
        self.artifacts[step] = manifest.get("artifacts", {})
        self.transformations[step] = manifest.get("transformations", {})
        self.train_paths[step] = manifest.get("train_dir")
        self.train_artifacts[step] = manifest.get("train_artifacts", {})
        self.train_models[step] = manifest.get("train_models", {})
        self.train_transformations[step] = manifest.get("train_transformations", {})
        return

    # ------------------------------------------------------------------- #
    # 1️⃣  Inference → load artefacts from the training run               #
    # ------------------------------------------------------------------- #
    if not self.train_mode:
        train_step_dir = os.path.join("artifacts", f"run_{self.global_train_hash}", step)
        train_manifest_dir = os.path.join(train_step_dir, "manifest.json")
        train_manifest = {}
        if os.path.exists(train_manifest_dir):
            print(f"[{step.upper()}] Reusing training artefacts from {train_step_dir}")
            train_manifest = json.load(open(train_manifest_dir, "r"))
            self.train_paths[step] = train_step_dir
            self.train_artifacts[step] = train_manifest.get("train_artifacts", {})
            self.train_models[step] = train_manifest.get("train_models", {})
            self.train_transformations[step] = train_manifest.get("train_transformations", {})
            self.train_artifacts[step] = train_manifest.get("artifacts", {})
            self.train_transformations[step] = train_manifest.get("transformations", {})

            # self.train_metrics[step] = {}
            # self.train_dataframes[step] = {}
            #return
        # Nothing to reuse → spec mandates failure
        else:
            raise FileNotFoundError(
                f"[{step.upper()}] Expected training artefacts at {train_step_dir} but none found."
            )


    # ------------------------------------------------------------------- #
    # 2️⃣  Training mode – compute and persist                            #
    # ------------------------------------------------------------------- #
    cfg = self.config["init"]
    seed = cfg.get("seed", 42)
    np.random.seed(seed)

    previous_step = "numeric_conversion"
    test_df = self.dataframes[previous_step]["test_num"]
    train_df = self.dataframes[previous_step]["train_num"] if self.train_mode else test_df
    val_df = self.dataframes[previous_step]["val_num"] if self.train_mode else None
    excluded_df = self.dataframes[previous_step].get("excluded_num") if self.train_mode else None

    target_col = cfg["target_col"]  # if self.train_mode else None
    id_col = cfg["id_col"]
    exclude_cols = [c for c in (target_col, id_col) if c in train_df.columns]

    numeric_cols = [
        c
        for c in train_df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    # ------------------------------------------------ centre / scale ----
    if self.train_mode:
        centre_func = np.mean if cfg.get("t1", True) else np.median
        scale_func = np.std if cfg.get("s1", True) else lambda x: np.subtract(*np.percentile(x, [75, 25]))
    else:
        centre_func = np.mean if train_manifest.get("config", {}).get("t1", True) else np.median
        scale_func = np.std if train_manifest.get("config", {}).get("s1", True) else lambda x: np.subtract(*np.percentile(x, [75, 25]))
    # Compute centre/scale stats

    # centre_stats = train_df[numeric_cols].agg(centre_func)
    if self.train_mode:
        centre_func_name = "mean" if cfg.get("t1", True) else "median"
    else:
        centre_func_name = "mean" if train_manifest.get("config", {}).get("t1", True) else "median"

    centre_stats = train_df[numeric_cols].agg(centre_func_name)


    scale_stats = train_df[numeric_cols].agg(scale_func).replace(0, 1.0)

    # ------------------------------------------------ transform ---------
    test_scaled = _standardise(test_df, numeric_cols, centre_stats, scale_stats)
    train_scaled = _standardise(train_df, numeric_cols, centre_stats, scale_stats) if self.train_mode else test_scaled
    val_scaled = _standardise(val_df, numeric_cols, centre_stats, scale_stats) if self.train_mode else None  # test_scaled
    
    excluded_scaled = (
        _standardise(excluded_df, numeric_cols, centre_stats, scale_stats)
        if excluded_df is not None
        else None
    ) if self.train_mode else None
    # If no excluded_df, set to None

    # ------------------------------------------------ persist artefacts -
    os.makedirs(run_step_dir, exist_ok=True)

    out_files: Dict[str, str] = {
        "train_scaled_csv": os.path.join(run_step_dir, "train_scaled.csv") if self.train_mode else None,
        "val_scaled_csv": os.path.join(run_step_dir, "val_scaled.csv") if self.train_mode else None,
        "test_scaled_csv": os.path.join(run_step_dir, "test_scaled.csv"),
        "scaling_stats_json": os.path.join(run_step_dir, "scaling_stats.json"),
    }
    if excluded_scaled is not None and self.train_mode:
        out_files["excluded_scaled_csv"] = os.path.join(run_step_dir, "excluded_scaled.csv")
    
    test_scaled.to_csv(out_files["test_scaled_csv"], index=False)
    if self.train_mode:
        train_scaled.to_csv(out_files["train_scaled_csv"], index=False)
        val_scaled.to_csv(out_files["val_scaled_csv"], index=False)
        if excluded_scaled is not None:
            excluded_scaled.to_csv(out_files["excluded_scaled_csv"], index=False)

    centre_function = ""
    scale_function = ""

    if self.train_mode:
        centre_function = "mean" if cfg.get("t1", True) else "median"
        scale_function = "std" if cfg.get("s1", True) else "iqr"
    else:
        centre_function = "mean" if train_manifest.get("config", {}).get("t1", True) else "median"
        scale_function = "std" if train_manifest.get("config", {}).get("s1", True) else "iqr"


    stats_payload: Dict[str, Any] = {
        "centre_function": centre_function,
        "scale_function": scale_function,
        "centre": convert_numpy_types(centre_stats.to_dict()),
        "scale": convert_numpy_types(scale_stats.to_dict()),
        "numeric_columns": numeric_cols,
        "excluded_columns": exclude_cols,
    }
    
    with open(out_files["scaling_stats_json"], "w") as fh:
        json.dump(stats_payload, fh, indent=2)

    self.paths[step] = run_step_dir
    self.artifacts[step] = out_files
    self.transformations[step] = stats_payload
    self.dataframes[step].update(
        {
            "train_sca": train_scaled if self.train_mode else None,
            "val_sca": val_scaled if self.train_mode else None,
            "test_sca": test_scaled,
        }
    )
    if excluded_scaled is not None:
        self.dataframes[step]["excluded_sca"] = excluded_scaled if self.train_mode else None

    metadata = {
            "step": step,
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "global_hash": param_hash,
            "train_mode": self.train_mode,
            "paths": self.paths.get(step),
            "artifacts": self.artifacts.get(step),
            "models": self.models.get(step),
            "transformations": self.transformations.get(step),
            "config": self.config.get(step),
            "global_train_hash": self.global_train_hash if self.train_mode else None,
            "train_paths": self.train_paths.get(step),
            "train_artifacts": self.train_artifacts.get(step),
            "train_models": self.train_models.get(step),
            "train_transformations": self.train_transformations.get(step),
            "train_metadata": self.train_metadata.get(step) if self.train_mode else None,
            "train_config": self.train_config.get(step),
        }

    self.metadata[step] = metadata

    outputs = {}
    if self.train_mode:
        outputs = {k: os.path.basename(v) for k, v in out_files.items()}
    else:
        outputs = {"test_scaled_csv": os.path.basename(out_files["test_scaled_csv"])}


    manifest = {
        "step": step,
        "param_hash": param_hash,  # still recorded for traceability
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "t1": cfg.get("t1", True),
            "s1": cfg.get("s1", True),
            "seed": seed,
        },
        "output_dir": run_step_dir,
        "outputs": outputs,  # {k: os.path.basename(v) for k, v in out_files.items()},
        "transformations": self.transformations[step],
        "artifacts": self.artifacts[step],
        "paths": self.paths[step],
        "metadata": metadata,

    }

    with open(run_manifest_dir, "w") as fh:
        json.dump(manifest, fh, indent=2)

    # ------------------------------------------------ MLflow / registry -
    if cfg.get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{param_hash}"):
            mlflow.set_tags({"step": step, "global_hash": param_hash})
            mlflow.log_artifacts(run_step_dir, artifact_path=step)

    log_registry(step, param_hash, manifest["config"], run_step_dir)

    # ------------------------------------------------ update pipeline ---

    print(
        f"[{step.upper()}] Done - artefacts at {run_step_dir}  "
        f"(test {len(test_scaled)}"
        + (f", train {len(train_scaled)}, val {len(val_scaled)}" if self.train_mode else "")
        + ")"
    )
