"""
 This module contains functions for partitioning a dataset into training, validation, and test sets.
 
 It includes functionality for stratified downsampling, random downsampling, and saving the resulting
 datasets to files. The module also supports logging with MLflow and creating a manifest file for
 tracking purposes.

 It is designed to be used as part of a larger machine learning pipeline.
"""

import os
import json
from datetime import datetime, timezone
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from ml_pipeline.utils import log_registry, DEFAULT_TEST_HASH  # already lives in utils.py


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def identify_stratification_columns(df, target, use_stratification, cardinality_threshold):
    """Identify columns to use for stratification."""
    stratify_cols = [target]
    if use_stratification:
        for col in df.columns:
            if col != target and df[col].nunique() <= cardinality_threshold:
                stratify_cols.append(col)
    return stratify_cols


def identify_classes(df, target, step):
    """Identify minority and majority classes in the dataset."""
    class_counts = df[target].value_counts()
    if len(class_counts) != 2:
        print(f"[{step.upper()}] Warning: Expected binary classification but found {len(class_counts)} classes")
    
    minority_class = class_counts.index[-1]  # Least frequent class
    majority_class = class_counts.index[0]   # Most frequent class
    
    df_minority = df[df[target] == minority_class]
    df_majority = df[df[target] == majority_class]
    
    return minority_class, majority_class, df_minority, df_majority, class_counts


def perform_stratified_downsampling(df_majority, df_minority, strat_cols_for_downsampling, seed):
    """Downsample majority class with stratification."""
    majority_strat_key = df_majority[strat_cols_for_downsampling].astype(str).agg("_".join, axis=1)
    minority_strat_key = df_minority[strat_cols_for_downsampling].astype(str).agg("_".join, axis=1)
    
    minority_dist = minority_strat_key.value_counts(normalize=True)
    
    df_majority_downsampled = pd.DataFrame(columns=df_majority.columns)
    df_excluded = pd.DataFrame(columns=df_majority.columns)
    
    for stratum, proportion in minority_dist.items():
        stratum_df = df_majority[majority_strat_key == stratum]
        if not stratum_df.empty:
            n_samples = max(1, int(proportion * len(df_minority)))
            if len(stratum_df) > n_samples:
                sampled = stratum_df.sample(n=n_samples, random_state=seed)
                df_majority_downsampled = pd.concat([df_majority_downsampled, sampled])
                df_excluded = pd.concat([df_excluded, stratum_df.drop(sampled.index)])
            else:
                df_majority_downsampled = pd.concat([df_majority_downsampled, stratum_df])
    
    return df_majority_downsampled, df_excluded


def balance_samples(df_majority_downsampled, df_minority, df_majority, df_excluded, seed):
    """Balance the number of samples between majority and minority classes."""
    if len(df_majority_downsampled) > len(df_minority):
        excess = len(df_majority_downsampled) - len(df_minority)
        excess_indices = df_majority_downsampled.sample(n=excess, random_state=seed).index
        df_excluded = pd.concat([df_excluded, df_majority_downsampled.loc[excess_indices]])
        df_majority_downsampled = df_majority_downsampled.drop(excess_indices)
    elif len(df_majority_downsampled) < len(df_minority):
        shortage = len(df_minority) - len(df_majority_downsampled)
        if len(df_excluded) >= shortage:
            additional = df_excluded.sample(n=shortage, random_state=seed)
            df_majority_downsampled = pd.concat([df_majority_downsampled, additional])
            df_excluded = df_excluded.drop(additional.index)
        else:
            additional_needed = shortage - len(df_excluded)
            df_majority_downsampled = pd.concat([df_majority_downsampled, df_excluded])
            df_excluded = pd.DataFrame(columns=df_majority.columns)
            additional = df_majority.sample(n=additional_needed, random_state=seed)
            df_majority_downsampled = pd.concat([df_majority_downsampled, additional])
            df_excluded = df_majority.drop(df_majority_downsampled.index)
    
    return df_majority_downsampled, df_excluded


def random_downsample(df_majority, df_minority, seed):
    """Simple random downsampling of majority class."""
    df_majority_downsampled = df_majority.sample(n=len(df_minority), random_state=seed)
    df_excluded = df_majority.drop(df_majority_downsampled.index)
    return df_majority_downsampled, df_excluded


# --------------------------------------------------------------------------- #
# Train / Val / Test split helper                                             #
# --------------------------------------------------------------------------- #
def perform_data_splits(
    df: pd.DataFrame,
    stratify_cols: list[str],
    test_size: float | int,
    val_ratio: float,
    seed: int,
):
    """
    Split *df* into train, validation and test sets while preserving the joint
    distribution of all columns in *stratify_cols*.

    Safety guards:
    1. Ensure the **test** split has at least one sample per stratum.
    2. Ensure the **validation** split has at least one sample per stratum.

    If either constraint would be violated, the corresponding split fraction is
    automatically increased to the minimum feasible value and a message is
    printed.  The adjustment keeps the pipeline running without user
    intervention.
    """
    # Build a single stratification key
    stratify_key = df[stratify_cols].astype(str).agg("_".join, axis=1)
    n_classes = stratify_key.nunique()

    # ───────────────────────────────── 1️⃣  Test split guard
    if isinstance(test_size, float):
        test_rows_req = max(int(round(len(df) * test_size)), 1)
    else:
        test_rows_req = test_size
    if test_rows_req < n_classes:
        test_rows_req = n_classes
        test_size = test_rows_req / len(df)
        print(
            f"[PARTITIONING] Auto‑bumped test_size to {test_size:.3f} "
            f"so that each of the {n_classes} strata appears at least once."
        )

    # First split: Train+Val vs Test
    df_train_val, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=stratify_key,
        random_state=seed,
    )

    # ───────────────────────────────── 2️⃣  Validation split guard
    stratify_key_tv = df_train_val[stratify_cols].astype(str).agg("_".join, axis=1)
    n_classes_tv = stratify_key_tv.nunique()

    if isinstance(val_ratio, float):
        val_rows_req = max(int(round(len(df_train_val) * val_ratio)), 1)
    else:  # val_ratio should always be float, but guard just in case
        val_rows_req = val_ratio
    if val_rows_req < n_classes_tv:
        val_rows_req = n_classes_tv
        val_ratio = val_rows_req / len(df_train_val)
        print(
            f"[PARTITIONING] Auto‑bumped val_ratio to {val_ratio:.3f} "
            f"so that each of the {n_classes_tv} strata appears at least once."
        )

    # Second split: Train vs Val
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_ratio,
        stratify=stratify_key_tv,
        random_state=seed,
    )

    return df_train, df_val, df_test, stratify_key



def save_outputs(df_train, df_val, df_test, df_excluded,
                  id_col, stratify_keys, step_dir, train_mode):
    """Save all partition outputs to files (hash omitted from filenames per SPEC§25)."""
    df_test.to_csv(os.path.join(step_dir, "test.csv"), index=False)
    if train_mode:
        df_train.to_csv(os.path.join(step_dir, "train.csv"), index=False)
        df_val.to_csv(os.path.join(step_dir, "val.csv"), index=False)
        if df_excluded is not None:
            df_excluded.to_csv(os.path.join(step_dir, "excluded_majority.csv"), index=False)

    id_map = {
        "test": df_test[id_col].tolist(),
    }
    if train_mode:
        id_map.update({
            "train": df_train[id_col].tolist(),
            "val": df_val[id_col].tolist(),
            "excluded_majority": df_excluded[id_col].tolist()
        })
    
        with open(os.path.join(step_dir, "id_partition_map.json"), "w") as f:
            json.dump(id_map, f, indent=2)

        stratify_keys.to_csv(os.path.join(step_dir, "stratify_keys.csv"), index=False)  # stratify_keys.to_json(os.path.join(step_dir, "stratify_keys.json"), orient="records", lines=True)  # Save the stratification keys as a JSON file


def create_manifest(step, param_hash, config, step_dir):
    """Create and save the manifest file."""    
    outputs = {"test_csv": "test.csv",}

    if config["train_mode"]:
        outputs.update({
            "train_csv": "train.csv",
            "val_csv": "val.csv",
            "excluded_majority_csv": "excluded_majority.csv",
            "id_partition_map_json": "id_partition_map.json",
            "stratify_keys_csv": "stratify_keys.csv"
        })

    manifest = {
        "step": step,
        "param_hash": param_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "output_dir": step_dir,
        "training_mode": config["train_mode"],
        "outputs": outputs
    }
    with open(os.path.join(step_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def load_checkpoint(step_dir, train_mode):
    """Load data from existing checkpoint (filenames hash‑free)."""
    dataframes = {}

    mode_datasets = ["test"]
    if train_mode:
        mode_datasets += ["train", "val", "excluded_majority"]

    for split in mode_datasets:
        path = os.path.join(step_dir, f"{split}.csv")
        if os.path.exists(path):
            dataframes[split] = pd.read_csv(path)
    return dataframes


# --------------------------------------------------------------------------- #
# Main pipeline step                                                          #
# --------------------------------------------------------------------------- #
def partitioning(self) -> None:
    """
    Partition feature‑engineered data into train, validation, and test sets.
    """
    step = "partitioning"
    config_init = self.config["init"]
    param_hash = self.global_hash
    run_step_dir   = os.path.join("artifacts", f"run_{param_hash}", step)
    run_manifest_dir   = os.path.join(run_step_dir, "manifest.json")
    os.makedirs(run_step_dir, exist_ok=True)
    self.dataframes[step] = {}
    self.paths[step] = run_step_dir
    # ----------------------------------------------------------------
    # 0  Skip‑guard – artefacts already in *current* run folder
    # ----------------------------------------------------------------
    if os.path.exists(run_manifest_dir):
        print(f"[{step.upper()}] Skipping — checkpoint exists at {run_step_dir}")
        manifest = json.load(open(run_manifest_dir, "r"))
        self.paths[step] = run_step_dir
        self.dataframes[step].update(load_checkpoint(run_step_dir, self.train_mode))  #self.global_hash, step
        # removed: self.hashes no longer used.  # SPEC§3
        # self.dataframes[step].update(_load_existing_numeric(self, run_step_dir))
        self.artifacts[step] = manifest.get("artifacts", {})
        self.transformations[step] = manifest.get("transformations", {})
        self.config[step] = manifest.get("config", {})
        self.metadata[step] = manifest.get("metadata", {}) 
        self.train_paths[step] = manifest.get("train_dir")
        self.train_artifacts[step] = manifest.get("train_artifacts", {})
        self.train_models[step] = manifest.get("train_models", {})
        self.train_transformations[step] = manifest.get("train_transformations", {})
        return


    # ----------------------------------------------------------------
    # 1  Inference mode → load built artefacts NOT from TRAINING run
    # ----------------------------------------------------------------
    if not self.train_mode:
        # step_dir   = os.path.join("artifacts", f"run_{self.global_hash}", step)
        # train_step_dir = os.path.join("artifacts", f"run_{self.global_train_hash}", step)
        # train_manifest_dir = os.path.join(train_step_dir, "manifest.json")
        # train_manifest = {}
        """if os.path.exists(train_manifest_dir):
            print(f"[{step.upper()}] Reusing training artefacts from {train_step_dir}")
            train_manifest = json.load(open(train_manifest_dir, "r"))
            self.train_paths[step] = train_step_dir
            self.paths[step] = train_step_dir
            # self.dataframes[step].update(load_checkpoint(train_step_dir, self.train_mode))  #self.global_hash, step
            self.train_paths[step] = train_step_dir
            self.train_artifacts[step] = train_manifest.get("train_artifacts", {})
            self.train_models[step] = train_manifest.get("train_models", {})
            self.train_transformations[step] = train_manifest.get("train_transformations", {})
            self.train_artifacts[step] = train_manifest.get("artifacts", {})
            self.train_transformations[step] = train_manifest.get("transformations", {})
            # self.train_metrics[step] = {}
            # self.train_dataframes[step] = {}
            return"""
        # Nothing to reuse → spec mandates failure
    
        # ‑‑‑ nothing to reuse → raise, as required by SPEC §5
        #raise FileNotFoundError(
            #f"[{step.upper()}] Expected training artefacts at {step_dir} but none found."
        #)
    
        # self.dataframes[step]["test"]=self.dataframes["feature_engineering"]["feature_engineered"]
        self.dataframes[step].update({
            "test": self.dataframes["feature_engineering"]["feature_engineered"],  # self.dataframes["test"],
            #"train": self.dataframes["train"],
            #"val": self.dataframes["val"],
            #"excluded": self.dataframes["excluded"]
            
        })

        save_outputs(
            df_test=self.dataframes[step]["test"], # df_train, df_val,  df_excluded,
            id_col=config_init["id_col"], # self.dataframes["stratification_keys"], 
            step_dir=run_step_dir,  # , param_hash=param_hash
            df_train=None, 
            df_val=None, 
            df_excluded=None, 
            stratify_keys=None,
            train_mode=self.train_mode
        )


        if config_init.get("use_mlflow", False):
            with mlflow.start_run(run_name=f"{step}_{param_hash}"):
                mlflow.set_tags({"step": step, "param_hash": param_hash})
                mlflow.log_artifacts(run_step_dir, artifact_path=step)

        log_registry(step, self.global_hash, config_init, run_step_dir)  # SPEC§7

        print(f"[{step.upper()}] Partitioning step in inference mode does nothing. Data saved to {run_step_dir}")
        print(f"[{step.upper()}] Inference records: {len(self.dataframes[step]['test'])}")
        # print(f"[{step.upper()}] Excluded samples: {len(df_excluded)}")
        print(f"[{step.upper()}] Total records processed: "
            f"{len(self.dataframes[step]['test'])}")

        self.paths[step] = run_step_dir
        create_manifest(step, param_hash, config_init, run_step_dir)
        return
        # raise FileNotFoundError(
        #    f"[{step.upper()}] Expected training artefacts at {train_step_dir} but none found."
        # )


    # ----------------------------------------------------------------
    # 3️⃣  Training mode – continue with normal computation
    # ----------------------------------------------------------------

    
    df = self.dataframes["feature_engineering"]["feature_engineered"]
    target = config_init["target_col"]
    id_col = config_init["id_col"]
    use_stratification = config_init["use_stratification"]
    use_downsampling = config_init.get("use_downsampling", True)
    seed = config_init["seed"]

    # ------------------------------------------------------------------- #
    # SPEC§1: use run‑level directory: artifacts/run_<hash>/<step>/       #
    # ------------------------------------------------------------------- #
    param_hash = self.global_hash  # keep var name to minimise diff
    
    manifest_file = os.path.join(run_step_dir, "manifest.json")

    # ------------------------------------------------------------------- #
    # SPEC§14 skip‑guard                                                  #
    # ------------------------------------------------------------------- #
    if os.path.exists(manifest_file):
        print(f"[{step.upper()}] Skipping — checkpoint exists at {run_step_dir}")
        self.paths[step] = run_step_dir
        # removed: self.hashes no longer used.  # SPEC§3
        checkpoint_data = load_checkpoint(run_step_dir, self.train_mode)  # self.global_hash, step
        self.dataframes[step].update(checkpoint_data)
        return

    # os.makedirs(step_dir, exist_ok=True)
    stratify_cols = identify_stratification_columns(
        df, target, use_stratification, config_init["stratify_cardinality_threshold"]
    )
    
    df_for_splitting = df
    df_excluded = pd.DataFrame(columns=df.columns)
    
    if use_downsampling:
        minority_class, majority_class, df_minority, df_majority, _ = identify_classes(df, target, step)
        
        if len(df_majority) < len(df_minority):
            print(f"[{step.upper()}] Warning: 'Majority' class {majority_class} ({len(df_majority)} samples) "
                  f"is smaller than 'minority' class {minority_class} ({len(df_minority)} samples). "
                  f"Skipping downsampling.")
        else:
            if use_stratification:
                strat_cols_for_downsampling = [col for col in stratify_cols if col != target]
                
                if strat_cols_for_downsampling:
                    df_majority_downsampled, df_excluded = perform_stratified_downsampling(
                        df_majority, df_minority, strat_cols_for_downsampling, seed
                    )
                    df_majority_downsampled, df_excluded = balance_samples(
                        df_majority_downsampled, df_minority, df_majority, df_excluded, seed
                    )
                else:
                    df_majority_downsampled, df_excluded = random_downsample(df_majority, df_minority, seed)
            else:
                df_majority_downsampled, df_excluded = random_downsample(df_majority, df_minority, seed)
            
            df_balanced = pd.concat([df_minority, df_majority_downsampled], axis=0).sample(frac=1, random_state=seed)
            df_for_splitting = df_balanced

    stratify_key = df_for_splitting[stratify_cols].astype(str).agg("_".join, axis=1)
    self.dataframes[step]["stratification_keys"] = pd.Series(stratify_key)

    val_ratio = config_init["val_size"] / (config_init["train_size"] + config_init["val_size"])
    df_train, df_val, df_test, _ = perform_data_splits(
        df_for_splitting, stratify_cols, config_init["test_size"], val_ratio, seed
    )
    
    save_outputs(
        df_train=df_train, df_val=df_val, df_test=df_test, df_excluded=df_excluded,
        id_col=id_col, stratify_keys=self.dataframes[step]["stratification_keys"],
        step_dir=run_step_dir, train_mode=self.train_mode  # , param_hash
    )


    self.dataframes[step].update({
        "train": df_train,
        "val": df_val,
        "test": df_test,
        "excluded": df_excluded
    })
    print(f"[{step.upper()}] Partitioning completed. Data saved to {run_step_dir}")
    print(f"[{step.upper()}] Train samples: {len(df_train)}, Val samples: {len(df_val)}, Test samples: {len(df_test)}")
    print(f"[{step.upper()}] Excluded samples: {len(df_excluded)}")
    print(f"[{step.upper()}] Total samples processed: "
          f"{len(df_train) + len(df_val) + len(df_test) + len(df_excluded)}")

    self.paths[step] = run_step_dir
    # removed: self.hashes no longer used.  # SPEC§3
    create_manifest(step, param_hash, config_init, run_step_dir)

    if config_init.get("use_mlflow", False):
        with mlflow.start_run(run_name=f"{step}_{param_hash}"):
            mlflow.set_tags({"step": step, "param_hash": param_hash})
            mlflow.log_artifacts(run_step_dir, artifact_path=step)

    log_registry(step, self.global_hash, config_init, run_step_dir)  # SPEC§7
