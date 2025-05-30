# -*- coding: utf-8 -*-
"""
Smoke‑tests for ml_pipeline.model_baseline  (SPEC §19)

Fix: indices were reset *before* the .drop() calls, triggering a KeyError.
We now keep the original indices for drop‑based splitting and reset them
only afterwards.
"""
import os
import shutil
import traceback

import numpy as np
import pandas as pd

from ml_pipeline.base import MLPipeline
from ml_pipeline.utils import DEFAULT_TEST_HASH

step = "model_baseline"

# ------------------------------------------------------------------ #
# Helper                                                             #
# ------------------------------------------------------------------ #
def build_cfg(train_mode: bool, **kw) -> dict:
    cfg: dict = {
        "train_mode": train_mode,
        "model_name": "dummy_model",
        "model_hash": "abcd1234",
        "dataset_name": "dummy_ds",
        "feature_names": ["f1", "f2"],
        "target_col": "target",
        "id_col": "id",
        "random_state": 123,
        "use_mlflow": False,
        "n_estimators": 50,
    }
    if not train_mode:
        cfg["train_hash"] = kw.get("train_hash")
    return cfg

def safe(label: str, fn):
    try:
        fn()
        print(f"[OK ] {label}")
    except Exception as exc:
        print(f"[ERR] {label} → {exc}")
        traceback.print_exc()

# ------------------------------------------------------------------ #
# Create deterministic numeric data                                  #
# ------------------------------------------------------------------ #
np.random.seed(42)
n = 150
mock_df = pd.DataFrame(
    {
        "id": range(1, n + 1),
        "f1": np.random.normal(0, 1, n),
        "f2": np.random.normal(5, 2, n),
        "target": np.random.choice([0, 1], n, p=[0.75, 0.25]),
    }
)

# simple split – keep indices intact for `.drop()`
train_df = mock_df.sample(frac=0.6, random_state=1)
val_df = mock_df.drop(train_df.index).sample(frac=0.4, random_state=2)
test_df = mock_df.drop(train_df.index).drop(val_df.index)
excl_df = mock_df.sample(20, random_state=3)

# finally reset indices
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
excl_df.reset_index(drop=True, inplace=True)

# ------------------------------------------------------------------ #
# Clean slate                                                         #
# ------------------------------------------------------------------ #
artefact_root = os.path.join("artifacts", f"run_{DEFAULT_TEST_HASH}", step)
if os.path.exists(artefact_root):
    shutil.rmtree(artefact_root)

# 1️⃣  Training – fresh artefacts
pipe_train = MLPipeline(build_cfg(True))
pipe_train.global_hash = DEFAULT_TEST_HASH
pipe_train.global_train_hash = DEFAULT_TEST_HASH
pipe_train.dataframes = {
    "train_sca": train_df,
    "val_sca": val_df,
    "test_sca": test_df,
    "excluded_sca": excl_df,
}
print("\n>>> TRAINING RUN (fresh artefacts)")
safe("TRAIN‑fresh", pipe_train.model_baseline)

# 2️⃣  Training – skip‑guard
pipe_train_skip = MLPipeline(build_cfg(True))
pipe_train_skip.global_hash = DEFAULT_TEST_HASH
pipe_train_skip.global_train_hash = DEFAULT_TEST_HASH
pipe_train_skip.dataframes = {
    "train_sca": train_df,
    "val_sca": val_df,
    "test_sca": test_df,
}
print("\n>>> TRAINING RUN (should skip)")
safe("TRAIN‑skip‑guard", pipe_train_skip.model_baseline)

# 3️⃣  Inference – artefacts present
infer_hash_ok = "abcabcabcabc"
pipe_infer_ok = MLPipeline(build_cfg(False, train_hash=DEFAULT_TEST_HASH))
pipe_infer_ok.global_hash = infer_hash_ok
pipe_infer_ok.global_train_hash = DEFAULT_TEST_HASH
pipe_infer_ok.dataframes = {"test_sca": test_df}
print("\n>>> INFERENCE RUN (artefacts present)")
safe("INFER‑reuse", pipe_infer_ok.model_baseline)

# 4️⃣  Inference – artefacts missing (should fail)
missing_train_hash = "feedfeedfeed"
miss_dir = os.path.join("artifacts", f"run_{missing_train_hash}", step)
if os.path.exists(miss_dir):
    shutil.rmtree(miss_dir)

pipe_infer_fail = MLPipeline(build_cfg(False, train_hash=missing_train_hash))
pipe_infer_fail.global_hash = "deadbeef0000"
pipe_infer_fail.global_train_hash = missing_train_hash
print("\n>>> INFERENCE RUN (artefacts missing – should fail)")
try:
    pipe_infer_fail.model_baseline()
    print("❌  ERROR: Missing‑artefact inference did *not* fail as expected")
except FileNotFoundError as e:
    print(f"✅  Caught expected error → {e}")