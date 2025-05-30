# -*- coding: utf-8 -*-
"""
Smoke‑tests for ml_pipeline.scaling  (SPEC §19)

Scenarios exercised
-------------------
1. Training with no pre‑existing artefacts  (fresh run)
2. Training with artefacts present          (skip‑guard hit)
3. Inference with required artefacts        (reuse training outputs)
4. Inference without artefacts              (must fail clearly)

All tests rely on the shared DEFAULT_TEST_HASH to guarantee artefact
continuity across pipeline steps (SPEC §17).
"""

import shutil
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ml_pipeline.base import MLPipeline
from ml_pipeline.utils import DEFAULT_TEST_HASH

step = "scaling"

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
        "seed": 123,
        "t1": True,   # mean
        "s1": True,   # std
        "use_mlflow": False,
        # mandatory for inference
        "train_hash": DEFAULT_TEST_HASH,
    }
    cfg.update(kw)
    return cfg

def safe(label: str, fn):
    try:
        fn()
        print(f"[OK ] {label}")
    except Exception as exc:
        print(f"[ERR] {label} → {exc}")
        traceback.print_exc()

# ------------------------------------------------------------------ #
# Mock numeric‑only data (as produced by numeric_conversion)         #
# ------------------------------------------------------------------ #
np.random.seed(42)
n = 120
mock_train = pd.DataFrame(
    {
        "id": range(1, n + 1),
        "feature_A": np.random.normal(10, 2, n),
        "feature_B": np.random.normal(50, 5, n),
        "target": np.random.choice([0, 1], n, p=[0.8, 0.2]),
    }
)
mock_val = mock_train.sample(30, random_state=1).reset_index(drop=True)
mock_test = mock_train.sample(30, random_state=2).reset_index(drop=True)
mock_excluded = mock_train.sample(10, random_state=3).reset_index(drop=True)

# ------------------------------------------------------------------ #
# Clean slate for DEFAULT_TEST_HASH                                  #
# ------------------------------------------------------------------ #
artefact_root = Path("artifacts") / f"run_{DEFAULT_TEST_HASH}" / step
if artefact_root.exists():
    shutil.rmtree(artefact_root)

# 1️⃣  Training – fresh artefacts
pipe_train = MLPipeline(build_cfg(True))
pipe_train.global_hash = DEFAULT_TEST_HASH
pipe_train.global_train_hash = DEFAULT_TEST_HASH
pipe_train.dataframes = {
    "train_num": mock_train,
    "val_num": mock_val,
    "test_num": mock_test,
    "excluded_num": mock_excluded,
}
print("\n>>> TRAINING RUN (fresh artefacts)")
safe("TRAIN‑fresh", pipe_train.scaling)

# 2️⃣  Training – skip‑guard hit
pipe_train_skip = MLPipeline(build_cfg(True))
pipe_train_skip.global_hash = DEFAULT_TEST_HASH
pipe_train_skip.global_train_hash = DEFAULT_TEST_HASH
pipe_train_skip.dataframes = {
    "train_num": mock_train,
    "val_num": mock_val,
    "test_num": mock_test,
}
print("\n>>> TRAINING RUN (should skip)")
safe("TRAIN‑skip‑guard", pipe_train_skip.scaling)

# 3️⃣  Inference – artefacts present
infer_hash_ok = "abcdefabcdef"
pipe_infer_ok = MLPipeline(build_cfg(False, train_hash=DEFAULT_TEST_HASH))
pipe_infer_ok.global_hash = infer_hash_ok
pipe_infer_ok.global_train_hash = DEFAULT_TEST_HASH
pipe_infer_ok.dataframes = {
    "test_num": mock_test
}  # only need test for this step
print("\n>>> INFERENCE RUN (artefacts present)")
safe("INFER‑reuse", pipe_infer_ok.scaling)

# 4️⃣  Inference – artefacts missing (expect failure)
missing_train_hash = "feedfeedfeed"
artefact_miss_dir = Path("artifacts") / f"run_{missing_train_hash}" / step
if artefact_miss_dir.exists():
    shutil.rmtree(artefact_miss_dir)

pipe_infer_fail = MLPipeline(build_cfg(False, train_hash=missing_train_hash))
pipe_infer_fail.global_hash = "deadbeef0000"
pipe_infer_fail.global_train_hash = missing_train_hash
print("\n>>> INFERENCE RUN (artefacts missing – should fail)")
try:
    pipe_infer_fail.scaling()
    print("❌  ERROR: Missing‑artefact inference did *not* fail as expected")
except FileNotFoundError as e:
    print(f"✅  Caught expected error → {e}")