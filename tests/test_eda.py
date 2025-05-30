# -*- coding: utf-8 -*-
"""
Smoke‑tests for ml_pipeline.eda  (spec §19)

Scenarios exercised
-------------------
1. Training with no pre‑existing artefacts  (fresh run)
2. Training with artefacts present          (skip‑guard hit)
3. Inference with required artefacts        (reuse training outputs)
4. Inference without artefacts              (must fail clearly)
"""

import os
import shutil
import traceback
import pandas as pd

from ml_pipeline.base import MLPipeline
from ml_pipeline.utils import DEFAULT_TEST_HASH

step = "eda"
# Clean up previous artefacts
artefact_root = os.path.join("artifacts", f"run_{DEFAULT_TEST_HASH}", step)
if os.path.exists(artefact_root):
    shutil.rmtree(artefact_root)  # Clean up previous artefacts

df_demo = pd.DataFrame(
    {
        "account_id": [1, 2],
        "merchant_id": [10, 20],
        "amount": [100.5, 200.0],
        "timestamp": ["2023-01-01", "2023-01-02"],
        "is_fraud": [0, 1],
    }
)

# Shared config builder
def make_cfg(train_mode: bool, **overrides):
    base = {
        "train_mode": train_mode,
        "model_name": "demo_model",
        "model_hash": "abc1234",
        "dataset_name": "demo_ds",
        "target_col": "is_fraud",
        "use_mlflow": False,
        "feature_names": ["amount", "timestamp"],
    }
    return {**base, **overrides}

def safe(label: str, fn):
    try:
        fn()
        print(f"[OK ] {label}")
    except Exception as exc:
        print(f"[ERR] {label} → {exc}")
        traceback.print_exc()

# 1. Training – fresh run
cfg_train = make_cfg(True)
pipe_1 = MLPipeline(cfg_train, data_source="raw", raw_data=df_demo)
pipe_1.global_hash = DEFAULT_TEST_HASH
pipe_1.global_train_hash = DEFAULT_TEST_HASH
pipe_1.run_dir = os.path.join("artifacts", f"run_{DEFAULT_TEST_HASH}")
safe("TRAIN‑v1: EDA fresh", lambda: pipe_1.eda())

# 2. Training – same config, skip‑guard
pipe_2 = MLPipeline(cfg_train, data_source="raw", raw_data=df_demo)
pipe_2.global_hash = DEFAULT_TEST_HASH
pipe_2.global_train_hash = DEFAULT_TEST_HASH
pipe_2.run_dir = os.path.join("artifacts", f"run_{DEFAULT_TEST_HASH}")
safe("TRAIN‑v2: EDA skip‑guard", lambda: pipe_2.eda())

# 3. Inference – correct train_hash
cfg_infer = make_cfg(False, train_hash=DEFAULT_TEST_HASH)
pipe_3 = MLPipeline(cfg_infer, data_source="raw", raw_data=df_demo)
pipe_3.global_hash = DEFAULT_TEST_HASH  # set explicitly
pipe_3.global_train_hash = DEFAULT_TEST_HASH
pipe_3.run_dir = os.path.join("artifacts", f"run_{DEFAULT_TEST_HASH}")
safe("INFER: EDA re‑use", lambda: pipe_3.eda())

# 4. Inference – bad train_hash (must fail)
bad_hash = "not_a_real_hash"
cfg_fail = make_cfg(False, train_hash=bad_hash)
cfg_fail["model_name"] = "new_model"  # force a new inference hash
cfg_fail["model_hash"] = "zzzz9999"
pipe_4 = MLPipeline(cfg_fail, data_source="raw", raw_data=df_demo)
pipe_4.global_hash = "badf00ddead0"  # override inference run hash
pipe_4.global_train_hash = bad_hash
pipe_4.run_dir = os.path.join("artifacts", f"run_{pipe_4.global_hash}")
try:
    pipe_4.eda()
    raise SystemExit("[TEST] Expected failure not raised")
except AssertionError as e:
    print(f"[OK ] INFER‑fail: Correctly failed → {e}")
