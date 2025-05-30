# -*- coding: utf-8 -*- 
"""
Smoke‑tests for ml_pipeline.feature_engineering  (spec §19)

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
from datetime import datetime, timedelta

from ml_pipeline.base import MLPipeline
from ml_pipeline.utils import DEFAULT_TEST_HASH

step = "feature_engineering"
# Clean up previous artefacts
artefact_root = os.path.join("artifacts", f"run_{DEFAULT_TEST_HASH}", step)
if os.path.exists(artefact_root):
    shutil.rmtree(artefact_root)  # Clean up previous artefacts


now = datetime.now()
df_demo = pd.DataFrame({
    "account_id": [1, 2],
    "merchant_id": [101, 102],
    "amount": [100.0, 200.0],
    "is_fraud": [0, 1],
    "timestamp": [(now - timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S") for i in [1, 2]],
    "account_creation_date_client": [(now - timedelta(days=100 + i)).strftime("%Y-%m-%d") for i in [1, 2]],
    "account_creation_date_merchant": [(now - timedelta(days=1000 + i)).strftime("%Y-%m-%d") for i in [1, 2]],
    "constant_col": [1, 1],  # To test dropping of constant columns
})

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
pipe_1.dataframes["raw"] = df_demo
safe("TRAIN‑v1: Feature engineering fresh", lambda: pipe_1.feature_engineering())

# 2. Training – same config, skip‑guard
pipe_2 = MLPipeline(cfg_train, data_source="raw", raw_data=df_demo)
pipe_2.global_hash = DEFAULT_TEST_HASH
pipe_2.global_train_hash = DEFAULT_TEST_HASH
pipe_2.run_dir = os.path.join("artifacts", f"run_{DEFAULT_TEST_HASH}")
pipe_2.dataframes["raw"] = df_demo
safe("TRAIN‑v2: Feature engineering skip‑guard", lambda: pipe_2.feature_engineering())

# 3. Inference – valid train_hash reuse
cfg_infer = make_cfg(False, train_hash=DEFAULT_TEST_HASH)
pipe_3 = MLPipeline(cfg_infer, data_source="raw", raw_data=df_demo)
pipe_3.global_hash = DEFAULT_TEST_HASH
pipe_3.global_train_hash = DEFAULT_TEST_HASH
pipe_3.run_dir = os.path.join("artifacts", f"run_{DEFAULT_TEST_HASH}")
pipe_3.dataframes["raw"] = df_demo
safe("INFER: Feature engineering reuse", lambda: pipe_3.feature_engineering())

# 4. Inference – invalid train_hash (should fail)
bad_hash = "not_real_hash123"
cfg_fail = make_cfg(False, train_hash=bad_hash)
cfg_fail["model_name"] = "new_model"
cfg_fail["model_hash"] = "ffff9999"
pipe_4 = MLPipeline(cfg_fail, data_source="raw", raw_data=df_demo)
pipe_4.global_hash = "feedbead1234"
pipe_4.global_train_hash = bad_hash
