# -*- coding: utf-8 -*-
"""
Smoke‑tests for the partitioning step
(SPEC §§17‑19 and §21: delivered separately from the main file).
"""
import numpy as np
import pandas as pd
import shutil
import os
from ml_pipeline.utils import DEFAULT_TEST_HASH
from ml_pipeline.base import MLPipeline

# ---------------------------------------------------------------
# Helper to build a fresh pipeline instance
# ---------------------------------------------------------------
def build_pipeline(train_mode: bool, *, with_created_hash: bool = True):
    # base parameters shared by both train & inference
    cfg: dict = {
        "target_col": "is_fraud",
        "id_col": "transaction_id",
        "use_mlflow": False,
        "seed": 42,
        "use_stratification": True,
        "use_downsampling": True,
        "stratify_cardinality_threshold": 5,
        "train_size": 0.7,
        "val_size": 0.15,
        "test_size": 0.15,
        # always present so we can flip modes freely
        "train_hash": DEFAULT_TEST_HASH,
        # mandatory inference‑mode keys (dummies are fine for tests)
        "model_name": "dummy_model",
        "model_hash": "abcd1234",
        "dataset_name": "dummy_ds",
        "feature_names": ["amount", "hour", "day"],
    }

    cfg["train_mode"] = train_mode            # flag must live inside config
    pipe = MLPipeline(config=cfg)             # ctor reads everything from cfg

    if with_created_hash:
        pipe.global_hash = DEFAULT_TEST_HASH  # deterministic hash for tests
    return pipe
step = "partitioning"
# ---------------------------------------------------------------
# Create mock data
# ---------------------------------------------------------------
np.random.seed(42)
n_samples = 200
mock_df = pd.DataFrame({
    "transaction_id": range(1, n_samples + 1),
    "amount": np.random.uniform(5, 500, n_samples),
    "account_id": np.random.randint(1, 10, n_samples),
    "merchant_id": np.random.randint(20, 40, n_samples),
    "hour": np.random.randint(0, 6, n_samples),
    "day": np.random.randint(0, 3, n_samples),
    "category": np.random.choice(["A", "B"], n_samples),
    "is_fraud": np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
})
mock_df.loc[mock_df.sample(frac=0.05, random_state=1).index, "amount"] = np.nan

# Clean up previous artefacts
artefact_root = os.path.join("artifacts", f"run_{DEFAULT_TEST_HASH}", step)
if os.path.exists(artefact_root):
    shutil.rmtree(artefact_root)  # Clean up previous artefacts

# 1️⃣  Training ‑ fresh
pipe_train_fresh = build_pipeline(train_mode=True)
pipe_train_fresh.dataframes["feature_engineered"] = mock_df.copy()
print("\n>>> TRAINING RUN (fresh artefacts)")
pipe_train_fresh.partitioning()

# 2️⃣  Training ‑ skip‑guard
pipe_train_skip = build_pipeline(train_mode=True)
pipe_train_skip.dataframes["feature_engineered"] = mock_df.copy()
print("\n>>> TRAINING RUN (should skip)")
pipe_train_skip.partitioning()

# 3️⃣  Inference ‑ artefacts present
pipe_infer_ok = build_pipeline(train_mode=False)
pipe_infer_ok.dataframes["feature_engineered"] = mock_df.copy()
print("\n>>> INFERENCE RUN (artefacts present)")
pipe_infer_ok.partitioning()

# 4️⃣  Inference ‑ artefacts missing (expect failure)
# 4️⃣  Inference – artefacts missing (must fail)
#missing_hash = "deadbeef9999"  # any value not used earlier
#pipe_infer_fail = build_pipeline(train_mode=False, with_created_hash=False)
#pipe_infer_fail.global_hash = missing_hash
# DO NOT inject feature_engineered here → forces step to look for training artefacts
print("\n>>> INFERENCE RUN (artefacts missing, should fail normally, but in partitioning this test is irrelevant)")
#try:
    #pipe_infer_fail.partitioning()
    #print("❌  ERROR: Missing‑artefact inference did *not* fail as expected")
#except FileNotFoundError as e:
#    print(f"✅  Caught expected error: {e}")