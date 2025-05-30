
"""
    Smoke‑tests for numeric_conversion
    ----------------------------------
    1. Training run - fresh artefacts (compute).
    2. Training run - artefacts present (skip).
    3. Inference - artefacts present (load).
    4. Inference - artefacts missing (fail).
"""
import os
import shutil
import numpy as np
import pandas as pd
import pytest
from ml_pipeline.base import MLPipeline
from ml_pipeline.utils import DEFAULT_TEST_HASH

step = "numeric_conversion"

# ------------------------------------------------------------------ #
# Mock data set‑up                                                   #
# ------------------------------------------------------------------ #
@pytest.fixture(scope="module")
def mock_data():
    np.random.seed(42)
    n_samples = 200
    df = pd.DataFrame(
    {
        "id": range(1, n_samples + 1),
        "numeric_col": np.random.normal(0, 1, n_samples),
        "integer_col": np.random.randint(0, 100, n_samples),
        "low_card_cat": np.random.choice(["A", "B", "C"], n_samples),
        "mid_card_cat": np.random.choice([f"Cat_{i}" for i in range(20)], n_samples),
        "high_card_cat": np.random.choice([f"ID_{i}" for i in range(150)], n_samples),
        "id_like_col": [f"U{i:05d}" for i in range(n_samples)],
        "missing_values_col": np.random.choice([np.nan, 1, 2, 3], n_samples),
        "target": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        }
    )
    df.loc[np.random.choice(n_samples, 10), "low_card_cat"] = ""

    val = df.sample(50, random_state=123).reset_index(drop=True)
    test = df.sample(50, random_state=456).reset_index(drop=True)
    excl = df.sample(20, random_state=789).reset_index(drop=True)
    return df, val, test, excl




# ------------------------------------------------------------------ #
# Helper to build a minimally valid pipeline config                  #
# ------------------------------------------------------------------ #
def build_pipeline(*, train_mode: bool, run_hash: str, train_hash: str | None = None, feature_names: list[str] | None = None) -> MLPipeline:
    cfg = {
        # ----- keys required by base.py / hashing rules
        "train_mode": train_mode,
        "model_name": "dummy_model",
        "model_hash": "dummy_model_hash",
        "dataset_name": "mock_dataset",
        "feature_names": feature_names or [],  # sorted(mock_train.columns.tolist()),
        "extra_inference_settings": {},
        # ----- numeric‑conversion hyper‑params
        "seed": 42,
        "target_col": "target",
        "id_col": "id",
        "use_mlflow": False,
        "c1": 5,
        "c2": 0.2,
        "b1": True,
        "c3": 1.5,
        "id_like_exempt": True,
        "central_tendency": "median",
    }
    if not train_mode:
        cfg["train_hash"] = train_hash
    pipe = MLPipeline(config=cfg)
    pipe.global_hash = run_hash
    pipe.global_train_hash = run_hash if train_mode else train_hash
    return pipe

# ---------- clean slate for DEFAULT_TEST_HASH --------------------- #
def cleanup(run_hash):
    """Remove artifacts for a given run hash."""
    train_run_dir = os.path.join("artifacts", f"run_{run_hash}", step)
    if os.path.exists(train_run_dir):
        shutil.rmtree(train_run_dir)


def test_training_compute_and_skip(mock_data):
    # ------------------------------------------------------------------ #
    # 1️⃣  TRAINING - fresh artefacts                                    #
    # ------------------------------------------------------------------ #
    mock_train, mock_val, mock_test, excl = mock_data
    cleanup(DEFAULT_TEST_HASH)

    # 1) fresh training
    # pipe_train = build_pipeline(True, DEFAULT_TEST_HASH, feature_names=list(train.columns))

    pipe_train = build_pipeline(train_mode=True, run_hash=DEFAULT_TEST_HASH, feature_names=list(mock_train.columns))
    pipe_train.dataframes = {
        "train": mock_train,
        "val": mock_val,
        "test": mock_test,
        "excluded": mock_train.sample(n=20, random_state=789),
    }
    print("\n>>> TRAINING RUN (fresh artefacts)")
    pipe_train.numeric_conversion()
    manifest = f"artifacts/run_{DEFAULT_TEST_HASH}/{step}/manifest.json"
    assert os.path.exists(manifest)

    # 2) skip on second pass
    # ------------------------------------------------------------------ #
    # 2️⃣  TRAINING - artefacts present (skip)                           #
    # ------------------------------------------------------------------ #
    pipe_train_skip = build_pipeline(train_mode=True, run_hash=DEFAULT_TEST_HASH, feature_names=list(mock_train.columns))
    pipe_train_skip.dataframes = {"train":mock_train,"val":mock_val,"test":mock_test}
    # should not raise
    print("\n>>> TRAINING RUN (should skip)")
    pipe_train_skip.numeric_conversion()



def test_inference_load_and_fail(mock_data):
    # ------------------------------------------------------------------ #
    # 3️⃣  INFERENCE - artefacts present                                 #
    # ------------------------------------------------------------------ #

    # 3) inference with artifacts present
    print(DEFAULT_TEST_HASH)
    cleanup(DEFAULT_TEST_HASH)
    train, val, test, _ = mock_data
    trainer = build_pipeline(
        train_mode=True, run_hash=DEFAULT_TEST_HASH, feature_names=list(train.columns))

    # first create training artifacts
    trainer = build_pipeline(
        train_mode=True, run_hash=DEFAULT_TEST_HASH, feature_names=list(train.columns)
    )
    trainer.dataframes = {"train":train,"val":val,"test":test}
    trainer.numeric_conversion()

    infer_hash_ok = "abcabcabcabc"
    pipe_infer_ok = build_pipeline(
        train_mode=False, run_hash=infer_hash_ok, train_hash=DEFAULT_TEST_HASH,
        feature_names=list(train.columns)
    )
    
    pipe_infer_ok.dataframes = {"test": test}
    print("\n>>> INFERENCE RUN (artefacts present)")
    pipe_infer_ok.numeric_conversion()  

    # ------------------------------------------------------------------ #
    # 4️⃣  INFERENCE - artefacts missing (must fail)                      #
    # ------------------------------------------------------------------ #

    # 4) inference missing → should raise
    missing_train_hash = "feedfeedfeed"
    infer_hash_fail = "deadbeef0000"
    cleanup(missing_train_hash)

    # create a mock test dataframe
    pipe_infer_fail = build_pipeline(
        train_mode=False, run_hash=infer_hash_fail, train_hash=missing_train_hash
    )

    pipe_infer_fail.dataframes = {"test": test}
    print("\n>>> INFERENCE RUN (artefacts missing - should fail)")
    try:
        pipe_infer_fail.numeric_conversion()
        print("❌  ERROR: inference did NOT fail although artefacts are missing")
    except FileNotFoundError as e:
        print(f"✅  Caught expected error → {e}")




# print("\nSmoke-tests finished.")