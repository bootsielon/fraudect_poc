# ──────────────────────────────────────────────────────────────────────
# SMOKE‑TEST SUITE  (base‑only edition – spec §17 – §24)
# ──────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
"""
Base‑only smoke‑test.

1. Training: verify hash, run‑dir creation, load_data().
2. Training again: same config ⇒ identical hash; no directory errors.
3. Inference with artefacts from #1: verify deterministic new hash
    and correct `global_train_hash` binding.
4. Inference with bad `train_hash`: expect failure when accessing
    training artefacts directory (simulated here).
"""

import os
# import shutil
import traceback
import pandas as pd

source = "csv" # if pipe_train.train_mode else "xlsx"

csv_test_path = "data/kagglebankfraud/Variant I.csv"
csv_file_path = "data/kagglebankfraud/Base.csv"  # "fraud_poc.db",
# csv_file_path = "data/kagglebankfraud/Variant II.csv"  # "fraud_poc.db",
# ------------------------------------------------------------------
# Shared constant hash (spec §17)
# ------------------------------------------------------------------
try:
    from ml_pipeline.utils import DEFAULT_TEST_HASH
except Exception:                                    # pragma: no cover
    DEFAULT_TEST_HASH = "deadbeefcaf0"

# fresh workspace for repeatability
# if os.path.exists("artifacts"):
    # shutil.rmtree("artifacts")

df_demo = pd.DataFrame(
    {
        "account_id": [1, 2],
        "merchant_id": [10, 20],
        "amount": [100.5, None],
        "timestamp": ["2022-01-01", "2022-01-02"],
        "is_fraud": [0, 1],
    }
)

# Helper
def safe(label: str, fn):
    try:
        fn()
        print(f"[OK ] {label}")
    except Exception as exc:
        print(f"[ERR] {label} → {exc}")
        traceback.print_exc()

# 1. Training – first run
train_cfg = {
    "train_mode": True,
    "random_seed": 123,
    "model_name": "demo_model",
    "model_hash": "abcd1234",
    "dataset_name": "demo_ds",
    "feature_names": ["amount", "timestamp"],
    "global_hash": DEFAULT_TEST_HASH,  # will be overridden
}
#pipe_train = MLPipeline(train_cfg, data_source="raw", raw_data=df_demo)
pipe_train = MLPipeline(train_cfg, data_source="raw", raw_data=df_demo)

safe("TRAIN‑v1: load_data()", lambda: pipe_train._load_data())

TRAIN_HASH = pipe_train.global_hash
print("TRAIN_HASH =", TRAIN_HASH)
assert os.path.isdir(pipe_train.run_dir), "run directory missing"

# 2. Training – identical config
pipe_train2 = MLPipeline(train_cfg, data_source="raw", raw_data=df_demo)
assert (
    pipe_train2.global_hash == TRAIN_HASH
), "hash mismatch for identical config"
safe("TRAIN‑v2: load_data()", lambda: pipe_train2._load_data())

# 3. Inference – with correct train_hash
infer_cfg = {
    "train_mode": False,
    "train_hash": TRAIN_HASH,
    "model_name": "demo_model",
    "model_hash": "abcd1234",
    "dataset_name": "demo_ds",
    "feature_names": ["timestamp", "amount"],
    "inference_extra": {},
    "csv_path": "data/kagglebankfraud/Variant I.csv",  # "fraud_poc.db",
    
}



pipe_infer = MLPipeline(infer_cfg, data_source="raw", raw_data=df_demo)
assert (
    pipe_infer.global_train_hash == TRAIN_HASH
), "global_train_hash not set correctly"

# Only call load_data() — no external steps
safe("INFER: load_data()", lambda: pipe_infer._load_data())

# 4. Inference – bad train_hash (should fail when we try to access artefacts)
bad_infer_cfg = infer_cfg | {"train_hash": "not_a_real_hash"}
pipe_bad = MLPipeline(bad_infer_cfg, data_source="raw", raw_data=df_demo)
try:
    # simulate a later step trying to read the training directory
    bad_dir = os.path.join("artifacts", f"run_{pipe_bad.global_train_hash}")
    if not os.path.exists(bad_dir):
        raise FileNotFoundError(
            f"Expected training artefacts at {bad_dir} (simulated failure)"
        )
    print("[ERR] bad inference did NOT fail as expected")
except FileNotFoundError as exc:
    print(f"[OK ] bad inference failed correctly → {exc}")