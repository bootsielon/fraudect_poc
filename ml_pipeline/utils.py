# === ml_pipeline/utils.py =============================================
"""
Utility helpers shared across the ML pipeline.

Spec compliance highlights
──────────────────────────
• Provides deterministic `make_param_hash` (spec §1‑A).  
• Adds `DEFAULT_TEST_HASH` for cross‑file smoke‑tests (spec §17).  
• Keeps legacy / deprecated helpers but comments them out per spec §8.  
• All imports are absolute (spec §11).  
• No timestamp‑based hashing. Any timestamps written are for *human*
  consumption only and never enter the hash.
"""

from __future__ import annotations

import json
import hashlib
import os
from datetime import datetime, timezone
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3

# ──────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────
DEFAULT_TEST_HASH = "deadbeefcaf0"  # shared across smoke‑tests (spec §17)

# ──────────────────────────────────────────────────────────────────────
# HASHING HELPERS
# ──────────────────────────────────────────────────────────────────────
def make_param_hash(obj: Any, length: int = 12) -> str:
    """
    Deterministic hash: SHA‑256 over canonical JSON (sorted keys),
    truncated to `length` hex characters.  (Complies with spec §1‑A.)
    """
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:length]


# legacy hashing utilities ─ kept but commented‑out for auditability
# ----------------------------------------------------------------------
# def stable_hash(obj: Any, length: int = 12) -> str:  # deprecated alias
#     return make_param_hash(obj, length)
#
# def timestamp_hash(config):
#     """☠️ Deprecated: timestamp‑based, non‑deterministic."""
#     from datetime import datetime, timezone
#     config = convert_numpy_types(config)
#     config["_ts"] = datetime.now(timezone.utc).isoformat()
#     return make_param_hash(config)

# ──────────────────────────────────────────────────────────────────────
# DATA LOADER (LEGACY)
# ──────────────────────────────────────────────────────────────────────
def load_data_old(db_path: str = "fraud_poc.db") -> pd.DataFrame:
    """
    Legacy helper that merges the three core tables from the SQLite DB.
    Kept only for notebooks and ad‑hoc use; modern code uses `MLPipeline.load_data`.
    """
    conn = sqlite3.connect(db_path)
    df_clients       = pd.read_sql_query("SELECT * FROM clients"      , conn)
    df_merchants     = pd.read_sql_query("SELECT * FROM merchants"    , conn)
    df_transactions  = pd.read_sql_query("SELECT * FROM transactions" , conn)
    conn.close()

    df_clients   = df_clients.rename(columns={"account_creation_date": "account_creation_date_client"})
    df_merchants = df_merchants.rename(columns={"account_creation_date": "account_creation_date_merchant"})

    return (
        df_transactions
        .merge(df_clients , on="account_id")
        .merge(df_merchants, on="merchant_id")
    )

# The following duplicate loader accepted a `self` param by mistake.
# It is now commented‑out to avoid confusion (spec §8 – never delete).
# ----------------------------------------------------------------------
# def load_data(self) -> pd.DataFrame:
#     """Deprecated duplicate of MLPipeline.load_data()."""
#     raise NotImplementedError(
#         "Use MLPipeline.load_data(); this stub is kept for backward compat."
#     )

# ──────────────────────────────────────────────────────────────────────
# JSON‑SERIALISATION HELPERS
# ──────────────────────────────────────────────────────────────────────
def convert_numpy_types(obj: Any):
    """
    Recursively convert NumPy / pandas scalar types to native Python types
    so the object becomes JSON‑serialisable.
    """
    import numpy as np

    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(v) for v in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd and isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    else:
        return obj


# secondary alias kept for backward compatibility
def convert_numpy_types_(obj: Any):
    """Deprecated alias that points to `convert_numpy_types`."""
    return convert_numpy_types(obj)


# ──────────────────────────────────────────────────────────────────────
# GLOBAL REGISTRY LOGGER
# ──────────────────────────────────────────────────────────────────────
def log_registry(step: str, param_hash: str, config: dict, output_dir: str) -> None:
    """
    Append one JSON‑line to `artifacts/global_registry.jsonl`.
    A timestamp is logged for human traceability; it never feeds into hashing.
    """
    registry_path = "artifacts/global_registry.jsonl"
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)

    entry = {
        "step"       : step,
        "param_hash" : param_hash,
        "timestamp"  : datetime.now(timezone.utc).isoformat(),  # allowed
        "config"     : config,
        "output_dir" : output_dir,
    }

    with open(registry_path, "a") as fh:
        fh.write(json.dumps(entry) + "\n")


# ──────────────────────────────────────────────────────────────────────
# MATPLOTLIB ARTIFACT HELPER
# ──────────────────────────────────────────────────────────────────────
def save_plot_as_artifact(fig, artifact_path: str, artifacts_dict: dict, artifact_key: str) -> None:
    """
    Save a matplotlib figure as an artefact and register its path.
    """
    fig.savefig(artifact_path)
    plt.close(fig)
    artifacts_dict[artifact_key] = artifact_path

    # ──────────────────────────────────────────────────────────────────────
# SMOKE‑TEST SUITE  (spec §17‑§19)
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Smoke‑tests for utility helpers:

    1. `make_param_hash` is deterministic and order‑independent.
    2. `convert_numpy_types` makes a NumPy‑laden dict JSON‑serialisable.
    3. `log_registry` appends a line to the global registry file.
    4. `save_plot_as_artifact` writes a PNG and records it.

    These tests use DEFAULT_TEST_HASH so artefacts align with other files.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import json, os

    # ------------------------------------------------------------------
    # 1. make_param_hash determinism
    # ------------------------------------------------------------------
    obj_a = {"x": 1, "y": 2}
    obj_b = {"y": 2, "x": 1}  # same content, different order
    h1 = make_param_hash(obj_a)
    h2 = make_param_hash(obj_b)
    assert h1 == h2, "Hash should be order‑independent"
    print("[OK ] make_param_hash deterministic:", h1)

    # ------------------------------------------------------------------
    # 2. convert_numpy_types
    # ------------------------------------------------------------------
    complex_obj = {
        "int"   : np.int32(5),
        "float" : np.float64(3.14),
        "bool"  : np.bool_(True),
        "array" : np.array([1, 2, 3]),
    }
    converted = convert_numpy_types(complex_obj)
    try:
        json.dumps(converted)  # should succeed
        print("[OK ] convert_numpy_types JSON‑serialisable")
    except TypeError as exc:                       # pragma: no cover
        print("[ERR] convert_numpy_types failed:", exc)

    # ------------------------------------------------------------------
    # 3. log_registry
    # ------------------------------------------------------------------
    log_registry(
        step="utils_smoke_test",
        param_hash=DEFAULT_TEST_HASH,
        config={"test": True},
        output_dir="artifacts/run_utils_smoke",
    )
    registry_file = "artifacts/global_registry.jsonl"
    last_line = open(registry_file).read().strip().splitlines()[-1]
    parsed = json.loads(last_line)
    assert parsed["step"] == "utils_smoke_test", "Registry entry mismatch"
    print("[OK ] log_registry wrote entry")

    # ------------------------------------------------------------------
    # 4. save_plot_as_artifact
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    artifact_dir = "artifacts/run_utils_smoke"
    os.makedirs(artifact_dir, exist_ok=True)
    png_path = os.path.join(artifact_dir, "diag.png")
    artefacts_dict = {}
    save_plot_as_artifact(fig, png_path, artefacts_dict, "diag_png")
    assert os.path.exists(png_path), "PNG not saved"
    assert artefacts_dict["diag_png"] == png_path, "Artefact dict mismatch"
    print("[OK ] save_plot_as_artifact saved", png_path)