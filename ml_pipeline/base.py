# === ml_pipeline/base.py ==============================================
from __future__ import annotations

import os
# import json   # currently unused
# import hashlib  # currently unused
from typing import Any
from copy import deepcopy
from datetime import datetime, timezone

import pandas as pd

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
from ml_pipeline.utils import log_registry, make_param_hash  # absolute import (spec §11)

# ----------------------------------------------------------------------
# Pipeline step factories – lazily bound lambdas
# ----------------------------------------------------------------------
from ml_pipeline.eda                       import eda
from ml_pipeline.feature_engineering       import feature_engineering
from ml_pipeline.partitioning              import partitioning
from ml_pipeline.numeric_conversion        import numeric_conversion
from ml_pipeline.scaling                   import scaling
from ml_pipeline.model_baseline            import model_baseline


class MLPipeline:
    """
    Configuration‑driven ML pipeline orchestrator.

    A *single* deterministic `self.global_hash` identifies the entire run.
    In inference mode, `self.global_train_hash` points to the hash of the
    training run whose artefacts must be re‑used (spec §4 & §5).
    """

    # ──────────────────────────────────────────────────────────────────
    # INITIALISER
    # ──────────────────────────────────────────────────────────────────
    def __init__(
        self,
        input_config      : dict[str, Any],
        *,
        data_source : str               = "csv",  # "sqlite", "xlsx", "raw"
        db_path     : str               = "fraud_poc.db",
        xlsx_path   : str | None        = None,
        csv_path    : str | None        = None,
        raw_data    : pd.DataFrame | None = None,
    ) -> None:
        step = "init"
        # ------------------------------------------------ core config --
        self.config: dict[str, Any] = {}
        self.config[step] = deepcopy(input_config)
        self.train_mode: bool = bool(self.config[step].get("train_mode", True))
        self.config[step]["train_mode"] = self.train_mode  # echo back
        self.csv_path = self.config[step].get("csv_path")
        # -------------------- data handles ----------------------------
        self.data_source    = self.config[step].get("data_source")  # data_source
        self.db_path        = self.config[step].get("sqlite_path")
        self.xlsx_path      = self.config[step].get("excel_path")  # updated to use config
        # self.csv_path       = self.config[step].get("csv_path")   # updated to use config
        self.raw_data       = self.config[step].get("raw_path")   # updated to use config

        # ------------------------------------------------ bookkeeping --
        self.dataframes: dict[str, dict[str, pd.DataFrame] | pd.DataFrame] = {}
        self.paths     : dict[str, str]          = {}
        self.models    : dict[str, object]       = {}
        self.metrics   : dict[str, dict]         = {}
        self.artifacts : dict[str, dict]         = {}
        self.transformations: dict[str, dict]    = {}
        self.metadata  : dict[str, Any]            = {}
        # ------------------------------------------------ run hash -----
        self.dataframes[step] = {}
        self.dataframes[step]["raw"] = pd.read_csv(self.csv_path)  # raw_data

        if self.train_mode:
            full_config = {
                "user_config": self.config[step],
                "data_source": data_source,
                "db_path"    : db_path,
                "xlsx_path"  : xlsx_path,
                "csv_path"  : csv_path,
            }
            self.global_hash = make_param_hash(full_config)
            self.global_train_hash = self.global_hash
        else:
            key_tuple = (
                self.config[step]["model_name"],
                self.config[step]["model_hash"],
                self.config[step]["dataset_name"],
                tuple(sorted(self.config[step]["feature_names"])),
                self.config[step].get("inference_extra", {}),
                # current time
                datetime.now(timezone.utc).isoformat(),
            )
            self.global_hash = make_param_hash(key_tuple)
            self.global_train_hash = self.config[step]["train_hash"]
            # self.dataframes: dict[str, pd.DataFrame] = {}
        self.train_paths     : dict[str, str]          = {}
        self.train_models    : dict[str, object]       = {}
        # self.train_metrics   : dict[str, dict]         = {}
        self.train_artifacts : dict[str, dict]         = {}
        self.train_transformations: dict[str, dict]    = {}
        # self.train_manifest  : dict[str, Any]            = {}
        self.train_metadata  : dict[str, Any]            = {}
        self.train_config    : dict[str, Any] = {}
        # expose hashes
        self.config[step]["global_hash"]       = self.global_hash
        self.config[step]["global_train_hash"] = self.global_train_hash

        # ------------------------------------------------ run directory -
        self.run_dir = os.path.join("artifacts", f"run_{self.global_hash}")
        first_time = not os.path.exists(self.run_dir)
        os.makedirs(self.run_dir, exist_ok=True)
        self.train_dir = os.path.join("artifacts", f"run_{self.global_train_hash}")
        os.makedirs(self.train_dir, exist_ok=True)
        if first_time:  # write human timestamp (not hashed)
            with open(os.path.join(self.run_dir, "created_at.txt"), "w") as fh:
                # from datetime import datetime, timezone
                fh.write(datetime.now(timezone.utc).isoformat())

        # ------------------------------------------------ bind steps ---
        self.load_data                 = self._load_data
        self.eda                       = lambda: eda(self)
        self.feature_engineering       = lambda: feature_engineering(self)
        self.partitioning              = lambda: partitioning(self)
        self.numeric_conversion        = lambda: numeric_conversion(self)
        self.scaling                   = lambda: scaling(self)
        self.model_baseline            = lambda: model_baseline(self)

    # ──────────────────────────────────────────────────────────────────
    # DATA LOADING  (self‑contained, satisfies spec §24a)
    # ──────────────────────────────────────────────────────────────────
    def _load_data(self) -> pd.DataFrame:
        """
        Load and merge raw data from SQLite / Excel / CSV or the supplied DF.
        """
        print(f"[DEBUG] _load_data() loading from: {self.csv_path}")
        assert os.path.exists(self.csv_path), f"File does not exist: {self.csv_path}"
        df = pd.read_csv(self.csv_path)
        print(f"[DEBUG] Loaded in _load_data: shape={df.shape}, columns={df.columns.tolist()}")
        return df


    def _load_data_old(self) -> pd.DataFrame:
        """
        Load and merge raw data from SQLite / Excel / CSV or the supplied DF.
        """
        df_clients   = pd.DataFrame()
        df_merchants = pd.DataFrame()
        df = pd.DataFrame()
        conn = None


        if self.data_source == "raw":
            if self.raw_data is None:
                raise ValueError("raw_data=None while data_source='raw'")
            return self.raw_data.copy()

        elif self.data_source == "sqlite" or self.data_source == "xlsx":
            if self.data_source == "sqlite":
                import sqlite3
                conn = sqlite3.connect(self.db_path)
                df_clients   = pd.read_sql("SELECT * FROM clients"      , conn)
                df_merchants = pd.read_sql("SELECT * FROM merchants"    , conn)
                df           = pd.read_sql("SELECT * FROM transactions" , conn)
            elif self.data_source == "xlsx":
                xls          = pd.ExcelFile(self.xlsx_path)
                df_clients   = pd.read_excel(xls, sheet_name="clients")
                df_merchants = pd.read_excel(xls, sheet_name="merchants")
                df           = pd.read_excel(xls, sheet_name="transactions")

            df_clients   = df_clients.rename(columns={"account_creation_date": "account_creation_date_client"})
            df_merchants = df_merchants.rename(columns={"account_creation_date": "account_creation_date_merchant"})
            df = df.merge(df_clients , on="account_id",   how="left")
            df = df.merge(df_merchants, on="merchant_id", how="left")
            
        elif self.data_source == "csv":
            # df_clients   = pd.read_csv(self.csv_path["clients"])
            # df_merchants = pd.read_csv(self.csv_path["merchants"])
            # df_tx        = pd.read_csv(self.csv_path["transactions"])
            if isinstance(self.csv_path, str) and os.path.exists(self.csv_path):
                return pd.read_csv(self.csv_path)
            else:
                raise ValueError(f"Unknown data_source: {self.csv_path}")


        # unify column names
        if self.data_source == "sqlite" and conn is not None:
                    conn.close()
        return (


            df
        )

    # ──────────────────────────────────────────────────────────────────
    # PIPELINE RUNNERS
    # ──────────────────────────────────────────────────────────────────
    def run_all(self) -> None:
        """Execute the core steps needed for a baseline model."""
        self.eda()
        self.feature_engineering()
        self.partitioning()
        self.numeric_conversion()
        self.scaling()
        self.model_baseline()



    # ──────────────────────────────────────────────────────────────────
    # REGISTRY HELPER
    # ──────────────────────────────────────────────────────────────────
    def _register_step(
        self,
        step: str,
        step_dir: str,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        """
        Record artefact paths and push a single line to the global registry
        (spec §7).  Called by individual step modules.
        """
        self.paths[step] = step_dir
        log_registry(step, self.global_hash, cfg or {}, step_dir)

