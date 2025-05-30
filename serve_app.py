# serve_app.py  – Streamlit front‑end for the SPEC‑compliant pipeline
from __future__ import annotations

import json

from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st
import os
from ml_pipeline.base import MLPipeline

ARTIFACTS_ROOT = Path("artifacts")


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
import streamlit as st

def check_password():
    def password_entered():
        if st.session_state["password"] == "demo123":
            st.session_state["authenticated"] = True
        else:
            st.session_state["authenticated"] = False
            st.error("❌ Incorrect password")

    if "authenticated" not in st.session_state:
        st.text_input("Enter password", type="password", on_change=password_entered, key="password")
        st.stop()
    elif not st.session_state["authenticated"]:
        st.text_input("Enter password", type="password", on_change=password_entered, key="password")
        st.stop()




def discover_trained_runs(root: Path) -> List[str]:
    """
    Return all run‑hashes that contain a trained baseline model.
    Expected location: artifacts/run_<hash>/model_baseline/model.json
    """
    hashes: list[str] = []
    if not root.exists():
        return hashes

    for run_dir in root.iterdir():
        if run_dir.is_dir() and run_dir.name.startswith("run_"):
            if (run_dir / "model_baseline" / "model.json").exists():
                hashes.append(run_dir.name.replace("run_", "", 1))
    return sorted(hashes)


def load_feature_names(train_hash: str) -> List[str]:
    fn_file = ARTIFACTS_ROOT / f"run_{train_hash}" / "model_baseline" / "feature_names.json"
    if fn_file.exists():
        with open(fn_file, encoding="utf-8") as fh:
            return json.load(fh)["feature_names"]
    return []


import os
import pandas as pd

def read_uploaded(uploaded_file):
    if uploaded_file is None:
        raise ValueError("No file uploaded.")

    uploaded_file.seek(0)  # Reset pointer before reading

    # Save to known location inside repo
    os.makedirs("uploaded_data", exist_ok=True)
    save_path = os.path.join("uploaded_data", uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

    if os.path.getsize(save_path) == 0:
        raise ValueError(f"File was saved as {save_path}, but it's empty.")


    uploaded_file.seek(0)
    contents = uploaded_file.read()

    print(f"[DEBUG] Length of uploaded_file.read(): {len(contents)} bytes")
    print(f"[DEBUG] Preview of contents:\n{contents[:200]!r}")

    # Write to disk
    os.makedirs("uploaded_data", exist_ok=True)
    save_path = os.path.join("uploaded_data", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(contents)

    # Verify file written
    print(f"[DEBUG] File written to: {save_path} ({os.path.getsize(save_path)} bytes)")

    df = pd.read_csv(save_path)
    print(f"[DEBUG] DataFrame shape: {df.shape}")
    print(f"[DEBUG] DataFrame columns: {df.columns.tolist()}")

    if df.empty or df.columns.size == 0:
        raise ValueError(f"CSV at {save_path} has no usable columns.")

    return df, save_path



def read_uploaded_old(file) -> pd.DataFrame | None:
    """Accept CSV or Excel (clients / merchants / transactions)."""
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file), "csv"
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        try:
            xls = pd.ExcelFile(file)
            df_clients = pd.read_excel(xls, "clients")
            df_merchants = pd.read_excel(xls, "merchants")
            df_tx = pd.read_excel(xls, "transactions")
        except Exception as exc:
            st.error(f"Excel reading error: {exc}")
            return None

        return (
            df_tx.merge(df_clients, on="account_id", how="left")
                .merge(df_merchants, on="merchant_id", how="left")
        ), "excel"
    elif name.endswith("sqlite"):
        pass # TODO: implement SQLite reading
    else:
        # st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return file, "raw"  # for testing only


def build_inference_cfg(train_hash: str, feature_names: list[str], upl: any) -> dict:
    """Minimal config that satisfies MLPipeline hashing rules in inference."""
    df_raw, csv_path = read_uploaded(upl)
    return {
        "train_mode": False,
        "train_hash": train_hash,
        "model_name": "deployed_baseline",
        "model_hash": train_hash,
        "dataset_name": "uploaded_ds",
        "feature_names": feature_names or ["dummy"],
        "target_col": "fraud_bool",  # "is_fraud",
        "id_col": "account_id",  # "transaction_id",
        "use_mlflow": False,
        "csv_path": csv_path,
    }


# ──────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    st.title("Fraud‑Prediction Demo")

    # 1️⃣  Select a trained run
    st.header("1. Pick a trained model run")
    runs = discover_trained_runs(ARTIFACTS_ROOT)
    if not runs:
        st.error("No trained baseline models found under artifacts/run_*")
        return

    chosen_hash = st.selectbox("Run hash", runs)
    feats = load_feature_names(chosen_hash)
    st.caption(f"{len(feats)} feature columns found for run `{chosen_hash}`.")

    # 2️⃣  Upload data
    st.header("2. Upload new transactions")
    upl = st.file_uploader("CSV or Excel (with sheets clients / merchants / transactions)",
                           type=["csv", "xlsx"])

    if upl is None:
        st.info("Waiting for a file…")
        return

    df_raw, _ = read_uploaded(upl)
    # if df_raw is None:
        # return

    st.dataframe(df_raw.head(), use_container_width=True)

    # 3️⃣  Run pipeline
    st.header("3. Run inference")
    if st.button("Start prediction"):
        with st.spinner("Running pipeline…"):
            cfg = build_inference_cfg(chosen_hash, feats, upl)
            pipe = MLPipeline(cfg)  #, data_source=type, raw_data=df_raw)
            pipe.dataframes["init"]["raw"] = df_raw          # <- rende disponibile il raw DF

            # run_all works in inference: each step reuses artefacts
            pipe.run_all()
            
            # scored_df = pipe.dataframes.get("test_sca")
            # if scored_df is None:
                # scored_df = pipe.dataframes.get("raw")
            # the most transformed DF available is test_sca (scaling step)
            scored_df = pipe.dataframes["scaling"].get("test_sca") # or pipe.dataframes.get("raw")
            if scored_df is None:
                # something went wrong upstream – stop early and help the user
                raise RuntimeError(
                    "Scaling step did not create 'test_sca'. "
                    "Check that numeric_conversion() and scaling() ran in inference mode."
                )            
            preds = pipe.train_models["model_baseline"]["baseline"].predict_proba(
                scored_df[feats])[:, 1]
            scored_df = scored_df.copy()
            scored_df["fraud_score"] = preds

        st.success("Done!")
        st.dataframe(scored_df.head(), use_container_width=True)
        
        # 4️⃣  Download
        st.header("4. Download results")
        st.download_button(
            label="Download CSV",
            data=scored_df.to_csv(index=False).encode(),
            file_name="fraud_scores.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    check_password()
    main()
    