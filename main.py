from ml_pipeline.base import MLPipeline
from ml_pipeline.utils import make_param_hash
from cleanup import cleanup_artifacts
from gen_data import gen_data

def main() -> None:
    artifact_dirs = [
        "mlruns",
        "artifacts",  
    ]
    # Path to the generated database
    db_path = "fraud_poc.db"
    csv_path = "data/kagglebankfraud/Base.csv"
    configs = [
        {
            # Step 0
            "cleanup_artifacts": False,
            #"db_path": db_path, #"fraud_poc.db",
            # "gen_data": True,
            # "n": 10000,
            "data_source": "csv",
            "load_data": True,
            "load_data_path": "data/kagglebankfraud/Base.csv",  # "fraud_poc.db",
            "csv_path": csv_path,


            # Step 1
            "seed": 400,  # 42,
            "target_col": "fraud_bool",# "is_fraud",
            "id_col": "account_id",  # "transaction_id",
            "use_mlflow": True,
            
            "train_size": 0.7,
            "test_size": 0.2,
            "val_size": 0.1,
            "use_stratification": False,
            "use_downsampling": True,
            # "stratify_cols": ["account_id", "merchant_id"],
            "stratify_cardinality_threshold": 5,

            # Step 3
            "c1": 10,
            "c2": 0.01,
            "b1": True,
            "c3": 5,

            # Step 4
            "s1": False,
            "t1": False,

            # Step 8
            "shap_cutoff": 1 - 1e-5,  # 0.95,

            # Step 10
            "use_cluster_select": [True, False],
            "save_fs_mods": False,

            # Step 11
            "k_folds": 5,
            "opt_metric": "f1",
            "minimize_metric": False,

            # Step 12: Baseline hyperparameters
            "baseline_hyperparams": {
                "n_estimators": 320,
                "max_depth": 4,
                "learning_rate": 0.1,
                "subsample": 0.5,
                "colsample_bytree": 0.5,
                "use_label_encoder": False,
                "eval_metric": "logloss"
            }
        },
        # Add more config dicts here
    ]

    for cfg in configs:
        param_hash = make_param_hash(cfg)
        print(f"\n=== Running pipeline for hash: {param_hash} ===")
        pipeline = MLPipeline(cfg)

        if cfg["cleanup_artifacts"]:
            # Clean up artifacts and database
            cleanup_artifacts(artifact_dirs, db_path)
        if not cfg["load_data"]:
            # Generate new data
            gen_data(n=10000, random_seed=402)
        else:
            # Load data from the specified path
            print(f"Loading data from {cfg['load_data_path']}")
            # Here you would load your data into the pipeline or set it up as needed
            # For example, you might want to load a DataFrame or a database connection

            # data = load_data(cfg['load_data_path'])
            #pipeline.set_data(data)
            pipeline.load_data()
        pipeline.run_all()


if __name__ == "__main__":
    # List of artifact directories to clean
    main()