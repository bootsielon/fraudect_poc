import os
import shutil

def cleanup_artifacts(artifact_dirs, db_path):
    """
    Remove specified artifact directories and database file to reset the environment.

    Args:
        artifact_dirs (list): List of directories to clean up.
        db_path (str): Path to the database file to remove.
    """
    for dir_path in artifact_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"Removed: {dir_path}")
        else:
            print(f"Directory not found, skipping: {dir_path}")

    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed database: {db_path}")
    else:
        print(f"Database file not found, skipping: {db_path}")

if __name__ == "__main__":
    # List of artifact directories to clean
    artifact_dirs = [
        "mlruns",
        "artifacts",
        "artifacts/eda",
        "artifacts/step1",
        "artifacts/step2",
        "artifacts/step3",
        "artifacts/step4",
        "artifacts/step5",
        "artifacts/step6",
        "artifacts/step7",
        "artifacts/step8",
        "artifacts/step9",
        "artifacts/step10",
        "artifacts/step11",
        "artifacts/step12",    ]
    # Path to the generated database
    db_path = "fraud_poc.db"
    cleanup_artifacts(artifact_dirs, db_path)
