import pytest
import pandas as pd
import numpy as np
import os
import json
import shutil
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, mock_open
from ..feature_engineering import feature_engineering

# Relative import for the function to test


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline object with necessary attributes for testing"""
    mock = MagicMock()
    mock.train_mode = True
    mock.config = {"target_col": "is_fraud", "use_mlflow": False}
    mock.dataframes = {}
    mock.paths = {}
    mock.hashes = {}
    mock.artifacts = {}
    return mock


@pytest.fixture
def sample_data():
    """Create sample data for testing feature engineering"""
    now = datetime.now()
    data = pd.DataFrame({
        "client_id": [1, 2, 3, 4, 5],
        "merchant_id": [101, 102, 103, 101, 102],
        "amount": [100.0, 50.0, 200.0, 75.0, 125.0],
        "is_fraud": [0, 0, 1, 0, 1],
        "timestamp": [
            now - timedelta(days=1),
            now - timedelta(days=2),
            now - timedelta(days=3),
            now - timedelta(days=4),
            now - timedelta(days=5)
        ],
        "account_creation_date_client": [
            now - timedelta(days=100),
            now - timedelta(days=200),
            now - timedelta(days=300),
            now - timedelta(days=400),
            now - timedelta(days=500)
        ],
        "account_creation_date_merchant": [
            now - timedelta(days=1000),
            now - timedelta(days=1200),
            now - timedelta(days=1300),
            now - timedelta(days=1400),
            now - timedelta(days=1500)
        ],
        "constant_col": [1, 1, 1, 1, 1]  # This should be identified and dropped
    })
    
    # Convert datetime columns to string format similar to a database
    for dt_col in ['timestamp', 'account_creation_date_client', 'account_creation_date_merchant']:
        data[dt_col] = data[dt_col].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return data


@pytest.fixture(autouse=True)
def setup_teardown():
    """Setup and teardown for each test - create and cleanup artifacts directory"""
    os.makedirs("artifacts", exist_ok=True)
    yield
    if os.path.exists("artifacts"):
        shutil.rmtree("artifacts")


def test_basic_feature_engineering_train_mode(mock_pipeline, sample_data):
    """Test basic feature engineering in training mode"""
    # Setup
    mock_pipeline.dataframes["raw"] = sample_data.copy()
    
    # Execute
    feature_engineering(mock_pipeline)
    
    # Verify
    assert "feature_engineered" in mock_pipeline.dataframes
    df_result = mock_pipeline.dataframes["feature_engineered"]
    
    # Check that new features were created
    assert "transaction_hour" in df_result.columns
    assert "transaction_dayofweek" in df_result.columns
    assert "account_creation_date_client_age_days" in df_result.columns
    assert "account_creation_date_merchant_age_days" in df_result.columns
    
    # Check that constant columns were dropped
    assert "constant_col" not in df_result.columns


@patch("os.path.exists")
def test_checkpoint_loading(mock_exists, mock_pipeline, sample_data):
    """Test loading from checkpoint when manifest file exists"""
    # Setup
    mock_exists.return_value = True
    mock_pipeline.dataframes["raw"] = sample_data.copy()
    
    # Mock file operations
    mock_manifest = {
        "outputs": {
            "dropped_features": ["constant_col"],
            "engineered_csv": "path/to/csv.csv"
        }
    }
    
    with patch("builtins.open", mock_open(read_data=json.dumps(mock_manifest))):
        with patch("pandas.read_csv", return_value=pd.DataFrame({"mock": [1, 2, 3]})):
            # Execute
            feature_engineering(mock_pipeline)
    
    # Verify
    assert "feature_engineered" in mock_pipeline.dataframes
    assert mock_pipeline.artifacts["feature_engineering"] is not None


@patch("mlflow.start_run")
@patch("mlflow.log_artifacts")
@patch("mlflow.set_tags")
def test_mlflow_logging(mock_set_tags, mock_log_artifacts, mock_start_run, mock_pipeline, sample_data):
    """Test MLflow logging when enabled"""
    # Setup
    mock_pipeline.dataframes["raw"] = sample_data.copy()
    mock_pipeline.config["use_mlflow"] = True
    
    # Execute
    feature_engineering(mock_pipeline)
    
    # Verify MLflow was called
    mock_start_run.assert_called_once()
    mock_log_artifacts.assert_called_once()
    mock_set_tags.assert_called_once()


def test_timestamp_feature_creation(mock_pipeline, sample_data):
    """Test specific timestamp-derived features"""
    # Setup
    mock_pipeline.dataframes["raw"] = sample_data.copy()
    
    # Execute
    feature_engineering(mock_pipeline)
    
    # Verify timestamp features
    df_result = mock_pipeline.dataframes["feature_engineered"]
    
    # Test the timestamp column was processed correctly
    assert "transaction_hour" in df_result.columns
    assert "transaction_dayofweek" in df_result.columns
    
    # Test account creation date features
    for col_prefix in ["account_creation_date_client", "account_creation_date_merchant"]:
        assert f"{col_prefix}_year" in df_result.columns
        assert f"{col_prefix}_month" in df_result.columns
        assert f"{col_prefix}_day" in df_result.columns
        assert f"{col_prefix}_hour" in df_result.columns
        assert f"{col_prefix}_dayofweek" in df_result.columns
        assert f"{col_prefix}_age_days" in df_result.columns
        assert f"{col_prefix}_age_years" in df_result.columns


def test_constant_column_dropping(mock_pipeline):
    """Test that columns with a single unique value are dropped"""
    # Setup with multiple constant columns
    mock_pipeline.train_mode = True
    mock_pipeline.dataframes["raw"] = pd.DataFrame({
        "normal_col": [1, 2, 3, 4, 5],
        "constant_col1": [7, 7, 7, 7, 7],
        "constant_col2": ["same", "same", "same", "same", "same"],
        "low_variance_col": [1, 1, 1, 1, 2]
    })
    
    # Execute
    feature_engineering(mock_pipeline)
    
    # Verify
    df_result = mock_pipeline.dataframes["feature_engineered"]
    assert "normal_col" in df_result.columns
    assert "constant_col1" not in df_result.columns
    assert "constant_col2" not in df_result.columns
    assert "low_variance_col" in df_result.columns  # Should be kept as it has > 1 unique values


def test_inference_mode(mock_pipeline, sample_data):
    """Test feature engineering in inference mode"""
    # Setup
    mock_pipeline.train_mode = False
    mock_pipeline.config["train_hash"] = "test_hash"
    mock_pipeline.dataframes["raw"] = sample_data.copy()
    
    # Mock existing manifest with dropped features
    manifest_data = {
        "outputs": {
            "dropped_features": ["constant_col", "another_col"],
            "engineered_csv": "path/to/csv"
        }
    }
    
    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = True
        with patch("builtins.open", mock_open(read_data=json.dumps(manifest_data))):
            with patch("pandas.read_csv", return_value=pd.DataFrame()):
                # Execute
                feature_engineering(mock_pipeline)
    
    # Verify paths and hashes are set correctly
    assert "feature_engineering" in mock_pipeline.paths
    assert "feature_engineering" in mock_pipeline.hashes
    def test_file_output_structure(mock_pipeline, sample_data):
        """Test that the output file structure is correctly created"""
        mock_pipeline.dataframes["raw"] = sample_data.copy()
        
        # Execute
        feature_engineering(mock_pipeline)
        
        # Verify files were created
        step_dir = mock_pipeline.paths["feature_engineering"]
        assert os.path.exists(step_dir)
        assert os.path.exists(os.path.join(step_dir, "manifest.json"))
        assert os.path.exists(os.path.join(step_dir, f"feature_engineering_{mock_pipeline.hashes['feature_engineering']}.csv"))
        assert os.path.exists(os.path.join(step_dir, f"dropped_features_{mock_pipeline.hashes['feature_engineering']}.json"))


    def test_empty_dataframe_handling(mock_pipeline):
        """Test feature engineering with an empty dataframe"""
        mock_pipeline.dataframes["raw"] = pd.DataFrame()
        
        # Execute
        feature_engineering(mock_pipeline)
        
        # Verify
        assert "feature_engineered" in mock_pipeline.dataframes
        assert mock_pipeline.dataframes["feature_engineered"].empty
        assert "feature_engineering" in mock_pipeline.paths
        assert "feature_engineering" in mock_pipeline.hashes


    @patch("ml_pipeline.utils.log_registry")
    def test_log_registry_integration(mock_log_registry, mock_pipeline, sample_data):
        """Test that log_registry is called with correct parameters"""
        mock_pipeline.dataframes["raw"] = sample_data.copy()
        
        # Execute
        feature_engineering(mock_pipeline)
        
        # Verify
        mock_log_registry.assert_called_once()
        args = mock_log_registry.call_args[0]
        assert args[0] == "feature_engineering"  # step
        assert isinstance(args[1], str)  # hash
        assert isinstance(args[2], dict)  # config
        assert os.path.exists(args[3])  # output_dir


    @pytest.mark.parametrize("date_format", [
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%m/%d/%Y %H:%M"
    ])
    def test_different_date_formats(mock_pipeline, date_format):
        """Test feature engineering with different date formats"""
        now = datetime.now()
        
        # Create data with specified date format
        data = pd.DataFrame({
            "client_id": [1, 2, 3],
            "timestamp": [
                (now - timedelta(days=i)).strftime(date_format) 
                for i in range(3)
            ],
            "account_creation_date_client": [
                (now - timedelta(days=i*30)).strftime(date_format)
                for i in range(3)
            ],
            "is_fraud": [0, 1, 0]
        })
        
        mock_pipeline.dataframes["raw"] = data
        
        # Execute - should handle different date formats gracefully
        feature_engineering(mock_pipeline)
        
        # Verify new features were created
        df_result = mock_pipeline.dataframes["feature_engineered"]
        assert "transaction_hour" in df_result.columns
        assert "account_creation_date_client_year" in df_result.columns


    def test_missing_date_values(mock_pipeline):
        """Test feature engineering with missing date values"""
        data = pd.DataFrame({
            "client_id": [1, 2, 3, 4],
            "timestamp": ["2023-01-01 12:30:45", "2023-01-02 13:20:15", None, "2023-01-04 10:15:30"],
            "account_creation_date_client": ["2022-06-01 09:00:00", None, "2022-08-15 14:30:00", None],
            "amount": [100.0, 200.0, 150.0, 300.0],
            "is_fraud": [0, 1, 0, 1]
        })
        
        mock_pipeline.dataframes["raw"] = data
        
        # Execute - should handle missing dates without crashing
        feature_engineering(mock_pipeline)
        
        # Verify
        df_result = mock_pipeline.dataframes["feature_engineered"]
        assert "transaction_hour" in df_result.columns
        assert "account_creation_date_client_year" in df_result.columns
        
        # Check that NaN values in timestamp column result in NaN derived features
        assert df_result["transaction_hour"].isna().sum() > 0


    def test_hash_consistency(mock_pipeline, sample_data):
        """Test that the same data produces the same hash"""
        mock_pipeline.dataframes["raw"] = sample_data.copy()
        
        # First run
        feature_engineering(mock_pipeline)
        first_hash = mock_pipeline.hashes["feature_engineering"]
        
        # Clean up
        del mock_pipeline.dataframes["feature_engineered"]
        del mock_pipeline.paths["feature_engineering"]
        del mock_pipeline.hashes["feature_engineering"]
        
        # Second run with the same data
        mock_pipeline.dataframes["raw"] = sample_data.copy()
        feature_engineering(mock_pipeline)
        second_hash = mock_pipeline.hashes["feature_engineering"]
        
        # Hashes should be the same for identical inputs
        assert first_hash == second_hash


    def test_no_train_hash_in_inference_mode():
        """Test error handling when train_hash is not provided in inference mode"""
        mock_pipeline = MagicMock()
        mock_pipeline.train_mode = False
        mock_pipeline.config = {"target_col": "is_fraud", "use_mlflow": False}
        # Intentionally omit the train_hash
        mock_pipeline.dataframes = {"raw": pd.DataFrame({"a": [1, 2]})}
        mock_pipeline.paths = {}
        mock_pipeline.hashes = {}
        mock_pipeline.artifacts = {}
        
        # Should raise KeyError when train_hash is missing in inference mode
        with pytest.raises(KeyError):
            feature_engineering(mock_pipeline)


    @patch("os.path.exists")
    def test_inference_no_manifest(mock_exists, mock_pipeline, sample_data):
        """Test assertion when no manifest file exists for inference"""
        mock_pipeline.train_mode = False
        mock_pipeline.config = {"train_hash": "test_hash", "use_mlflow": False}
        mock_pipeline.dataframes["raw"] = sample_data.copy()
        
        # Mock file existence check to return False
        mock_exists.return_value = False
        
        # Should assert when manifest file doesn't exist
        with pytest.raises(AssertionError) as excinfo:
            feature_engineering(mock_pipeline)
        
        # Check error message
        assert "No manifest file found for inference" in str(excinfo.value)


    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_convert_numpy_types_called(mock_json_dump, mock_file_open, mock_pipeline, sample_data):
        """Test that numpy types are properly converted before JSON serialization"""
        mock_pipeline.dataframes["raw"] = sample_data.copy()
        
        with patch("ml_pipeline.utils.convert_numpy_types", side_effect=lambda x: x) as mock_convert:
            feature_engineering(mock_pipeline)
            
            # verify convert_numpy_types was called at least twice
            # (once for manifest and once for config)
            assert mock_convert.call_count >= 2


    @pytest.mark.parametrize("use_mlflow", [True, False])
    def test_mlflow_config_parameter(use_mlflow, mock_pipeline, sample_data):
        """Test behavior with different 'use_mlflow' configurations"""
        mock_pipeline.dataframes["raw"] = sample_data.copy()
        mock_pipeline.config["use_mlflow"] = use_mlflow
        
        with patch("mlflow.start_run") as mock_start_run:
            feature_engineering(mock_pipeline)
            
            if use_mlflow:
                mock_start_run.assert_called_once()
            else:
                mock_start_run.assert_not_called()
                def test_malformed_date_handling(mock_pipeline):
                    """Test how feature engineering handles malformed date entries"""
                    # Setup with some malformed dates
                    data = pd.DataFrame({
                        "client_id": [1, 2, 3, 4],
                        "timestamp": ["2023-01-01", "invalid-date", "2023/01/03", None],
                        "account_creation_date_client": ["2022-06-01", "not-a-date", None, "2022-01-01"],
                        "amount": [100.0, 200.0, 150.0, 300.0],
                        "is_fraud": [0, 1, 0, 1]
                    })
                    
                    mock_pipeline.dataframes["raw"] = data
                    
                    # Execute - should handle malformed dates without crashing
                    feature_engineering(mock_pipeline)
                    
                    # Verify execution completes and results are reasonable