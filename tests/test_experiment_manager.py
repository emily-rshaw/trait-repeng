# tests/test_experiment_manager.py
import pytest
import os
import sys
import sqlite3
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from experiment_manager import ExperimentManager, main

def test_create_experiment(test_db_path):
    mgr = ExperimentManager(db_path=test_db_path)

    # Attempt to create an experiment
    experiment_id = mgr.create_experiment(
        model_name="test_model",
        quantization_level="fp16",
        trait_id=123,
        trait_max_or_min="max",
        description="Test experiment"
    )
    assert experiment_id is not None

    # Possibly query the DB to ensure the row is inserted
    rows = mgr.cursor.execute("SELECT * FROM experiments").fetchall()
    assert len(rows) == 1
    assert rows[0]["trait_id"] == 123

    mgr.close()


def test_create_run(test_db_path):
    mgr = ExperimentManager(db_path=test_db_path)

    # First create an experiment
    exp_id = mgr.create_experiment(
        model_name="test_model",
        quantization_level="int8",
        trait_id=456,
        trait_max_or_min="min",
        description="Another test"
    )
    # Then create a run
    run_id = mgr.create_run(exp_id, {
        "seed": 325,
        "max_new_tokens": 256,
        "num_samples": 4,
        "max_seq_len": 27,
        "source_layer_idx": 10,
        "target_layer_idx": 20,
        "num_factors": 512,
        "forward_batch_size": 1,
        "backward_batch_size": 1,
        "factor_batch_size": 128,
        "num_eval": 128,
        "system_prompt": "You are a test prompt",
        "dim_output_projection": 32,
        "beta": 1.0,
        "max_iters": 10,
        "target_ratio": 0.5,
        "input_scale": None,
        "run_description": "Test run"
    })
    assert run_id is not None

    # Check the runs table
    rows = mgr.cursor.execute("SELECT * FROM runs").fetchall()
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == exp_id
    assert rows[0]["seed"] == 325

    mgr.close()

def test_create_run_with_target_vector_id(test_db_path):
    mgr = ExperimentManager(db_path=test_db_path)
    exp_id = mgr.create_experiment(
        model_name="test_model", 
        quantization_level="int8", 
        trait_id=456, 
        trait_max_or_min="min", 
        description="Test"
    )
    
    # Create run with target_vector_id
    run_id = mgr.create_run(exp_id, {
        "target_vector_id": 789,
        "seed": 325,
        "max_new_tokens": 256,
        "num_samples": 4,
        "max_seq_len": 27,
        "source_layer_idx": 10,
        "target_layer_idx": 20,
        "num_factors": 512,
        "forward_batch_size": 1,
        "backward_batch_size": 1,
        "factor_batch_size": 128,
        "num_eval": 128,
        "system_prompt": "Test prompt",
        "dim_output_projection": 32,
        "beta": 1.0,
        "max_iters": 10,
        "target_ratio": 0.5,
        "input_scale": None
    })
    
    rows = mgr.cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchall()
    assert rows[0]["target_vector_id"] == 789
    mgr.close()

def test_create_multiple_runs(test_db_path):
    mgr = ExperimentManager(db_path=test_db_path)
    exp_id = mgr.create_experiment(
        model_name="test_model",
        quantization_level="int8",
        trait_id=456,
        trait_max_or_min="min",
        description="Multiple runs test"
    )
    
    run_params_list = []
    for seed in [101, 102]:
        rp = {
            "seed": seed,
            "max_new_tokens": 256,
            "num_samples": 4,
            "max_seq_len": 27,
            "source_layer_idx": 10,
            "target_layer_idx": 20,
            "num_factors": 512,
            "forward_batch_size": 1,
            "backward_batch_size": 1,
            "factor_batch_size": 128,
            "num_eval": 128,
            "system_prompt": "You are a test",
            "dim_output_projection": 32,
            "beta": 1.0,
            "max_iters": 10,
            "target_ratio": 0.5,
            "input_scale": None,
            "run_description": f"Test run with seed={seed}"
        }
        run_params_list.append(rp)
    
    run_ids = mgr.create_multiple_runs(exp_id, run_params_list)
    
    assert len(run_ids) == 2
    rows = mgr.cursor.execute("SELECT * FROM runs").fetchall()
    assert len(rows) == 2
    assert rows[0]["seed"] == 101
    assert rows[1]["seed"] == 102
    
    mgr.close()

@patch('experiment_manager.run_dct')
def test_execute_run(mock_run_dct, test_db_path):
    mock_run_dct.return_value = "Success"
    
    mgr = ExperimentManager(db_path=test_db_path)
    exp_id = mgr.create_experiment(
        model_name="test_model",
        quantization_level="int8",
        trait_id=456,
        trait_max_or_min="min",
        description="Execute run test"
    )
    
    run_id = mgr.create_run(exp_id, {
        "seed": 325,
        "max_new_tokens": 256,
        "num_samples": 4,
        "max_seq_len": 27,
        "source_layer_idx": 10,
        "target_layer_idx": 20,
        "num_factors": 512,
        "forward_batch_size": 1,
        "backward_batch_size": 1,
        "factor_batch_size": 128,
        "num_eval": 128,
        "system_prompt": "Test prompt",
        "dim_output_projection": 32,
        "beta": 1.0,
        "max_iters": 10,
        "target_ratio": 0.5,
        "input_scale": None,
        "run_description": "Test execution"
    })
    
    mgr.execute_run(run_id)
    
    # The function uses DB_PATH constant from experiment_manager, not the test_db_path
    mock_run_dct.assert_called_once_with(run_id, db_path="results/database/experiments.db")
    mgr.close()

def test_delete_experiment(test_db_path):
    mgr = ExperimentManager(db_path=test_db_path)
    
    # Create experiment and runs
    exp_id = mgr.create_experiment(
        model_name="test_model",
        quantization_level="int8",
        trait_id=456,
        trait_max_or_min="min",
        description="Delete test"
    )
    
    run_id = mgr.create_run(exp_id, {
        "seed": 325,
        "max_new_tokens": 256,
        "num_samples": 4,
        "max_seq_len": 27,
        "source_layer_idx": 10,
        "target_layer_idx": 20,
        "num_factors": 512,
        "forward_batch_size": 1,
        "backward_batch_size": 1,
        "factor_batch_size": 128,
        "num_eval": 128,
        "system_prompt": "Test prompt",
        "dim_output_projection": 32,
        "beta": 1.0,
        "max_iters": 10,
        "target_ratio": 0.5,
        "input_scale": None,
        "run_description": "Will be deleted"
    })
    
    # Verify data exists
    exp_rows = mgr.cursor.execute("SELECT * FROM experiments WHERE experiment_id = ?", (exp_id,)).fetchall()
    run_rows = mgr.cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchall()
    assert len(exp_rows) == 1
    assert len(run_rows) == 1
    
    # Delete experiment
    mgr.delete_experiment(exp_id)
    
    # Verify data is deleted
    exp_rows = mgr.cursor.execute("SELECT * FROM experiments WHERE experiment_id = ?", (exp_id,)).fetchall()
    run_rows = mgr.cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchall()
    assert len(exp_rows) == 0
    assert len(run_rows) == 0
    
    mgr.close()

def test_delete_run(test_db_path):
    mgr = ExperimentManager(db_path=test_db_path)
    
    # Create experiment and run
    exp_id = mgr.create_experiment(
        model_name="test_model",
        quantization_level="int8",
        trait_id=456,
        trait_max_or_min="min",
        description="Delete run test"
    )
    
    run_id = mgr.create_run(exp_id, {
        "seed": 325,
        "max_new_tokens": 256,
        "num_samples": 4,
        "max_seq_len": 27,
        "source_layer_idx": 10,
        "target_layer_idx": 20,
        "num_factors": 512,
        "forward_batch_size": 1,
        "backward_batch_size": 1,
        "factor_batch_size": 128,
        "num_eval": 128,
        "system_prompt": "Test prompt",
        "dim_output_projection": 32,
        "beta": 1.0,
        "max_iters": 10,
        "target_ratio": 0.5,
        "input_scale": None,
        "run_description": "Will be deleted"
    })
    
    # Verify run exists
    run_rows = mgr.cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchall()
    assert len(run_rows) == 1
    
    # Delete run
    mgr.delete_run(run_id)
    
    # Verify run is deleted
    run_rows = mgr.cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchall()
    assert len(run_rows) == 0
    
    # Experiment should still exist
    exp_rows = mgr.cursor.execute("SELECT * FROM experiments WHERE experiment_id = ?", (exp_id,)).fetchall()
    assert len(exp_rows) == 1
    
    mgr.close()

@patch('experiment_manager.ExperimentManager.execute_run')
@patch('experiment_manager.ExperimentManager.create_multiple_runs')
@patch('experiment_manager.ExperimentManager.create_experiment')
def test_main(mock_create_experiment, mock_create_multiple_runs, mock_execute_run):
    # Setup mocks
    mock_create_experiment.return_value = 999
    mock_create_multiple_runs.return_value = [1001, 1002]
    
    # Run the main function
    main()
    
    # Verify calls
    mock_create_experiment.assert_called_once()
    mock_create_multiple_runs.assert_called_once()
    assert mock_execute_run.call_count == 2
    mock_execute_run.assert_any_call(1001)
    mock_execute_run.assert_any_call(1002)

@patch('experiment_manager.main')
def test_run_main_if_module_is_main(mock_main):
    """Test the function that handles the entry point logic."""
    # Import the module
    from experiment_manager import run_main_if_module_is_main
    
    # Save original __name__
    import sys
    original_name = sys.modules["experiment_manager"].__name__
    
    try:
        # Set __name__ to "__main__" to trigger the condition
        sys.modules["experiment_manager"].__name__ = "__main__"
        
        # Call the function
        run_main_if_module_is_main()
        
        # Verify main was called
        mock_main.assert_called_once()
    finally:
        # Restore original name
        sys.modules["experiment_manager"].__name__ = original_name