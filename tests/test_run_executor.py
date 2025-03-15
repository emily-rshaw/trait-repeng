# tests/test_run_executor.py
import pytest
import os
import sys
import sqlite3
from unittest.mock import patch, MagicMock, ANY
import numpy as np
import torch

# Add scripts directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from run_executor import (
    execute_run,
    select_trait_dataset,
    load_examples,
    prepare_model_and_tokenizer,
    extract_after_assistant
)

class TestRunExecutorUtilities:
    """Tests for the utility functions in run_executor.py"""

    def test_select_trait_dataset(self, tmp_path):
        # Create a temporary traits directory with test files
        traits_dir = tmp_path / "traits"
        traits_dir.mkdir()
        
        # Create test files
        (traits_dir / "max_extraversion.csv").write_text("test data")
        (traits_dir / "min_extraversion.csv").write_text("test data")
        
        # Test valid case
        result = select_trait_dataset(str(traits_dir), "extraversion", "max")
        assert "max_extraversion.csv" in result
        
        # Test invalid trait name
        with pytest.raises(ValueError, match="No dataset found for trait"):
            select_trait_dataset(str(traits_dir), "nonexistent", "max")
        
        # Test invalid direction
        with pytest.raises(ValueError, match="No dataset found for trait"):
            select_trait_dataset(str(traits_dir), "extraversion", "invalid")
            
        # Test nonexistent directory
        with pytest.raises(ValueError, match="does not exist"):
            select_trait_dataset("/nonexistent/path", "extraversion", "max")

    def test_extract_after_assistant(self):
        # Test basic extraction
        text = "user: What's the weather?\nassistant: It's sunny"
        result = extract_after_assistant(text)
        expected = "It's sunny"
        assert result == expected
        
        # Test with capitalized "Assistant"
        text = "user: What's the weather?\nAssistant: It's sunny"
        result = extract_after_assistant(text)
        expected = "It's sunny"
        assert result == expected
        
        # Test with no assistant in text
        text = "This text has no mention of anything else"
        result = extract_after_assistant(text)
        assert result == text  # Should return the original text
        
        # Test with assistant at the beginning
        text = "assistant: This is the full response"
        result = extract_after_assistant(text)
        expected = "This is the full response"
        assert result == expected


@pytest.fixture
def mock_db_setup(test_db_path):
    """Set up a test database with minimal required data for testing"""
    conn = sqlite3.connect(test_db_path)
    cursor = conn.cursor()
    
    # Insert a test trait
    cursor.execute("""
        INSERT INTO traits (trait_id, trait_name, trait_description)
        VALUES (1, 'trust', 'Tendency to trust others')
    """)
    
    # Insert a test experiment
    cursor.execute("""
        INSERT INTO experiments (
            experiment_id, date_time, model_name, quantization_level, 
            trait_id, trait_max_or_min, description
        ) VALUES (1, '2025-03-15T12:00:00', 'test_model', 'fp16', 1, 'max', 'Test experiment')
    """)
    
    # Insert a test run with input_scale=NULL so calibration function will be called
    cursor.execute("""
        INSERT INTO runs (
            run_id, experiment_id, seed, max_new_tokens, num_samples, 
            max_seq_len, source_layer_idx, target_layer_idx, num_factors,
            forward_batch_size, backward_batch_size, factor_batch_size,
            num_eval, system_prompt, dim_output_projection,
            beta, max_iters, target_ratio, input_scale, run_description
        ) VALUES (
            1, 1, 42, 20, 2, 15, 1, 2, 10, 
            1, 1, 2, 2, 'Test system prompt', 8,
            1.0, 5, 0.5, NULL, 'Test run'
        )
    """)
    
    conn.commit()
    conn.close()
    
    return test_db_path


class TestRunExecutor:
    """Tests for the main execute_run function"""
    
    @patch('run_executor.prepare_model_and_tokenizer')
    @patch('run_executor.load_examples')
    @patch('run_executor.create_sliced_model')
    @patch('run_executor.compute_activations')
    @patch('run_executor.dct.DeltaActivations')
    @patch('run_executor.dct.SteeringCalibrator')
    @patch('run_executor.train_dct_model')
    @patch('run_executor.dct.SlicedModel')
    @patch('run_executor.rank_vectors')
    @patch('run_executor.dct.ModelEditor')
    @patch('run_executor.evaluate_vectors_and_capture')
    @patch('torch.manual_seed')
    @patch('torch.set_default_device')
    @patch('torch.cuda.is_available')
    def test_execute_run(
        self, mock_cuda_available, mock_set_default_device, mock_manual_seed, 
        mock_evaluate, mock_model_editor, mock_rank_vectors, 
        mock_sliced_model, mock_train_dct, mock_calibrator, 
        mock_delta_acts, mock_compute_activations, mock_create_sliced,
        mock_load_examples, mock_prepare_model, mock_db_setup
    ):
        # Mock torch CUDA usage
        mock_cuda_available.return_value = False
        
        # Set up mocks
        mock_model = MagicMock()
        mock_model.config.hidden_size = 768
        mock_model.config.num_hidden_layers = 12
        
        mock_tokenizer = MagicMock()
        mock_prepare_model.return_value = (mock_model, mock_tokenizer)
        
        mock_examples = [MagicMock() for _ in range(4)]
        mock_targets = [MagicMock() for _ in range(4)]
        mock_test_examples = [MagicMock() for _ in range(2)]
        mock_test_targets = [MagicMock() for _ in range(2)]
        mock_load_examples.return_value = (mock_examples, mock_targets, mock_test_examples, mock_test_targets)
        
        mock_sliced = MagicMock()
        mock_create_sliced.return_value = mock_sliced
        
        mock_X = torch.zeros(2, 15, 768)
        mock_Y = torch.zeros(2, 15, 768)
        mock_compute_activations.return_value = (mock_X, mock_Y)
        
        mock_delta_acts_inst = MagicMock()
        mock_delta_acts.return_value = mock_delta_acts_inst
        
        mock_calibrator_inst = MagicMock()
        mock_calibrator_inst.calibrate.return_value = 2.0
        mock_calibrator.return_value = mock_calibrator_inst
        
        mock_exp_dct = MagicMock()
        mock_U = torch.zeros(10, 10)
        mock_V = torch.zeros(10, 10)
        mock_train_dct.return_value = (mock_exp_dct, mock_U, mock_V)
        
        mock_delta_acts_end = MagicMock()
        mock_sliced_model.return_value = mock_delta_acts_end
        
        mock_scores = torch.tensor([0.9, 0.8])
        mock_indices = torch.tensor([5, 3])
        mock_rank_vectors.return_value = (mock_scores, mock_indices)
        
        mock_editor_inst = MagicMock()
        mock_model_editor.return_value = mock_editor_inst
        
        mock_eval_results = {
            "unsteered": [("prompt1", "assistant: response1"), ("prompt2", "assistant: response2")],
            "steered": {
                5: [("prompt1", "assistant: steered1"), ("prompt2", "assistant: steered2")],
                3: [("prompt1", "assistant: steered3"), ("prompt2", "assistant: steered4")]
            }
        }
        mock_evaluate.return_value = mock_eval_results
        
        # Execute the test with multiple patches to avoid CUDA calls
        with patch('torch.Tensor.cuda', return_value=torch.Tensor()):
            with patch('torch.zeros', return_value=torch.zeros((2, 2))):
                with patch('run_executor.vmap', MagicMock()):
                    # Call execute_run with the mocked database
                    result = execute_run(1, db_path=mock_db_setup)
        
        # Assertions
        assert result["run_id"] == 1
        assert result["trait_name"] == "trust"
        assert result["direction"] == "max"
        assert "duration_in_seconds" in result
        assert "input_scale" in result
        assert result["num_vectors_stored"] == 2  # From mock_indices
        assert result["run_description"] == "Test run"
        
        # Verify that key functions were called
        mock_prepare_model.assert_called_once_with('test_model')
        mock_load_examples.assert_called_once()
        mock_create_sliced.assert_called_once_with(mock_model, 1, 2)
        mock_compute_activations.assert_called_once()
        mock_rank_vectors.assert_called_once()
        mock_evaluate.assert_called_once()
        
        # Check if database is updated correctly by querying it
        conn = sqlite3.connect(mock_db_setup)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Check if duration_in_seconds was updated
        run_row = cursor.execute("SELECT * FROM runs WHERE run_id = 1").fetchone()
        assert run_row["duration_in_seconds"] is not None
        
        # Check if steering vectors were inserted
        vectors = cursor.execute("SELECT * FROM steering_vectors").fetchall()
        assert len(vectors) == 2
        
        # Check if outputs were inserted
        outputs = cursor.execute("SELECT * FROM outputs").fetchall()
        assert len(outputs) >= 4  # 2 unsteered + 2x2 steered
        
        conn.close()


if __name__ == "__main__":
    pytest.main(["-v", "test_run_executor.py"])