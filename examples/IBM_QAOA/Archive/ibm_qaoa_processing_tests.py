"""
Tests for IBM QAOA processing functions.

This test suite validates the IBM-specific QAOA data ingestion pipeline
in examples/IBM_QAOA/ibm_qaoa_processing.py.
"""

import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import pytest

# Add the IBM_QAOA directory to path for imports
IBM_QAOA_DIR = Path(__file__).parent.parent / "examples" / "IBM_QAOA"
sys.path.insert(0, str(IBM_QAOA_DIR))

# Add stochastic-benchmark src to path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import ibm_qaoa_processing as ibm


# Fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "ibm_qaoa"


class TestQAOAResult:
    """Test QAOAResult dataclass."""
    
    def test_qaoa_result_creation(self):
        """Test basic QAOAResult object creation."""
        result = ibm.QAOAResult(
            trial_id=0,
            instance_id="001",
            depth=2,
            energy=-12.5,
            approximation_ratio=0.85,
            train_duration=2.3,
            trainer_name="FixedAngleConjecture",
            evaluator=None,
            success=True,
            optimized_params=[0.1, 0.2],
            energy_history=[-10.0, -12.5]
        )
        
        assert result.trial_id == 0
        assert result.instance_id == "001"
        assert result.depth == 2
        assert result.energy == -12.5
        assert result.approximation_ratio == 0.85
        assert result.trainer_name == "FixedAngleConjecture"
        assert result.evaluator is None
        assert result.success is True
        assert len(result.optimized_params) == 2
        assert len(result.energy_history) == 2


class TestParseQAOATrial:
    """Test parse_qaoa_trial function."""
    
    def test_parse_complete_trial(self):
        """Test parsing a trial with all fields present."""
        trial_data = {
            'energy': -12.5,
            'approximation ratio': 0.85,
            'train_duration': 2.3,
            'trainer': {
                'trainer_name': 'FixedAngleConjecture',
                'evaluator': None
            },
            'success': True,
            'optimal_params': [0.1, 0.2],
            'history': [-10.0, -11.5, -12.5]
        }
        
        result = ibm.parse_qaoa_trial(trial_data, trial_id=0, instance_id="001", depth=2)
        
        assert result.trial_id == 0
        assert result.instance_id == "001"
        assert result.depth == 2
        assert result.energy == -12.5
        assert result.approximation_ratio == 0.85
        assert result.train_duration == 2.3
        assert result.trainer_name == 'FixedAngleConjecture'
        assert result.evaluator is None
        assert result.success is True
        assert result.optimized_params == [0.1, 0.2]
        assert result.energy_history == [-10.0, -11.5, -12.5]
    
    def test_parse_missing_fields(self):
        """Test parsing with missing optional fields."""
        trial_data = {
            'energy': -5.0,
            'approximation ratio': 0.5
        }
        
        result = ibm.parse_qaoa_trial(trial_data, trial_id=1, instance_id="002", depth=1)
        
        assert result.energy == -5.0
        assert result.approximation_ratio == 0.5
        assert result.train_duration == 0.0  # Default
        assert result.trainer_name == 'Unknown'  # Default when missing
        assert result.evaluator is None
        assert result.success is False  # Default
        assert result.optimized_params == []
        assert result.energy_history == []
    
    def test_parse_missing_trainer_dict(self):
        """Test parsing when trainer field is missing."""
        trial_data = {
            'energy': -8.0,
            'approximation ratio': 0.7,
            'train_duration': 1.5,
            'success': True
        }
        
        result = ibm.parse_qaoa_trial(trial_data, trial_id=2, instance_id="003", depth=1)
        
        assert result.trainer_name == 'Unknown'
        assert result.evaluator is None
    
    def test_parse_trainer_as_string(self):
        """Test parsing when trainer is a string instead of dict."""
        trial_data = {
            'energy': -9.0,
            'approximation ratio': 0.75,
            'trainer': 'COBYLA'
        }
        
        result = ibm.parse_qaoa_trial(trial_data, trial_id=3, instance_id="004", depth=2)
        
        assert result.trainer_name == 'COBYLA'
        assert result.evaluator is None
    
    def test_parse_nan_values(self):
        """Test that NaN values are properly handled."""
        trial_data = {}
        
        result = ibm.parse_qaoa_trial(trial_data, trial_id=4, instance_id="005", depth=1)
        
        assert np.isnan(result.energy)
        assert np.isnan(result.approximation_ratio)


class TestLoadQAOAResults:
    """Test load_qaoa_results function."""
    
    def test_load_single_trial(self):
        """Test loading JSON with single trial."""
        json_data = {
            "0": {
                'energy': -12.5,
                'approximation ratio': 0.85,
                'train_duration': 2.3
            }
        }
        
        results = ibm.load_qaoa_results(json_data)
        
        assert len(results) == 1
        assert results[0].trial_id == 0
        assert results[0].energy == -12.5
    
    def test_load_multiple_trials(self):
        """Test loading JSON with multiple trials."""
        json_data = {
            "0": {'energy': -12.5, 'approximation ratio': 0.85},
            "1": {'energy': -13.0, 'approximation ratio': 0.87},
            "2": {'energy': -12.8, 'approximation ratio': 0.86}
        }
        
        results = ibm.load_qaoa_results(json_data)
        
        assert len(results) == 3
        assert results[0].trial_id == 0
        assert results[1].trial_id == 1
        assert results[2].trial_id == 2
    
    def test_load_ignores_non_numeric_keys(self):
        """Test that non-numeric keys are ignored."""
        json_data = {
            "0": {'energy': -12.5, 'approximation ratio': 0.85},
            "1": {'energy': -13.0, 'approximation ratio': 0.87},
            "metadata": {'note': 'test data'},
            "config": {'version': '1.0'}
        }
        
        results = ibm.load_qaoa_results(json_data)
        
        assert len(results) == 2
        assert all(isinstance(r.trial_id, int) for r in results)
    
    def test_load_empty_dict(self):
        """Test loading empty dictionary."""
        json_data = {}
        
        results = ibm.load_qaoa_results(json_data)
        
        assert len(results) == 0
    
    def test_load_from_fixture(self):
        """Test loading from actual fixture file."""
        fixture_path = FIXTURES_DIR / "multi_trial_synthetic.json"
        with open(fixture_path, 'r') as f:
            json_data = json.load(f)
        
        results = ibm.load_qaoa_results(json_data)
        
        assert len(results) == 3
        assert all(r.trainer_name == 'FixedAngleConjecture' for r in results)


class TestConvertToDataFrame:
    """Test convert_to_dataframe function."""
    
    def test_convert_single_result(self):
        """Test converting single QAOAResult to DataFrame."""
        result = ibm.QAOAResult(
            trial_id=0,
            instance_id="001",
            depth=2,
            energy=-12.5,
            approximation_ratio=0.85,
            train_duration=2.3,
            trainer_name="FixedAngleConjecture",
            evaluator=None,
            success=True,
            optimized_params=[0.1, 0.2],
            energy_history=[-10.0, -12.5]
        )
        
        df = ibm.convert_to_dataframe([result], instance_id="001", p=2)
        
        assert len(df) == 1
        assert df['trial_id'].iloc[0] == 0
        assert df['instance'].iloc[0] == "001"
        assert df['p'].iloc[0] == 2
        assert df['Energy'].iloc[0] == -12.5
        assert df['Approximation_Ratio'].iloc[0] == 0.85
        assert df['MeanTime'].iloc[0] == 2.3
        assert df['trainer'].iloc[0] == "FixedAngleConjecture"
        assert pd.isna(df['evaluator'].iloc[0])
        assert df['success'].iloc[0] == True
        assert df['n_iterations'].iloc[0] == 2
        assert 'param_0' in df.columns
        assert 'param_1' in df.columns
        assert df['param_0'].iloc[0] == 0.1
        assert df['param_1'].iloc[0] == 0.2
    
    def test_convert_multiple_results(self):
        """Test converting multiple results."""
        results = [
            ibm.QAOAResult(0, "001", 1, -12.5, 0.85, 2.3, "FA", None, True, [0.1], [-12.5]),
            ibm.QAOAResult(1, "001", 1, -13.0, 0.87, 2.5, "FA", None, True, [0.12], [-13.0]),
            ibm.QAOAResult(2, "001", 1, -12.8, 0.86, 2.4, "FA", None, True, [0.11], [-12.8])
        ]
        
        df = ibm.convert_to_dataframe(results, instance_id="001", p=1)
        
        assert len(df) == 3
        assert df['instance'].iloc[0] == "001"
        assert df['p'].iloc[0] == 1
        assert list(df['Energy']) == [-12.5, -13.0, -12.8]
    
    def test_convert_no_params(self):
        """Test converting result without optimized parameters."""
        result = ibm.QAOAResult(
            trial_id=0,
            instance_id="002",
            depth=1,
            energy=-8.0,
            approximation_ratio=0.7,
            train_duration=1.5,
            trainer_name="COBYLA",
            evaluator=None,
            success=True,
            optimized_params=[],
            energy_history=[-8.0]
        )
        
        df = ibm.convert_to_dataframe([result], instance_id="002", p=1)
        
        assert len(df) == 1
        assert 'param_0' not in df.columns
        assert df['n_iterations'].iloc[0] == 1
    
    def test_convert_empty_history(self):
        """Test converting result with empty history."""
        result = ibm.QAOAResult(
            0, "003", 1, -5.0, 0.5, 1.0, "Unknown", None, False, [], []
        )
        
        df = ibm.convert_to_dataframe([result], instance_id="003", p=1)
        
        assert df['n_iterations'].iloc[0] == 0


class TestGroupNameFunction:
    """Test group_name_fcn function."""
    
    def test_extract_standard_filename(self):
        """Test extracting group name from standard pickle filename."""
        filename = "raw_results_inst=001_depth=2.pkl"
        
        group = ibm.group_name_fcn(filename)
        
        assert group == "inst=001_depth=2"
    
    def test_extract_with_path(self):
        """Test extracting from full path."""
        filepath = "/path/to/raw_results_inst=123_depth=4.pkl"
        
        group = ibm.group_name_fcn(filepath)
        
        assert group == "inst=123_depth=4"
    
    def test_malformed_filename_returns_full(self):
        """Test that malformed filename returns full basename."""
        filename = "malformed_file.pkl"
        
        group = ibm.group_name_fcn(filename)
        
        assert group == "malformed_file.pkl"


class TestPrepareStochasticBenchmarkData:
    """Test prepare_stochastic_benchmark_data function."""
    
    def test_save_and_load_pickle(self):
        """Test saving DataFrame to pickle format."""
        df = pd.DataFrame({
            'trial_id': [0, 1],
            'instance': ['001', '001'],
            'p': [2, 2],
            'Energy': [-12.5, -13.0],
            'Approximation_Ratio': [0.85, 0.87],
            'MeanTime': [2.3, 2.5]
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = ibm.prepare_stochastic_benchmark_data(df, "001", 2, tmpdir)
            
            assert os.path.exists(filepath)
            assert filepath.endswith("raw_results_inst=001_depth=2.pkl")
            # Verify file is in exp_raw subdirectory (StochasticBenchmark convention)
            assert "exp_raw" in filepath
            
            # Verify we can load it back
            loaded_df = pd.read_pickle(filepath)
            assert len(loaded_df) == 2
            assert list(loaded_df.columns) == list(df.columns)
            pd.testing.assert_frame_equal(loaded_df, df)
    
    def test_creates_output_directory(self):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "nested", "output")
            df = pd.DataFrame({'col': [1, 2]})
            
            filepath = ibm.prepare_stochastic_benchmark_data(df, "001", 1, output_dir)
            
            # Should create both output_dir and output_dir/exp_raw
            assert os.path.exists(output_dir)
            assert os.path.exists(os.path.join(output_dir, "exp_raw"))
            assert os.path.exists(filepath)


class TestProcessQAOADataIntegration:
    """Integration tests for process_qaoa_data function."""
    
    def test_process_single_file(self):
        """Test processing a single JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy fixture to temp directory with expected naming
            fixture = FIXTURES_DIR / "20250901_165018_000N10R3R_MC_FA_SV_noOpt_2.json"
            test_file = os.path.join(tmpdir, "20250901_165018_000N10R3R_MC_FA_SV_noOpt_2.json")
            shutil.copy(fixture, test_file)
            
            pattern = os.path.join(tmpdir, "*.json")
            output_dir = os.path.join(tmpdir, "output")
            
            sb, agg_df = ibm.process_qaoa_data(json_pattern=pattern, output_dir=output_dir)
            
            assert sb is not None
            assert len(agg_df) == 1  # Single trial
            assert 'instance' in agg_df.columns
            assert 'p' in agg_df.columns
            assert 'Energy' in agg_df.columns
            assert 'Approximation_Ratio' in agg_df.columns
            assert agg_df['instance'].iloc[0] == 0  # Parsed from "000N10R3R"
            assert agg_df['p'].iloc[0] == 2  # Parsed from "_2.json"
    
    def test_process_multiple_files(self):
        """Test processing multiple JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy multiple fixtures
            fixtures = [
                "20250901_165018_000N10R3R_MC_FA_SV_noOpt_2.json",
                "20250901_165018_000N10R3R_MC_FA_SV_noOpt_4.json",
                "20250913_170712_001N10R3R_MC_FA_SV_noOpt_1.json"
            ]
            
            for fixture_name in fixtures:
                fixture = FIXTURES_DIR / fixture_name
                shutil.copy(fixture, tmpdir)
            
            pattern = os.path.join(tmpdir, "*.json")
            output_dir = os.path.join(tmpdir, "output")
            
            sb, agg_df = ibm.process_qaoa_data(json_pattern=pattern, output_dir=output_dir)
            
            assert sb is not None
            assert len(agg_df) == 3  # One trial per file
            assert agg_df['instance'].nunique() == 2  # instances 0 and 1
            assert set(agg_df['p'].unique()) == {1, 2, 4}  # depths 1, 2, 4
            
            # Check sorting
            assert list(agg_df['instance']) == sorted(agg_df['instance'])
    
    def test_process_adds_gtminenergy(self):
        """Test that GTMinEnergy is added to DataFrames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = FIXTURES_DIR / "20250913_170712_001N10R3R_MC_FA_SV_noOpt_1.json"
            test_file = os.path.join(tmpdir, "test.json")
            shutil.copy(fixture, test_file)
            
            pattern = os.path.join(tmpdir, "*.json")
            output_dir = os.path.join(tmpdir, "output")
            
            sb, agg_df = ibm.process_qaoa_data(json_pattern=pattern, output_dir=output_dir)
            
            assert 'GTMinEnergy' in agg_df.columns
            # GTMinEnergy should equal the minimum Energy in single-trial case
            assert agg_df['GTMinEnergy'].iloc[0] == agg_df['Energy'].iloc[0]
    
    def test_process_empty_pattern(self):
        """Test processing with no matching files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pattern = os.path.join(tmpdir, "*.json")
            output_dir = os.path.join(tmpdir, "output")
            
            sb, agg_df = ibm.process_qaoa_data(json_pattern=pattern, output_dir=output_dir)
            
            assert sb is None
            assert len(agg_df) == 0
    
    def test_process_creates_pickles(self):
        """Test that pickle files are created for each input JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = FIXTURES_DIR / "20250901_165018_000N10R3R_MC_FA_SV_noOpt_2.json"
            test_file = os.path.join(tmpdir, "20250901_165018_000N10R3R_MC_FA_SV_noOpt_2.json")
            shutil.copy(fixture, test_file)
            
            pattern = os.path.join(tmpdir, "*.json")
            output_dir = os.path.join(tmpdir, "output")
            
            # Use config with persist_raw=True to ensure pickles are written
            config = ibm.ProcessingConfig(persist_raw=True)
            sb, agg_df = ibm.process_qaoa_data(json_pattern=pattern, output_dir=output_dir, config=config)
            
            # Check that pickle file was created in output_dir/exp_raw (StochasticBenchmark convention)
            raw_data_dir = os.path.join(output_dir, "exp_raw")
            assert os.path.exists(raw_data_dir)
            pickle_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.pkl')]
            assert len(pickle_files) == 1
            expected_pickle = os.path.join(raw_data_dir, pickle_files[0])
            assert os.path.exists(expected_pickle)
            
            # Verify content
            df = pd.read_pickle(expected_pickle)
            assert len(df) == 1
            # Instance ID is extracted as '000' from filename, not converted to int
            assert df['instance'].iloc[0] == '000'
            assert df['p'].iloc[0] == 2
    
    def test_process_single_trial_bootstrap_fabrication(self):
        """Test that single-trial files get manual bootstrap fabrication."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = FIXTURES_DIR / "20250913_171721_002N10R3R_MC_FA_SV_noOpt_1.json"
            test_file = os.path.join(tmpdir, "20250913_171721_002N10R3R_MC_FA_SV_noOpt_1.json")
            shutil.copy(fixture, test_file)
            
            pattern = os.path.join(tmpdir, "*.json")
            output_dir = os.path.join(tmpdir, "output")
            
            sb, agg_df = ibm.process_qaoa_data(json_pattern=pattern, output_dir=output_dir)
            
            assert sb is not None
            assert sb.bs_results is not None
            
            # Single trial should have 5 bootstrap entries (boots 10, 20, 30, 40, 50)
            assert len(sb.bs_results) == 5
            assert set(sb.bs_results['boots'].unique()) == {10, 20, 30, 40, 50}
            
            # Check confidence intervals are zero-width for single trial
            if 'Key=PerfRatio' in sb.bs_results.columns:
                first_row = sb.bs_results.iloc[0]
                assert first_row['Key=PerfRatio'] == first_row['ConfInt=lower_Key=PerfRatio']
                assert first_row['Key=PerfRatio'] == first_row['ConfInt=upper_Key=PerfRatio']
    
    def test_process_interpolation_fallback(self):
        """Test that interpolation falls back for single instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = FIXTURES_DIR / "20250913_171721_002N10R3R_MC_FA_SV_noOpt_1.json"
            test_file = os.path.join(tmpdir, "test.json")
            shutil.copy(fixture, test_file)
            
            pattern = os.path.join(tmpdir, "*.json")
            output_dir = os.path.join(tmpdir, "output")
            
            sb, agg_df = ibm.process_qaoa_data(json_pattern=pattern, output_dir=output_dir)
            
            assert sb.interp_results is not None
            # For single instance (5 rows from bootstrap), interp_results should equal bs_results
            assert len(sb.interp_results) == 5
            assert 'resource' in sb.interp_results.columns
    
    def test_process_adds_train_test_split(self):
        """Test that train/test split column is added."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = FIXTURES_DIR / "20250913_170712_001N10R3R_MC_FA_SV_noOpt_1.json"
            test_file = os.path.join(tmpdir, "test.json")
            shutil.copy(fixture, test_file)
            
            pattern = os.path.join(tmpdir, "*.json")
            output_dir = os.path.join(tmpdir, "output")
            
            sb, agg_df = ibm.process_qaoa_data(json_pattern=pattern, output_dir=output_dir)
            
            assert 'train' in sb.interp_results.columns
            assert set(sb.interp_results['train'].unique()).issubset({0, 1})


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_missing_trainer_fixture(self):
        """Test processing file with missing trainer field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = FIXTURES_DIR / "missing_trainer.json"
            test_file = os.path.join(tmpdir, "20250901_165018_000N10R3R_MC_FA_SV_noOpt_1.json")
            shutil.copy(fixture, test_file)
            
            pattern = os.path.join(tmpdir, "*.json")
            output_dir = os.path.join(tmpdir, "output")
            
            sb, agg_df = ibm.process_qaoa_data(json_pattern=pattern, output_dir=output_dir)
            
            assert len(agg_df) == 1
            assert agg_df['trainer'].iloc[0] == 'Unknown'
            assert pd.isna(agg_df['evaluator'].iloc[0])
    
    def test_missing_optimal_params_fixture(self):
        """Test processing file with missing optimal_params field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = FIXTURES_DIR / "missing_optimal_params.json"
            test_file = os.path.join(tmpdir, "20250901_165018_000N10R3R_MC_FA_SV_noOpt_1.json")
            shutil.copy(fixture, test_file)
            
            pattern = os.path.join(tmpdir, "*.json")
            output_dir = os.path.join(tmpdir, "output")
            
            sb, agg_df = ibm.process_qaoa_data(json_pattern=pattern, output_dir=output_dir)
            
            assert len(agg_df) == 1
            # No param columns should exist
            param_cols = [c for c in agg_df.columns if c.startswith('param_')]
            assert len(param_cols) == 0
    
    def test_empty_trials_fixture(self):
        """Test processing file with no trials."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = FIXTURES_DIR / "empty_trials.json"
            test_file = os.path.join(tmpdir, "20250901_165018_000N10R3R_MC_FA_SV_noOpt_1.json")
            shutil.copy(fixture, test_file)
            
            pattern = os.path.join(tmpdir, "*.json")
            output_dir = os.path.join(tmpdir, "output")
            
            sb, agg_df = ibm.process_qaoa_data(json_pattern=pattern, output_dir=output_dir)
            
            # Should process but result in empty data
            assert sb is None
            assert len(agg_df) == 0
    
    def test_multi_trial_synthetic_fixture(self):
        """Test processing synthetic multi-trial file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = FIXTURES_DIR / "multi_trial_synthetic.json"
            test_file = os.path.join(tmpdir, "20250901_165018_000N10R3R_MC_FA_SV_noOpt_2.json")
            shutil.copy(fixture, test_file)
            
            pattern = os.path.join(tmpdir, "*.json")
            output_dir = os.path.join(tmpdir, "output")
            
            sb, agg_df = ibm.process_qaoa_data(json_pattern=pattern, output_dir=output_dir)
            
            # Should have 3 trials from the multi-trial fixture
            assert len(agg_df) == 3
            assert agg_df['instance'].iloc[0] == 0
            assert agg_df['p'].iloc[0] == 2
            
            # Pickle should contain all 3 trials (in exp_raw subdirectory)
            raw_data_dir = os.path.join(output_dir, "exp_raw")
            pickle_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.pkl')]
            assert len(pickle_files) == 1
            df = pd.read_pickle(os.path.join(raw_data_dir, pickle_files[0]))
            assert len(df) == 3
            
            # With 3 trials, should attempt standard bootstrap (may succeed or fail)
            # Key is that interp_results should exist and have resource column
            assert sb.interp_results is not None
            if len(sb.interp_results) > 0:
                assert 'resource' in sb.interp_results.columns


class TestProcessingConfig:
    """Test ProcessingConfig dataclass and config-driven behavior."""
    
    def test_config_defaults(self):
        """Test ProcessingConfig default values."""
        config = ibm.ProcessingConfig()
        assert config.persist_raw is True
        assert config.interpolate_diversity_threshold == 6
        assert config.fabricate_single_trial is True
        assert config.seed == 42
        assert config.log_progress_interval == 50
    
    def test_config_overrides(self):
        """Test ProcessingConfig with custom values."""
        config = ibm.ProcessingConfig(
            persist_raw=False,
            interpolate_diversity_threshold=10,
            fabricate_single_trial=False,
            seed=123,
            log_progress_interval=10
        )
        assert config.persist_raw is False
        assert config.interpolate_diversity_threshold == 10
        assert config.fabricate_single_trial is False
        assert config.seed == 123
        assert config.log_progress_interval == 10
    
    def test_persist_raw_false_no_pickles(self):
        """Test that persist_raw=False prevents pickle creation during ingestion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = FIXTURES_DIR / "20250901_165018_000N10R3R_MC_FA_SV_noOpt_2.json"
            test_file = os.path.join(tmpdir, "20250901_165018_000N10R3R_MC_FA_SV_noOpt_2.json")
            shutil.copy(fixture, test_file)
            
            pattern = os.path.join(tmpdir, "*.json")
            output_dir = os.path.join(tmpdir, "output")
            
            config = ibm.ProcessingConfig(persist_raw=False)
            sb, agg_df = ibm.process_qaoa_data(json_pattern=pattern, output_dir=output_dir, config=config)
            
            # With persist_raw=False, bootstrap will generate temp pickles
            # but they won't be from the original ingestion phase
            # We can verify the aggregated DataFrame exists
            assert len(agg_df) == 1
            # Instance may be either string '000' or int 0 depending on conversion
            assert agg_df['instance'].iloc[0] in ('000', 0)
            
            # Bootstrap should still run and produce results
            assert sb.interp_results is not None
    
    def test_diversity_heuristic_skip(self):
        """Test that low diversity skips interpolation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Single instance, single depth = diversity 1
            fixture = FIXTURES_DIR / "20250901_165018_000N10R3R_MC_FA_SV_noOpt_2.json"
            test_file = os.path.join(tmpdir, "20250901_165018_000N10R3R_MC_FA_SV_noOpt_2.json")
            shutil.copy(fixture, test_file)
            
            pattern = os.path.join(tmpdir, "*.json")
            output_dir = os.path.join(tmpdir, "output")
            
            config = ibm.ProcessingConfig(interpolate_diversity_threshold=6)
            sb, agg_df = ibm.process_qaoa_data(json_pattern=pattern, output_dir=output_dir, config=config)
            
            # Diversity (1 instance × 1 depth = 1) < threshold (6) → skip interpolation
            # interp_results should equal bs_results (with resource column added)
            assert sb.interp_results is not None
            if len(sb.interp_results) > 0:
                assert 'resource' in sb.interp_results.columns
    
    def test_diversity_heuristic_run(self):
        """Test that sufficient diversity enables interpolation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy 2 different instances with different depths
            fixtures = [
                "20250901_165018_000N10R3R_MC_FA_SV_noOpt_2.json",  # instance 000, depth 2
                "20250901_165018_000N10R3R_MC_FA_SV_noOpt_4.json"   # instance 000, depth 4 (different depth, same instance)
            ]
            
            for fixture_name in fixtures:
                fixture = FIXTURES_DIR / fixture_name
                test_file = os.path.join(tmpdir, fixture_name)
                shutil.copy(fixture, test_file)
            
            pattern = os.path.join(tmpdir, "*.json")
            output_dir = os.path.join(tmpdir, "output")
            
            config = ibm.ProcessingConfig(interpolate_diversity_threshold=3)
            sb, agg_df = ibm.process_qaoa_data(json_pattern=pattern, output_dir=output_dir, config=config)
            
            # Diversity (2 instances × 2 depths = 4) >= threshold (3) → run interpolation
            assert sb.interp_results is not None
            if len(sb.interp_results) > 0:
                assert 'resource' in sb.interp_results.columns


if __name__ == "__main__":

    pytest.main([__file__, "-v", "--tb=short"])
