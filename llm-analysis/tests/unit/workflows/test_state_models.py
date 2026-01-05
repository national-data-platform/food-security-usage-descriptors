"""
Unit tests for workflow state models.

Tests for AnalysisState, DatasetMention, and DatasetJoin dataclasses
including serialization, deserialization, and state management methods.
"""

import pytest
import json
from datetime import datetime
from dataclasses import asdict

from pub_analysis_agent.workflows.state_models import (
    AnalysisState,
    DatasetMention,
    DatasetJoin
)


class TestDatasetMention:
    """Test cases for DatasetMention dataclass."""
    
    def test_dataset_mention_creation(self):
        """Test basic DatasetMention creation."""
        mention = DatasetMention(
            name="NHANES",
            confidence_score_mention=0.95,
            confidence_score_use=0.85,
            text_quote="We used NHANES data for analysis",
            context="We used NHANES data for analysis in our study"
        )
        
        assert mention.name == "NHANES"
        assert mention.confidence_score_mention == 0.95
        assert mention.confidence_score_use == 0.85
        assert mention.text_quote == "We used NHANES data for analysis"
        assert mention.context == "We used NHANES data for analysis in our study"
    
    def test_dataset_mention_minimal(self):
        """Test DatasetMention with minimal required fields."""
        mention = DatasetMention(
            name="BRFSS", 
            confidence_score_mention=0.8,
            confidence_score_use=0.7,
            text_quote="BRFSS dataset",
            context="BRFSS context"
        )
        
        assert mention.name == "BRFSS"
        assert mention.confidence_score_mention == 0.8
        assert mention.confidence_score_use == 0.7
        assert mention.text_quote == "BRFSS dataset"
        assert mention.context == "BRFSS context"


class TestDatasetJoin:
    """Test cases for DatasetJoin dataclass."""
    
    def test_dataset_join_creation(self):
        """Test basic DatasetJoin creation."""
        join = DatasetJoin(
            dataset1="NHANES",
            dataset2="BRFSS",
            join_type="inner",
            confidence_score=0.9,
            methodology="Joined by demographic variables"
        )
        
        assert join.dataset1 == "NHANES"
        assert join.dataset2 == "BRFSS"
        assert join.join_type == "inner"
        assert join.confidence_score == 0.9
        assert join.methodology == "Joined by demographic variables"
    
    def test_dataset_join_minimal(self):
        """Test DatasetJoin with minimal required fields."""
        join = DatasetJoin(
            dataset1="Dataset A",
            dataset2="Dataset B",
            join_type="left",
            confidence_score=0.7
        )
        
        assert join.dataset1 == "Dataset A"
        assert join.dataset2 == "Dataset B"
        assert join.join_type == "left"
        assert join.confidence_score == 0.7


class TestAnalysisState:
    """Test cases for AnalysisState dataclass."""
    
    @pytest.fixture
    def sample_state(self):
        """Create a sample AnalysisState for testing."""
        return AnalysisState(
            publication_id="pub_123",
            workflow_id="workflow_456"
        )
    
    @pytest.fixture
    def sample_dataset_mention(self):
        """Create a sample DatasetMention for testing."""
        return DatasetMention(
            name="NHANES",
            confidence_score_mention=0.95,
            confidence_score_use=0.85,
            text_quote="NHANES health analysis",
            context="Used for health analysis"
        )
    
    @pytest.fixture
    def sample_dataset_join(self):
        """Create a sample DatasetJoin for testing."""
        return DatasetJoin(
            dataset1="NHANES",
            dataset2="BRFSS",
            join_type="inner",
            confidence_score=0.9
        )
    
    def test_analysis_state_creation(self, sample_state):
        """Test basic AnalysisState creation."""
        assert sample_state.publication_id == "pub_123"
        assert sample_state.workflow_id == "workflow_456"
        assert sample_state.grobid_content is None
        assert sample_state.raw_text is None
        assert sample_state.is_data_analysis is None
        assert sample_state.has_datasets is None
        assert len(sample_state.validated_datasets) == 0
        assert len(sample_state.newly_discovered_datasets) == 0
        assert len(sample_state.dataset_joins) == 0
        assert sample_state.final_json is None
        assert sample_state.current_step is None
        assert len(sample_state.completed_steps) == 0
        assert sample_state.error_message is None
        assert isinstance(sample_state.created_at, datetime)
        assert isinstance(sample_state.updated_at, datetime)
    
    def test_update_step(self, sample_state):
        """Test step update functionality."""
        # Update to first step
        sample_state.update_step("parse_grobid")
        
        assert sample_state.current_step == "parse_grobid"
        assert len(sample_state.completed_steps) == 0
        assert sample_state.error_message is None
        
        # Update to second step
        old_updated_at = sample_state.updated_at
        sample_state.update_step("classify_data_analysis")
        
        assert sample_state.current_step == "classify_data_analysis"
        assert "parse_grobid" in sample_state.completed_steps
        assert len(sample_state.completed_steps) == 1
        assert sample_state.updated_at > old_updated_at
    
    def test_update_step_with_error(self, sample_state):
        """Test step update with error message."""
        sample_state.update_step("extract_datasets", "Connection failed")
        
        assert sample_state.current_step == "extract_datasets"
        assert sample_state.error_message == "Connection failed"
    
    def test_add_validated_dataset(self, sample_state, sample_dataset_mention):
        """Test adding validated dataset."""
        old_updated_at = sample_state.updated_at
        sample_state.add_validated_dataset(sample_dataset_mention)
        
        assert len(sample_state.validated_datasets) == 1
        assert sample_state.validated_datasets[0] == sample_dataset_mention
        assert sample_state.updated_at > old_updated_at
    
    def test_add_new_dataset(self, sample_state, sample_dataset_mention):
        """Test adding newly discovered dataset."""
        old_updated_at = sample_state.updated_at
        sample_state.add_new_dataset(sample_dataset_mention)
        
        assert len(sample_state.newly_discovered_datasets) == 1
        assert sample_state.newly_discovered_datasets[0] == sample_dataset_mention
        assert sample_state.updated_at > old_updated_at
    
    def test_add_dataset_join(self, sample_state, sample_dataset_join):
        """Test adding dataset join."""
        old_updated_at = sample_state.updated_at
        sample_state.add_dataset_join(sample_dataset_join)
        
        assert len(sample_state.dataset_joins) == 1
        assert sample_state.dataset_joins[0] == sample_dataset_join
        assert sample_state.updated_at > old_updated_at
    
    def test_get_all_datasets(self, sample_state):
        """Test getting all datasets (validated + newly discovered)."""
        dataset1 = DatasetMention(name="Dataset1", confidence_score_mention=0.9, confidence_score_use=0.8, text_quote="Dataset1 quote", context="Dataset1 context")
        dataset2 = DatasetMention(name="Dataset2", confidence_score_mention=0.8, confidence_score_use=0.7, text_quote="Dataset2 quote", context="Dataset2 context")
        dataset3 = DatasetMention(name="Dataset3", confidence_score_mention=0.7, confidence_score_use=0.6, text_quote="Dataset3 quote", context="Dataset3 context")
        
        sample_state.add_validated_dataset(dataset1)
        sample_state.add_validated_dataset(dataset2)
        sample_state.add_new_dataset(dataset3)
        
        all_datasets = sample_state.get_all_datasets()
        
        assert len(all_datasets) == 3
        assert dataset1 in all_datasets
        assert dataset2 in all_datasets
        assert dataset3 in all_datasets
    
    def test_is_complete(self, sample_state):
        """Test workflow completion check."""
        assert not sample_state.is_complete()
        
        sample_state.final_json = {"result": "complete"}
        assert sample_state.is_complete()
    
    def test_has_error(self, sample_state):
        """Test error state check."""
        assert not sample_state.has_error()
        
        sample_state.error_message = "Something went wrong"
        assert sample_state.has_error()
    
    def test_to_dict_serialization(self, sample_state):
        """Test state serialization to dictionary."""
        # Add some data to the state
        sample_state.is_data_analysis = True
        sample_state.grobid_content = {"text": "sample text"}
        sample_state.add_validated_dataset(
            DatasetMention(name="NHANES", confidence_score_mention=0.9, confidence_score_use=0.8, text_quote="NHANES quote", context="NHANES context")
        )
        
        state_dict = sample_state.to_dict()
        
        assert isinstance(state_dict, dict)
        assert state_dict["publication_id"] == "pub_123"
        assert state_dict["workflow_id"] == "workflow_456"
        assert state_dict["is_data_analysis"] is True
        assert state_dict["grobid_content"] == {"text": "sample text"}
        assert len(state_dict["validated_datasets"]) == 1
        assert isinstance(state_dict["created_at"], str)  # Should be ISO string
        assert isinstance(state_dict["updated_at"], str)  # Should be ISO string
    
    def test_from_dict_deserialization(self, sample_state):
        """Test state deserialization from dictionary."""
        # First serialize
        sample_state.is_data_analysis = True
        sample_state.add_validated_dataset(
            DatasetMention(name="BRFSS", confidence_score_mention=0.8, confidence_score_use=0.7, text_quote="BRFSS quote", context="BRFSS context")
        )
        state_dict = sample_state.to_dict()
        
        # Then deserialize
        new_state = AnalysisState.from_dict(state_dict)
        
        assert new_state.publication_id == sample_state.publication_id
        assert new_state.workflow_id == sample_state.workflow_id
        assert new_state.is_data_analysis == sample_state.is_data_analysis
        assert len(new_state.validated_datasets) == 1
        assert new_state.validated_datasets[0].name == "BRFSS"
        assert isinstance(new_state.created_at, datetime)
        assert isinstance(new_state.updated_at, datetime)
    
    def test_to_json_serialization(self, sample_state):
        """Test state serialization to JSON string."""
        sample_state.is_data_analysis = False
        json_str = sample_state.to_json()
        
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["publication_id"] == "pub_123"
        assert parsed["is_data_analysis"] is False
    
    def test_from_json_deserialization(self, sample_state):
        """Test state deserialization from JSON string."""
        # Add some complex data
        sample_state.add_dataset_join(
            DatasetJoin(
                dataset1="A", dataset2="B", 
                join_type="left", confidence_score=0.7
            )
        )
        
        # Serialize to JSON
        json_str = sample_state.to_json()
        
        # Deserialize from JSON
        new_state = AnalysisState.from_json(json_str)
        
        assert new_state.publication_id == sample_state.publication_id
        assert len(new_state.dataset_joins) == 1
        assert new_state.dataset_joins[0].dataset1 == "A"
        assert new_state.dataset_joins[0].join_type == "left"
    
    def test_serialization_roundtrip(self, sample_state):
        """Test complete serialization roundtrip (dict -> JSON -> dict)."""
        # Add complex data
        sample_state.is_data_analysis = True
        sample_state.grobid_content = {"sections": ["intro", "methods"]}
        sample_state.add_validated_dataset(
            DatasetMention(name="NHANES", confidence_score_mention=0.95, confidence_score_use=0.85, text_quote="NHANES quote", context="NHANES context")
        )
        sample_state.add_new_dataset(
            DatasetMention(name="BRFSS", confidence_score_mention=0.85, confidence_score_use=0.75, text_quote="BRFSS quote", context="BRFSS context")
        )
        sample_state.add_dataset_join(
            DatasetJoin(
                dataset1="NHANES", dataset2="BRFSS",
                join_type="inner", confidence_score=0.9,
                methodology="Demographic join"
            )
        )
        sample_state.update_step("final_step")
        sample_state.final_json = {"status": "complete"}
        
        # Full roundtrip: state -> dict -> JSON -> dict -> state
        dict1 = sample_state.to_dict()
        json_str = json.dumps(dict1)
        dict2 = json.loads(json_str)
        new_state = AnalysisState.from_dict(dict2)
        
        # Verify all data is preserved
        assert new_state.publication_id == sample_state.publication_id
        assert new_state.is_data_analysis == sample_state.is_data_analysis
        assert new_state.grobid_content == sample_state.grobid_content
        assert len(new_state.validated_datasets) == 1
        assert len(new_state.newly_discovered_datasets) == 1
        assert len(new_state.dataset_joins) == 1
        assert new_state.current_step == sample_state.current_step
        assert new_state.final_json == sample_state.final_json
        assert new_state.validated_datasets[0].name == "NHANES"
        assert new_state.dataset_joins[0].methodology == "Demographic join"
    
    def test_serialization_error_handling(self):
        """Test error handling in serialization methods."""
        # This test ensures that malformed data doesn't crash serialization
        state = AnalysisState(publication_id="test")
        
        # Test with valid data first
        result = state.to_dict()
        assert isinstance(result, dict)
        
        # Test deserialization with invalid datetime string
        invalid_data = {
            "publication_id": "test",
            "created_at": "invalid-datetime",
            "updated_at": "2023-01-01T00:00:00"
        }
        
        with pytest.raises(ValueError):
            AnalysisState.from_dict(invalid_data)
    
    def test_deserialization_with_dict_objects(self):
        """Test deserialization when datasets/joins are stored as dicts."""
        data = {
            "publication_id": "test",
            "validated_datasets": [
                {"name": "NHANES", "confidence_score_mention": 0.9, "confidence_score_use": 0.8, "text_quote": "NHANES quote", "context": "NHANES context"}
            ],
            "dataset_joins": [
                {"dataset1": "A", "dataset2": "B", "join_type": "inner", "confidence_score": 0.8, "methodology": None}
            ],
            "created_at": "2023-01-01T00:00:00",
            "updated_at": "2023-01-01T00:00:00"
        }
        
        state = AnalysisState.from_dict(data)
        
        assert len(state.validated_datasets) == 1
        assert isinstance(state.validated_datasets[0], DatasetMention)
        assert state.validated_datasets[0].name == "NHANES"
        
        assert len(state.dataset_joins) == 1
        assert isinstance(state.dataset_joins[0], DatasetJoin)
        assert state.dataset_joins[0].dataset1 == "A" 