"""
Unit tests for SchemaValidator.

This module contains comprehensive unit tests for the SchemaValidator class,
testing schema validation, field type validation, and error handling.
"""

import pytest
from datetime import datetime, UTC
from typing import Dict, Any

from pub_analysis_agent.models.schema_validator import (
    SchemaValidator, ConsolidatedAnalysisSchema, WorkflowStatus,
    PublicationMetadata, AnalysisFlags, DatasetAnalysis, CodeExtraction,
    LinkExtraction, WorkflowMetadata, LLMMetadataSchema, ErrorInformation
)


class TestSchemaValidator:
    """Test cases for SchemaValidator."""
    
    @pytest.fixture
    def schema_validator(self):
        """Create a SchemaValidator instance."""
        return SchemaValidator()
    
    @pytest.fixture
    def valid_json_data(self):
        """Create valid JSON data for testing."""
        return {
            "publication_id": "test_pub_123",
            "workflow_id": "workflow_456",
            "analysis_timestamp": datetime.now(UTC).isoformat(),
            "workflow_status": "completed",
            "publication_metadata": {
                "publication_id": "test_pub_123",
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
                "title": "Test Publication",
                "authors": ["Author 1", "Author 2"],
                "abstract": "Test abstract",
                "publication_date": "2024-01-01",
                "journal": "Test Journal",
                "doi": "10.1234/test.123"
            },
            "analysis_flags": {
                "is_data_analysis": True,
                "has_datasets": True,
                "analysis_classification": {
                    "is_data_analysis": True,
                    "has_datasets": True,
                    "dataset_count": 2,
                    "code_snippets_count": 3,
                    "external_links_count": 1,
                    "github_repos_count": 1
                }
            },
            "dataset_analysis": {
                "validated_datasets": [
                    {
                        "name": "Dataset 1",
                        "confidence": 0.9,
                        "context": "Test context",
                        "section": "Methods",
                        "page_number": 5
                    }
                ],
                "newly_discovered_datasets": [],
                "dataset_joins": [],
                "summary": {
                    "total_validated_datasets": 1,
                    "total_new_datasets": 0,
                    "total_dataset_joins": 0,
                    "total_unique_datasets": 1
                }
            },
            "code_extraction": {
                "extracted_code_snippets": [
                    {
                        "content": "print('Hello World')",
                        "language": "python",
                        "context": "Test code context",
                        "relevance_score": 8.5,
                        "description": "Test code",
                        "purpose": "Testing",
                        "start_position": 100,
                        "end_position": 120
                    }
                ],
                "extraction_metadata": {
                    "total_code_blocks": 1,
                    "total_links_found": 2,
                    "programming_languages": ["python"],
                    "processing_time": 1.5,
                    "extraction_errors": [],
                    "extraction_timestamp": datetime.now(UTC).isoformat()
                },
                "summary": {
                    "total_code_snippets": 1,
                    "programming_languages": ["python"],
                    "average_relevance_score": 8.5
                }
            },
            "link_extraction": {
                "external_links": [
                    {
                        "url": "https://example.com",
                        "link_type": "dataset",
                        "title": "Example Dataset",
                        "description": "Test dataset",
                        "context": "Test link context",
                        "is_accessible": True,
                        "relevance_score": 7.0
                    }
                ],
                "github_repositories": [
                    {
                        "url": "https://github.com/test/repo",
                        "owner": "test",
                        "repository": "repo",
                        "path": "/src",
                        "branch": "main",
                        "is_valid": True,
                        "description": "Test repository",
                        "language": "Python",
                        "stars": 100
                    }
                ],
                "summary": {
                    "total_external_links": 1,
                    "total_github_repos": 1,
                    "accessible_links": 1,
                    "valid_github_repos": 1,
                    "average_link_relevance": 7.0
                }
            },
            "workflow_metadata": {
                "current_step": "json_assembly",
                "completed_steps": ["triage", "dataset_validation", "code_extraction", "json_assembly"],
                "workflow_duration": 120.5,
                "step_count": 4,
                "total_steps": 7,
                "completion_percentage": 57.14
            },
            "llm_metadata": {
                "model_name": "gpt-4",
                "model_version": "latest",
                "tokens_used": 1500,
                "response_time": 2.5,
                "temperature": 0.7,
                "max_tokens": 2000
            }
        }
    
    def test_initialization(self, schema_validator):
        """Test SchemaValidator initialization."""
        assert schema_validator.schema_version == "1.0"
    
    def test_validate_consolidated_json_valid(self, schema_validator, valid_json_data):
        """Test validation of valid JSON data."""
        result = schema_validator.validate_consolidated_json(valid_json_data)
        
        assert result["success"] is True
        assert result["validated_data"] is not None
        assert len(result["errors"]) == 0
        assert len(result["warnings"]) == 0
    
    def test_validate_consolidated_json_missing_required_field(self, schema_validator, valid_json_data):
        """Test validation with missing required field."""
        del valid_json_data["publication_id"]
        
        result = schema_validator.validate_consolidated_json(valid_json_data)
        
        assert result["success"] is False
        assert result["validated_data"] is None
        assert len(result["errors"]) > 0
        assert any("publication_id" in error["field"] for error in result["errors"])
    
    def test_validate_consolidated_json_invalid_workflow_status(self, schema_validator, valid_json_data):
        """Test validation with invalid workflow status."""
        valid_json_data["workflow_status"] = "invalid_status"
        
        result = schema_validator.validate_consolidated_json(valid_json_data)
        
        assert result["success"] is False
        assert len(result["errors"]) > 0
        assert any("workflow_status" in error["field"] for error in result["errors"])
    
    def test_validate_consolidated_json_invalid_timestamp(self, schema_validator, valid_json_data):
        """Test validation with invalid timestamp."""
        valid_json_data["analysis_timestamp"] = "invalid_timestamp"
        
        result = schema_validator.validate_consolidated_json(valid_json_data)
        
        assert result["success"] is False
        assert len(result["errors"]) > 0
        assert any("analysis_timestamp" in error["field"] for error in result["errors"])
    
    def test_validate_consolidated_json_failed_status_without_error(self, schema_validator, valid_json_data):
        """Test validation when status is failed but no error information provided."""
        valid_json_data["workflow_status"] = "failed"
        valid_json_data["error_information"] = None
        
        result = schema_validator.validate_consolidated_json(valid_json_data)
        
        assert result["success"] is False
        assert len(result["errors"]) > 0
        # Print the actual error message for debugging
        print(f"Actual error: {result['errors']}")
        # Check for any error related to failed status
        assert any("failed" in error["message"].lower() or "error_information" in error["message"] for error in result["errors"])
    
    def test_validate_consolidated_json_failed_status_with_error(self, schema_validator, valid_json_data):
        """Test validation when status is failed with proper error information."""
        valid_json_data["workflow_status"] = "failed"
        valid_json_data["error_information"] = {
            "error_message": "Test error",
            "error_timestamp": datetime.now(UTC).isoformat(),
            "current_step_at_error": "json_assembly"
        }
        
        result = schema_validator.validate_consolidated_json(valid_json_data)
        
        assert result["success"] is True
        assert len(result["errors"]) == 0
    
    def test_validate_field_types_valid(self, schema_validator, valid_json_data):
        """Test field type validation with valid data."""
        errors = schema_validator.validate_field_types(valid_json_data)
        assert len(errors) == 0
    
    def test_validate_field_types_invalid_string_field(self, schema_validator, valid_json_data):
        """Test field type validation with invalid string field."""
        valid_json_data["publication_id"] = 123  # Should be string
        
        errors = schema_validator.validate_field_types(valid_json_data)
        assert len(errors) > 0
        assert any("publication_id" in error["field"] for error in errors)
    
    def test_validate_field_types_invalid_dict_field(self, schema_validator, valid_json_data):
        """Test field type validation with invalid dict field."""
        valid_json_data["publication_metadata"] = "not_a_dict"  # Should be dict
        
        errors = schema_validator.validate_field_types(valid_json_data)
        assert len(errors) > 0
        assert any("publication_metadata" in error["field"] for error in errors)
    
    def test_validate_required_fields_valid(self, schema_validator, valid_json_data):
        """Test required fields validation with valid data."""
        errors = schema_validator.validate_required_fields(valid_json_data)
        assert len(errors) == 0
    
    def test_validate_required_fields_missing(self, schema_validator, valid_json_data):
        """Test required fields validation with missing fields."""
        del valid_json_data["publication_id"]
        del valid_json_data["workflow_status"]
        
        errors = schema_validator.validate_required_fields(valid_json_data)
        assert len(errors) == 2
        assert any("publication_id" in error["field"] for error in errors)
        assert any("workflow_status" in error["field"] for error in errors)
    
    def test_validate_data_formats_valid(self, schema_validator, valid_json_data):
        """Test data format validation with valid data."""
        errors = schema_validator.validate_data_formats(valid_json_data)
        assert len(errors) == 0
    
    def test_validate_data_formats_invalid_timestamp(self, schema_validator, valid_json_data):
        """Test data format validation with invalid timestamp."""
        valid_json_data["analysis_timestamp"] = "invalid_timestamp"
        
        errors = schema_validator.validate_data_formats(valid_json_data)
        assert len(errors) > 0
        assert any("analysis_timestamp" in error["field"] for error in errors)
    
    def test_validate_data_formats_invalid_workflow_status(self, schema_validator, valid_json_data):
        """Test data format validation with invalid workflow status."""
        valid_json_data["workflow_status"] = "invalid_status"
        
        errors = schema_validator.validate_data_formats(valid_json_data)
        assert len(errors) > 0
        assert any("workflow_status" in error["field"] for error in errors)
    
    def test_get_validation_summary(self, schema_validator, valid_json_data):
        """Test validation summary generation."""
        validation_result = schema_validator.validate_consolidated_json(valid_json_data)
        summary = schema_validator.get_validation_summary(validation_result)
        
        assert summary["success"] is True
        assert summary["total_errors"] == 0
        assert summary["total_warnings"] == 0
        assert summary["schema_version"] == "1.0"
        assert "validation_timestamp" in summary
    
    def test_validate_schema_version_compatibility(self, schema_validator, valid_json_data):
        """Test schema version compatibility validation."""
        is_compatible = schema_validator.validate_schema_version_compatibility(valid_json_data)
        assert is_compatible is True
    
    def test_workflow_status_enum(self):
        """Test WorkflowStatus enum values."""
        assert WorkflowStatus.PENDING == "pending"
        assert WorkflowStatus.IN_PROGRESS == "in_progress"
        assert WorkflowStatus.COMPLETED == "completed"
        assert WorkflowStatus.FAILED == "failed"
    
    def test_publication_metadata_validation(self):
        """Test PublicationMetadata model validation."""
        metadata = PublicationMetadata(
            publication_id="test",
            created_at=datetime.now(UTC).isoformat(),
            updated_at=datetime.now(UTC).isoformat(),
            title="Test Title",
            authors=["Author 1"],
            abstract="Test abstract"
        )
        
        assert metadata.publication_id == "test"
        assert metadata.title == "Test Title"
        assert len(metadata.authors) == 1
    
    def test_publication_metadata_invalid_timestamp(self):
        """Test PublicationMetadata with invalid timestamp."""
        with pytest.raises(ValueError, match="Invalid ISO timestamp format"):
            PublicationMetadata(
                publication_id="test",
                created_at="invalid_timestamp",
                updated_at=datetime.now(UTC).isoformat()
            )
    
    def test_dataset_analysis_validation(self):
        """Test DatasetAnalysis model validation."""
        analysis = DatasetAnalysis(
            validated_datasets=[],
            newly_discovered_datasets=[],
            dataset_joins=[],
            summary={
                "total_validated_datasets": 0,
                "total_new_datasets": 0,
                "total_dataset_joins": 0,
                "total_unique_datasets": 0
            }
        )
        
        assert len(analysis.validated_datasets) == 0
        assert analysis.summary.total_validated_datasets == 0
    
    def test_code_extraction_validation(self):
        """Test CodeExtraction model validation."""
        extraction = CodeExtraction(
            extracted_code_snippets=[],
            summary={
                "total_code_snippets": 0,
                "programming_languages": [],
                "average_relevance_score": 0.0
            }
        )
        
        assert len(extraction.extracted_code_snippets) == 0
        assert extraction.summary.total_code_snippets == 0
    
    def test_link_extraction_validation(self):
        """Test LinkExtraction model validation."""
        extraction = LinkExtraction(
            external_links=[],
            github_repositories=[],
            summary={
                "total_external_links": 0,
                "total_github_repos": 0,
                "accessible_links": 0,
                "valid_github_repos": 0,
                "average_link_relevance": 0.0
            }
        )
        
        assert len(extraction.external_links) == 0
        assert extraction.summary.total_external_links == 0
    
    def test_workflow_metadata_validation(self):
        """Test WorkflowMetadata model validation."""
        metadata = WorkflowMetadata(
            current_step="json_assembly",
            completed_steps=["triage", "dataset_validation"],
            workflow_duration=60.0,
            step_count=2,
            total_steps=7,
            completion_percentage=28.57
        )
        
        assert metadata.current_step == "json_assembly"
        assert len(metadata.completed_steps) == 2
        assert metadata.completion_percentage == pytest.approx(28.57, rel=0.01)
    
    def test_llm_metadata_validation(self):
        """Test LLMMetadataSchema model validation."""
        metadata = LLMMetadataSchema(
            model_name="gpt-4",
            model_version="latest",
            tokens_used=1000,
            response_time=2.0,
            temperature=0.7,
            max_tokens=2000
        )
        
        assert metadata.model_name == "gpt-4"
        assert metadata.tokens_used == 1000
        assert metadata.temperature == 0.7
    
    def test_error_information_validation(self):
        """Test ErrorInformation model validation."""
        error_info = ErrorInformation(
            error_message="Test error",
            error_timestamp=datetime.now(UTC).isoformat(),
            current_step_at_error="json_assembly"
        )
        
        assert error_info.error_message == "Test error"
        assert error_info.current_step_at_error == "json_assembly"
    
    def test_error_information_invalid_timestamp(self):
        """Test ErrorInformation with invalid timestamp."""
        with pytest.raises(ValueError, match="Invalid ISO timestamp format"):
            ErrorInformation(
                error_message="Test error",
                error_timestamp="invalid_timestamp",
                current_step_at_error="json_assembly"
            ) 