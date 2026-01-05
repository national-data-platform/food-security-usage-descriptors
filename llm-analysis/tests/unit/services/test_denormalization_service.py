"""
Unit tests for DenormalizationService.

This module tests the denormalization service functionality,
including document transformation, batch processing, and validation.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from pub_analysis_agent.services.denormalization_service import DenormalizationService
from pub_analysis_agent.models.schema_validator import (
    ConsolidatedAnalysisSchema,
    WorkflowStatus,
    PublicationMetadata,
    AnalysisFlags,
    AnalysisClassification,
    DatasetAnalysis,
    DatasetAnalysisSummary,
    DatasetMentionSchema,
    CodeExtraction,
    CodeAnalysisSummary,
    CodeSnippetSchema,
    LinkExtraction,
    LinkAnalysisSummary,
    ExternalLinkSchema,
    GitHubRepositorySchema,
    WorkflowMetadata,
    LLMMetadataSchema,
    ErrorInformation
)


class TestDenormalizationService:
    """Test DenormalizationService functionality."""
    
    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        return DenormalizationService()
    
    @pytest.fixture
    def sample_mongo_doc(self):
        """Create a sample MongoDB document for testing."""
        return {
            "publication_id": "test_pub_123",
            "workflow_id": "workflow_456",
            "analysis_timestamp": "2024-01-15T10:30:00Z",
            "workflow_status": "completed",
            "publication_metadata": {
                "publication_id": "test_pub_123",
                "created_at": "2024-01-15T10:00:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
                "title": "Test Publication Title",
                "authors": ["Author 1", "Author 2"],
                "abstract": "This is a test abstract",
                "publication_date": "2024-01-15",
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
                    "external_links_count": 5,
                    "github_repos_count": 2
                }
            },
            "dataset_analysis": {
                "validated_datasets": [
                    {
                        "name": "Dataset A",
                        "confidence": 0.9,
                        "context": "Dataset A was used for analysis",
                        "section": "Methods",
                        "page_number": 5
                    },
                    {
                        "name": "Dataset B",
                        "confidence": 0.8,
                        "context": "Dataset B was referenced",
                        "section": "Introduction",
                        "page_number": 2
                    }
                ],
                "newly_discovered_datasets": [
                    {
                        "name": "Dataset C",
                        "confidence": 0.7,
                        "context": "New dataset C discovered",
                        "section": "Results",
                        "page_number": 8
                    }
                ],
                "dataset_joins": [
                    {
                        "dataset1": "Dataset A",
                        "dataset2": "Dataset B",
                        "join_type": "inner",
                        "confidence": 0.85,
                        "description": "Join between A and B"
                    }
                ],
                "summary": {
                    "total_validated_datasets": 2,
                    "total_new_datasets": 1,
                    "total_dataset_joins": 1,
                    "total_unique_datasets": 3
                }
            },
            "code_extraction": {
                "extracted_code_snippets": [
                    {
                        "content": "import pandas as pd",
                        "language": "python",
                        "context": "Data processing code",
                        "relevance_score": 8.5,
                        "description": "Pandas import",
                        "purpose": "Data loading",
                        "start_position": 100,
                        "end_position": 120
                    },
                    {
                        "content": "df = pd.read_csv('data.csv')",
                        "language": "python",
                        "context": "Data loading code",
                        "relevance_score": 9.0,
                        "description": "CSV loading",
                        "purpose": "Data import",
                        "start_position": 150,
                        "end_position": 180
                    }
                ],
                "extraction_metadata": {
                    "total_code_blocks": 2,
                    "total_links_found": 0,
                    "programming_languages": ["python"],
                    "processing_time": 1.5,
                    "extraction_errors": [],
                    "extraction_timestamp": "2024-01-15T10:25:00Z"
                },
                "summary": {
                    "total_code_snippets": 2,
                    "programming_languages": ["python"],
                    "average_relevance_score": 8.75
                }
            },
            "link_extraction": {
                "external_links": [
                    {
                        "url": "https://example.com/data",
                        "link_type": "dataset",
                        "title": "Example Dataset",
                        "description": "External dataset link",
                        "context": "Dataset reference",
                        "is_accessible": True,
                        "relevance_score": 7.5
                    }
                ],
                "github_repositories": [
                    {
                        "url": "https://github.com/example/repo",
                        "owner": "example",
                        "repository": "repo",
                        "path": "/data",
                        "branch": "main",
                        "is_valid": True,
                        "description": "Example repository",
                        "language": "Python",
                        "stars": 100
                    }
                ],
                "summary": {
                    "total_external_links": 1,
                    "total_github_repos": 1,
                    "accessible_links": 1,
                    "valid_github_repos": 1,
                    "average_link_relevance": 7.5
                }
            },
            "workflow_metadata": {
                "current_step": "completed",
                "completed_steps": ["triage", "dataset_discovery", "code_extraction"],
                "workflow_duration": 180.5,
                "step_count": 3,
                "total_steps": 7,
                "completion_percentage": 42.86
            },
            "llm_metadata": {
                "model_name": "gpt-4",
                "model_version": "1.0",
                "tokens_used": 1500,
                "response_time": 2.5,
                "temperature": 0.1,
                "max_tokens": 2000
            }
        }
    
    def test_initialization(self, service):
        """Test service initialization."""
        assert service is not None
        assert hasattr(service, 'logger')
    
    def test_denormalize_analysis_result_success(self, service, sample_mongo_doc):
        """Test successful denormalization of analysis result."""
        result = service.denormalize_analysis_result(sample_mongo_doc)
        
        # Check base fields
        assert result["publication_id"] == "test_pub_123"
        assert result["workflow_id"] == "workflow_456"
        assert result["workflow_status"] == "completed"
        assert result["created_at"] == "2024-01-15T10:30:00Z"
        
        # Check publication metadata
        assert result["publication_title"] == "Test Publication Title"
        assert result["publication_authors"] == ["Author 1", "Author 2"]
        assert result["publication_abstract"] == "This is a test abstract"
        assert result["authors_text"] == "Author 1 Author 2"
        
        # Check analysis flags
        assert result["is_data_analysis"] is True
        assert result["has_datasets"] is True
        assert result["dataset_count"] == 2
        
        # Check dataset analysis
        assert len(result["validated_datasets"]) == 2
        assert len(result["newly_discovered_datasets"]) == 1
        assert len(result["dataset_joins"]) == 1
        assert "Dataset A" in result["all_dataset_names"]
        assert "Dataset B" in result["all_dataset_names"]
        assert "Dataset C" in result["all_dataset_names"]
        
        # Check code extraction
        assert len(result["code_snippets"]) == 2
        assert "python" in result["all_programming_languages"]
        assert result["total_code_snippets"] == 2
        
        # Check link extraction
        assert len(result["external_links"]) == 1
        assert len(result["github_repositories"]) == 1
        assert result["total_links"] == 2
        
        # Check search fields
        assert "search_text" in result
        assert "autocomplete_text" in result
        assert result["has_content"] is True
    
    def test_denormalize_analysis_result_with_errors(self, service, sample_mongo_doc):
        """Test denormalization with error information."""
        sample_mongo_doc["error_information"] = {
            "error_message": "Test error occurred",
            "error_timestamp": "2024-01-15T10:35:00Z",
            "current_step_at_error": "code_extraction"
        }
        
        result = service.denormalize_analysis_result(sample_mongo_doc)
        
        assert result["error_message"] == "Test error occurred"
        assert result["error_timestamp"] == "2024-01-15T10:35:00Z"
        assert result["error_current_step"] == "code_extraction"
        assert result["has_errors"] is True
    
    def test_denormalize_analysis_result_missing_optional_fields(self, service):
        """Test denormalization with missing optional fields."""
        minimal_doc = {
            "publication_id": "test_pub_123",
            "workflow_id": "workflow_456",
            "analysis_timestamp": "2024-01-15T10:30:00Z",
            "workflow_status": "completed",
            "publication_metadata": {
                "publication_id": "test_pub_123",
                "created_at": "2024-01-15T10:00:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
                "title": None,
                "authors": None,
                "abstract": None,
                "publication_date": None,
                "journal": None,
                "doi": None
            },
            "analysis_flags": {
                "is_data_analysis": None,
                "has_datasets": None,
                "analysis_classification": {
                    "is_data_analysis": None,
                    "has_datasets": None,
                    "dataset_count": 0,
                    "code_snippets_count": 0,
                    "external_links_count": 0,
                    "github_repos_count": 0
                }
            },
            "dataset_analysis": {
                "validated_datasets": [],
                "newly_discovered_datasets": [],
                "dataset_joins": [],
                "summary": {
                    "total_validated_datasets": 0,
                    "total_new_datasets": 0,
                    "total_dataset_joins": 0,
                    "total_unique_datasets": 0
                }
            },
            "code_extraction": {
                "extracted_code_snippets": [],
                "extraction_metadata": None,
                "summary": {
                    "total_code_snippets": 0,
                    "programming_languages": [],
                    "average_relevance_score": 0.0
                }
            },
            "link_extraction": {
                "external_links": [],
                "github_repositories": [],
                "summary": {
                    "total_external_links": 0,
                    "total_github_repos": 0,
                    "accessible_links": 0,
                    "valid_github_repos": 0,
                    "average_link_relevance": 0.0
                }
            },
            "workflow_metadata": {
                "current_step": None,
                "completed_steps": [],
                "workflow_duration": 0.0,
                "step_count": 0,
                "total_steps": 7,
                "completion_percentage": 0.0
            }
        }
        
        result = service.denormalize_analysis_result(minimal_doc)
        
        # Check that optional fields are handled gracefully
        assert result["publication_title"] == ""
        assert result["publication_authors"] == []
        assert result["publication_abstract"] == ""
        assert result["is_data_analysis"] is False
        assert result["has_datasets"] is False
        assert result["dataset_count"] == 0
        assert len(result["validated_datasets"]) == 0
        assert len(result["code_snippets"]) == 0
        assert len(result["external_links"]) == 0
    
    def test_denormalize_analysis_result_invalid_document(self, service):
        """Test denormalization with invalid document structure."""
        invalid_doc = {
            "publication_id": "test_pub_123",
            # Missing required fields
        }
        
        with pytest.raises(Exception):
            service.denormalize_analysis_result(invalid_doc)
    
    def test_batch_denormalize_success(self, service, sample_mongo_doc):
        """Test successful batch denormalization."""
        docs = [sample_mongo_doc, sample_mongo_doc]
        
        results = service.batch_denormalize(docs)
        
        assert len(results) == 2
        assert all(isinstance(doc, dict) for doc in results)
        assert all(doc["publication_id"] == "test_pub_123" for doc in results)
    
    def test_batch_denormalize_with_errors(self, service, sample_mongo_doc):
        """Test batch denormalization with some failing documents."""
        invalid_doc = {"invalid": "document"}
        docs = [sample_mongo_doc, invalid_doc, sample_mongo_doc]
        
        results = service.batch_denormalize(docs)
        
        # Should return only successful denormalizations
        assert len(results) == 2
        assert all(doc["publication_id"] == "test_pub_123" for doc in results)
    
    def test_validate_denormalized_document_valid(self, service, sample_mongo_doc):
        """Test validation of valid denormalized document."""
        denormalized_doc = service.denormalize_analysis_result(sample_mongo_doc)
        
        validation_result = service.validate_denormalized_document(denormalized_doc)
        
        assert validation_result["valid"] is True
        assert len(validation_result["issues"]) == 0
        assert validation_result["document_size"] > 0
    
    def test_validate_denormalized_document_invalid(self, service):
        """Test validation of invalid denormalized document."""
        invalid_doc = {
            "publication_id": "",  # Empty required field
            "dataset_count": "not_a_number",  # Wrong type
            "publication_authors": "not_a_list"  # Wrong type
        }
        
        validation_result = service.validate_denormalized_document(invalid_doc)
        
        assert validation_result["valid"] is False
        assert len(validation_result["issues"]) > 0
        assert any("Missing required field" in issue for issue in validation_result["issues"])
        assert any("must be an integer" in issue for issue in validation_result["issues"])
        assert any("must be a list" in issue for issue in validation_result["issues"])
    
    def test_search_fields_generation(self, service, sample_mongo_doc):
        """Test generation of search-optimized fields."""
        result = service.denormalize_analysis_result(sample_mongo_doc)
        
        # Check search text contains key information
        search_text = result["search_text"]
        assert "Test Publication Title" in search_text
        assert "This is a test abstract" in search_text
        assert "Author 1 Author 2" in search_text
        assert "Dataset A" in search_text
        assert "Dataset B" in search_text
        assert "Dataset C" in search_text
        assert "python" in search_text
        
        # Check autocomplete text
        autocomplete_text = result["autocomplete_text"]
        assert "Test Publication Title" in autocomplete_text
        assert "Dataset A" in autocomplete_text
        assert "python" in autocomplete_text
        
        # Check content metrics
        assert result["content_length"] > 0
        assert result["has_content"] is True
    
    def test_dataset_analysis_denormalization(self, service, sample_mongo_doc):
        """Test specific dataset analysis denormalization."""
        result = service.denormalize_analysis_result(sample_mongo_doc)
        
        # Check validated datasets
        validated_datasets = result["validated_datasets"]
        assert len(validated_datasets) == 2
        assert validated_datasets[0]["name"] == "Dataset A"
        assert validated_datasets[0]["confidence"] == 0.9
        assert validated_datasets[1]["name"] == "Dataset B"
        assert validated_datasets[1]["confidence"] == 0.8
        
        # Check newly discovered datasets
        new_datasets = result["newly_discovered_datasets"]
        assert len(new_datasets) == 1
        assert new_datasets[0]["name"] == "Dataset C"
        assert new_datasets[0]["confidence"] == 0.7
        
        # Check dataset joins
        dataset_joins = result["dataset_joins"]
        assert len(dataset_joins) == 1
        assert dataset_joins[0]["dataset1"] == "Dataset A"
        assert dataset_joins[0]["dataset2"] == "Dataset B"
        assert dataset_joins[0]["join_type"] == "inner"
        
        # Check summary
        summary = result["dataset_analysis_summary"]
        assert summary["total_validated_datasets"] == 2
        assert summary["total_new_datasets"] == 1
        assert summary["total_dataset_joins"] == 1
        assert summary["total_unique_datasets"] == 3
        
        # Check aggregated fields
        assert len(result["all_dataset_names"]) == 3
        assert "Dataset A" in result["all_dataset_names"]
        assert "Dataset B" in result["all_dataset_names"]
        assert "Dataset C" in result["all_dataset_names"]
        assert result["total_datasets"] == 3
    
    def test_code_extraction_denormalization(self, service, sample_mongo_doc):
        """Test specific code extraction denormalization."""
        result = service.denormalize_analysis_result(sample_mongo_doc)
        
        # Check code snippets
        code_snippets = result["code_snippets"]
        assert len(code_snippets) == 2
        assert code_snippets[0]["content"] == "import pandas as pd"
        assert code_snippets[0]["language"] == "python"
        assert code_snippets[0]["relevance_score"] == 8.5
        assert code_snippets[1]["content"] == "df = pd.read_csv('data.csv')"
        assert code_snippets[1]["language"] == "python"
        assert code_snippets[1]["relevance_score"] == 9.0
        
        # Check summary
        summary = result["code_extraction_summary"]
        assert summary["total_code_snippets"] == 2
        assert summary["programming_languages"] == ["python"]
        assert summary["average_relevance_score"] == 8.75
        
        # Check metadata
        metadata = result["code_extraction_metadata"]
        assert metadata["total_code_blocks"] == 2
        assert metadata["programming_languages"] == ["python"]
        assert metadata["processing_time"] == 1.5
        
        # Check aggregated fields
        assert result["all_programming_languages"] == ["python"]
        assert result["total_code_snippets"] == 2
    
    def test_link_extraction_denormalization(self, service, sample_mongo_doc):
        """Test specific link extraction denormalization."""
        result = service.denormalize_analysis_result(sample_mongo_doc)
        
        # Check external links
        external_links = result["external_links"]
        assert len(external_links) == 1
        assert external_links[0]["url"] == "https://example.com/data"
        assert external_links[0]["link_type"] == "dataset"
        assert external_links[0]["is_accessible"] is True
        assert external_links[0]["relevance_score"] == 7.5
        
        # Check GitHub repositories
        github_repos = result["github_repositories"]
        assert len(github_repos) == 1
        assert github_repos[0]["url"] == "https://github.com/example/repo"
        assert github_repos[0]["owner"] == "example"
        assert github_repos[0]["repository"] == "repo"
        assert github_repos[0]["is_valid"] is True
        assert github_repos[0]["stars"] == 100
        
        # Check summary
        summary = result["link_extraction_summary"]
        assert summary["total_external_links"] == 1
        assert summary["total_github_repos"] == 1
        assert summary["accessible_links"] == 1
        assert summary["valid_github_repos"] == 1
        assert summary["average_link_relevance"] == 7.5
        
        # Check aggregated fields
        assert len(result["all_urls"]) == 2
        assert "https://example.com/data" in result["all_urls"]
        assert "https://github.com/example/repo" in result["all_urls"]
        assert result["total_links"] == 2 