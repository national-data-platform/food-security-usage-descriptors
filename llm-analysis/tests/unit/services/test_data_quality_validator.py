"""
Unit tests for DataQualityValidator.

This module contains comprehensive unit tests for the DataQualityValidator class,
testing data quality checks, completeness validation, and quality metrics calculation.
"""

import pytest
from datetime import datetime, UTC
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from pub_analysis_agent.services.data_quality_validator import (
    DataQualityValidator, QualityMetrics, CompletenessScore, QualityIssue,
    QualityLevel, ValidationSeverity
)


class TestDataQualityValidator:
    """Test cases for DataQualityValidator."""
    
    @pytest.fixture
    def dataset_service_mock(self):
        """Create a mock dataset service."""
        mock_service = Mock()
        mock_service.get_datasets_by_aliases = AsyncMock(return_value=[{"name": "MNIST"}])
        return mock_service
    
    @pytest.fixture
    def quality_validator(self, dataset_service_mock):
        """Create a DataQualityValidator instance."""
        return DataQualityValidator(dataset_service_mock)
    
    @pytest.fixture
    def valid_analysis_data(self):
        """Create valid analysis data for testing."""
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
                        "name": "MNIST",
                        "confidence": 0.9,
                        "context": "Test context",
                        "section": "Methods",
                        "page_number": 5
                    },
                    {
                        "name": "CIFAR-10",
                        "confidence": 0.8,
                        "context": "Test context 2",
                        "section": "Methods",
                        "page_number": 6
                    }
                ],
                "newly_discovered_datasets": [],
                "dataset_joins": [],
                "summary": {
                    "total_validated_datasets": 2,
                    "total_new_datasets": 0,
                    "total_dataset_joins": 0,
                    "total_unique_datasets": 2
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
                    },
                    {
                        "content": "import numpy as np\nx = np.array([1, 2, 3])",
                        "language": "python",
                        "context": "Data processing code",
                        "relevance_score": 9.0,
                        "description": "Data processing",
                        "purpose": "Data manipulation",
                        "start_position": 200,
                        "end_position": 250
                    }
                ],
                "extraction_metadata": {
                    "total_code_blocks": 2,
                    "total_links_found": 2,
                    "programming_languages": ["python"],
                    "processing_time": 1.5,
                    "extraction_errors": [],
                    "extraction_timestamp": datetime.now(UTC).isoformat()
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
            },
            "final_json": {
                "publication_id": "test_pub_123",
                "workflow_id": "workflow_456",
                "analysis_timestamp": datetime.now(UTC).isoformat(),
                "workflow_status": "completed"
            }
        }
    
    def test_initialization(self, quality_validator):
        """Test DataQualityValidator initialization."""
        assert quality_validator.dataset_service is not None
    
    @pytest.mark.asyncio
    async def test_validate_analysis_quality_valid_data(self, quality_validator, valid_analysis_data):
        """Test quality validation with valid data."""
        metrics = await quality_validator.validate_analysis_quality(valid_analysis_data)
        
        assert isinstance(metrics, QualityMetrics)
        assert metrics.overall_quality_level in [QualityLevel.EXCELLENT, QualityLevel.GOOD]
        assert metrics.completeness_score.overall_score > 80
        assert metrics.data_consistency_score > 80
        assert metrics.logical_validation_score > 80
    
    @pytest.mark.asyncio
    async def test_validate_completeness_missing_critical_fields(self, quality_validator):
        """Test completeness validation with missing critical fields."""
        incomplete_data = {
            "publication_id": "test_pub_123",
            # Missing other critical fields
        }
        
        metrics = await quality_validator.validate_analysis_quality(incomplete_data)
        
        assert metrics.completeness_score.overall_score < 50
        assert len(metrics.completeness_score.missing_critical_fields) > 0
        assert metrics.overall_quality_level == QualityLevel.CRITICAL
    
    @pytest.mark.asyncio
    async def test_validate_completeness_missing_optional_fields(self, quality_validator):
        """Test completeness validation with missing optional fields."""
        data_without_optional = {
            "publication_id": "test_pub_123",
            "workflow_id": "workflow_456",
            "analysis_timestamp": datetime.now(UTC).isoformat(),
            "workflow_status": "in_progress",
            "publication_metadata": {
                "publication_id": "test_pub_123",
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat()
                # Missing title, authors, etc.
            },
            "analysis_flags": {
                "is_data_analysis": True,
                "has_datasets": False,
                "analysis_classification": {
                    "is_data_analysis": True,
                    "has_datasets": False,
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
                "current_step": "json_assembly",
                "completed_steps": ["triage", "dataset_validation", "code_extraction", "json_assembly"],
                "workflow_duration": 120.5,
                "step_count": 4,
                "total_steps": 7,
                "completion_percentage": 57.14
            }
        }
        
        metrics = await quality_validator.validate_analysis_quality(data_without_optional)
        
        assert len(metrics.completeness_score.missing_optional_fields) > 0
        assert metrics.overall_quality_level in [QualityLevel.GOOD, QualityLevel.FAIR]
    
    @pytest.mark.asyncio
    async def test_validate_data_consistency_count_mismatch(self, quality_validator):
        """Test data consistency validation with count mismatches."""
        inconsistent_data = {
            "publication_id": "test_pub_123",
            "workflow_id": "workflow_456",
            "analysis_timestamp": datetime.now(UTC).isoformat(),
            "workflow_status": "completed",
            "publication_metadata": {
                "publication_id": "test_pub_123",
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
                "title": "Test Publication"
            },
            "analysis_flags": {
                "is_data_analysis": True,
                "has_datasets": True,
                "analysis_classification": {
                    "is_data_analysis": True,
                    "has_datasets": True,
                    "dataset_count": 5,  # Mismatch with actual datasets
                    "code_snippets_count": 10,  # Mismatch with actual code snippets
                    "external_links_count": 1,
                    "github_repos_count": 1
                }
            },
            "dataset_analysis": {
                "validated_datasets": [
                    {"name": "Dataset 1", "confidence": 0.9}
                ],  # Only 1 dataset but count says 5
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
                    {"content": "print('test')", "language": "python", "context": "test", "relevance_score": 8.0}
                ],  # Only 1 snippet but count says 10
                "summary": {
                    "total_code_snippets": 1,
                    "programming_languages": ["python"],
                    "average_relevance_score": 8.0
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
                "current_step": "json_assembly",
                "completed_steps": ["triage", "dataset_validation", "code_extraction", "json_assembly"],
                "workflow_duration": 120.5,
                "step_count": 4,
                "total_steps": 7,
                "completion_percentage": 57.14
            }
        }
        
        metrics = await quality_validator.validate_analysis_quality(inconsistent_data)
        
        # Should find consistency issues
        consistency_issues = [i for i in metrics.quality_issues if "mismatch" in i.issue_type]
        assert len(consistency_issues) > 0
        assert metrics.data_consistency_score < 100
    
    @pytest.mark.asyncio
    async def test_validate_logical_rules_inconsistent_flags(self, quality_validator):
        """Test logical validation with inconsistent flags."""
        inconsistent_flags_data = {
            "publication_id": "test_pub_123",
            "workflow_id": "workflow_456",
            "analysis_timestamp": datetime.now(UTC).isoformat(),
            "workflow_status": "completed",
            "publication_metadata": {
                "publication_id": "test_pub_123",
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
                "title": "Test Publication"
            },
            "analysis_flags": {
                "is_data_analysis": False,  # Inconsistent with code snippets
                "has_datasets": False,  # Inconsistent with datasets
                "analysis_classification": {
                    "is_data_analysis": False,
                    "has_datasets": False,
                    "dataset_count": 0,
                    "code_snippets_count": 0,
                    "external_links_count": 0,
                    "github_repos_count": 0
                }
            },
            "dataset_analysis": {
                "validated_datasets": [
                    {"name": "Dataset 1", "confidence": 0.9}
                ],  # Has datasets but flag says False
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
                    {"content": "print('test')", "language": "python", "context": "test", "relevance_score": 8.0}
                ],  # Has code but flag says False
                "summary": {
                    "total_code_snippets": 1,
                    "programming_languages": ["python"],
                    "average_relevance_score": 8.0
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
                "current_step": "json_assembly",
                "completed_steps": ["triage", "dataset_validation", "code_extraction", "json_assembly"],
                "workflow_duration": 120.5,
                "step_count": 4,
                "total_steps": 7,
                "completion_percentage": 57.14
            }
        }
        
        metrics = await quality_validator.validate_analysis_quality(inconsistent_flags_data)
        
        # Should find logical inconsistency issues
        logical_issues = [i for i in metrics.quality_issues if "logical" in i.issue_type]
        assert len(logical_issues) > 0
        assert metrics.logical_validation_score < 100
    
    @pytest.mark.asyncio
    async def test_validate_dataset_references_unknown_dataset(self, quality_validator):
        """Test dataset reference validation with unknown datasets."""
        # Mock dataset service to return empty results for unknown datasets
        quality_validator.dataset_service.get_datasets_by_aliases = AsyncMock(return_value=[])
        
        data_with_unknown_datasets = {
            "publication_id": "test_pub_123",
            "workflow_id": "workflow_456",
            "analysis_timestamp": datetime.now(UTC).isoformat(),
            "workflow_status": "completed",
            "publication_metadata": {
                "publication_id": "test_pub_123",
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
                "title": "Test Publication"
            },
            "analysis_flags": {
                "is_data_analysis": True,
                "has_datasets": True,
                "analysis_classification": {
                    "is_data_analysis": True,
                    "has_datasets": True,
                    "dataset_count": 1,
                    "code_snippets_count": 0,
                    "external_links_count": 0,
                    "github_repos_count": 0
                }
            },
            "dataset_analysis": {
                "validated_datasets": [
                    {"name": "UnknownDataset", "confidence": 0.9}  # Unknown dataset
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
                "extracted_code_snippets": [],
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
                "current_step": "json_assembly",
                "completed_steps": ["triage", "dataset_validation", "code_extraction", "json_assembly"],
                "workflow_duration": 120.5,
                "step_count": 4,
                "total_steps": 7,
                "completion_percentage": 57.14
            }
        }
        
        metrics = await quality_validator.validate_analysis_quality(data_with_unknown_datasets)
        
        # Should find unknown dataset issues
        unknown_dataset_issues = [i for i in metrics.quality_issues if "unknown_dataset" in i.issue_type]
        assert len(unknown_dataset_issues) > 0
    
    @pytest.mark.asyncio
    async def test_validate_code_quality_invalid_scores(self, quality_validator):
        """Test code quality validation with invalid scores."""
        data_with_invalid_code = {
            "publication_id": "test_pub_123",
            "workflow_id": "workflow_456",
            "analysis_timestamp": datetime.now(UTC).isoformat(),
            "workflow_status": "completed",
            "publication_metadata": {
                "publication_id": "test_pub_123",
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
                "title": "Test Publication"
            },
            "analysis_flags": {
                "is_data_analysis": True,
                "has_datasets": False,
                "analysis_classification": {
                    "is_data_analysis": True,
                    "has_datasets": False,
                    "dataset_count": 0,
                    "code_snippets_count": 1,
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
                "extracted_code_snippets": [
                    {
                        "content": "",  # Empty content
                        "language": "",  # Missing language
                        "context": "test",
                        "relevance_score": 15.0,  # Invalid score > 10
                        "description": "Test code"
                    }
                ],
                "summary": {
                    "total_code_snippets": 1,
                    "programming_languages": [],
                    "average_relevance_score": 15.0
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
                "current_step": "json_assembly",
                "completed_steps": ["triage", "dataset_validation", "code_extraction", "json_assembly"],
                "workflow_duration": 120.5,
                "step_count": 4,
                "total_steps": 7,
                "completion_percentage": 57.14
            }
        }
        
        metrics = await quality_validator.validate_analysis_quality(data_with_invalid_code)
        
        # Should find code quality issues
        code_quality_issues = [i for i in metrics.quality_issues if "code" in i.issue_type or "relevance_score" in i.issue_type]
        assert len(code_quality_issues) > 0
    
    @pytest.mark.asyncio
    async def test_validate_link_quality_invalid_urls(self, quality_validator):
        """Test link quality validation with invalid URLs."""
        data_with_invalid_links = {
            "publication_id": "test_pub_123",
            "workflow_id": "workflow_456",
            "analysis_timestamp": datetime.now(UTC).isoformat(),
            "workflow_status": "completed",
            "publication_metadata": {
                "publication_id": "test_pub_123",
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
                "title": "Test Publication"
            },
            "analysis_flags": {
                "is_data_analysis": True,
                "has_datasets": False,
                "analysis_classification": {
                    "is_data_analysis": True,
                    "has_datasets": False,
                    "dataset_count": 0,
                    "code_snippets_count": 0,
                    "external_links_count": 1,
                    "github_repos_count": 1
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
                "summary": {
                    "total_code_snippets": 0,
                    "programming_languages": [],
                    "average_relevance_score": 0.0
                }
            },
            "link_extraction": {
                "external_links": [
                    {
                        "url": "invalid-url",  # Invalid URL
                        "link_type": "dataset",
                        "title": "Example Dataset",
                        "description": "Test dataset",
                        "context": "Test link context",
                        "is_accessible": True,
                        "relevance_score": 15.0  # Invalid score
                    }
                ],
                "github_repositories": [
                    {
                        "url": "not-a-github-url",  # Invalid GitHub URL
                        "owner": "",  # Missing owner
                        "repository": "",  # Missing repository
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
                    "average_link_relevance": 15.0
                }
            },
            "workflow_metadata": {
                "current_step": "json_assembly",
                "completed_steps": ["triage", "dataset_validation", "code_extraction", "json_assembly"],
                "workflow_duration": 120.5,
                "step_count": 4,
                "total_steps": 7,
                "completion_percentage": 57.14
            }
        }
        
        metrics = await quality_validator.validate_analysis_quality(data_with_invalid_links)
        
        # Should find link quality issues
        link_quality_issues = [i for i in metrics.quality_issues if "url" in i.issue_type or "github" in i.issue_type]
        assert len(link_quality_issues) > 0
    
    def test_calculate_completeness_score(self, quality_validator, valid_analysis_data):
        """Test completeness score calculation."""
        issues = []
        score = quality_validator._calculate_completeness_score(valid_analysis_data, issues)
        
        assert isinstance(score, CompletenessScore)
        assert score.overall_score > 80
        assert score.publication_metadata_score > 80
        assert score.dataset_analysis_score > 80
        assert score.code_extraction_score > 80
        assert score.link_extraction_score > 80
        assert score.workflow_metadata_score > 80
    
    def test_calculate_consistency_score(self, quality_validator):
        """Test consistency score calculation."""
        # Test with no consistency issues
        no_issues = []
        score = quality_validator._calculate_consistency_score(no_issues)
        assert score == 100.0
        
        # Test with consistency issues
        consistency_issues = [
            QualityIssue(
                field="test",
                message="Test consistency issue",
                severity=ValidationSeverity.WARNING,
                issue_type="consistency_mismatch",
                current_value=1,
                expected_value=2
            )
        ]
        score = quality_validator._calculate_consistency_score(consistency_issues)
        assert score < 100.0
    
    def test_calculate_logical_score(self, quality_validator):
        """Test logical validation score calculation."""
        # Test with no logical issues
        no_issues = []
        score = quality_validator._calculate_logical_score(no_issues)
        assert score == 100.0
        
        # Test with logical issues
        logical_issues = [
            QualityIssue(
                field="test",
                message="Test logical issue",
                severity=ValidationSeverity.ERROR,
                issue_type="logical_inconsistency",
                current_value=False,
                expected_value=True
            )
        ]
        score = quality_validator._calculate_logical_score(logical_issues)
        assert score < 100.0
    
    def test_determine_quality_level(self, quality_validator):
        """Test quality level determination."""
        # Test with high completeness and no critical issues
        high_completeness = CompletenessScore(
            overall_score=95.0,
            publication_metadata_score=100.0,
            dataset_analysis_score=90.0,
            code_extraction_score=95.0,
            link_extraction_score=90.0,
            workflow_metadata_score=100.0,
            missing_critical_fields=[],
            missing_optional_fields=[]
        )
        no_issues = []
        
        level = quality_validator._determine_quality_level(high_completeness, no_issues)
        assert level == QualityLevel.EXCELLENT
        
        # Test with critical issues
        critical_issues = [
            QualityIssue(
                field="test",
                message="Critical issue",
                severity=ValidationSeverity.CRITICAL,
                issue_type="missing_critical_field",
                current_value=None,
                expected_value="present"
            )
        ]
        
        level = quality_validator._determine_quality_level(high_completeness, critical_issues)
        assert level == QualityLevel.CRITICAL
    
    def test_get_quality_report(self, quality_validator):
        """Test quality report generation."""
        # Create sample metrics
        completeness_score = CompletenessScore(
            overall_score=85.0,
            publication_metadata_score=90.0,
            dataset_analysis_score=80.0,
            code_extraction_score=85.0,
            link_extraction_score=80.0,
            workflow_metadata_score=90.0,
            missing_critical_fields=[],
            missing_optional_fields=["publication_metadata.authors"]
        )
        
        quality_issues = [
            QualityIssue(
                field="test_field",
                message="Test issue",
                severity=ValidationSeverity.WARNING,
                issue_type="test_issue",
                current_value="test",
                expected_value="expected",
                suggestion="Fix this"
            )
        ]
        
        metrics = QualityMetrics(
            completeness_score=completeness_score,
            data_consistency_score=90.0,
            logical_validation_score=95.0,
            overall_quality_level=QualityLevel.GOOD,
            total_issues=1,
            critical_issues=0,
            warnings=1,
            info_issues=0,
            quality_issues=quality_issues,
            validation_timestamp=datetime.now(UTC)
        )
        
        report = quality_validator.get_quality_report(metrics)
        
        assert "overall_quality" in report
        assert "issue_summary" in report
        assert "completeness_details" in report
        assert "quality_issues" in report
        assert report["overall_quality"]["level"] == "good"
        assert report["issue_summary"]["total_issues"] == 1
        assert len(report["quality_issues"]) == 1
    
    def test_quality_level_enum(self):
        """Test QualityLevel enum values."""
        assert QualityLevel.EXCELLENT == "excellent"
        assert QualityLevel.GOOD == "good"
        assert QualityLevel.FAIR == "fair"
        assert QualityLevel.POOR == "poor"
        assert QualityLevel.CRITICAL == "critical"
    
    def test_validation_severity_enum(self):
        """Test ValidationSeverity enum values."""
        assert ValidationSeverity.INFO == "info"
        assert ValidationSeverity.WARNING == "warning"
        assert ValidationSeverity.ERROR == "error"
        assert ValidationSeverity.CRITICAL == "critical"
    
    def test_quality_issue_dataclass(self):
        """Test QualityIssue dataclass."""
        issue = QualityIssue(
            field="test_field",
            message="Test message",
            severity=ValidationSeverity.WARNING,
            issue_type="test_type",
            current_value="test_value",
            expected_value="expected_value",
            suggestion="Test suggestion"
        )
        
        assert issue.field == "test_field"
        assert issue.message == "Test message"
        assert issue.severity == ValidationSeverity.WARNING
        assert issue.issue_type == "test_type"
        assert issue.current_value == "test_value"
        assert issue.expected_value == "expected_value"
        assert issue.suggestion == "Test suggestion"
    
    def test_completeness_score_dataclass(self):
        """Test CompletenessScore dataclass."""
        score = CompletenessScore(
            overall_score=85.0,
            publication_metadata_score=90.0,
            dataset_analysis_score=80.0,
            code_extraction_score=85.0,
            link_extraction_score=80.0,
            workflow_metadata_score=90.0,
            missing_critical_fields=[],
            missing_optional_fields=["field1", "field2"]
        )
        
        assert score.overall_score == 85.0
        assert score.publication_metadata_score == 90.0
        assert score.dataset_analysis_score == 80.0
        assert score.code_extraction_score == 85.0
        assert score.link_extraction_score == 80.0
        assert score.workflow_metadata_score == 90.0
        assert len(score.missing_critical_fields) == 0
        assert len(score.missing_optional_fields) == 2 