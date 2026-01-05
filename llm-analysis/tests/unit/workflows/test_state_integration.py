"""
Unit tests for state integration functionality.

Tests the integration of extracted data with LangGraph state management,
including data conversion, validation, and serialization.
"""

import pytest
import json
from datetime import datetime, UTC
from unittest.mock import Mock, AsyncMock

from src.pub_analysis_agent.workflows.state_models import (
    AnalysisState,
    ExtractedCodeSnippet,
    ExtractedExternalLink,
    ExtractedGitHubRepository,
    ExtractionMetadata
)
from src.pub_analysis_agent.workflows.state_converters import (
    convert_code_snippets_to_state,
    convert_external_links_to_state,
    convert_github_repos_to_state,
    convert_extraction_result_to_state,
    validate_state_data_structure,
    convert_state_to_extraction_summary
)


class TestExtractedDataStructures:
    """Test cases for extracted data structures."""
    
    def test_extracted_code_snippet_creation(self):
        """Test ExtractedCodeSnippet creation and validation."""
        snippet = ExtractedCodeSnippet(
            content="import pandas as pd",
            language="python",
            context="Data loading section",
            relevance_score=8.5,
            description="Pandas import statement",
            purpose="data_processing",
            start_position=100,
            end_position=120
        )
        
        assert snippet.content == "import pandas as pd"
        assert snippet.language == "python"
        assert snippet.relevance_score == 8.5
        assert snippet.purpose == "data_processing"
        assert snippet.start_position == 100
    
    def test_extracted_external_link_creation(self):
        """Test ExtractedExternalLink creation and validation."""
        link = ExtractedExternalLink(
            url="https://zenodo.org/record/12345",
            link_type="data_repository",
            title="Research Dataset",
            description="Dataset for ML analysis",
            context="Data availability section",
            is_accessible=True,
            relevance_score=9.0
        )
        
        assert link.url == "https://zenodo.org/record/12345"
        assert link.link_type == "data_repository"
        assert link.is_accessible is True
        assert link.relevance_score == 9.0
    
    def test_extracted_github_repository_creation(self):
        """Test ExtractedGitHubRepository creation and validation."""
        repo = ExtractedGitHubRepository(
            url="https://github.com/user/repo",
            owner="user",
            repository="repo",
            path="src/main.py",
            branch="main",
            is_valid=True,
            description="Machine learning repository",
            language="Python",
            stars=250
        )
        
        assert repo.url == "https://github.com/user/repo"
        assert repo.owner == "user"
        assert repo.repository == "repo"
        assert repo.stars == 250
        assert repo.language == "Python"
    
    def test_extraction_metadata_creation(self):
        """Test ExtractionMetadata creation and validation."""
        metadata = ExtractionMetadata(
            total_code_blocks=5,
            total_links_found=10,
            programming_languages=["python", "r"],
            processing_time=2.5,
            extraction_errors=["Minor warning"],
            extraction_timestamp="2024-01-01T12:00:00Z"
        )
        
        assert metadata.total_code_blocks == 5
        assert metadata.total_links_found == 10
        assert metadata.programming_languages == ["python", "r"]
        assert metadata.processing_time == 2.5
        assert metadata.extraction_errors == ["Minor warning"]


class TestAnalysisStateExtensions:
    """Test cases for AnalysisState extensions."""
    
    @pytest.fixture
    def sample_state(self):
        """Create a sample AnalysisState for testing."""
        return AnalysisState(
            publication_id="test_pub_123",
            workflow_id="workflow_456"
        )
    
    @pytest.fixture
    def sample_extraction_data(self):
        """Create sample extraction data for testing."""
        code_snippets = [
            ExtractedCodeSnippet(
                content="import numpy as np",
                language="python",
                context="imports section",
                relevance_score=7.5
            ),
            ExtractedCodeSnippet(
                content="library(ggplot2)",
                language="r",
                context="visualization section",
                relevance_score=8.0
            )
        ]
        
        external_links = [
            ExtractedExternalLink(
                url="https://zenodo.org/record/12345",
                link_type="data_repository",
                relevance_score=9.0
            )
        ]
        
        github_repos = [
            ExtractedGitHubRepository(
                url="https://github.com/user/repo",
                owner="user",
                repository="repo",
                is_valid=True,
                stars=150
            )
        ]
        
        metadata = ExtractionMetadata(
            total_code_blocks=2,
            total_links_found=2,
            programming_languages=["python", "r"],
            processing_time=1.5
        )
        
        return {
            "code_snippets": code_snippets,
            "external_links": external_links,
            "github_repos": github_repos,
            "metadata": metadata
        }
    
    def test_update_extraction_results(self, sample_state, sample_extraction_data):
        """Test updating state with extraction results."""
        initial_updated_time = sample_state.updated_at
        
        sample_state.update_extraction_results(
            code_snippets=sample_extraction_data["code_snippets"],
            external_links=sample_extraction_data["external_links"],
            github_repos=sample_extraction_data["github_repos"],
            metadata=sample_extraction_data["metadata"]
        )
        
        assert len(sample_state.extracted_code) == 2
        assert len(sample_state.extracted_links) == 1
        assert len(sample_state.extracted_github_repos) == 1
        assert sample_state.extraction_metadata is not None
        assert sample_state.updated_at > initial_updated_time
    
    def test_add_extracted_items_individually(self, sample_state):
        """Test adding extracted items individually."""
        # Add code snippet
        snippet = ExtractedCodeSnippet(
            content="print('hello')",
            language="python",
            context="test",
            relevance_score=5.0
        )
        sample_state.add_extracted_code_snippet(snippet)
        assert len(sample_state.extracted_code) == 1
        
        # Add external link
        link = ExtractedExternalLink(
            url="https://example.com",
            link_type="other",
            relevance_score=6.0
        )
        sample_state.add_extracted_link(link)
        assert len(sample_state.extracted_links) == 1
        
        # Add GitHub repository
        repo = ExtractedGitHubRepository(
            url="https://github.com/test/repo",
            owner="test",
            repository="repo"
        )
        sample_state.add_extracted_github_repo(repo)
        assert len(sample_state.extracted_github_repos) == 1
    
    def test_get_all_extracted_content(self, sample_state, sample_extraction_data):
        """Test getting all extracted content in structured format."""
        sample_state.update_extraction_results(**sample_extraction_data)
        
        all_content = sample_state.get_all_extracted_content()
        
        assert "code_snippets" in all_content
        assert "external_links" in all_content
        assert "github_repositories" in all_content
        assert "metadata" in all_content
        
        assert len(all_content["code_snippets"]) == 2
        assert len(all_content["external_links"]) == 1
        assert len(all_content["github_repositories"]) == 1
    
    def test_has_extraction_results(self, sample_state, sample_extraction_data):
        """Test checking if state has extraction results."""
        assert not sample_state.has_extraction_results()
        
        sample_state.update_extraction_results(**sample_extraction_data)
        assert sample_state.has_extraction_results()
    
    def test_get_extraction_summary(self, sample_state, sample_extraction_data):
        """Test getting extraction summary."""
        sample_state.update_extraction_results(**sample_extraction_data)
        
        summary = sample_state.get_extraction_summary()
        
        assert summary["code_snippets_count"] == 2
        assert summary["external_links_count"] == 1
        assert summary["github_repos_count"] == 1
        assert summary["total_programming_languages"] == 2
    
    def test_validate_extraction_data_integrity(self, sample_state):
        """Test data integrity validation."""
        # Add valid data
        valid_snippet = ExtractedCodeSnippet(
            content="valid code",
            language="python",
            context="test",
            relevance_score=7.0
        )
        sample_state.add_extracted_code_snippet(valid_snippet)
        
        # Add invalid data
        invalid_snippet = ExtractedCodeSnippet(
            content="",  # Empty content
            language="python",
            context="test",
            relevance_score=15.0  # Invalid score
        )
        sample_state.add_extracted_code_snippet(invalid_snippet)
        
        errors = sample_state.validate_extraction_data_integrity()
        
        assert len(errors) >= 2  # Should find empty content and invalid score
        assert any("empty content" in error for error in errors)
        assert any("invalid relevance score" in error for error in errors)


class TestStateSerializationAndDeserialization:
    """Test cases for state serialization and deserialization."""
    
    @pytest.fixture
    def populated_state(self):
        """Create a state populated with extraction data."""
        state = AnalysisState(publication_id="test_123")
        
        # Add extraction data
        snippet = ExtractedCodeSnippet(
            content="import pandas as pd",
            language="python",
            context="imports",
            relevance_score=8.0,
            description="Pandas import"
        )
        state.add_extracted_code_snippet(snippet)
        
        link = ExtractedExternalLink(
            url="https://zenodo.org/record/123",
            link_type="data_repository",
            relevance_score=9.0,
            title="Research Data"
        )
        state.add_extracted_link(link)
        
        repo = ExtractedGitHubRepository(
            url="https://github.com/user/repo",
            owner="user",
            repository="repo",
            is_valid=True,
            stars=50
        )
        state.add_extracted_github_repo(repo)
        
        metadata = ExtractionMetadata(
            total_code_blocks=1,
            total_links_found=2,
            programming_languages=["python"],
            processing_time=1.0
        )
        state.extraction_metadata = metadata
        
        return state
    
    def test_state_serialization_to_dict(self, populated_state):
        """Test serializing state to dictionary."""
        state_dict = populated_state.to_dict()
        
        assert state_dict["publication_id"] == "test_123"
        assert len(state_dict["extracted_code"]) == 1
        assert len(state_dict["extracted_links"]) == 1
        assert len(state_dict["extracted_github_repos"]) == 1
        assert state_dict["extraction_metadata"] is not None
        
        # Check that datetime fields are serialized as strings
        assert isinstance(state_dict["created_at"], str)
        assert isinstance(state_dict["updated_at"], str)
    
    def test_state_serialization_to_json(self, populated_state):
        """Test serializing state to JSON string."""
        json_str = populated_state.to_json()
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["publication_id"] == "test_123"
        assert len(parsed["extracted_code"]) == 1
    
    def test_state_deserialization_from_dict(self, populated_state):
        """Test deserializing state from dictionary."""
        state_dict = populated_state.to_dict()
        restored_state = AnalysisState.from_dict(state_dict)
        
        assert restored_state.publication_id == populated_state.publication_id
        assert len(restored_state.extracted_code) == len(populated_state.extracted_code)
        assert len(restored_state.extracted_links) == len(populated_state.extracted_links)
        assert len(restored_state.extracted_github_repos) == len(populated_state.extracted_github_repos)
        
        # Check that extracted data is properly restored
        assert isinstance(restored_state.extracted_code[0], ExtractedCodeSnippet)
        assert isinstance(restored_state.extracted_links[0], ExtractedExternalLink)
        assert isinstance(restored_state.extracted_github_repos[0], ExtractedGitHubRepository)
        assert isinstance(restored_state.extraction_metadata, ExtractionMetadata)
    
    def test_state_deserialization_from_json(self, populated_state):
        """Test deserializing state from JSON string."""
        json_str = populated_state.to_json()
        restored_state = AnalysisState.from_json(json_str)
        
        assert restored_state.publication_id == populated_state.publication_id
        assert len(restored_state.extracted_code) == len(populated_state.extracted_code)
        assert restored_state.extracted_code[0].content == populated_state.extracted_code[0].content


class TestStateConverters:
    """Test cases for state converter functions."""
    
    def test_convert_code_snippets_to_state(self):
        """Test converting code snippets to state format."""
        # Mock CodeSnippet objects
        mock_snippets = []
        
        # Create a mock snippet with enum language
        mock_snippet1 = Mock()
        mock_snippet1.content = "import pandas as pd"
        mock_snippet1.language = Mock()
        mock_snippet1.language.value = "python"
        mock_snippet1.context = "imports section"
        mock_snippet1.relevance_score = 8.0
        mock_snippet1.description = "Pandas import"
        mock_snippet1.purpose = "data_processing"
        mock_snippet1.start_position = 0
        mock_snippet1.end_position = 20
        mock_snippets.append(mock_snippet1)
        
        # Create a mock snippet with string language
        mock_snippet2 = Mock()
        mock_snippet2.content = "library(ggplot2)"
        mock_snippet2.language = "r"
        mock_snippet2.context = "visualization"
        mock_snippet2.relevance_score = 7.5
        mock_snippet2.description = "R library import"
        mock_snippet2.purpose = "visualization"
        mock_snippet2.start_position = 50
        mock_snippet2.end_position = 70
        mock_snippets.append(mock_snippet2)
        
        converted = convert_code_snippets_to_state(mock_snippets)
        
        assert len(converted) == 2
        assert isinstance(converted[0], ExtractedCodeSnippet)
        assert converted[0].content == "import pandas as pd"
        assert converted[0].language == "python"
        assert converted[1].language == "r"
    
    def test_convert_external_links_to_state(self):
        """Test converting external links to state format."""
        # Mock ExternalLink objects
        mock_links = []
        
        mock_link = Mock()
        mock_link.url = "https://zenodo.org/record/123"
        mock_link.link_type = Mock()
        mock_link.link_type.value = "data_repository"
        mock_link.title = "Research Dataset"
        mock_link.description = "ML dataset"
        mock_link.context = "data section"
        mock_link.is_accessible = True
        mock_link.relevance_score = 9.0
        mock_links.append(mock_link)
        
        converted = convert_external_links_to_state(mock_links)
        
        assert len(converted) == 1
        assert isinstance(converted[0], ExtractedExternalLink)
        assert converted[0].url == "https://zenodo.org/record/123"
        assert converted[0].link_type == "data_repository"
    
    def test_convert_github_repos_to_state(self):
        """Test converting GitHub repositories to state format."""
        # Mock GitHubInfo objects
        mock_repos = []
        
        mock_repo = Mock()
        mock_repo.url = "https://github.com/user/repo"
        mock_repo.owner = "user"
        mock_repo.repository = "repo"
        mock_repo.path = "src/main.py"
        mock_repo.branch = "main"
        mock_repo.is_valid = True
        mock_repo.description = "ML repository"
        mock_repo.language = "Python"
        mock_repo.stars = 150
        mock_repos.append(mock_repo)
        
        converted = convert_github_repos_to_state(mock_repos)
        
        assert len(converted) == 1
        assert isinstance(converted[0], ExtractedGitHubRepository)
        assert converted[0].url == "https://github.com/user/repo"
        assert converted[0].owner == "user"
        assert converted[0].stars == 150
    
    def test_convert_extraction_result_to_state(self):
        """Test converting complete extraction result to state format."""
        # Mock ExtractionResult
        mock_result = Mock()
        mock_result.code_snippets = []
        mock_result.external_links = []
        mock_result.github_repositories = []
        mock_result.programming_languages = {"python", "r"}
        mock_result.total_code_blocks = 0
        mock_result.total_links_found = 0
        mock_result.processing_time = 1.5
        mock_result.errors = []
        
        converted = convert_extraction_result_to_state(mock_result)
        
        assert "code_snippets" in converted
        assert "external_links" in converted
        assert "github_repos" in converted
        assert "metadata" in converted
        assert isinstance(converted["metadata"], ExtractionMetadata)
        assert converted["metadata"].processing_time == 1.5
    
    def test_validate_state_data_structure(self):
        """Test validation of state data structure."""
        # Valid structure
        valid_data = {
            "code_snippets": [ExtractedCodeSnippet("code", "python", "context", 7.0)],
            "external_links": [ExtractedExternalLink("https://example.com", "other")],
            "github_repos": [ExtractedGitHubRepository("https://github.com/u/r", "u", "r")],
            "metadata": ExtractionMetadata()
        }
        
        errors = validate_state_data_structure(valid_data)
        assert len(errors) == 0
        
        # Invalid structure - missing field
        invalid_data = {
            "code_snippets": [],
            "external_links": []
            # Missing github_repos and metadata
        }
        
        errors = validate_state_data_structure(invalid_data)
        assert len(errors) >= 2  # Should report missing fields
    
    def test_convert_state_to_extraction_summary(self):
        """Test converting state to extraction summary."""
        state = AnalysisState(publication_id="test")
        
        # Add some extraction data
        state.add_extracted_code_snippet(
            ExtractedCodeSnippet("code", "python", "context", 8.0)
        )
        state.add_extracted_link(
            ExtractedExternalLink("https://example.com", "other", relevance_score=7.0, is_accessible=True)
        )
        state.add_extracted_github_repo(
            ExtractedGitHubRepository("https://github.com/u/r", "u", "r", is_valid=True, language="Python", stars=100)
        )
        
        summary = convert_state_to_extraction_summary(state)
        
        assert summary["total_items"] == 3
        assert summary["code_snippets"]["count"] == 1
        assert summary["external_links"]["count"] == 1
        assert summary["github_repositories"]["count"] == 1
        assert summary["external_links"]["accessible_count"] == 1
        assert summary["github_repositories"]["total_stars"] == 100


if __name__ == "__main__":
    pytest.main([__file__]) 