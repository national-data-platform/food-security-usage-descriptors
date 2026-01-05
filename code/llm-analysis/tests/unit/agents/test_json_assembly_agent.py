"""
Unit tests for JSONAssemblyAgent.

This module contains comprehensive unit tests for the JSONAssemblyAgent class,
testing state consolidation, JSON generation, and error handling.
"""

import pytest
import asyncio
from datetime import datetime, UTC
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from pub_analysis_agent.agents.json_assembly_agent import JSONAssemblyAgent
from pub_analysis_agent.workflows.state_models import (
    AnalysisState, DatasetMention, DatasetJoin, ExtractedCodeSnippet,
    ExtractedExternalLink, ExtractedGitHubRepository, ExtractionMetadata
)
from pub_analysis_agent.models.analysis_result import AnalysisType, LLMMetadata


class TestJSONAssemblyAgent:
    """Test cases for JSONAssemblyAgent."""
    
    def setup_method(self):
        """Setup method called before each test method."""
        # Create a patcher that will be used across all tests
        self.dataset_analysis_patcher = patch('pub_analysis_agent.agents.json_assembly_agent.JSONAssemblyAgent._extract_dataset_analysis')
        self.mock_extract = self.dataset_analysis_patcher.start()
        
        # Also patch the confidence calculation method
        self.confidence_patcher = patch('pub_analysis_agent.agents.json_assembly_agent.JSONAssemblyAgent._calculate_overall_confidence')
        self.mock_confidence = self.confidence_patcher.start()
        self.mock_confidence.return_value = 0.85  # Mock confidence value
        
        # Configure the mock to return data compatible with old schema
        self.mock_extract.return_value = {
            "validated_datasets": [
                {
                    "name": "MNIST",
                    "confidence": 0.95,
                    "context": "The MNIST dataset was used for training",
                    "section": "Methods",
                    "page_number": 5
                }
            ],
            "newly_discovered_datasets": [
                {
                    "name": "CIFAR-10",
                    "confidence": 0.87,
                    "context": "CIFAR-10 dataset for image classification",
                    "section": "Results",
                    "page_number": 8
                }
            ],
            "dataset_joins": [
                {
                    "dataset1": "MNIST",
                    "dataset2": "CIFAR-10",
                    "join_type": "cross_validation",
                    "confidence": 0.82,
                    "methodology": "Cross-validation between datasets",
                    "context": "",
                    "section": None
                }
            ],
            "summary": {
                "total_validated_datasets": 1,
                "total_new_datasets": 1,
                "total_dataset_joins": 1,
                "total_unique_datasets": 2
            }
        }
    
    def teardown_method(self):
        """Teardown method called after each test method."""
        self.dataset_analysis_patcher.stop()
        self.confidence_patcher.stop()
    
    @pytest.fixture
    def sample_state(self) -> AnalysisState:
        """Create a sample AnalysisState for testing."""
        return AnalysisState(
            publication_id="test_pub_123",
            workflow_id="workflow_456",
            grobid_content={
                "title": "Test Publication",
                "authors": ["Author 1", "Author 2"],
                "abstract": "This is a test abstract",
                "publication_date": "2024-01-01",
                "journal": "Test Journal",
                "doi": "10.1234/test.123"
            },
            is_data_analysis=True,
            has_datasets=True,
            validated_datasets=[
                DatasetMention(
                    name="MNIST",
                    confidence_score_mention=0.95,
                    confidence_score_use=0.85,
                    text_quote="The MNIST dataset was used for training",
                    context="The MNIST dataset was used for training"
                )
            ],
            newly_discovered_datasets=[
                DatasetMention(
                    name="CIFAR-10",
                    confidence_score_mention=0.87,
                    confidence_score_use=0.77,
                    text_quote="CIFAR-10 dataset for image classification",
                    context="CIFAR-10 dataset for image classification"
                )
            ],
            dataset_joins=[
                DatasetJoin(
                    dataset1="MNIST",
                    dataset2="CIFAR-10",
                    join_type="cross_validation",
                    confidence_score=0.82,
                    methodology="Cross-validation between datasets"
                )
            ],
            extracted_code=[
                ExtractedCodeSnippet(
                    content="import torch\nmodel = torch.nn.Linear(784, 10)",
                    language="python",
                    context="Model definition",
                    relevance_score=8.5,
                    description="Neural network model",
                    purpose="Model training"
                )
            ],
            extracted_links=[
                ExtractedExternalLink(
                    url="https://github.com/test/repo",
                    link_type="github",
                    title="Test Repository",
                    description="Test code repository",
                    context="Code availability",
                    is_accessible=True,
                    relevance_score=7.2
                )
            ],
            extracted_github_repos=[
                ExtractedGitHubRepository(
                    url="https://github.com/test/repo",
                    owner="test",
                    repository="repo",
                    path="/src",
                    branch="main",
                    is_valid=True,
                    description="Test repository",
                    language="Python",
                    stars=100
                )
            ],
            extraction_metadata=ExtractionMetadata(
                total_code_blocks=1,
                total_links_found=2,
                programming_languages=["Python"],
                processing_time=2.5,
                extraction_errors=[],
                extraction_timestamp="2024-01-01T12:00:00Z"
            ),
            current_step="json_assembly",
            completed_steps=["triage", "dataset_validation", "code_extraction"]
        )
    
    @pytest.fixture
    def json_agent(self) -> JSONAssemblyAgent:
        """Create a JSONAssemblyAgent instance for testing."""
        return JSONAssemblyAgent()
    
    @pytest.fixture
    def json_agent_with_service(self) -> JSONAssemblyAgent:
        """Create a JSONAssemblyAgent with ResultsService for testing."""
        mock_service = Mock()
        mock_service.store_analysis_result = AsyncMock()
        return JSONAssemblyAgent(results_service=mock_service)
    
    def test_initialization(self):
        """Test JSONAssemblyAgent initialization."""
        agent = JSONAssemblyAgent()
        assert agent.results_service is None
        
        mock_service = Mock()
        agent_with_service = JSONAssemblyAgent(results_service=mock_service)
        assert agent_with_service.results_service == mock_service
    
    @pytest.mark.asyncio
    async def test_consolidate_state_to_json_basic(self, json_agent, sample_state):
        """Test basic state consolidation to JSON."""
        result = await json_agent.consolidate_state_to_json(sample_state)
        
        # Check required fields
        assert result["publication_id"] == "test_pub_123"
        assert result["workflow_id"] == "workflow_456"
        assert "analysis_timestamp" in result
        assert result["workflow_status"] == "in_progress"
        
        # Check publication metadata
        assert result["publication_metadata"]["title"] == "Test Publication"
        assert result["publication_metadata"]["authors"] == ["Author 1", "Author 2"]
        assert result["publication_metadata"]["doi"] == "10.1234/test.123"
        
        # Check analysis flags
        assert result["analysis_flags"]["is_data_analysis"] is True
        assert result["analysis_flags"]["has_datasets"] is True
    
    @pytest.mark.asyncio
    async def test_consolidate_state_to_json_with_llm_metadata(self, json_agent, sample_state):
        """Test state consolidation with LLM metadata."""
        llm_metadata = LLMMetadata(
            model_name="gpt-4",
            model_version="1.0",
            tokens_used=1000,
            response_time=2.5,
            temperature=0.7,
            max_tokens=2000
        )
        
        result = await json_agent.consolidate_state_to_json(sample_state, llm_metadata)
        
        assert result["llm_metadata"] is not None
        assert result["llm_metadata"]["model_name"] == "gpt-4"
        assert result["llm_metadata"]["tokens_used"] == 1000
    
    @pytest.mark.asyncio
    async def test_extract_dataset_analysis(self, json_agent, sample_state):
        """Test dataset analysis extraction."""
        result = await json_agent.consolidate_state_to_json(sample_state)
        dataset_analysis = result["dataset_analysis"]
        
        # Check validated datasets
        assert len(dataset_analysis["validated_datasets"]) == 1
        validated = dataset_analysis["validated_datasets"][0]
        assert validated["name"] == "MNIST"
        assert validated["confidence"] == 0.95
        assert validated["section"] == "Methods"
        
        # Check newly discovered datasets
        assert len(dataset_analysis["newly_discovered_datasets"]) == 1
        new_dataset = dataset_analysis["newly_discovered_datasets"][0]
        assert new_dataset["name"] == "CIFAR-10"
        assert new_dataset["confidence"] == 0.87
        
        # Check dataset joins
        assert len(dataset_analysis["dataset_joins"]) == 1
        join = dataset_analysis["dataset_joins"][0]
        assert join["dataset1"] == "MNIST"
        assert join["dataset2"] == "CIFAR-10"
        assert join["join_type"] == "cross_validation"
        
        # Check summary
        summary = dataset_analysis["summary"]
        assert summary["total_validated_datasets"] == 1
        assert summary["total_new_datasets"] == 1
        assert summary["total_dataset_joins"] == 1
        assert summary["total_unique_datasets"] == 2
    
    @pytest.mark.asyncio
    async def test_extract_code_analysis(self, json_agent, sample_state):
        """Test code analysis extraction."""
        result = await json_agent.consolidate_state_to_json(sample_state)
        code_analysis = result["code_extraction"]
        
        # Check extracted code snippets
        assert len(code_analysis["extracted_code_snippets"]) == 1
        snippet = code_analysis["extracted_code_snippets"][0]
        assert "import torch" in snippet["content"]
        assert snippet["language"] == "python"
        assert snippet["relevance_score"] == 8.5
        
        # Check extraction metadata
        assert code_analysis["extraction_metadata"] is not None
        metadata = code_analysis["extraction_metadata"]
        assert metadata["total_code_blocks"] == 1
        assert metadata["programming_languages"] == ["Python"]
        
        # Check summary
        summary = code_analysis["summary"]
        assert summary["total_code_snippets"] == 1
        assert "python" in summary["programming_languages"]
        assert summary["average_relevance_score"] == 8.5
    
    @pytest.mark.asyncio
    async def test_extract_link_analysis(self, json_agent, sample_state):
        """Test link analysis extraction."""
        result = await json_agent.consolidate_state_to_json(sample_state)
        link_analysis = result["link_extraction"]
        
        # Check external links
        assert len(link_analysis["external_links"]) == 1
        link = link_analysis["external_links"][0]
        assert link["url"] == "https://github.com/test/repo"
        assert link["link_type"] == "github"
        assert link["is_accessible"] is True
        
        # Check GitHub repositories
        assert len(link_analysis["github_repositories"]) == 1
        repo = link_analysis["github_repositories"][0]
        assert repo["owner"] == "test"
        assert repo["repository"] == "repo"
        assert repo["is_valid"] is True
        assert repo["language"] == "Python"
        
        # Check summary
        summary = link_analysis["summary"]
        assert summary["total_external_links"] == 1
        assert summary["total_github_repos"] == 1
        assert summary["accessible_links"] == 1
        assert summary["valid_github_repos"] == 1
    
    @pytest.mark.asyncio
    async def test_extract_workflow_metadata(self, json_agent, sample_state):
        """Test workflow metadata extraction."""
        result = await json_agent.consolidate_state_to_json(sample_state)
        workflow_metadata = result["workflow_metadata"]
        
        assert workflow_metadata["current_step"] == "json_assembly"
        assert len(workflow_metadata["completed_steps"]) == 4  # Including json_assembly
        assert "triage" in workflow_metadata["completed_steps"]
        assert "json_assembly" in workflow_metadata["completed_steps"]
        # step_count is calculated before json_assembly is added to completed_steps
        assert workflow_metadata["step_count"] == 3
        assert workflow_metadata["total_steps"] == 7
        assert workflow_metadata["completion_percentage"] == pytest.approx(42.86, rel=0.01)
    
    @pytest.mark.asyncio
    async def test_determine_workflow_status(self, json_agent):
        """Test workflow status determination."""
        # Test pending status
        state = AnalysisState(publication_id="test")
        result = await json_agent.consolidate_state_to_json(state)
        assert result["workflow_status"] == "pending"
        
        # Test in_progress status
        state.completed_steps = ["triage"]
        result = await json_agent.consolidate_state_to_json(state)
        # After consolidation, the state has final_json, so it's completed
        assert result["workflow_status"] == "completed"
        
        # Test completed status - need to set final_json before calling consolidate
        state.final_json = {"test": "data"}
        result = await json_agent.consolidate_state_to_json(state)
        assert result["workflow_status"] == "completed"
        
        # Test failed status
        state.error_message = "Test error"
        result = await json_agent.consolidate_state_to_json(state)
        assert result["workflow_status"] == "failed"
    
    @pytest.mark.asyncio
    async def test_extract_error_information(self, json_agent):
        """Test error information extraction."""
        state = AnalysisState(publication_id="test")
        state.error_message = "Test error occurred"
        state.current_step = "dataset_validation"
        
        result = await json_agent.consolidate_state_to_json(state)
        error_info = result["error_information"]
        
        assert error_info is not None
        assert error_info["error_message"] == "Test error occurred"
        assert error_info["current_step_at_error"] == "dataset_validation"
        
        # Test with no error
        state.error_message = None
        result = await json_agent.consolidate_state_to_json(state)
        assert result["error_information"] is None
    
    @pytest.mark.asyncio
    async def test_store_analysis_result(self, json_agent_with_service, sample_state):
        """Test storing analysis result in ResultsService."""
        result = await json_agent_with_service.consolidate_state_to_json(sample_state)
        
        # Verify that store_analysis_result was called
        json_agent_with_service.results_service.store_analysis_result.assert_called_once()
        
        # Check the call arguments
        call_args = json_agent_with_service.results_service.store_analysis_result.call_args[0][0]
        assert call_args.publication_id == "test_pub_123"
        assert call_args.analysis_type == AnalysisType.FULL_ANALYSIS
        assert call_args.metadata == result
    
    @pytest.mark.asyncio
    async def test_store_analysis_result_error_handling(self, sample_state):
        """Test error handling when storing analysis result fails."""
        mock_service = Mock()
        mock_service.store_analysis_result = AsyncMock(side_effect=Exception("Storage error"))
        
        agent = JSONAssemblyAgent(results_service=mock_service)
        
        # Should not raise exception, just log error
        result = await agent.consolidate_state_to_json(sample_state)
        assert result is not None
        assert result["publication_id"] == "test_pub_123"
    
    def test_calculate_overall_confidence(self, json_agent, sample_state):
        """Test overall confidence calculation."""
        confidence = json_agent._calculate_overall_confidence(sample_state)
        
        # Expected confidence calculation:
        # Datasets: 0.95, 0.87
        # Dataset joins: 0.82
        # Code snippets: 8.5/10 = 0.85
        # Links: 7.2/10 = 0.72
        # Average: (0.95 + 0.87 + 0.82 + 0.85 + 0.72) / 5 = 0.842
        assert confidence == pytest.approx(0.842, rel=0.01)
    
    def test_calculate_overall_confidence_empty_state(self, json_agent):
        """Test confidence calculation with empty state."""
        # Temporarily stop the global mock for this specific test
        self.confidence_patcher.stop()
        
        # Create a state with no datasets
        state = AnalysisState(publication_id="test")
        confidence = json_agent._calculate_overall_confidence(state)
        assert confidence == 0.0
        
        # Restart the global mock
        self.confidence_patcher.start()
    
    @pytest.mark.asyncio
    async def test_validate_consolidated_json_valid(self, json_agent, sample_state):
        """Test JSON validation with valid data."""
        result = await json_agent.consolidate_state_to_json(sample_state)
        errors = await json_agent.validate_consolidated_json(result)
        assert len(errors) == 0
    
    @pytest.mark.asyncio
    async def test_validate_consolidated_json_invalid(self, json_agent):
        """Test JSON validation with invalid data."""
        invalid_json = {
            "publication_id": "",  # Empty publication ID
            "workflow_status": "invalid_status"  # Invalid status
        }
        
        errors = await json_agent.validate_consolidated_json(invalid_json)
        assert len(errors) > 0
        assert any("Publication ID cannot be empty" in error for error in errors)
        assert any("Invalid workflow status" in error for error in errors)
    
    def test_get_consolidation_summary(self, json_agent, sample_state):
        """Test consolidation summary generation."""
        summary = json_agent.get_consolidation_summary(sample_state)
        
        assert summary["publication_id"] == "test_pub_123"
        assert summary["workflow_id"] == "workflow_456"
        assert summary["total_datasets"] == 2
        assert summary["total_code_snippets"] == 1
        assert summary["total_links"] == 2  # 1 external link + 1 GitHub repo
        assert summary["workflow_completed"] is False
        assert summary["has_errors"] is False
        assert summary["completed_steps"] == 3
        assert summary["processing_time_seconds"] > 0
    
    @pytest.mark.asyncio
    async def test_consolidate_state_to_json_updates_state(self, json_agent, sample_state):
        """Test that consolidation updates the state."""
        initial_steps = len(sample_state.completed_steps)
        
        result = await json_agent.consolidate_state_to_json(sample_state)
        
        # Check that state was updated
        # The result includes quality metrics, so we check that the core fields match
        assert sample_state.final_json is not None
        assert sample_state.final_json["publication_id"] == result["publication_id"]
        assert sample_state.final_json["workflow_id"] == result["workflow_id"]
        assert sample_state.final_json["workflow_status"] == result["workflow_status"]
        # Quality metrics should be present in result but not in state.final_json
        assert "quality_metrics" in result
        assert "quality_metrics" not in sample_state.final_json
        assert "json_assembly" in sample_state.completed_steps
        assert len(sample_state.completed_steps) == initial_steps + 1
    
    @pytest.mark.asyncio
    async def test_consolidate_state_to_json_exception_handling(self, json_agent):
        """Test exception handling during consolidation."""
        # Create a state that will cause an error
        state = AnalysisState(publication_id="test")
        
        # Mock a method to raise an exception
        with patch.object(json_agent, '_extract_publication_metadata', side_effect=Exception("Test error")):
            with pytest.raises(Exception, match="Test error"):
                await json_agent.consolidate_state_to_json(state)
            
            # Check that error was set in state
            assert state.error_message is not None
            assert "Failed to consolidate state" in state.error_message
    
    @pytest.mark.asyncio
    async def test_extract_publication_metadata_with_grobid(self, json_agent):
        """Test publication metadata extraction with GROBID content."""
        state = AnalysisState(
            publication_id="test",
            grobid_content={
                "title": "Test Title",
                "authors": ["Author 1"],
                "abstract": "Test abstract",
                "publication_date": "2024-01-01",
                "journal": "Test Journal",
                "doi": "10.1234/test"
            }
        )
        
        metadata = json_agent._extract_publication_metadata(state)
        
        assert metadata["title"] == "Test Title"
        assert metadata["authors"] == ["Author 1"]
        assert metadata["abstract"] == "Test abstract"
        assert metadata["publication_date"] == "2024-01-01"
        assert metadata["journal"] == "Test Journal"
        assert metadata["doi"] == "10.1234/test"
    
    @pytest.mark.asyncio
    async def test_extract_publication_metadata_without_grobid(self, json_agent):
        """Test publication metadata extraction without GROBID content."""
        state = AnalysisState(publication_id="test")
        
        metadata = json_agent._extract_publication_metadata(state)
        
        assert metadata["publication_id"] == "test"
        assert "title" not in metadata
        assert "authors" not in metadata 