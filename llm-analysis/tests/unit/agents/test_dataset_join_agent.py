"""
Unit tests for DatasetJoinAnalysisAgent.

Tests the dataset join analysis functionality including LLM integration,
text processing, and result validation.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from datetime import datetime, UTC
from typing import List, Dict, Any

from pub_analysis_agent.agents.dataset_join_agent import (
    DatasetJoinAnalysisAgent,
    JoinAnalysisConfig,
    DatasetJoinAnalysis,
    IntegrationChallenge,
    LessonLearned,
    ValidationMethod,
    RiskAssessment,
    JoinAnalysisResult,
    dataset_join_analysis_agent_step
)
from pub_analysis_agent.services.llm_service import LLMService
from pub_analysis_agent.services.dataset_service import DatasetService
from pub_analysis_agent.models.dataset import Dataset
from pub_analysis_agent.workflows.state_models import AnalysisState, DatasetJoin


class TestJoinAnalysisConfig:
    """Test JoinAnalysisConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = JoinAnalysisConfig()
        
        assert config.confidence_threshold == 6.0
        assert config.max_text_length == 10000
        assert config.temperature == 0.1
        assert config.max_tokens == 800
        assert config.context_window == 500
        assert config.min_join_confidence == 5.0
        assert config.max_joins_per_call == 10
        assert config.enable_methodology_extraction is True
        assert config.enable_challenge_documentation is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = JoinAnalysisConfig(
            confidence_threshold=8.0,
            max_text_length=5000,
            temperature=0.2,
            enable_methodology_extraction=False
        )
        
        assert config.confidence_threshold == 8.0
        assert config.max_text_length == 5000
        assert config.temperature == 0.2
        assert config.enable_methodology_extraction is False


class TestDatasetJoinAnalysis:
    """Test DatasetJoinAnalysis dataclass."""
    
    def test_basic_creation(self):
        """Test basic join analysis creation."""
        join = DatasetJoinAnalysis(
            dataset1="Dataset A",
            dataset2="Dataset B",
            join_type="merge",
            confidence_score=8.5,
            context="Datasets were merged for analysis"
        )
        
        assert join.dataset1 == "Dataset A"
        assert join.dataset2 == "Dataset B"
        assert join.join_type == "merge"
        assert join.confidence_score == 8.5
        assert join.context == "Datasets were merged for analysis"
        assert join.methodology is None
        assert join.integration_challenges is None
    
    def test_full_creation(self):
        """Test full join analysis creation with all fields."""
        join = DatasetJoinAnalysis(
            dataset1="Comprehensive Dataset A",
            dataset2="Comprehensive Dataset B",
            join_type="fusion",
            confidence_score=9.0,
            methodology="Key-based join using common identifiers",
            join_keys=["id", "timestamp"],
            integration_challenges=["Schema mismatch", "Data quality issues"],
            success_metrics={"data_loss": "2%", "success_rate": "95%"},
            context="Full integration context",
            section="Methodology",
            text_position=100,
            analysis_reasoning="Clear integration patterns identified",
            software_tools=["Python", "Pandas"],
            programming_language="Python",
            data_preprocessing_steps=["Cleaning", "Normalization"],
            quality_control_measures=["Validation", "Cross-checking"]
        )
        
        assert join.dataset1 == "Comprehensive Dataset A"
        assert join.dataset2 == "Comprehensive Dataset B"
        assert join.join_type == "fusion"
        assert join.confidence_score == 9.0
        assert join.methodology == "Key-based join using common identifiers"
        assert join.join_keys == ["id", "timestamp"]
        assert join.integration_challenges == ["Schema mismatch", "Data quality issues"]
        assert join.success_metrics == {"data_loss": "2%", "success_rate": "95%"}
        assert join.software_tools == ["Python", "Pandas"]
        assert join.programming_language == "Python"


class TestJoinAnalysisResult:
    """Test JoinAnalysisResult dataclass."""
    
    def test_basic_creation(self):
        """Test basic result creation."""
        result = JoinAnalysisResult(
            joins_identified=[],
            total_joins_found=0,
            methodology_details_extracted=0,
            challenges_documented=0,
            processing_time=1.5,
            llm_calls_made=2
        )
        
        assert result.total_joins_found == 0
        assert result.methodology_details_extracted == 0
        assert result.challenges_documented == 0
        assert result.processing_time == 1.5
        assert result.llm_calls_made == 2
        assert result.errors == []
    
    def test_with_joins(self):
        """Test result creation with joins."""
        joins = [
            DatasetJoinAnalysis(
                dataset1="Dataset A",
                dataset2="Dataset B",
                join_type="merge",
                confidence_score=8.0,
                context="Test join"
            )
        ]
        
        result = JoinAnalysisResult(
            joins_identified=joins,
            total_joins_found=1,
            methodology_details_extracted=1,
            challenges_documented=1,
            processing_time=2.5,
            llm_calls_made=3,
            errors=["Test error"]
        )
        
        assert result.total_joins_found == 1
        assert result.methodology_details_extracted == 1
        assert result.challenges_documented == 1
        assert len(result.joins_identified) == 1
        assert result.errors == ["Test error"]


class TestDatasetJoinAnalysisAgent:
    """Test DatasetJoinAnalysisAgent class."""
    
    @pytest.fixture
    def mock_llm_service(self):
        """Create mock LLM service."""
        return AsyncMock(spec=LLMService)
    
    @pytest.fixture
    def mock_dataset_service(self):
        """Create mock dataset service."""
        return AsyncMock(spec=DatasetService)
    
    @pytest.fixture
    def agent(self, mock_llm_service, mock_dataset_service):
        """Create agent instance with mocked services."""
        return DatasetJoinAnalysisAgent(mock_llm_service, mock_dataset_service)
    
    @pytest.fixture
    def sample_state(self):
        """Create sample analysis state."""
        state = AnalysisState(
            publication_id="test_pub_123",
            grobid_content={
                "metadata": {"title": "Test Publication"},
                "fulltext": {
                    "abstract": {
                        "sections": [{"text": "This study combines Dataset A and Dataset B."}]
                    },
                    "body": {
                        "sections": [
                            {
                                "heading": "Methodology",
                                "sentences": [
                                    {"text": "We merged Dataset A with Dataset B using common keys."}
                                ]
                            }
                        ]
                    }
                }
            }
        )
        state.raw_text = "This study combines Dataset A and Dataset B. We merged Dataset A with Dataset B using common keys."
        return state
    
    def test_initialization(self, agent):
        """Test agent initialization."""
        assert agent.llm_service is not None
        assert agent.dataset_service is not None
        assert agent.config is not None
        assert agent._known_datasets_cache is None
        assert agent._dataset_names_set is None
    
    def test_initialization_with_config(self, mock_llm_service, mock_dataset_service):
        """Test agent initialization with custom config."""
        config = JoinAnalysisConfig(confidence_threshold=8.0)
        agent = DatasetJoinAnalysisAgent(mock_llm_service, mock_dataset_service, config)
        
        assert agent.config.confidence_threshold == 8.0
    
    def test_setup_prompt_templates(self, agent):
        """Test prompt template setup."""
        assert agent.join_detection_prompt is not None
        assert agent.methodology_extraction_prompt is not None
        assert agent.challenge_documentation_prompt is not None
    
    def test_extract_text_content(self, agent, sample_state):
        """Test text content extraction."""
        text = agent._extract_text_content(sample_state)
        
        assert "Dataset A" in text
        assert "Dataset B" in text
        assert "METHODOLOGY" in text
    
    def test_extract_text_content_empty_state(self, agent):
        """Test text extraction with empty state."""
        empty_state = AnalysisState(publication_id="test")
        text = agent._extract_text_content(empty_state)
        
        assert text == ""
    
    def test_extract_text_content_no_grobid(self, agent):
        """Test text extraction without GROBID content."""
        state = AnalysisState(
            publication_id="test",
            raw_text="This combines Dataset A and Dataset B."
        )
        text = agent._extract_text_content(state)
        
        assert "Dataset A" in text
        assert "Dataset B" in text
    
    def test_preprocess_text(self, agent):
        """Test text preprocessing."""
        text = "  This is a test   text  with   multiple   spaces.  "
        processed = agent._preprocess_text(text)
        
        assert processed == "This is a test text with multiple spaces."
    
    def test_preprocess_text_truncation(self, agent):
        """Test text preprocessing with truncation."""
        long_text = "A" * (agent.config.max_text_length + 100)
        processed = agent._preprocess_text(long_text)
        
        assert len(processed) <= agent.config.max_text_length
        assert processed.endswith("...")
    
    def test_preprocess_text_empty(self, agent):
        """Test preprocessing empty text."""
        assert agent._preprocess_text("") == ""
        assert agent._preprocess_text(None) == ""
    
    @pytest.mark.asyncio
    async def test_get_known_datasets(self, agent, mock_dataset_service):
        """Test getting known datasets."""
        mock_datasets = [
            Dataset(name="Dataset A", domain="Computer Science"),
            Dataset(name="Dataset B", domain="Biology")
        ]
        mock_dataset_service.get_all_known_datasets.return_value = mock_datasets
        
        datasets = await agent._get_known_datasets()
        
        assert len(datasets) == 2
        assert datasets[0].name == "Dataset A"
        assert datasets[1].name == "Dataset B"
        assert agent._dataset_names_set == {"dataset a", "dataset b"}
    
    @pytest.mark.asyncio
    async def test_get_known_datasets_error(self, agent, mock_dataset_service):
        """Test getting known datasets with error."""
        mock_dataset_service.get_all_known_datasets.side_effect = Exception("Database error")
        
        datasets = await agent._get_known_datasets()
        
        assert datasets == []
        assert agent._dataset_names_set == set()
    
    def test_parse_join_detection_response_valid_json(self, agent):
        """Test parsing valid JSON response."""
        response = '''
        {
          "dataset_joins": [
            {
              "dataset1": "Dataset A",
              "dataset2": "Dataset B",
              "join_type": "merge",
              "confidence_score": 8.5,
              "context": "Datasets were merged",
              "section": "methodology",
              "analysis_reasoning": "Clear integration pattern"
            }
          ]
        }
        '''
        
        joins = agent._parse_join_detection_response(response, "test text")
        
        assert len(joins) == 1
        assert joins[0].dataset1 == "Dataset A"
        assert joins[0].dataset2 == "Dataset B"
        assert joins[0].join_type == "merge"
        assert joins[0].confidence_score == 8.5
    
    def test_parse_join_detection_response_invalid_json(self, agent):
        """Test parsing invalid JSON response."""
        response = "invalid json"
        
        joins = agent._parse_join_detection_response(response, "test text")
        
        # Should use fallback extraction
        assert isinstance(joins, list)
    
    def test_parse_join_detection_response_missing_fields(self, agent):
        """Test parsing response with missing fields."""
        response = '''
        {
          "dataset_joins": [
            {
              "dataset1": "Dataset A",
              "dataset2": "Dataset B"
            }
          ]
        }
        '''
        
        joins = agent._parse_join_detection_response(response, "test text")
        
        assert len(joins) == 1
        assert joins[0].dataset1 == "Dataset A"
        assert joins[0].dataset2 == "Dataset B"
        assert joins[0].join_type == "unknown"
        assert joins[0].confidence_score == 0
    
    def test_fallback_join_extraction(self, agent):
        """Test fallback join extraction."""
        response = "Dataset A and Dataset B"
        
        joins = agent._fallback_join_extraction(response, "test text")
        
        assert len(joins) == 1
        assert joins[0].dataset1 == "Dataset A"
        assert joins[0].dataset2 == "Dataset B"
        assert joins[0].confidence_score == 5.0
    
    def test_parse_methodology_response(self, agent):
        """Test parsing methodology response."""
        response = '''
        {
          "methodology": "Key-based join using patient_id with data preprocessing",
          "join_keys": ["patient_id", "timestamp"],
          "software_tools": ["Python", "Pandas"],
          "programming_language": "Python",
          "data_preprocessing_steps": ["Cleaning", "Normalization"],
          "quality_control_measures": ["Validation", "Cross-checking"],
          "integration_approach": "key-based",
          "join_algorithm": "hash_join",
          "matching_strategy": "exact_match"
        }
        '''
        
        join = DatasetJoinAnalysis(
            dataset1="Dataset A",
            dataset2="Dataset B",
            join_type="merge",
            confidence_score=8.0
        )
        
        agent._parse_methodology_response(response, join)
        
        assert join.methodology == "Key-based join using patient_id with data preprocessing"
        assert join.join_keys == ["patient_id", "timestamp"]
        assert join.software_tools == ["Python", "Pandas"]
        assert join.programming_language == "Python"
        assert join.data_preprocessing_steps == ["Cleaning", "Normalization"]
        assert join.quality_control_measures == ["Validation", "Cross-checking"]
        assert join.integration_approach == "key-based"
        assert join.join_algorithm == "hash_join"
        assert join.matching_strategy == "exact_match"
    
    def test_parse_methodology_response_fuzzy_matching(self, agent):
        """Test parsing methodology response with fuzzy matching approach."""
        response = '''
        {
          "methodology": "Fuzzy matching using string similarity algorithms",
          "join_keys": ["name", "address"],
          "software_tools": ["Python", "rapidfuzz"],
          "programming_language": "Python",
          "data_preprocessing_steps": ["String normalization", "Tokenization"],
          "quality_control_measures": ["Similarity threshold validation"],
          "integration_approach": "fuzzy_matching",
          "join_algorithm": "similarity_based",
          "matching_strategy": "fuzzy_match"
        }
        '''
        
        join = DatasetJoinAnalysis(
            dataset1="Dataset A",
            dataset2="Dataset B",
            join_type="fusion",
            confidence_score=7.5
        )
        
        agent._parse_methodology_response(response, join)
        
        assert join.integration_approach == "fuzzy_matching"
        assert join.join_algorithm == "similarity_based"
        assert join.matching_strategy == "fuzzy_match"
        assert "rapidfuzz" in join.software_tools
    
    def test_parse_methodology_response_record_linkage(self, agent):
        """Test parsing methodology response with record linkage approach."""
        response = '''
        {
          "methodology": "Probabilistic record linkage using blocking strategies",
          "join_keys": ["ssn", "dob", "name"],
          "software_tools": ["R", "RecordLinkage"],
          "programming_language": "R",
          "data_preprocessing_steps": ["Blocking", "Comparison"],
          "quality_control_measures": ["EM algorithm validation"],
          "integration_approach": "record_linkage",
          "join_algorithm": "blocking",
          "matching_strategy": "probabilistic"
        }
        '''
        
        join = DatasetJoinAnalysis(
            dataset1="Dataset A",
            dataset2="Dataset B",
            join_type="linkage",
            confidence_score=9.0
        )
        
        agent._parse_methodology_response(response, join)
        
        assert join.integration_approach == "record_linkage"
        assert join.join_algorithm == "blocking"
        assert join.matching_strategy == "probabilistic"
        assert "R" in join.programming_language
    
    def test_validate_methodology_data_invalid_approach(self, agent):
        """Test validation of methodology data with invalid integration approach."""
        join = DatasetJoinAnalysis(
            dataset1="Dataset A",
            dataset2="Dataset B",
            join_type="merge",
            confidence_score=8.0
        )
        
        # Set invalid values
        join.integration_approach = "invalid_approach"
        join.join_algorithm = "invalid_algorithm"
        join.matching_strategy = "invalid_strategy"
        
        # Validate should correct these
        agent._validate_methodology_data(join)
        
        assert join.integration_approach == "other"
        assert join.join_algorithm == "other"
        assert join.matching_strategy == "other"
    
    def test_validate_methodology_data_none_lists(self, agent):
        """Test validation of methodology data with None lists."""
        join = DatasetJoinAnalysis(
            dataset1="Dataset A",
            dataset2="Dataset B",
            join_type="merge",
            confidence_score=8.0
        )
        
        # Set None values
        join.join_keys = None
        join.software_tools = None
        join.data_preprocessing_steps = None
        join.quality_control_measures = None
        
        # Validate should initialize empty lists
        agent._validate_methodology_data(join)
        
        assert join.join_keys == []
        assert join.software_tools == []
        assert join.data_preprocessing_steps == []
        assert join.quality_control_measures == []
    
    def test_parse_challenge_response(self, agent):
        """Test parsing challenge response with structured data."""
        response = '''
        {
          "integration_challenges": [
            {
              "category": "data_quality",
              "description": "Missing values in key fields",
              "severity": "high",
              "impact": "Reduced dataset size by 15%"
            },
            {
              "category": "schema_mismatch",
              "description": "Different field naming conventions",
              "severity": "medium",
              "impact": "Required manual field mapping"
            }
          ],
          "success_metrics": {
            "data_loss_percentage": "5%",
            "integration_success_rate": "95%",
            "quality_improvement": "Data quality score improved from 85% to 92%",
            "performance_metrics": "Processing time: 2.5 hours",
            "before_integration_stats": {
              "dataset1_records": "10000",
              "dataset2_records": "8000",
              "data_quality_score": "85%"
            },
            "after_integration_stats": {
              "merged_records": "17500",
              "data_quality_score": "92%",
              "processing_time": "2.5 hours"
            },
            "cost_benefit_analysis": "Benefits outweighed costs despite data loss"
          },
          "lessons_learned": [
            {
              "category": "technical",
              "lesson": "Data validation is crucial before integration",
              "recommendation": "Implement automated data quality checks"
            },
            {
              "category": "methodological",
              "lesson": "Schema alignment saves significant time",
              "recommendation": "Establish data standards early in project"
            }
          ],
          "validation_methods": [
            {
              "method": "cross_validation",
              "description": "Used k-fold cross-validation on merged dataset",
              "results": "Model performance improved by 8%"
            },
            {
              "method": "manual_review",
              "description": "Expert review of sample records",
              "results": "Identified 3% of records with integration errors"
            }
          ],
          "risk_assessment": {
            "identified_risks": ["Data loss", "Processing delays"],
            "mitigation_strategies": ["Backup procedures", "Parallel processing"],
            "residual_risks": ["Minor data inconsistencies"]
          }
        }
        '''
        
        join = DatasetJoinAnalysis(
            dataset1="Dataset A",
            dataset2="Dataset B",
            join_type="merge",
            confidence_score=8.0
        )
        
        agent._parse_challenge_response(response, join)
        
        # Test integration challenges
        assert len(join.integration_challenges) == 2
        assert join.integration_challenges[0].category == "data_quality"
        assert join.integration_challenges[0].severity == "high"
        assert join.integration_challenges[1].category == "schema_mismatch"
        
        # Test success metrics
        assert join.success_metrics["data_loss_percentage"] == "5%"
        assert join.success_metrics["integration_success_rate"] == "95%"
        assert "before_integration_stats" in join.success_metrics
        assert "after_integration_stats" in join.success_metrics
        
        # Test lessons learned
        assert len(join.lessons_learned) == 2
        assert join.lessons_learned[0].category == "technical"
        assert join.lessons_learned[1].category == "methodological"
        
        # Test validation methods
        assert len(join.validation_methods) == 2
        assert join.validation_methods[0].method == "cross_validation"
        assert join.validation_methods[1].method == "manual_review"
        
        # Test risk assessment
        assert join.risk_assessment is not None
        assert len(join.risk_assessment.identified_risks) == 2
        assert len(join.risk_assessment.mitigation_strategies) == 2
    
    def test_parse_challenge_response_fallback_format(self, agent):
        """Test parsing challenge response with fallback string format."""
        response = '''
        {
          "integration_challenges": ["Schema mismatch", "Data quality issues"],
          "success_metrics": {
            "data_loss_percentage": "3%",
            "integration_success_rate": "97%"
          },
          "lessons_learned": ["Need better validation", "Schema alignment is key"],
          "validation_methods": ["cross_validation", "manual_review"]
        }
        '''
        
        join = DatasetJoinAnalysis(
            dataset1="Dataset A",
            dataset2="Dataset B",
            join_type="merge",
            confidence_score=8.0
        )
        
        agent._parse_challenge_response(response, join)
        
        # Test fallback parsing for string format
        assert len(join.integration_challenges) == 2
        assert join.integration_challenges[0].category == "other"
        assert join.integration_challenges[0].description == "Schema mismatch"
        assert join.integration_challenges[1].description == "Data quality issues"
        
        # Test lessons learned fallback
        assert len(join.lessons_learned) == 2
        assert join.lessons_learned[0].lesson == "Need better validation"
        assert join.lessons_learned[1].lesson == "Schema alignment is key"
        
        # Test validation methods fallback
        assert len(join.validation_methods) == 2
        assert join.validation_methods[0].description == "cross_validation"
        assert join.validation_methods[1].description == "manual_review"
    
    def test_validate_challenge_data_invalid_categories(self, agent):
        """Test validation of challenge data with invalid categories."""
        join = DatasetJoinAnalysis(
            dataset1="Dataset A",
            dataset2="Dataset B",
            join_type="merge",
            confidence_score=8.0
        )
        
        # Create challenges with invalid categories
        join.integration_challenges = [
            IntegrationChallenge(
                category="invalid_category",
                description="Test challenge",
                severity="invalid_severity",
                impact="Test impact"
            )
        ]
        
        join.lessons_learned = [
            LessonLearned(
                category="invalid_lesson_category",
                lesson="Test lesson",
                recommendation="Test recommendation"
            )
        ]
        
        join.validation_methods = [
            ValidationMethod(
                method="invalid_method",
                description="Test validation",
                results="Test results"
            )
        ]
        
        # Validate should correct these
        agent._validate_challenge_data(join)
        
        assert join.integration_challenges[0].category == "other"
        assert join.integration_challenges[0].severity == "medium"
        assert join.lessons_learned[0].category == "other"
        assert join.validation_methods[0].method == "other"
    
    def test_parse_challenge_response_scale_challenges(self, agent):
        """Test parsing challenge response with scale-related challenges."""
        response = '''
        {
          "integration_challenges": [
            {
              "category": "scale",
              "description": "Memory constraints with large datasets",
              "severity": "critical",
              "impact": "Required distributed processing approach"
            },
            {
              "category": "temporal_alignment",
              "description": "Different time periods and sampling rates",
              "severity": "high",
              "impact": "Needed temporal interpolation and resampling"
            }
          ],
          "success_metrics": {
            "data_loss_percentage": "2%",
            "integration_success_rate": "98%",
            "performance_metrics": "Processing time: 8 hours using Spark",
            "before_integration_stats": {
              "dataset1_records": "1000000",
              "dataset2_records": "800000",
              "data_quality_score": "88%"
            },
            "after_integration_stats": {
              "merged_records": "1750000",
              "data_quality_score": "94%",
              "processing_time": "8 hours"
            }
          },
          "lessons_learned": [
            {
              "category": "technical",
              "lesson": "Distributed processing is essential for large-scale integration",
              "recommendation": "Use Spark or similar frameworks for datasets > 1M records"
            }
          ],
          "validation_methods": [
            {
              "method": "statistical_test",
              "description": "Applied statistical tests on sample data",
              "results": "95% confidence in integration quality"
            }
          ]
        }
        '''
        
        join = DatasetJoinAnalysis(
            dataset1="Large Dataset A",
            dataset2="Large Dataset B",
            join_type="distributed_merge",
            confidence_score=9.0
        )
        
        agent._parse_challenge_response(response, join)
        
        # Test scale challenges
        assert len(join.integration_challenges) == 2
        assert join.integration_challenges[0].category == "scale"
        assert join.integration_challenges[0].severity == "critical"
        assert join.integration_challenges[1].category == "temporal_alignment"
        
        # Test large-scale metrics
        assert join.success_metrics["before_integration_stats"]["dataset1_records"] == "1000000"
        assert join.success_metrics["after_integration_stats"]["merged_records"] == "1750000"
        assert "Spark" in join.success_metrics["performance_metrics"]
    
    def test_deduplicate_joins(self, agent):
        """Test join deduplication."""
        joins = [
            DatasetJoinAnalysis("Dataset A", "Dataset B", "merge", 8.0),
            DatasetJoinAnalysis("Dataset B", "Dataset A", "merge", 7.0),  # Duplicate
            DatasetJoinAnalysis("Dataset C", "Dataset D", "fusion", 9.0)
        ]
        
        unique_joins = agent._deduplicate_joins(joins)
        
        assert len(unique_joins) == 2
        assert unique_joins[0].dataset1 == "Dataset A"
        assert unique_joins[1].dataset1 == "Dataset C"
    
    def test_post_process_joins(self, agent):
        """Test post-processing of joins."""
        joins = [
            DatasetJoinAnalysis("Dataset A", "Dataset B", "merge", 4.0),  # Below threshold
            DatasetJoinAnalysis("Dataset C", "Dataset D", "fusion", 8.0),
            DatasetJoinAnalysis("Dataset E", "Dataset F", "join", 9.0)
        ]
        
        processed_joins = agent._post_process_joins(joins)
        
        assert len(processed_joins) == 2  # One filtered out
        assert processed_joins[0].confidence_score == 9.0  # Sorted by confidence
        assert processed_joins[1].confidence_score == 8.0
    
    @pytest.mark.asyncio
    async def test_analyze_dataset_joins_success(self, agent, mock_llm_service, mock_dataset_service, sample_state):
        """Test successful join analysis."""
        # Mock dataset service
        mock_datasets = [Dataset(name="Dataset A"), Dataset(name="Dataset B")]
        mock_dataset_service.get_all_known_datasets.return_value = mock_datasets
        
        # Mock LLM responses
        mock_llm_service.generate_response.return_value = '''
        {
          "dataset_joins": [
            {
              "dataset1": "Dataset A",
              "dataset2": "Dataset B",
              "join_type": "merge",
              "confidence_score": 8.5,
              "context": "Datasets were merged",
              "section": "methodology"
            }
          ]
        }
        '''
        
        result = await agent.analyze_dataset_joins(sample_state)
        
        assert result.total_joins_found == 1
        assert result.methodology_details_extracted == 1
        assert result.challenges_documented == 1
        assert result.processing_time > 0
        assert result.llm_calls_made == 3  # Detection + methodology + challenges
    
    @pytest.mark.asyncio
    async def test_analyze_dataset_joins_no_text(self, agent, mock_dataset_service):
        """Test analysis with no text content."""
        empty_state = AnalysisState(publication_id="test")
        
        result = await agent.analyze_dataset_joins(empty_state)
        
        assert result.total_joins_found == 0
        assert result.errors == ["No text content available"]
    
    @pytest.mark.asyncio
    async def test_analyze_dataset_joins_error(self, agent, mock_llm_service, mock_dataset_service, sample_state):
        """Test analysis with error."""
        mock_llm_service.generate_response.side_effect = Exception("LLM error")
        
        result = await agent.analyze_dataset_joins(sample_state)
        
        assert result.total_joins_found == 0
        assert len(result.errors) == 1
        assert "Join analysis error" in result.errors[0]


class TestDatasetJoinAnalysisAgentStep:
    """Test the LangGraph step function."""
    
    @pytest.fixture
    def sample_state(self):
        """Create sample analysis state."""
        state = AnalysisState(
            publication_id="test_pub_123",
            grobid_content={
                "metadata": {"title": "Test Publication"},
                "fulltext": {
                    "abstract": {
                        "sections": [{"text": "This study combines Dataset A and Dataset B for comprehensive analysis."}]
                    },
                    "body": {
                        "sections": [
                            {
                                "heading": "Methodology",
                                "sentences": [
                                    {"text": "We merged Dataset A with Dataset B using common identifiers."},
                                    {"text": "The integration process involved key-based joins and data validation."}
                                ]
                            },
                            {
                                "heading": "Results",
                                "sentences": [
                                    {"text": "The combined dataset showed improved performance metrics."}
                                ]
                            }
                        ]
                    }
                }
            }
        )
        state.raw_text = "This study combines Dataset A and Dataset B for comprehensive analysis. We merged Dataset A with Dataset B using common identifiers. The integration process involved key-based joins and data validation. The combined dataset showed improved performance metrics."
        return state
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.llm_service.LLMModelConfig')
    @patch('pub_analysis_agent.config.settings.DatabaseSettings')
    @patch('pub_analysis_agent.services.mongodb_client.MongoDBClient')
    @patch('pub_analysis_agent.services.llm_service.LLMService')
    @patch('pub_analysis_agent.services.dataset_service.DatasetService')
    async def test_dataset_join_analysis_agent_step_success(self, mock_dataset_service_class, mock_llm_service_class, mock_mongodb_client_class, mock_db_settings_class, mock_model_config_class, sample_state):
        """Test successful step execution."""
        # Mock model config
        mock_model_config = Mock()
        mock_model_config_class.return_value = mock_model_config
        
        # Mock database settings
        mock_db_settings = Mock()
        mock_db_settings_class.return_value = mock_db_settings
        
        # Mock MongoDB client
        mock_mongodb_client = AsyncMock()
        mock_mongodb_client_class.return_value = mock_mongodb_client
        
        # Mock LLM service
        mock_llm_service = AsyncMock()
        mock_llm_service_class.return_value = mock_llm_service
        
        # Mock dataset service
        mock_dataset_service = AsyncMock()
        mock_dataset_service.get_all_known_datasets.return_value = []
        mock_dataset_service_class.return_value = mock_dataset_service
        
        # Mock LLM response
        mock_llm_service.generate_response.return_value = '''
        {
          "dataset_joins": [
            {
              "dataset1": "Dataset A",
              "dataset2": "Dataset B",
              "join_type": "merge",
              "confidence_score": 8.5,
              "context": "Datasets were merged",
              "section": "methodology"
            }
          ]
        }
        '''
        
        result = await dataset_join_analysis_agent_step(sample_state)
        
        assert len(result) == 1
        # Test DatasetRelationship structure instead of DatasetJoin
        assert result[0].source_datasets[0].name == "Dataset A"
        assert result[0].source_datasets[1].name == "Dataset B"
        assert result[0].relationship_type.value == "merge"
        assert result[0].relationship_metrics.confidence_score == 8.5
        assert sample_state.current_step == "dataset_join_analysis"
        assert "dataset_join_analysis" in sample_state.completed_steps
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.llm_service.LLMModelConfig')
    @patch('pub_analysis_agent.config.settings.DatabaseSettings')
    @patch('pub_analysis_agent.services.mongodb_client.MongoDBClient')
    @patch('pub_analysis_agent.services.llm_service.LLMService')
    @patch('pub_analysis_agent.services.dataset_service.DatasetService')
    async def test_dataset_join_analysis_agent_step_error(self, mock_dataset_service_class, mock_llm_service_class, mock_mongodb_client_class, mock_db_settings_class, mock_model_config_class, sample_state):
        """Test step execution with error."""
        # Mock model config
        mock_model_config = Mock()
        mock_model_config_class.return_value = mock_model_config
        
        # Mock database settings
        mock_db_settings = Mock()
        mock_db_settings_class.return_value = mock_db_settings
        
        # Mock MongoDB client
        mock_mongodb_client = AsyncMock()
        mock_mongodb_client_class.return_value = mock_mongodb_client
        
        # Mock LLM service to raise error
        mock_llm_service_class.side_effect = Exception("Service error")
        
        result = await dataset_join_analysis_agent_step(sample_state)
        
        assert result == []
        # After error, the step is updated with error information
        assert sample_state.current_step == "dataset_join_analysis"
        assert sample_state.error_message is not None
        assert "Service error" in sample_state.error_message


if __name__ == "__main__":
    pytest.main([__file__]) 