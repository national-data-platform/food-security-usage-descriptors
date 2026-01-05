"""
Unit tests for the CodeAnalysisLLM module.

Tests comprehensive code analysis functionality including language detection,
purpose categorization, implementation type classification, and relevance scoring.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any, List

from src.pub_analysis_agent.services.code_analysis_llm import (
    CodeAnalysisLLM,
    CodeAnalysisConfig,
    CodeAnalysisResult,
    CodePurpose,
    CodeImplementationType,
    ProgrammingLanguage
)
from src.pub_analysis_agent.services.llm_service import LLMService, LLMModelConfig


class TestCodeAnalysisLLM:
    """Test cases for CodeAnalysisLLM functionality."""
    
    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service for testing."""
        mock_service = AsyncMock(spec=LLMService)
        mock_service.add_prompt_template = MagicMock()
        mock_service.get_prompt_template = MagicMock()
        mock_service.generate = AsyncMock()
        return mock_service
    
    @pytest.fixture
    def analysis_config(self):
        """Create a test configuration for code analysis."""
        return CodeAnalysisConfig(
            temperature=0.2,
            max_tokens=500,
            confidence_threshold=0.7,
            relevance_threshold=6.0,
            context_window=200,
            batch_size=3
        )
    
    @pytest.fixture
    def code_analyzer(self, mock_llm_service, analysis_config):
        """Create a CodeAnalysisLLM instance for testing."""
        return CodeAnalysisLLM(mock_llm_service, analysis_config)
    
    def test_initialization(self, code_analyzer, mock_llm_service):
        """Test proper initialization of CodeAnalysisLLM."""
        assert code_analyzer.llm_service == mock_llm_service
        assert isinstance(code_analyzer.config, CodeAnalysisConfig)
        assert code_analyzer.config.temperature == 0.2
        assert code_analyzer.config.batch_size == 3
        
        # Verify prompt templates were added
        assert mock_llm_service.add_prompt_template.call_count == 2
    
    def test_heuristic_language_detection(self, code_analyzer):
        """Test heuristic-based programming language detection."""
        # Test Python detection
        python_code = """
        import pandas as pd
        import numpy as np
        
        def analyze_data(df):
            return df.mean()
        """
        detected = code_analyzer._detect_language_heuristically(python_code)
        assert detected == ProgrammingLanguage.PYTHON
        
        # Test R detection
        r_code = """
        library(ggplot2)
        data <- data.frame(x = 1:10, y = 1:10)
        plot <- ggplot(data, aes(x, y)) + geom_point()
        """
        detected = code_analyzer._detect_language_heuristically(r_code)
        assert detected == ProgrammingLanguage.R
        
        # Test SQL detection
        sql_code = """
        SELECT customer_id, COUNT(*) as order_count
        FROM orders
        WHERE order_date >= '2024-01-01'
        GROUP BY customer_id
        """
        detected = code_analyzer._detect_language_heuristically(sql_code)
        assert detected == ProgrammingLanguage.SQL
        
        # Test unknown code
        unknown_code = "random text without programming indicators"
        detected = code_analyzer._detect_language_heuristically(unknown_code)
        assert detected == ProgrammingLanguage.OTHER
    
    @pytest.mark.asyncio
    async def test_single_code_analysis_success(self, code_analyzer, mock_llm_service):
        """Test successful single code analysis."""
        # Mock LLM response
        mock_response = {
            "choices": [{
                "text": """[{
                    "snippet_id": 0,
                    "language": "python",
                    "language_confidence": 0.95,
                    "purpose": "data_analysis",
                    "purpose_confidence": 0.90,
                    "implementation_type": "actual_implementation",
                    "implementation_confidence": 0.85,
                    "relevance_score": 8.5,
                    "relevance_confidence": 0.80,
                    "description": "Data processing using pandas",
                    "key_features": ["pandas", "data manipulation"],
                    "libraries_used": ["pandas", "numpy"],
                    "complexity_level": "medium"
                }]"""
            }]
        }
        mock_llm_service.generate.return_value = mock_response
        
        # Mock prompt template
        mock_template = MagicMock()
        mock_template.render.return_value = "mocked prompt"
        mock_llm_service.get_prompt_template.return_value = mock_template
        
        # Test single code analysis
        code_content = "import pandas as pd\ndf = pd.read_csv('data.csv')"
        result = await code_analyzer.analyze_single_code(
            code_content=code_content,
            context="Data loading section",
            publication_context="Research on data analysis"
        )
        
        # Verify result
        assert isinstance(result, CodeAnalysisResult)
        assert result.language == ProgrammingLanguage.PYTHON
        assert result.language_confidence == 0.95
        assert result.purpose == CodePurpose.DATA_ANALYSIS
        assert result.purpose_confidence == 0.90
        assert result.implementation_type == CodeImplementationType.ACTUAL_IMPLEMENTATION
        assert result.relevance_score == 8.5
        assert result.description == "Data processing using pandas"
        assert "pandas" in result.key_features
        assert "pandas" in result.libraries_used
        assert result.complexity_level == "medium"
    
    @pytest.mark.asyncio
    async def test_batch_code_analysis_success(self, code_analyzer, mock_llm_service):
        """Test successful batch code analysis."""
        # Mock LLM response for multiple snippets
        mock_response = {
            "choices": [{
                "text": """[
                    {
                        "snippet_id": 0,
                        "language": "python",
                        "language_confidence": 0.95,
                        "purpose": "data_analysis",
                        "purpose_confidence": 0.90,
                        "implementation_type": "actual_implementation",
                        "implementation_confidence": 0.85,
                        "relevance_score": 8.5,
                        "relevance_confidence": 0.80,
                        "description": "Data loading with pandas",
                        "key_features": ["pandas", "csv"],
                        "libraries_used": ["pandas"],
                        "complexity_level": "low"
                    },
                    {
                        "snippet_id": 1,
                        "language": "python",
                        "language_confidence": 0.90,
                        "purpose": "visualization",
                        "purpose_confidence": 0.88,
                        "implementation_type": "actual_implementation",
                        "implementation_confidence": 0.80,
                        "relevance_score": 7.0,
                        "relevance_confidence": 0.75,
                        "description": "Creating plots with matplotlib",
                        "key_features": ["matplotlib", "plotting"],
                        "libraries_used": ["matplotlib"],
                        "complexity_level": "medium"
                    }
                ]"""
            }]
        }
        mock_llm_service.generate.return_value = mock_response
        
        # Mock prompt template
        mock_template = MagicMock()
        mock_template.render.return_value = "mocked prompt"
        mock_llm_service.get_prompt_template.return_value = mock_template
        
        # Test batch analysis
        code_snippets = [
            {
                "id": 0,
                "content": "import pandas as pd\ndf = pd.read_csv('data.csv')",
                "context": "Data loading section"
            },
            {
                "id": 1,
                "content": "import matplotlib.pyplot as plt\nplt.plot(x, y)",
                "context": "Visualization section"
            }
        ]
        
        results = await code_analyzer.analyze_code_batch(
            code_snippets=code_snippets,
            publication_context="Research paper on data analysis"
        )
        
        # Verify results
        assert len(results) == 2
        
        # First snippet
        assert results[0].language == ProgrammingLanguage.PYTHON
        assert results[0].purpose == CodePurpose.DATA_ANALYSIS
        assert results[0].relevance_score == 8.5
        assert results[0].description == "Data loading with pandas"
        
        # Second snippet
        assert results[1].language == ProgrammingLanguage.PYTHON
        assert results[1].purpose == CodePurpose.VISUALIZATION
        assert results[1].relevance_score == 7.0
        assert results[1].description == "Creating plots with matplotlib"
    
    @pytest.mark.asyncio
    async def test_batch_processing_large_dataset(self, code_analyzer, mock_llm_service):
        """Test batch processing for large datasets."""
        # Create a large number of code snippets (exceeding batch size)
        code_snippets = []
        for i in range(8):  # More than batch_size of 3
            code_snippets.append({
                "id": i,
                "content": f"# Code snippet {i}\nprint('Hello {i}')",
                "context": f"Context {i}"
            })
        
        # Mock LLM response (will be called multiple times for batches)
        mock_response = {
            "choices": [{
                "text": """[{
                    "snippet_id": 0,
                    "language": "python",
                    "language_confidence": 0.95,
                    "purpose": "example",
                    "purpose_confidence": 0.90,
                    "implementation_type": "example_code",
                    "implementation_confidence": 0.85,
                    "relevance_score": 5.0,
                    "relevance_confidence": 0.70,
                    "description": "Simple print statement",
                    "key_features": ["print"],
                    "libraries_used": [],
                    "complexity_level": "low"
                }]"""
            }]
        }
        mock_llm_service.generate.return_value = mock_response
        
        # Mock prompt template
        mock_template = MagicMock()
        mock_template.render.return_value = "mocked prompt"
        mock_llm_service.get_prompt_template.return_value = mock_template
        
        # Test batch processing
        results = await code_analyzer.analyze_code_batch(
            code_snippets=code_snippets,
            use_batch_processing=True
        )
        
        # Verify that all snippets were processed
        assert len(results) == 8
        
        # Verify that generate was called multiple times (for different batches)
        assert mock_llm_service.generate.call_count >= 2  # At least 2 batches for 8 items with batch_size 3
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_json(self, code_analyzer, mock_llm_service):
        """Test error handling when LLM returns invalid JSON."""
        # Mock invalid JSON response
        mock_response = {
            "choices": [{
                "text": "This is not valid JSON response"
            }]
        }
        mock_llm_service.generate.return_value = mock_response
        
        # Mock prompt template
        mock_template = MagicMock()
        mock_template.render.return_value = "mocked prompt"
        mock_llm_service.get_prompt_template.return_value = mock_template
        
        # Test analysis with invalid response
        code_snippets = [{
            "id": 0,
            "content": "print('hello')",
            "context": "test"
        }]
        
        results = await code_analyzer.analyze_code_batch(code_snippets)
        
        # Should return default results
        assert len(results) == 1
        assert results[0].language == ProgrammingLanguage.PYTHON  # From heuristic detection
        assert results[0].purpose == CodePurpose.OTHER
        assert results[0].relevance_score == 5.0
        assert results[0].description == "Analysis not available"
    
    @pytest.mark.asyncio
    async def test_error_handling_llm_exception(self, code_analyzer, mock_llm_service):
        """Test error handling when LLM service raises an exception."""
        # Mock LLM service to raise exception
        mock_llm_service.generate.side_effect = Exception("LLM service error")
        
        # Mock prompt template
        mock_template = MagicMock()
        mock_template.render.return_value = "mocked prompt"
        mock_llm_service.get_prompt_template.return_value = mock_template
        
        # Test analysis with LLM error
        code_snippets = [{
            "id": 0,
            "content": "print('hello')",
            "context": "test"
        }]
        
        results = await code_analyzer.analyze_code_batch(code_snippets)
        
        # Should return error results
        assert len(results) == 1
        assert results[0].error_message == "LLM service error"
        assert results[0].relevance_score == 5.0
    
    @pytest.mark.asyncio
    async def test_enhanced_language_detection(self, code_analyzer, mock_llm_service):
        """Test enhanced language detection using LLM for uncertain cases."""
        # Mock LLM response for language detection
        mock_response = {
            "choices": [{
                "text": """{
                    "primary_language": "javascript",
                    "confidence": 0.92,
                    "alternatives": [{"language": "typescript", "probability": 0.08}],
                    "indicators": ["function", "const", "=>"]
                }"""
            }]
        }
        mock_llm_service.generate.return_value = mock_response
        
        # Mock prompt template
        mock_template = MagicMock()
        mock_template.render.return_value = "mocked prompt"
        mock_llm_service.get_prompt_template.return_value = mock_template
        
        # Test with ambiguous code
        ambiguous_code = "const data = () => { return fetch('/api/data'); }"
        
        language, confidence = await code_analyzer.enhanced_language_detection(
            code_content=ambiguous_code,
            context="API call section"
        )
        
        assert language == ProgrammingLanguage.JAVASCRIPT
        assert confidence == 0.92
    
    def test_analysis_summary(self, code_analyzer):
        """Test analysis summary generation."""
        # Create mock analysis results
        results = [
            CodeAnalysisResult(
                language=ProgrammingLanguage.PYTHON,
                language_confidence=0.95,
                purpose=CodePurpose.DATA_ANALYSIS,
                purpose_confidence=0.90,
                implementation_type=CodeImplementationType.ACTUAL_IMPLEMENTATION,
                implementation_confidence=0.85,
                relevance_score=8.5,
                relevance_confidence=0.80,
                description="Data processing"
            ),
            CodeAnalysisResult(
                language=ProgrammingLanguage.PYTHON,
                language_confidence=0.90,
                purpose=CodePurpose.VISUALIZATION,
                purpose_confidence=0.88,
                implementation_type=CodeImplementationType.EXAMPLE_CODE,
                implementation_confidence=0.75,
                relevance_score=7.0,
                relevance_confidence=0.78,
                description="Plotting data"
            ),
            CodeAnalysisResult(
                language=ProgrammingLanguage.R,
                language_confidence=0.92,
                purpose=CodePurpose.MODELING,
                purpose_confidence=0.87,
                implementation_type=CodeImplementationType.ACTUAL_IMPLEMENTATION,
                implementation_confidence=0.90,
                relevance_score=9.0,
                relevance_confidence=0.85,
                description="Statistical modeling"
            )
        ]
        
        summary = code_analyzer.get_analysis_summary(results)
        
        # Verify summary statistics
        assert summary["total_snippets"] == 3
        assert summary["language_distribution"]["python"] == 2
        assert summary["language_distribution"]["r"] == 1
        assert summary["purpose_distribution"]["data_analysis"] == 1
        assert summary["purpose_distribution"]["visualization"] == 1
        assert summary["purpose_distribution"]["modeling"] == 1
        assert summary["implementation_type_distribution"]["actual_implementation"] == 2
        assert summary["implementation_type_distribution"]["example_code"] == 1
        assert summary["average_relevance_score"] == 8.17  # (8.5 + 7.0 + 9.0) / 3
        assert summary["high_relevance_snippets"] == 3  # 8.5, 7.0 and 9.0 >= 6.0 threshold
        assert summary["high_relevance_percentage"] == 100.0  # 3/3 * 100
    
    def test_empty_input_handling(self, code_analyzer):
        """Test handling of empty inputs."""
        # Test empty code content
        detected = code_analyzer._detect_language_heuristically("")
        assert detected == ProgrammingLanguage.OTHER
        
        # Test empty summary
        summary = code_analyzer.get_analysis_summary([])
        assert summary == {}
    
    @pytest.mark.asyncio
    async def test_empty_batch_handling(self, code_analyzer):
        """Test handling of empty batch."""
        results = await code_analyzer.analyze_code_batch([])
        assert results == []


if __name__ == "__main__":
    pytest.main([__file__]) 