"""
LLM-powered Code Analysis Module for Publication Analysis.

This module provides comprehensive code analysis capabilities using LLM services
to categorize, assess relevance, and extract metadata from code snippets found
in academic publications.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from .llm_service import LLMService, PromptTemplate

logger = logging.getLogger(__name__)


class CodePurpose(Enum):
    """Purpose categories for code snippets."""
    DATA_ANALYSIS = "data_analysis"
    VISUALIZATION = "visualization"
    PREPROCESSING = "preprocessing"
    MODELING = "modeling"
    CONFIGURATION = "configuration"
    EXAMPLE = "example"
    UTILITY = "utility"
    TESTING = "testing"
    OTHER = "other"


class CodeImplementationType(Enum):
    """Implementation type of code snippets."""
    ACTUAL_IMPLEMENTATION = "actual_implementation"
    EXAMPLE_CODE = "example_code"
    PSEUDOCODE = "pseudocode"
    CONFIGURATION = "configuration"
    COMMAND_LINE = "command_line"
    UNKNOWN = "unknown"


class ProgrammingLanguage(Enum):
    """Programming languages for code classification."""
    PYTHON = "python"
    R = "r"
    JAVASCRIPT = "javascript"
    SQL = "sql"
    BASH = "bash"
    YAML = "yaml"
    JSON = "json"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    MATLAB = "matlab"
    SCALA = "scala"
    JULIA = "julia"
    GO = "go"
    RUST = "rust"
    OTHER = "other"


@dataclass
class CodeAnalysisConfig:
    """Configuration for code analysis."""
    temperature: float = 0.7
    max_tokens: int = 4000
    confidence_threshold: float = 0.7
    relevance_threshold: float = 6.0
    context_window: int = 400000
    batch_size: int = 5


@dataclass
class CodeAnalysisResult:
    """Result of code analysis."""
    language: ProgrammingLanguage
    language_confidence: float
    purpose: CodePurpose
    purpose_confidence: float
    implementation_type: CodeImplementationType
    implementation_confidence: float
    relevance_score: float
    relevance_confidence: float
    description: str
    key_features: List[str] = field(default_factory=list)
    libraries_used: List[str] = field(default_factory=list)
    complexity_level: str = "medium"  # low, medium, high
    error_message: Optional[str] = None


class CodeAnalysisLLM:
    """
    LLM-powered code analysis for categorization and relevance assessment.
    
    This class provides comprehensive analysis of code snippets including:
    - Programming language identification with confidence
    - Purpose categorization (data analysis, visualization, etc.)
    - Implementation type (actual code vs example vs pseudocode)
    - Relevance scoring based on publication context
    - Confidence scoring for all categorizations
    """
    
    def __init__(
        self,
        llm_service: LLMService,
        config: Optional[CodeAnalysisConfig] = None
    ) -> None:
        """
        Initialize the CodeAnalysisLLM.
        
        Args:
            llm_service: LLM service for analysis
            config: Configuration for analysis parameters
        """
        self.llm_service = llm_service
        self.config = config or CodeAnalysisConfig()
        
        # Setup prompt templates
        self._setup_prompt_templates()
        
        # Language detection patterns for enhanced analysis
        self._setup_language_patterns()
        
        logger.info("CodeAnalysisLLM initialized")
    
    def _setup_prompt_templates(self) -> None:
        """Setup comprehensive prompt templates for code analysis."""
        
        # Comprehensive code analysis prompt
        comprehensive_analysis_prompt = PromptTemplate(
            name="comprehensive_code_analysis",
            template="""You are an expert code analyst specializing in academic research publications. Analyze the following code snippets and provide detailed categorization.

PUBLICATION CONTEXT:
{publication_context}

CODE SNIPPETS TO ANALYZE:
{code_snippets}

For each code snippet, provide a comprehensive analysis including:

1. **Programming Language**: Identify the programming language with confidence level (0.0-1.0)
2. **Purpose**: Categorize the code purpose from these options:
   - data_analysis: Code for analyzing, processing, or transforming data
   - visualization: Code for creating plots, charts, or visual representations
   - preprocessing: Code for cleaning, filtering, or preparing data
   - modeling: Code for building, training, or evaluating models (ML/statistical)
   - configuration: Configuration files, settings, or setup code
   - example: Example or tutorial code demonstrating concepts
   - utility: Helper functions, utilities, or general-purpose code
   - testing: Test cases, validation, or debugging code
   - other: Code that doesn't fit other categories

3. **Implementation Type**: Classify as:
   - actual_implementation: Real, working implementation code
   - example_code: Simplified examples or demonstrations
   - pseudocode: High-level algorithmic descriptions
   - configuration: Configuration files or settings
   - command_line: Command-line instructions or shell commands
   - unknown: Cannot determine the type

4. **Relevance Score**: Rate relevance to the research (0-10, where 10 = highly relevant)
5. **Description**: Brief description of what the code does
6. **Key Features**: List main programming concepts/techniques used
7. **Libraries/Dependencies**: Identify any libraries or frameworks mentioned
8. **Complexity Level**: Assess as low, medium, or high complexity

For each classification, provide a confidence score (0.0-1.0) indicating your certainty.

Format your response as a JSON array:
[
  {
    "snippet_id": <id>,
    "language": "<programming_language>",
    "language_confidence": <0.0-1.0>,
    "purpose": "<purpose_category>",
    "purpose_confidence": <0.0-1.0>,
    "implementation_type": "<implementation_type>",
    "implementation_confidence": <0.0-1.0>,
    "relevance_score": <0-10>,
    "relevance_confidence": <0.0-1.0>,
    "description": "<brief description>",
    "key_features": ["<feature1>", "<feature2>"],
    "libraries_used": ["<lib1>", "<lib2>"],
    "complexity_level": "<low|medium|high>"
  }
]

Be thorough but concise. Focus on accuracy and provide realistic confidence scores.""",
            variables=["publication_context", "code_snippets"]
        )
        
        # Language-specific analysis prompt for uncertain cases
        language_focused_prompt = PromptTemplate(
            name="language_identification",
            template="""You are a programming language expert. Analyze the following code snippet and identify the programming language with high precision.

CODE SNIPPET:
{code_snippet}

CONTEXT:
{context}

Consider:
- Syntax patterns and keywords
- Import/include statements
- Function/method definitions
- Variable declarations
- Comments style
- File extensions or language hints from context

Provide:
1. Most likely programming language
2. Confidence level (0.0-1.0)
3. Alternative possibilities if confidence < 0.9
4. Key indicators that led to this identification

Format as JSON:
{
  "primary_language": "<language>",
  "confidence": <0.0-1.0>,
  "alternatives": [{"language": "<lang>", "probability": <0.0-1.0>}],
  "indicators": ["<indicator1>", "<indicator2>"]
}""",
            variables=["code_snippet", "context"]
        )
        
        self.llm_service.add_prompt_template(comprehensive_analysis_prompt)
        self.llm_service.add_prompt_template(language_focused_prompt)
        
        logger.info("Code analysis prompt templates configured")
    
    def _setup_language_patterns(self) -> None:
        """Setup patterns for programming language detection."""
        self.language_indicators = {
            ProgrammingLanguage.PYTHON: [
                "import ", "from ", "def ", "class ", "print(", "__init__", 
                "if __name__", "pandas", "numpy", "matplotlib", "sklearn"
            ],
            ProgrammingLanguage.R: [
                "library(", "require(", "<-", "data.frame", "ggplot", "dplyr",
                "tibble", "readr", "function("
            ],
            ProgrammingLanguage.JAVASCRIPT: [
                "function", "const ", "let ", "var ", "console.log", "=>",
                "require(", "import ", "export ", "document."
            ],
            ProgrammingLanguage.SQL: [
                "SELECT", "FROM", "WHERE", "JOIN", "INSERT", "UPDATE",
                "DELETE", "CREATE TABLE", "ALTER TABLE"
            ],
            ProgrammingLanguage.BASH: [
                "#!/bin/bash", "#!/bin/sh", "echo ", "grep ", "awk ",
                "sed ", "$", "export ", "chmod ", "ls "
            ],
            ProgrammingLanguage.JAVA: [
                "public class", "private ", "public ", "static ", "void ",
                "System.out", "import java", "package "
            ],
            ProgrammingLanguage.CPP: [
                "#include", "std::", "cout", "cin", "namespace", "class ",
                "private:", "public:", "virtual "
            ],
            ProgrammingLanguage.MATLAB: [
                "function ", "end", "plot(", "figure", "xlabel", "ylabel",
                "subplot", "disp(", "clc", "clear"
            ]
        }
    
    async def analyze_code_batch(
        self,
        code_snippets: List[Dict[str, Any]],
        publication_context: str = "",
        use_batch_processing: bool = True
    ) -> List[CodeAnalysisResult]:
        """
        Analyze a batch of code snippets.
        
        Args:
            code_snippets: List of code snippet dictionaries with 'id', 'content', 'context'
            publication_context: Context from the publication for relevance assessment
            use_batch_processing: Whether to process snippets in batches
            
        Returns:
            List of CodeAnalysisResult objects
        """
        if not code_snippets:
            return []
        
        try:
            if use_batch_processing and len(code_snippets) > self.config.batch_size:
                # Process in batches to avoid token limits
                results = []
                for i in range(0, len(code_snippets), self.config.batch_size):
                    batch = code_snippets[i:i + self.config.batch_size]
                    batch_results = await self._analyze_batch_internal(batch, publication_context)
                    results.extend(batch_results)
                return results
            else:
                return await self._analyze_batch_internal(code_snippets, publication_context)
                
        except Exception as e:
            logger.error(f"Error in batch code analysis: {e}")
            # Return default results for error cases
            return [self._create_error_result(snippet, str(e)) for snippet in code_snippets]
    
    async def analyze_single_code(
        self,
        code_content: str,
        context: str = "",
        publication_context: str = ""
    ) -> CodeAnalysisResult:
        """
        Analyze a single code snippet.
        
        Args:
            code_content: The code content to analyze
            context: Local context around the code
            publication_context: Broader publication context
            
        Returns:
            CodeAnalysisResult with comprehensive analysis
        """
        snippet_data = [{
            "id": 0,
            "content": code_content,
            "context": context
        }]
        
        results = await self.analyze_code_batch(snippet_data, publication_context, False)
        return results[0] if results else self._create_error_result(snippet_data[0], "Analysis failed")
    
    async def _analyze_batch_internal(
        self,
        code_snippets: List[Dict[str, Any]],
        publication_context: str
    ) -> List[CodeAnalysisResult]:
        """
        Internal method for batch analysis.
        
        Args:
            code_snippets: Batch of code snippets to analyze
            publication_context: Publication context for relevance
            
        Returns:
            List of analysis results
        """
        try:
            # Prepare data for LLM
            snippets_for_llm = []
            for snippet in code_snippets:
                snippets_for_llm.append({
                    "id": snippet["id"],
                    "content": snippet["content"][:1000],  # Truncate for LLM
                    "context": snippet.get("context", "")[:self.config.context_window]
                })
            
            # Get comprehensive analysis prompt
            prompt_template = self.llm_service.get_prompt_template("comprehensive_code_analysis")
            prompt = prompt_template.render(
                publication_context=publication_context[:800],  # Limit context length
                code_snippets=json.dumps(snippets_for_llm, indent=2)
            )
            
            # Call LLM
            response = await self.llm_service.generate(
                prompt=prompt,
                parameters={
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens
                }
            )
            
            # Process response
            return self._process_analysis_response(response, code_snippets)
            
        except Exception as e:
            logger.error(f"Error in internal batch analysis: {e}")
            return [self._create_error_result(snippet, str(e)) for snippet in code_snippets]
    
    def _process_analysis_response(
        self,
        llm_response: Dict[str, Any],
        original_snippets: List[Dict[str, Any]]
    ) -> List[CodeAnalysisResult]:
        """
        Process LLM response and create analysis results.
        
        Args:
            llm_response: Response from LLM service
            original_snippets: Original code snippets
            
        Returns:
            List of CodeAnalysisResult objects
        """
        try:
            # Extract response text (chat/completions first, fallback to legacy completions)
            response_text = ""
            if "choices" in llm_response and llm_response["choices"]:
                first_choice = llm_response["choices"][0]
                message = first_choice.get("message") or {}
                response_text = (message.get("content") or "").strip()
                if not response_text:
                    response_text = first_choice.get("text", "").strip()
            
            if not response_text:
                logger.warning("Empty response from LLM")
                return [self._create_default_result(snippet) for snippet in original_snippets]
            
            # Parse JSON response
            try:
                analyses = json.loads(response_text)
                if not isinstance(analyses, list):
                    analyses = [analyses]
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM response as JSON: {e}")
                logger.debug(f"Response text: {response_text}")
                return [self._create_default_result(snippet) for snippet in original_snippets]
            
            # Create results
            results = []
            for snippet in original_snippets:
                snippet_id = snippet["id"]
                
                # Find corresponding analysis
                analysis = None
                for a in analyses:
                    if isinstance(a, dict) and a.get("snippet_id") == snippet_id:
                        analysis = a
                        break
                
                if analysis:
                    result = self._create_result_from_analysis(analysis, snippet)
                else:
                    result = self._create_default_result(snippet)
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing analysis response: {e}")
            return [self._create_error_result(snippet, str(e)) for snippet in original_snippets]
    
    def _create_result_from_analysis(
        self,
        analysis: Dict[str, Any],
        original_snippet: Dict[str, Any]
    ) -> CodeAnalysisResult:
        """
        Create CodeAnalysisResult from LLM analysis.
        
        Args:
            analysis: LLM analysis dictionary
            original_snippet: Original snippet data
            
        Returns:
            CodeAnalysisResult object
        """
        try:
            # Parse language
            language_str = analysis.get("language", "other").lower()
            try:
                language = ProgrammingLanguage(language_str)
            except ValueError:
                language = ProgrammingLanguage.OTHER
            
            # Parse purpose
            purpose_str = analysis.get("purpose", "other").lower()
            try:
                purpose = CodePurpose(purpose_str)
            except ValueError:
                purpose = CodePurpose.OTHER
            
            # Parse implementation type
            impl_type_str = analysis.get("implementation_type", "unknown").lower()
            try:
                implementation_type = CodeImplementationType(impl_type_str)
            except ValueError:
                implementation_type = CodeImplementationType.UNKNOWN
            
            return CodeAnalysisResult(
                language=language,
                language_confidence=float(analysis.get("language_confidence", 0.5)),
                purpose=purpose,
                purpose_confidence=float(analysis.get("purpose_confidence", 0.5)),
                implementation_type=implementation_type,
                implementation_confidence=float(analysis.get("implementation_confidence", 0.5)),
                relevance_score=float(analysis.get("relevance_score", 5.0)),
                relevance_confidence=float(analysis.get("relevance_confidence", 0.5)),
                description=analysis.get("description", "No description provided"),
                key_features=analysis.get("key_features", []),
                libraries_used=analysis.get("libraries_used", []),
                complexity_level=analysis.get("complexity_level", "medium")
            )
            
        except Exception as e:
            logger.error(f"Error creating result from analysis: {e}")
            return self._create_default_result(original_snippet)
    
    def _create_default_result(self, snippet: Dict[str, Any]) -> CodeAnalysisResult:
        """
        Create a default result when analysis fails.
        
        Args:
            snippet: Original snippet data
            
        Returns:
            Default CodeAnalysisResult
        """
        # Try basic language detection
        content = snippet.get("content", "")
        detected_language = self._detect_language_heuristically(content)
        
        return CodeAnalysisResult(
            language=detected_language,
            language_confidence=0.3,
            purpose=CodePurpose.OTHER,
            purpose_confidence=0.2,
            implementation_type=CodeImplementationType.UNKNOWN,
            implementation_confidence=0.2,
            relevance_score=5.0,
            relevance_confidence=0.3,
            description="Analysis not available",
            key_features=[],
            libraries_used=[],
            complexity_level="medium"
        )
    
    def _create_error_result(self, snippet: Dict[str, Any], error_message: str) -> CodeAnalysisResult:
        """
        Create an error result.
        
        Args:
            snippet: Original snippet data
            error_message: Error message
            
        Returns:
            CodeAnalysisResult with error information
        """
        result = self._create_default_result(snippet)
        result.error_message = error_message
        return result
    
    def _detect_language_heuristically(self, code_content: str) -> ProgrammingLanguage:
        """
        Detect programming language using heuristic patterns.
        
        Args:
            code_content: Code content to analyze
            
        Returns:
            Detected ProgrammingLanguage
        """
        if not code_content:
            return ProgrammingLanguage.OTHER
        
        code_lower = code_content.lower()
        
        # Score each language based on indicators
        language_scores = {}
        for language, indicators in self.language_indicators.items():
            score = sum(1 for indicator in indicators if indicator.lower() in code_lower)
            if score > 0:
                language_scores[language] = score
        
        # Return language with highest score
        if language_scores:
            best_language = max(language_scores, key=language_scores.get)
            return best_language
        
        return ProgrammingLanguage.OTHER
    
    async def enhanced_language_detection(
        self,
        code_content: str,
        context: str = ""
    ) -> Tuple[ProgrammingLanguage, float]:
        """
        Enhanced language detection using LLM for uncertain cases.
        
        Args:
            code_content: Code content to analyze
            context: Context around the code
            
        Returns:
            Tuple of (detected_language, confidence_score)
        """
        try:
            # First try heuristic detection
            heuristic_language = self._detect_language_heuristically(code_content)
            
            # If we have a good heuristic match, use it
            heuristic_indicators = self.language_indicators.get(heuristic_language, [])
            code_lower = code_content.lower()
            matches = sum(1 for indicator in heuristic_indicators if indicator.lower() in code_lower)
            
            if matches >= 3:  # High confidence from heuristics
                return heuristic_language, 0.8
            
            # Use LLM for uncertain cases
            prompt_template = self.llm_service.get_prompt_template("language_identification")
            prompt = prompt_template.render(
                code_snippet=code_content[:500],
                context=context[:200]
            )
            
            response = await self.llm_service.generate(
                prompt=prompt,
                parameters={"temperature": 0.7, "max_tokens": 4000}
            )
            
            # Process LLM response
            if "choices" in response and response["choices"]:
                first_choice = response["choices"][0]
                message = first_choice.get("message") or {}
                response_text = (message.get("content") or "").strip()
                if not response_text:
                    response_text = first_choice.get("text", "").strip()
                try:
                    result = json.loads(response_text)
                    language_str = result.get("primary_language", "other").lower()
                    confidence = float(result.get("confidence", 0.5))
                    
                    try:
                        detected_language = ProgrammingLanguage(language_str)
                        return detected_language, confidence
                    except ValueError:
                        pass
                except json.JSONDecodeError:
                    pass
            
            # Fallback to heuristic result
            return heuristic_language, 0.4
            
        except Exception as e:
            logger.error(f"Error in enhanced language detection: {e}")
            return self._detect_language_heuristically(code_content), 0.3
    
    def get_analysis_summary(self, results: List[CodeAnalysisResult]) -> Dict[str, Any]:
        """
        Generate summary statistics from analysis results.
        
        Args:
            results: List of analysis results
            
        Returns:
            Summary dictionary with statistics
        """
        if not results:
            return {}
        
        # Language distribution
        language_counts = {}
        for result in results:
            lang = result.language.value
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        # Purpose distribution
        purpose_counts = {}
        for result in results:
            purpose = result.purpose.value
            purpose_counts[purpose] = purpose_counts.get(purpose, 0) + 1
        
        # Implementation type distribution
        impl_type_counts = {}
        for result in results:
            impl_type = result.implementation_type.value
            impl_type_counts[impl_type] = impl_type_counts.get(impl_type, 0) + 1
        
        # Average scores and confidence
        relevance_scores = [r.relevance_score for r in results]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        confidence_scores = [r.relevance_confidence for r in results]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # High relevance count
        high_relevance_count = sum(1 for r in results if r.relevance_score >= self.config.relevance_threshold)
        
        return {
            "total_snippets": len(results),
            "language_distribution": language_counts,
            "purpose_distribution": purpose_counts,
            "implementation_type_distribution": impl_type_counts,
            "average_relevance_score": round(avg_relevance, 2),
            "average_confidence": round(avg_confidence, 2),
            "high_relevance_snippets": high_relevance_count,
            "high_relevance_percentage": round((high_relevance_count / len(results)) * 100, 1)
        } 