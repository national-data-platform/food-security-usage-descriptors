"""
Triage Analysis Agent for Publication Classification.

This module implements the TriageAgent that determines if a publication
qualifies as a data analysis paper using LLM analysis of full text content.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import re

from ..services.llm_service import LLMService, PromptTemplate
from ..workflows.state_models import AnalysisState

logger = logging.getLogger(__name__)


@dataclass
class TriageConfig:
    """Configuration for the TriageAgent."""
    confidence_threshold: float = 6.0  # Minimum confidence score (0-10) to classify as data analysis
    max_text_length: int = 400000  # Maximum text length to send to LLM
    temperature: float = 0.7  # Low temperature for consistent classification
    max_tokens: int = 4000  # Short response for classification
    include_abstract: bool = True
    include_methods: bool = True
    include_results: bool = True
    include_conclusion: bool = False


@dataclass
class TriageResult:
    """Result from triage analysis."""
    is_data_analysis: bool
    confidence_score: float
    reasoning: str
    text_features: Dict[str, Any]
    llm_response: Dict[str, Any]


class TriageAgent:
    """
    Agent to determine if a publication qualifies as a data analysis paper.
    
    Uses LLM analysis of full text content to produce boolean classification
    with confidence scoring. Distinguishes data analysis papers from those
    that only mention datasets.
    """
    
    def __init__(
        self,
        llm_service: LLMService,
        config: Optional[TriageConfig] = None
    ) -> None:
        """
        Initialize the TriageAgent.
        
        Args:
            llm_service: LLM service for text analysis
            config: Configuration for triage analysis
        """
        self.llm_service = llm_service
        self.config = config or TriageConfig()
        
        # Initialize prompt templates
        self._setup_prompt_templates()
        
        logger.info("TriageAgent initialized")
    
    def _setup_prompt_templates(self) -> None:
        """Setup prompt templates for triage analysis."""
        
        # Main classification prompt
        classification_prompt = PromptTemplate(
            name="triage_classification",
            template="""You are an expert research paper classifier. Your task is to determine if a paper is a DATA ANALYSIS paper.

CLASSIFICATION CRITERIA:

A DATA ANALYSIS paper is one that:
1. Performs statistical analysis, modeling, or computational analysis on datasets
2. Reports quantitative findings from data analysis
3. Uses data as the primary source of evidence for conclusions
4. Applies analytical methods to derive insights from data
5. Presents results from data processing, visualization, or statistical tests

NOT a data analysis paper if it only:
- Mentions datasets without analyzing them
- Describes data collection methods without analysis
- Cites other studies that used data
- Proposes theoretical frameworks without empirical analysis
- Reviews literature without original data analysis

PAPER CONTENT:
{text_content}

CRITICAL INSTRUCTIONS:
1. Analyze the content carefully for evidence of actual data analysis
2. Assign a confidence score from 0-10 (10 = definitely data analysis, 0 = definitely not)
3. Provide clear reasoning in 2-3 sentences explaining your decision
4. You MUST respond with ONLY valid JSON in this exact format - no additional text:

{{
  "is_data_analysis": true,
  "confidence_score": 8,
  "reasoning": "The paper presents statistical analysis of survey data with quantitative results and visualizations."
}}

IMPORTANT: Your response must be valid JSON only. Do not include any text before or after the JSON object.""",
            variables=["text_content"]
        )
        
        self.llm_service.add_prompt_template(classification_prompt)
        
        logger.info("Prompt templates configured for TriageAgent")
    
    async def analyze(self, state: AnalysisState) -> TriageResult:
        """
        Analyze if the publication is a data analysis paper.
        
        Args:
            state: Current analysis state with publication content
            
        Returns:
            TriageResult with classification and confidence
        """
        try:
            logger.info(f"Starting triage analysis for publication: {state.publication_id}")
            
            # Extract and preprocess text
            text_content = state.raw_text
            
            if not text_content:
                logger.warning("No text content available for analysis")
                return TriageResult(
                    is_data_analysis=False,
                    confidence_score=0.0,
                    reasoning="No text content available for analysis",
                    text_features={},
                    llm_response={}
                )
            
            # Extract text features
            text_features = self._extract_text_features(text_content)
            
            # Perform LLM analysis
            llm_result = await self._perform_llm_analysis(text_content)
            
            # Process results
            result = self._process_llm_result(llm_result, text_features)
            
            logger.info(
                f"Triage analysis completed - Classification: {result.is_data_analysis}, "
                f"Confidence: {result.confidence_score:.2f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in triage analysis: {e}")
            return TriageResult(
                is_data_analysis=False,
                confidence_score=0.0,
                reasoning=f"Analysis failed: {str(e)}",
                text_features={},
                llm_response={}
            )
    
    def _extract_text_content(self, state: AnalysisState) -> str:
        """
        Extract relevant text content from the publication.
        
        Args:
            state: Analysis state with GROBID content
            
        Returns:
            Preprocessed text content for analysis
        """
        if not state.grobid_content:
            return state.raw_text or ""
        
        content_parts = []
        grobid_data = state.grobid_content
        
        # Extract title
        if "title" in grobid_data:
            content_parts.append(f"TITLE: {grobid_data['title']}")
        
        # Extract abstract
        if self.config.include_abstract and "abstract" in grobid_data:
            content_parts.append(f"ABSTRACT: {grobid_data['abstract']}")
        
        # Extract sections based on configuration
        if "sections" in grobid_data and isinstance(grobid_data["sections"], list):
            for section in grobid_data["sections"]:
                if not isinstance(section, dict):
                    continue
                    
                section_title = section.get("title", "").lower()
                section_text = section.get("text", "")
                
                if not section_text:
                    continue
                
                # Include methods section
                if (self.config.include_methods and 
                    any(keyword in section_title for keyword in ["method", "approach", "analysis", "data"])):
                    content_parts.append(f"METHODS: {section_text}")
                
                # Include results section
                if (self.config.include_results and 
                    any(keyword in section_title for keyword in ["result", "finding", "outcome"])):
                    content_parts.append(f"RESULTS: {section_text}")
                
                # Include conclusion section if configured
                if (self.config.include_conclusion and 
                    any(keyword in section_title for keyword in ["conclusion", "discussion", "summary"])):
                    content_parts.append(f"CONCLUSION: {section_text}")
        
        # Combine and preprocess
        full_text = "\n\n".join(content_parts)
        return self._preprocess_text(full_text)
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for LLM analysis.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned and truncated text
        """
        if not text:
            return ""
        
        # Clean up whitespace and formatting
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)
        
        # Remove URLs and DOIs
        text = re.sub(r'https?://\S+', '[URL]', text)
        text = re.sub(r'doi:\S+', '[DOI]', text)
        
        # Truncate to maximum length
        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length] + "..."
            logger.info(f"Text truncated to {self.config.max_text_length} characters")
        
        return text
    
    def _extract_text_features(self, text: str) -> Dict[str, Any]:
        """
        Extract features from text that might indicate data analysis.
        
        Args:
            text: Preprocessed text content
            
        Returns:
            Dictionary of extracted features
        """
        text_lower = text.lower()
        
        # Data analysis keywords
        analysis_keywords = [
            'statistical analysis', 'regression', 'correlation', 'significance',
            'p-value', 'confidence interval', 'hypothesis test', 'anova',
            'chi-square', 't-test', 'modeling', 'predictive', 'classification',
            'clustering', 'machine learning', 'data mining', 'visualization'
        ]
        
        # Dataset keywords
        dataset_keywords = [
            'dataset', 'data set', 'database', 'survey data', 'cohort',
            'sample size', 'participants', 'observations', 'records'
        ]
        
        # Method keywords
        method_keywords = [
            'methodology', 'approach', 'algorithm', 'procedure',
            'technique', 'framework', 'model', 'analysis'
        ]
        
        # Count keyword occurrences
        analysis_count = sum(1 for keyword in analysis_keywords if keyword in text_lower)
        dataset_count = sum(1 for keyword in dataset_keywords if keyword in text_lower)
        method_count = sum(1 for keyword in method_keywords if keyword in text_lower)
        
        # Check for numerical results
        number_pattern = r'\d+\.?\d*\s*[%]?'
        numerical_mentions = len(re.findall(number_pattern, text))
        
        # Check for statistical notation
        statistical_patterns = [
            r'p\s*[<>=]\s*0\.\d+',  # p-values
            r'r\s*=\s*0\.\d+',      # correlation coefficients
            r'β\s*=\s*\d+\.?\d*',   # beta coefficients
            r'\(.*\d+.*\)',         # parenthetical numbers (often statistics)
        ]
        statistical_mentions = sum(
            len(re.findall(pattern, text_lower)) for pattern in statistical_patterns
        )
        
        return {
            'text_length': len(text),
            'analysis_keywords': analysis_count,
            'dataset_keywords': dataset_count,
            'method_keywords': method_count,
            'numerical_mentions': numerical_mentions,
            'statistical_mentions': statistical_mentions,
            'has_methods_section': 'methods:' in text_lower or 'method:' in text_lower,
            'has_results_section': 'results:' in text_lower or 'result:' in text_lower,
        }
    
    async def _perform_llm_analysis(self, text_content: str) -> Dict[str, Any]:
        """
        Perform LLM-based analysis of the text content.
        
        Args:
            text_content: Preprocessed text for analysis
            
        Returns:
            LLM response with classification
        """
        try:
            # Get the classification prompt
            prompt_template = self.llm_service.get_prompt_template("triage_classification")
            prompt = prompt_template.render(text_content=text_content)
            
            # Call LLM with specific parameters for classification
            response = await self.llm_service.generate(
                prompt=prompt,
                parameters={
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            raise
    
    def _process_llm_result(
        self, 
        llm_response: Dict[str, Any],
        text_features: Dict[str, Any]
    ) -> TriageResult:
        """
        Process LLM response and combine with text features.
        
        Args:
            llm_response: Response from LLM
            text_features: Extracted text features
            
        Returns:
            Processed triage result
        """
        try:
            # Extract text from LLM response (chat/completions first, fallback to legacy completions)
            response_text = ""
            if "choices" in llm_response and llm_response["choices"]:
                first_choice = llm_response["choices"][0]
                message = first_choice.get("message") or {}
                response_text = (message.get("content") or "").strip()
                if not response_text:
                    response_text = first_choice.get("text", "").strip()
            
            # Try to parse JSON response with improved extraction
            import json
            import re
            
            parsed_response = None
            
            # First try: direct JSON parsing
            try:
                parsed_response = json.loads(response_text)
            except json.JSONDecodeError:
                # Second try: extract JSON from text using regex
                json_match = re.search(r'\{[^{}]*"is_data_analysis"[^{}]*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        parsed_response = json.loads(json_match.group())
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to load JSON")
                        pass
            
            if parsed_response:
                is_data_analysis = parsed_response.get("is_data_analysis", False)
                confidence_score = float(parsed_response.get("confidence_score", 0.0))
                reasoning = parsed_response.get("reasoning", "No reasoning provided")
            else:
                logger.warning(f"Failed to parse LLM JSON response. Raw text: {response_text[:200]}...")
                # Fallback: analyze response text for boolean indicators
                response_lower = response_text.lower()
                is_data_analysis = "true" in response_lower or "yes" in response_lower
                confidence_score = 5.0  # Default medium confidence
                reasoning = f"Fallback parsing from text: {response_text[:100]}..."
            
            # Apply confidence threshold
            final_classification = is_data_analysis and confidence_score >= self.config.confidence_threshold
            
            # Adjust confidence based on text features
            adjusted_confidence = self._adjust_confidence_with_features(
                confidence_score, text_features
            )
            
            return TriageResult(
                is_data_analysis=final_classification,
                confidence_score=adjusted_confidence,
                reasoning=reasoning,
                text_features=text_features,
                llm_response=llm_response
            )
            
        except Exception as e:
            logger.error(f"Error processing LLM result: {e}")
            return TriageResult(
                is_data_analysis=False,
                confidence_score=0.0,
                reasoning=f"Processing failed: {str(e)}",
                text_features=text_features,
                llm_response=llm_response
            )
    
    def _adjust_confidence_with_features(
        self,
        base_confidence: float,
        text_features: Dict[str, Any]
    ) -> float:
        """
        Adjust confidence score based on extracted text features.
        
        Args:
            base_confidence: Base confidence from LLM
            text_features: Extracted text features
            
        Returns:
            Adjusted confidence score
        """
        adjusted = base_confidence
        
        # Boost confidence if many analysis keywords found
        if text_features.get('analysis_keywords', 0) >= 3:
            adjusted = min(10.0, adjusted + 0.5)
        
        # Boost confidence if statistical mentions found
        if text_features.get('statistical_mentions', 0) >= 2:
            adjusted = min(10.0, adjusted + 0.5)
        
        # Boost confidence if has both methods and results sections
        if (text_features.get('has_methods_section', False) and 
            text_features.get('has_results_section', False)):
            adjusted = min(10.0, adjusted + 0.3)
        
        # Reduce confidence if very few indicators
        total_indicators = (
            text_features.get('analysis_keywords', 0) +
            text_features.get('statistical_mentions', 0) +
            text_features.get('dataset_keywords', 0)
        )
        if total_indicators < 2:
            adjusted = max(0.0, adjusted - 1.0)
        
        return round(adjusted, 2)


# Workflow integration function
async def triage_agent_step(state: AnalysisState) -> bool:
    """
    Workflow step function for triage analysis.
    
    Args:
        state: Current analysis state
        
    Returns:
        Boolean indicating if publication is data analysis
    """
    from ..services.llm_service import LLMService, LLMModelConfig
    from ..config.settings import get_settings
    
    # Get settings and create LLM service with proper configuration
    settings = get_settings()
    model_config = LLMModelConfig(
        name=settings.llm.default_model,
        base_url=settings.llm.base_url,
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens
    )
    llm_service = LLMService(model_config)
    
    # Create and run triage agent
    agent = TriageAgent(llm_service)
    result = await agent.analyze(state)
    
    logger.info(
        f"Triage analysis result: {result.is_data_analysis} "
        f"(confidence: {result.confidence_score})"
    )
    
    return result.is_data_analysis 