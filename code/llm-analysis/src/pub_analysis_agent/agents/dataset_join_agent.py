"""
Dataset Join Analysis Agent for Publication Analysis.

This module implements the DatasetJoinAnalysisAgent that identifies and analyzes
instances where multiple datasets are integrated or merged in research publications.
"""

import logging
import re
import json
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, UTC
import threading

from ..services.llm_service import LLMService, PromptTemplate
from ..services.dataset_service import DatasetService
from ..models.dataset import Dataset
from ..workflows.state_models import AnalysisState, DatasetJoin
from .dataset_relationship_output import (
    DatasetRelationship,
    RelationshipOutputManager,
)

logger = logging.getLogger(__name__)


@dataclass
class JoinAnalysisConfig:
    """Configuration for the DatasetJoinAnalysisAgent."""
    confidence_threshold: float = 6.0  # Minimum confidence score (0-10) for join detection
    max_text_length: int = 400000  # Maximum text length to send to LLM
    temperature: float = 0.7  # Temperature for precise analysis
    max_tokens: int = 5000  # Response length for analysis
    context_window: int = 400000  # Characters around join mention for context
    min_join_confidence: float = 5.0  # Minimum confidence for join detection
    max_joins_per_call: int = 10  # Maximum joins to analyze per LLM call
    enable_methodology_extraction: bool = True  # Whether to extract methodology details
    enable_challenge_documentation: bool = True  # Whether to document integration challenges


@dataclass
class IntegrationChallenge:
    """Represents a specific integration challenge."""
    category: str
    description: str
    severity: str
    impact: str

@dataclass
class LessonLearned:
    """Represents a lesson learned from the integration."""
    category: str
    lesson: str
    recommendation: str

@dataclass
class ValidationMethod:
    """Represents a validation method used."""
    method: str
    description: str
    results: str

@dataclass
class RiskAssessment:
    """Represents risk assessment information."""
    identified_risks: List[str]
    mitigation_strategies: List[str]
    residual_risks: List[str]

@dataclass
class DatasetJoinAnalysis:
    """Represents a detailed analysis of a dataset join."""
    dataset1: str
    dataset2: str
    join_type: str
    confidence_score: float
    methodology: Optional[str] = None
    join_keys: Optional[List[str]] = None
    integration_challenges: Optional[List[IntegrationChallenge]] = None
    success_metrics: Optional[Dict[str, Any]] = None
    context: str = ""
    section: Optional[str] = None
    text_position: int = -1
    analysis_reasoning: Optional[str] = None
    software_tools: Optional[List[str]] = None
    programming_language: Optional[str] = None
    data_preprocessing_steps: Optional[List[str]] = None
    quality_control_measures: Optional[List[str]] = None
    integration_approach: Optional[str] = None
    join_algorithm: Optional[str] = None
    matching_strategy: Optional[str] = None
    lessons_learned: Optional[List[LessonLearned]] = None
    validation_methods: Optional[List[ValidationMethod]] = None
    risk_assessment: Optional[RiskAssessment] = None


@dataclass
class JoinAnalysisResult:
    """Result from dataset join analysis."""
    joins_identified: List[DatasetJoinAnalysis]
    total_joins_found: int
    methodology_details_extracted: int
    challenges_documented: int
    processing_time: float
    llm_calls_made: int
    errors: List[str] = field(default_factory=list)


class DatasetJoinAnalysisAgent:
    """
    Agent to identify and analyze dataset joins in research publications.
    
    Uses multi-step LLM analysis to identify dataset pairs, extract joining methodologies,
    and document integration challenges and outcomes.
    """
    
    def __init__(
        self,
        llm_service: LLMService,
        dataset_service: DatasetService,
        config: Optional[JoinAnalysisConfig] = None
    ) -> None:
        """
        Initialize the DatasetJoinAnalysisAgent.
        
        Args:
            llm_service: LLM service for text analysis
            dataset_service: MongoDB dataset service for querying known datasets
            config: Configuration for join analysis
        """
        self.llm_service = llm_service
        self.dataset_service = dataset_service
        self.config = config or JoinAnalysisConfig()
        
        # Cache for known datasets to avoid repeated queries
        self._known_datasets_cache: Optional[List[Dataset]] = None
        self._dataset_names_set: Optional[Set[str]] = None
        
        # Initialize relationship output manager
        self.output_manager = RelationshipOutputManager()
        
        # Initialize prompt templates
        self._setup_prompt_templates()
        
        logger.info("DatasetJoinAnalysisAgent initialized")
    
    def _setup_prompt_templates(self) -> None:
        """Set up prompt templates for LLM analysis."""
        
        # Prompt for initial join detection
        self.join_detection_prompt = PromptTemplate(
            name="join_detection",
            template="""You are an expert in analyzing research publications to identify dataset integration and joining operations.

Publication Context:
{publication_context}

Known Datasets in Publication:
{known_datasets_list}

Text Content to Analyze:
{text_content}

Your task is to identify instances where multiple datasets are integrated, merged, joined, or combined in this research. Look for:
- Dataset pairs that are merged or joined together
- Integration operations between different datasets
- Data fusion or combination activities
- Multi-dataset analysis scenarios

Return your analysis as a JSON object with this structure:
{{
  "dataset_joins": [
    {{
      "dataset1": "name of first dataset",
      "dataset2": "name of second dataset", 
      "join_type": "merge|join|fusion|integration|combination",
      "confidence_score": 8.5,
      "context": "brief description of the join operation",
      "section": "methodology|results|discussion",
      "analysis_reasoning": "why you identified this as a join"
    }}
  ]
}}

Focus on high-confidence joins (confidence >= 6.0) and provide specific context for each identification."""
        )
        
        # Prompt for detailed methodology extraction
        self.methodology_extraction_prompt = PromptTemplate(
            name="methodology_extraction",
            template="""You are an expert in analyzing dataset integration methodologies in research publications.

Join Context:
{join_context}

Text Content:
{text_content}

Extract comprehensive methodology details for this dataset join operation. Consider these integration techniques:
- Key-based joins (using common identifiers like IDs, timestamps, etc.)
- Fuzzy matching (approximate string matching, similarity algorithms)
- Record linkage (probabilistic matching, blocking strategies)
- Statistical integration (data fusion, ensemble methods)
- SQL joins (INNER, LEFT, RIGHT, OUTER joins)
- Machine learning approaches (clustering, classification for matching)
- ETL processes (Extract, Transform, Load)

VERY IMPORTANT: DO NOT USE ANY SPECIAL CHARACTER (*|"|'...) WITHIN THE JSON OTHER THAN A COMMA OR SEMICOLON.
VERY IMPORTANT: DO NOT USE ANY SPECIAL CHARACTER (*|"|'...) WITHIN THE JSON OTHER THAN A COMMA OR SEMICOLON.

Return as JSON with ALL fields (use null for missing information):
{{
  "methodology": "detailed description of how the join was performed, including technical approach and reasoning",
  "join_keys": ["primary_key", "foreign_key", "timestamp", "other_identifiers"],
  "software_tools": ["pandas", "sql", "r", "spark", "other_tools"],
  "programming_language": "Python|R|SQL|Java|Scala|other",
  "data_preprocessing_steps": ["cleaning", "normalization", "deduplication", "validation"],
  "quality_control_measures": ["cross_validation", "data_quality_checks", "consistency_validation"],
  "integration_approach": "key-based|fuzzy_matching|record_linkage|statistical|sql|ml_based|etl|other",
  "join_algorithm": "hash_join|nested_loop|merge_join|blocking|similarity_based|other",
  "matching_strategy": "exact_match|fuzzy_match|probabilistic|rule_based|ml_based|other"
}}"""
        )
        
        # Prompt for challenge and outcome documentation
        self.challenge_documentation_prompt = PromptTemplate(
            name="challenge_documentation",
            template="""You are an expert in analyzing dataset integration challenges and outcomes in research publications.

Join Context:
{join_context}

Text Content:
{text_content}

Analyze and document comprehensive integration challenges and outcomes. Consider these challenge categories:
- Schema mismatches (different data structures, field naming, data types)
- Data quality issues (missing values, inconsistencies, duplicates, outliers)
- Temporal alignment problems (different time periods, timezone issues, sampling rates)
- Scale challenges (large datasets, memory constraints, performance bottlenecks)
- Semantic differences (different definitions, units, classifications)
- Technical limitations (software constraints, computational resources)

VERY IMPORTANT: DO NOT USE ANY SPECIAL CHARACTER (*|"|'...) WITHIN THE JSON OTHER THAN A COMMA OR SEMICOLON.
VERY IMPORTANT: DO NOT USE ANY SPECIAL CHARACTER (*|"|'...) WITHIN THE JSON OTHER THAN A COMMA OR SEMICOLON.

Extract detailed outcome metrics and validation methods. Return as JSON with ALL fields (use null for missing information):
{{
  "integration_challenges": [
    {{
      "category": "schema_mismatch|data_quality|temporal_alignment|scale|semantic|technical|other",
      "description": "detailed description of the challenge",
      "severity": "low|medium|high|critical",
      "impact": "description of how it affected the integration"
    }}
  ],
  "success_metrics": {{
    "data_loss_percentage": "5%",
    "integration_success_rate": "95%",
    "quality_improvement": "description of quality improvements achieved",
    "performance_metrics": "processing time, memory usage, throughput",
    "before_integration_stats": {{
      "dataset1_records": "10000",
      "dataset2_records": "8000",
      "data_quality_score": "85%"
    }},
    "after_integration_stats": {{
      "merged_records": "17500",
      "data_quality_score": "92%",
      "processing_time": "2.5 hours"
    }},
    "cost_benefit_analysis": "description of costs vs benefits"
  }},
  "lessons_learned": [
    {{
      "category": "technical|methodological|organizational|other",
      "lesson": "specific lesson learned",
      "recommendation": "actionable recommendation for future projects"
    }}
  ],
  "validation_methods": [
    {{
      "method": "cross_validation|manual_review|statistical_test|expert_verification|other",
      "description": "how the method was applied",
      "results": "what was validated and outcomes"
    }}
  ],
  "risk_assessment": {{
    "identified_risks": ["risk1", "risk2"],
    "mitigation_strategies": ["strategy1", "strategy2"],
    "residual_risks": ["remaining risk1", "remaining risk2"]
  }}
}}"""
        )
    
    async def analyze_dataset_joins(self, state: AnalysisState) -> JoinAnalysisResult:
        """
        Analyze dataset joins in the publication.
        
        Args:
            state: Current analysis state containing publication data
            
        Returns:
            JoinAnalysisResult with identified joins and analysis details
        """
        start_time = datetime.now(UTC)
        llm_calls = 0
        errors = []
        
        try:
            # Extract text content
            text_content = state.raw_text
            
            if not text_content:
                logger.warning("No text content available for join analysis")
                return JoinAnalysisResult(
                    joins_identified=[],
                    total_joins_found=0,
                    methodology_details_extracted=0,
                    challenges_documented=0,
                    processing_time=(datetime.now(UTC) - start_time).total_seconds(),
                    llm_calls_made=llm_calls,
                    errors=["No text content available"]
                )
            
            if not state.is_data_analysis:
                return JoinAnalysisResult(
                    joins_identified=[],
                    total_joins_found=0,
                    methodology_details_extracted=0,
                    challenges_documented=0,
                    processing_time=(datetime.now(UTC) - start_time).total_seconds(),
                    llm_calls_made=llm_calls,
                    errors=["No data analysis found in the publication"]
                )

            # Get known datasets for context
            known_datasets = await self.dataset_service.get_dataset_by_publication(state.publication_id)

            if not known_datasets:
                logger.warning("No datasets found in MongoDB")
                return JoinAnalysisResult(
                    joins_identified=[],
                    total_joins_found=0,
                    methodology_details_extracted=0,
                    challenges_documented=0,
                    processing_time=(datetime.now(UTC) - start_time).total_seconds(),
                    llm_calls_made=llm_calls,
                    errors=["No datasets available for join analysis"]
                )
            
            # Step 1: Initial join detection
            logger.info("Starting dataset join detection")
            joins_identified = await self._detect_joins_with_llm(
                text_content, known_datasets, state
            )
            llm_calls += 1
            
            # Step 2: Extract methodology details for each join
            methodology_details_extracted = 0
            for join in joins_identified:
                if self.config.enable_methodology_extraction:
                    try:
                        await self._extract_methodology_details(join, text_content)
                        methodology_details_extracted += 1
                        llm_calls += 1
                    except Exception as e:
                        logger.error(f"Error extracting methodology for join {join.dataset1}-{join.dataset2}: {e}")
                        errors.append(f"Methodology extraction error: {e}")
            
            # Step 3: Document challenges and outcomes
            challenges_documented = 0
            for join in joins_identified:
                if self.config.enable_challenge_documentation:
                    try:
                        await self._document_challenges_and_outcomes(join, text_content)
                        challenges_documented += 1
                        llm_calls += 1
                    except Exception as e:
                        logger.error(f"Error documenting challenges for join {join.dataset1}-{join.dataset2}: {e}")
                        errors.append(f"Challenge documentation error: {e}")
            
            # Post-process and validate joins
            validated_joins = self._post_process_joins(joins_identified)
            
            processing_time = (datetime.now(UTC) - start_time).total_seconds()
            
            logger.info(f"Join analysis completed: {len(validated_joins)} joins identified")
            
            return JoinAnalysisResult(
                joins_identified=validated_joins,
                total_joins_found=len(validated_joins),
                methodology_details_extracted=methodology_details_extracted,
                challenges_documented=challenges_documented,
                processing_time=processing_time,
                llm_calls_made=llm_calls,
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Error during dataset join analysis: {e}")
            errors.append(f"Join analysis error: {e}")
            return JoinAnalysisResult(
                joins_identified=[],
                total_joins_found=0,
                methodology_details_extracted=0,
                challenges_documented=0,
                processing_time=(datetime.now(UTC) - start_time).total_seconds(),
                llm_calls_made=llm_calls,
                errors=errors
            )
    
    async def _get_known_datasets(self) -> List[Dataset]:
        """Get known datasets from the service."""
        if self._known_datasets_cache is None:
            try:
                self._known_datasets_cache = await self.dataset_service.get_all_known_datasets()
                self._dataset_names_set = {
                    dataset.name.lower() for dataset in self._known_datasets_cache
                }
                logger.info(f"Loaded {len(self._known_datasets_cache)} known datasets")
            except Exception as e:
                logger.error(f"Error fetching known datasets: {e}")
                self._known_datasets_cache = []
                self._dataset_names_set = set()
        
        return self._known_datasets_cache
    
    def _extract_text_content(self, state: AnalysisState) -> str:
        """Extract relevant text content for join analysis."""
        text_parts = []
        
        # Extract from GROBID content if available
        if state.grobid_content:
            # Abstract
            if 'fulltext' in state.grobid_content and 'abstract' in state.grobid_content['fulltext']:
                abstract_text = self._extract_section_text(state.grobid_content['fulltext']['abstract'])
                if abstract_text:
                    text_parts.append(f"ABSTRACT: {abstract_text}")
            
            # Body text - focus on methodology and results sections
            if 'fulltext' in state.grobid_content and 'body' in state.grobid_content['fulltext']:
                body = state.grobid_content['fulltext']['body']
                
                # Methodology sections
                methodology_text = self._extract_methodology_sections(body)
                if methodology_text:
                    text_parts.append(f"METHODOLOGY: {methodology_text}")
                
                # Results sections
                if 'sections' in body:
                    for section in body['sections']:
                        if section.get('heading', '').lower() in ['results', 'experimental results', 'evaluation']:
                            section_text = self._extract_section_text(section)
                            if section_text:
                                text_parts.append(f"RESULTS: {section_text}")
        
        # Fallback to raw text
        if not text_parts and state.raw_text:
            text_parts.append(state.raw_text)
        
        return "\n\n".join(text_parts)
    
    def _extract_section_text(self, section: Dict[str, Any]) -> str:
        """Extract text from a GROBID section."""
        if not section:
            return ""
        
        text_parts = []
        
        # Extract from sentences
        if 'sentences' in section:
            for sentence in section['sentences']:
                if isinstance(sentence, dict) and 'text' in sentence:
                    text_parts.append(sentence['text'])
                elif isinstance(sentence, str):
                    text_parts.append(sentence)
        
        # Extract from paragraphs
        if 'paragraphs' in section:
            for paragraph in section['paragraphs']:
                if isinstance(paragraph, dict) and 'text' in paragraph:
                    text_parts.append(paragraph['text'])
                elif isinstance(paragraph, str):
                    text_parts.append(paragraph)
        
        return " ".join(text_parts)
    
    def _extract_methodology_sections(self, body: Dict[str, Any]) -> str:
        """Extract methodology-related sections from body text."""
        if not body or 'sections' not in body:
            return ""
        
        methodology_keywords = [
            'method', 'methodology', 'approach', 'procedure', 'experiment',
            'data processing', 'data analysis', 'implementation', 'setup'
        ]
        
        methodology_parts = []
        
        for section in body['sections']:
            heading = section.get('heading', '').lower()
            
            # Check if section is methodology-related
            if any(keyword in heading for keyword in methodology_keywords):
                section_text = self._extract_section_text(section)
                if section_text:
                    methodology_parts.append(f"{heading.upper()}: {section_text}")
        
        return "\n\n".join(methodology_parts)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for LLM analysis."""
        if not text:
            return ""
        
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Truncate if too long (account for the "..." suffix)
        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length - 3] + "..."
        
        return text
    
    async def _detect_joins_with_llm(
        self, 
        text: str, 
        known_datasets: List[Dataset], 
        state: AnalysisState
    ) -> List[DatasetJoinAnalysis]:
        """Use LLM to detect dataset joins in the text."""
        # Prepare known datasets list
        known_datasets_list = []
        for dataset in known_datasets:
            # TODO: check score of dataset
            known_datasets_list.append(f"- {dataset.name}")
        
        known_datasets_str = "\n".join(known_datasets_list) if known_datasets_list else "None identified"
        
        # Prepare publication context
        publication_context = f"Publication ID: {state.publication_id}"
        if state.grobid_content and 'metadata' in state.grobid_content:
            metadata = state.grobid_content['metadata']
            if 'title' in metadata:
                publication_context += f"\nTitle: {metadata['title']}"
        
        # Prepare prompt variables
        prompt_vars = {
            "publication_context": publication_context,
            "known_datasets_list": known_datasets_str,
            "text_content": self._preprocess_text(text)
        }
        
        try:
            response = await self.llm_service.generate_response(
                prompt_template=self.join_detection_prompt,
                variables=prompt_vars,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            joins = self._parse_join_detection_response(response, text)
            logger.info(f"LLM detected {len(joins)} potential dataset joins")
            return joins
            
        except Exception as e:
            logger.error(f"Error in LLM join detection: {e}")
            raise
    
    def _parse_join_detection_response(self, response: str, original_text: str) -> List[DatasetJoinAnalysis]:
        """Parse the LLM response for join detection."""
        joins = []
        
        try:
            # Try to parse as JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in LLM response")
                return []
            
            json_data = json_match.group()
            
            data = json.loads(json_data)
            
            if 'dataset_joins' in data:
                for item in data['dataset_joins']:
                    try:
                        join = DatasetJoinAnalysis(
                            dataset1=item.get('dataset1', ''),
                            dataset2=item.get('dataset2', ''),
                            join_type=item.get('join_type', 'unknown'),
                            confidence_score=float(item.get('confidence_score', 0)),
                            context=item.get('context', ''),
                            section=item.get('section', ''),
                            analysis_reasoning=item.get('analysis_reasoning', '')
                        )
                        
                        # Validate join - only check basic requirements, confidence filtering happens later
                        if (join.dataset1 and join.dataset2 and 
                            join.dataset1 != join.dataset2):
                            joins.append(join)
                    
                    except (ValueError, KeyError) as e:
                        logger.warning(f"Error parsing join item: {e}")
                        continue
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON, using fallback extraction")
            joins = self._fallback_join_extraction(response, original_text)
        
        return joins
    
    def _fallback_join_extraction(self, response: str, original_text: str) -> List[DatasetJoinAnalysis]:
        """Fallback method to extract joins using regex patterns."""
        joins = []
        
        # Look for common join patterns in the response
        join_patterns = [
            r'(\w+(?:\s+\w+)*)\s+(?:and|with|combined with|merged with|joined with)\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+(?:integration|fusion|combination)\s+(?:with|of)\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+plus\s+(\w+(?:\s+\w+)*)',
        ]
        
        for pattern in join_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                dataset1 = match.group(1).strip()
                dataset2 = match.group(2).strip()
                
                # Clean dataset names - take only the first two words for multi-word datasets
                dataset1_words = dataset1.split()[:2]
                dataset2_words = dataset2.split()[:2]
                dataset1 = ' '.join(dataset1_words)
                dataset2 = ' '.join(dataset2_words)
                
                # Basic validation
                if (len(dataset1) > 2 and len(dataset2) > 2 and 
                    dataset1.lower() != dataset2.lower()):
                    
                    join = DatasetJoinAnalysis(
                        dataset1=dataset1,
                        dataset2=dataset2,
                        join_type="unknown",
                        confidence_score=5.0,  # Lower confidence for fallback
                        context=f"Extracted from pattern: {match.group(0)}",
                        analysis_reasoning="Fallback regex extraction"
                    )
                    joins.append(join)
        
        return joins
    
    async def _extract_methodology_details(self, join: DatasetJoinAnalysis, text: str) -> None:
        """Extract detailed methodology information for a specific join."""
        join_context = f"Dataset 1: {join.dataset1}\nDataset 2: {join.dataset2}\nJoin Type: {join.join_type}\nContext: {join.context}"
        
        prompt_vars = {
            "join_context": join_context,
            "text_content": self._preprocess_text(text)
        }
        
        try:
            response = await self.llm_service.generate_response(
                prompt_template=self.methodology_extraction_prompt,
                variables=prompt_vars,
                temperature=self.config.temperature,
                max_tokens=4000
            )
            
            self._parse_methodology_response(response, join)
            
        except Exception as e:
            logger.error(f"Error extracting methodology details: {e}")
            raise
    
    def _parse_methodology_response(self, response: str, join: DatasetJoinAnalysis) -> None:
        """Parse methodology extraction response."""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in LLM response")
                return []
            
            json_data = json_match.group()
            
            data = json.loads(json_data)
            
            
            # Extract basic methodology information
            join.methodology = data.get('methodology')
            join.join_keys = data.get('join_keys', [])
            join.software_tools = data.get('software_tools', [])
            join.programming_language = data.get('programming_language')
            join.data_preprocessing_steps = data.get('data_preprocessing_steps', [])
            join.quality_control_measures = data.get('quality_control_measures', [])
            
            # Extract new technical fields
            join.integration_approach = data.get('integration_approach')
            join.join_algorithm = data.get('join_algorithm')
            join.matching_strategy = data.get('matching_strategy')
            
            # Validate and clean extracted data
            
            self._validate_methodology_data(join)
            
        except json.JSONDecodeError:
            print(f"Methodology response: {response}")
            logger.warning("Failed to parse methodology response as JSON")
        except Exception as e:
            logger.error(f"Error parsing methodology response: {e}")
    
    def _validate_methodology_data(self, join: DatasetJoinAnalysis) -> None:
        """Validate and clean extracted methodology data."""
        # Ensure lists are not None
        if join.join_keys is None:
            join.join_keys = []
        if join.software_tools is None:
            join.software_tools = []
        if join.data_preprocessing_steps is None:
            join.data_preprocessing_steps = []
        if join.quality_control_measures is None:
            join.quality_control_measures = []
        
        # Clean and validate integration approach
        if join.integration_approach:
            valid_approaches = [
                'key-based', 'fuzzy_matching', 'record_linkage', 'statistical', 
                'sql', 'ml_based', 'etl', 'other'
            ]
            if join.integration_approach not in valid_approaches:
                logger.warning(f"Invalid integration approach: {join.integration_approach}")
                join.integration_approach = 'other'
        
        # Clean and validate join algorithm
        if join.join_algorithm:
            valid_algorithms = [
                'hash_join', 'nested_loop', 'merge_join', 'blocking', 
                'similarity_based', 'other'
            ]
            if join.join_algorithm not in valid_algorithms:
                logger.warning(f"Invalid join algorithm: {join.join_algorithm}")
                join.join_algorithm = 'other'
        
        # Clean and validate matching strategy
        if join.matching_strategy:
            valid_strategies = [
                'exact_match', 'fuzzy_match', 'probabilistic', 'rule_based', 
                'ml_based', 'other'
            ]
            if join.matching_strategy not in valid_strategies:
                logger.warning(f"Invalid matching strategy: {join.matching_strategy}")
                join.matching_strategy = 'other'
    
    async def _document_challenges_and_outcomes(self, join: DatasetJoinAnalysis, text: str) -> None:
        """Document challenges and outcomes for a specific join."""
        join_context = f"Dataset 1: {join.dataset1}\nDataset 2: {join.dataset2}\nJoin Type: {join.join_type}\nContext: {join.context}"
        
        prompt_vars = {
            "join_context": join_context,
            "text_content": self._preprocess_text(text)
        }
        
        try:
            response = await self.llm_service.generate_response(
                prompt_template=self.challenge_documentation_prompt,
                variables=prompt_vars,
                temperature=self.config.temperature,
                max_tokens=4000
            )
            
            self._parse_challenge_response(response, join)
            
        except Exception as e:
            logger.error(f"Error documenting challenges and outcomes: {e}")
            raise
    
    def _parse_challenge_response(self, response: str, join: DatasetJoinAnalysis) -> None:
        """Parse challenge documentation response."""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in LLM response")
                return []
            
            json_data = json_match.group()
            
            def escape_newlines_inside_strings(match):
                return match.group(0).replace("\n", "\\n")

            json_data = re.sub(r'"([^"\\]*(\\.[^"\\]*)*)"', escape_newlines_inside_strings, json_data)

            data = json.loads(json_data)
            
            # Parse integration challenges
            challenges_data = data.get('integration_challenges', [])
            join.integration_challenges = []
            for challenge_data in challenges_data:
                if isinstance(challenge_data, dict):
                    challenge = IntegrationChallenge(
                        category=challenge_data.get('category', 'other'),
                        description=challenge_data.get('description', ''),
                        severity=challenge_data.get('severity', 'medium'),
                        impact=challenge_data.get('impact', '')
                    )
                    join.integration_challenges.append(challenge)
                elif isinstance(challenge_data, str):
                    # Fallback for old format
                    challenge = IntegrationChallenge(
                        category='other',
                        description=challenge_data,
                        severity='medium',
                        impact='Not specified'
                    )
                    join.integration_challenges.append(challenge)
            
            # Parse success metrics
            join.success_metrics = data.get('success_metrics', {})
            
            # Parse lessons learned
            lessons_data = data.get('lessons_learned', [])
            join.lessons_learned = []
            for lesson_data in lessons_data:
                if isinstance(lesson_data, dict):
                    lesson = LessonLearned(
                        category=lesson_data.get('category', 'other'),
                        lesson=lesson_data.get('lesson', ''),
                        recommendation=lesson_data.get('recommendation', '')
                    )
                    join.lessons_learned.append(lesson)
                elif isinstance(lesson_data, str):
                    # Fallback for old format
                    lesson = LessonLearned(
                        category='other',
                        lesson=lesson_data,
                        recommendation='Not specified'
                    )
                    join.lessons_learned.append(lesson)
            
            # Parse validation methods
            validation_data = data.get('validation_methods', [])
            join.validation_methods = []
            for validation_item in validation_data:
                if isinstance(validation_item, dict):
                    validation = ValidationMethod(
                        method=validation_item.get('method', 'other'),
                        description=validation_item.get('description', ''),
                        results=validation_item.get('results', '')
                    )
                    join.validation_methods.append(validation)
                elif isinstance(validation_item, str):
                    # Fallback for old format
                    validation = ValidationMethod(
                        method='other',
                        description=validation_item,
                        results='Not specified'
                    )
                    join.validation_methods.append(validation)
            
            # Parse risk assessment
            risk_data = data.get('risk_assessment', {})
            if risk_data and isinstance(risk_data, dict):
                join.risk_assessment = RiskAssessment(
                    identified_risks=risk_data.get('identified_risks', []),
                    mitigation_strategies=risk_data.get('mitigation_strategies', []),
                    residual_risks=risk_data.get('residual_risks', [])
                )
            
            # Validate and clean parsed data
            self._validate_challenge_data(join)
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse challenge response as JSON")
        except Exception as e:
            logger.error(f"Error parsing challenge response: {e}")
    
    def _validate_challenge_data(self, join: DatasetJoinAnalysis) -> None:
        """Validate and clean challenge documentation data."""
        # Validate integration challenges
        if join.integration_challenges:
            valid_categories = [
                'schema_mismatch', 'data_quality', 'temporal_alignment', 
                'scale', 'semantic', 'technical', 'other'
            ]
            valid_severities = ['low', 'medium', 'high', 'critical']
            
            for challenge in join.integration_challenges:
                if challenge.category not in valid_categories:
                    logger.warning(f"Invalid challenge category: {challenge.category}")
                    challenge.category = 'other'
                if challenge.severity not in valid_severities:
                    logger.warning(f"Invalid challenge severity: {challenge.severity}")
                    challenge.severity = 'medium'
        
        # Validate lessons learned
        if join.lessons_learned:
            valid_categories = ['technical', 'methodological', 'organizational', 'other']
            
            for lesson in join.lessons_learned:
                if lesson.category not in valid_categories:
                    logger.warning(f"Invalid lesson category: {lesson.category}")
                    lesson.category = 'other'
        
        # Validate validation methods
        if join.validation_methods:
            valid_methods = [
                'cross_validation', 'manual_review', 'statistical_test', 
                'expert_verification', 'other'
            ]
            
            for validation in join.validation_methods:
                if validation.method not in valid_methods:
                    logger.warning(f"Invalid validation method: {validation.method}")
                    validation.method = 'other'
    
    def _post_process_joins(self, joins: List[DatasetJoinAnalysis]) -> List[DatasetJoinAnalysis]:
        """Post-process and validate identified joins."""
        # Filter by confidence threshold
        valid_joins = [
            join for join in joins 
            if join.confidence_score >= self.config.confidence_threshold
        ]
        
        # Remove duplicates
        unique_joins = self._deduplicate_joins(valid_joins)
        
        # Sort by confidence score (highest first)
        unique_joins.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return unique_joins
    
    def _deduplicate_joins(self, joins: List[DatasetJoinAnalysis]) -> List[DatasetJoinAnalysis]:
        """Remove duplicate joins based on dataset pairs."""
        seen_pairs = set()
        unique_joins = []
        
        for join in joins:
            # Create normalized pair key
            pair_key = tuple(sorted([join.dataset1.lower(), join.dataset2.lower()]))
            
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                unique_joins.append(join)
        
        return unique_joins
    
    def create_structured_output(self, joins: List[DatasetJoinAnalysis]) -> List[DatasetRelationship]:
        """
        Create structured output from join analysis results.
        
        Args:
            joins: List of join analysis results
            
        Returns:
            List of structured dataset relationships
        """
        relationships = []
        
        for join in joins:
            # Convert integration challenges to dict format
            integration_challenges = None
            if join.integration_challenges:
                integration_challenges = []
                for challenge in join.integration_challenges:
                    integration_challenges.append({
                        'category': challenge.category,
                        'description': challenge.description,
                        'severity': challenge.severity,
                        'impact': challenge.impact
                    })
            
            # Convert lessons learned to dict format
            lessons_learned = None
            if join.lessons_learned:
                lessons_learned = []
                for lesson in join.lessons_learned:
                    lessons_learned.append({
                        'category': lesson.category,
                        'lesson': lesson.lesson,
                        'recommendation': lesson.recommendation
                    })
            
            # Convert validation methods to dict format
            validation_methods = None
            if join.validation_methods:
                validation_methods = []
                for validation in join.validation_methods:
                    validation_methods.append({
                        'method': validation.method,
                        'description': validation.description,
                        'results': validation.results
                    })
            
            # Convert risk assessment to dict format
            risk_assessment = None
            if join.risk_assessment:
                risk_assessment = {
                    'identified_risks': join.risk_assessment.identified_risks,
                    'mitigation_strategies': join.risk_assessment.mitigation_strategies,
                    'residual_risks': join.risk_assessment.residual_risks
                }
            
            # Create structured relationship
            relationship = self.output_manager.create_relationship_from_analysis(
                dataset1=join.dataset1,
                dataset2=join.dataset2,
                join_type=join.join_type,
                confidence_score=join.confidence_score,
                methodology=join.methodology,
                join_keys=join.join_keys,
                integration_challenges=integration_challenges,
                success_metrics=join.success_metrics,
                lessons_learned=lessons_learned,
                validation_methods=validation_methods,
                risk_assessment=risk_assessment,
                publication_context=join.context,
                tags=[join.join_type, f"confidence_{join.confidence_score}"]
            )
            
            # Add additional metadata
            if join.software_tools:
                relationship.tags.extend(join.software_tools)
            if join.programming_language:
                relationship.tags.append(join.programming_language)
            if join.integration_approach:
                relationship.tags.append(join.integration_approach)
            
            relationships.append(relationship)
            
            # Add to output manager collection
            self.output_manager.add_relationship(relationship)
        
        return relationships
    
    def get_structured_output_collection(self) -> 'RelationshipCollection':
        """
        Get the current structured output collection.
        
        Returns:
            RelationshipCollection with all analyzed relationships
        """
        return self.output_manager.get_collection()
    
    def export_structured_output(self, filepath: str) -> None:
        """
        Export structured output to JSON file.
        
        Args:
            filepath: Path to export the JSON file
        """
        self.output_manager.export_to_json(filepath)
    
    def validate_structured_output(self) -> Dict[str, List[str]]:
        """
        Validate the structured output collection.
        
        Returns:
            Dictionary of validation errors by relationship ID
        """
        return self.output_manager.validate_collection()
    
    def get_output_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the structured output collection.
        
        Returns:
            Dictionary with collection statistics
        """
        return self.output_manager.get_statistics()


async def dataset_join_analysis_agent_step(state: AnalysisState) -> List[DatasetJoin]:
    """
    LangGraph step function for dataset join analysis.
    Updates the state with structured dataset_joins array, validates integrity,
    performs detailed logging, rollback on error and serialization for debug.
    """
    logger.info(f"[LangGraph] Starting dataset join analysis step for publication {state.publication_id}")
    state_lock = threading.Lock()
    
    # Serialize state before changes (for rollback/debug)
    state_before = None
    try:
        state_before = state.to_json() if hasattr(state, 'to_json') else str(state)
    except Exception as e:
        logger.warning(f"Failed to serialize state before changes: {e}")
        state_before = None
    
    try:
        # Initialize services
        from ..services.llm_service import LLMService, LLMModelConfig
        from ..services.dataset_service import DatasetService
        from ..services.mongodb_client import MongoDBClient
        from ..config.settings import DatabaseSettings
        from .dataset_relationship_output import DatasetRelationship

        model_config = LLMModelConfig(
            name="gpt-oss:120b",
            base_url="http://localhost:1234/v1/",
            temperature=0.7,
            max_tokens=4000
        )
        db_settings = DatabaseSettings()
        mongodb_client = MongoDBClient(db_settings)
        llm_service = LLMService(model_config)
        dataset_service = DatasetService(mongodb_client)
        agent = DatasetJoinAnalysisAgent(llm_service, dataset_service)

        # Execute analysis
        result = await agent.analyze_dataset_joins(state)
        structured_joins = agent.create_structured_output(result.joins_identified)

        # Integrity validation
        validation_errors = []
        for rel in structured_joins:
            errors = rel.validate()
            if errors:
                validation_errors.append({"id": rel.relationship_id, "errors": errors})
        if validation_errors:
            logger.error(f"Validation failed for dataset_joins: {validation_errors}")
            raise ValueError(f"Validation failure in dataset_joins: {validation_errors}")

        # State update (thread-safe)
        with state_lock:
            # Replace the dataset_joins array in the state
            if hasattr(state, 'dataset_joins'):
                state.dataset_joins = structured_joins
            else:
                logger.warning("State does not have dataset_joins field, creating field.")
                state.dataset_joins = structured_joins
            state.update_step("dataset_join_analysis")
            state.mark_step_completed("dataset_join_analysis")

        logger.info(f"[LangGraph] State updated with {len(structured_joins)} dataset_joins.")
        # Serialize state after changes
        try:
            state_after = state.to_json() if hasattr(state, 'to_json') else str(state)
            logger.debug(f"[LangGraph] State after update: {state_after}")
        except Exception as e:
            logger.warning(f"Failed to serialize state after changes: {e}")
        
        # Audit trail
        logger.info(f"[LangGraph] Audit: dataset_joins updated for publication {state.publication_id}")
        return structured_joins

    except Exception as e:
        logger.error(f"[LangGraph] Error updating state with dataset_joins: {e}")
        # Rollback state
        if state_before:
            try:
                if hasattr(state, 'from_json'):
                    # Restore original state
                    restored_state = state.from_json(state_before)
                    # Update current state with restored values
                    for attr, value in restored_state.__dict__.items():
                        setattr(state, attr, value)
                    logger.warning("State rollback completed successfully.")
                else:
                    logger.warning("Rollback not supported: from_json method missing.")
            except Exception as rollback_exc:
                logger.error(f"Failed to restore state during rollback: {rollback_exc}")
        
        # Update step with error (preserves error_message)
        state.update_step("dataset_join_analysis", str(e))
        return [] 