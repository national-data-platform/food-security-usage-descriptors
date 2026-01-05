"""
Dataset Discovery Agent for Publication Analysis.

This module implements the DatasetDiscoveryAgent that identifies datasets mentioned
in publications that are not in the existing known datasets collection using LLM analysis.
"""

import logging
import re
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, UTC

from ..services.llm_service import LLMService, PromptTemplate
from ..services.dataset_service import DatasetService
from ..models.dataset import Dataset
from ..workflows.state_models import AnalysisState, DatasetMention

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryConfig:
    """Configuration for the DatasetDiscoveryAgent."""
    confidence_threshold: float = 6.0  # Minimum confidence score (0-10) for dataset discovery
    max_text_length: int = 400000  # Maximum text length to send to LLM
    temperature: float = 0.7  # Temperature for creative discovery
    max_tokens: int = 40000  # Response length for discovery
    context_window: int = 400000  # Characters around mention for context
    min_dataset_name_length: int = 3  # Minimum length for dataset name
    max_datasets_per_call: int = 15  # Maximum datasets to discover per LLM call
    enable_unknown_dataset_discovery: bool = True  # Whether to discover unknown datasets
    unknown_dataset_threshold: float = 7.0  # Confidence threshold for unknown datasets


@dataclass
class DiscoveredDataset:
    """Represents a discovered dataset."""
    name: str
    confidence_score: float
    context: str
    confidence_score_mention: Optional[float] = None  # Confidence that dataset is mentioned (0-10)
    confidence_score_use: Optional[float] = None  # Confidence that dataset is actually used (0-10)
    text_quote: Optional[str] = None  # Exact quote from paper showing usage
    description: Optional[str] = None
    source: Optional[str] = None
    domain: Optional[str] = None
    size: Optional[str] = None
    format: Optional[str] = None
    access_info: Optional[str] = None
    text_position: int = -1
    section: Optional[str] = None
    discovery_reasoning: Optional[str] = None
    is_known_dataset: bool = False
    matched_known_dataset: Optional[str] = None


@dataclass
class DiscoveryResult:
    """Result from dataset discovery analysis."""
    discovered_datasets: List[DiscoveredDataset]
    total_datasets_found: int
    known_datasets_matched: int
    unknown_datasets_found: int
    processing_time: float
    llm_calls_made: int
    errors: List[str] = field(default_factory=list)


class DatasetDiscoveryAgent:
    """
    Agent to discover datasets mentioned in publications that are not in the known datasets collection.
    
    Uses LLM analysis to identify dataset mentions, compares against known datasets,
    and discovers new datasets with confidence scoring and context extraction.
    """
    
    def __init__(
        self,
        llm_service: LLMService,
        dataset_service: DatasetService,
        config: Optional[DiscoveryConfig] = None
    ) -> None:
        """
        Initialize the DatasetDiscoveryAgent.
        
        Args:
            llm_service: LLM service for text analysis
            dataset_service: MongoDB dataset service for querying known datasets
            config: Configuration for discovery
        """
        self.llm_service = llm_service
        self.dataset_service = dataset_service
        self.config = config or DiscoveryConfig()
        
        # Cache for known datasets to avoid repeated queries
        self._known_datasets_cache: Optional[List[Dataset]] = None
        self._dataset_names_set: Optional[Set[str]] = None
        
        # Initialize prompt templates
        self._setup_prompt_templates()
        
        logger.info("DatasetDiscoveryAgent initialized")
    
    def _setup_prompt_templates(self) -> None:
        """Setup prompt templates for dataset discovery."""
        
        # Main discovery prompt
        self.discovery_prompt = PromptTemplate(
            name="dataset_discovery",
            template="""You are a specialized research analyst and expert at identifying datasets mentioned in academic publications. Your expertise lies in distinguishing between mere mentions and actual dataset utilization in research papers.

CONTEXT:
- You are analyzing a publication about: {publication_context}
- Known datasets in our database: {known_datasets_list}
- Text to analyze: {text_content}

VALIDATION CRITERIA:
A dataset is GENUINELY USED if:
1. The paper performs analysis, modeling, or experiments on the dataset
2. Results, findings, or insights are derived from the dataset
3. The dataset is central to the research methodology
4. Specific characteristics, variables, or samples from the dataset are discussed
5. The dataset is used for data processing, statistical analysis, or machine learning
6. The research methodology explicitly describes using this dataset

A dataset is ONLY MENTIONED (not genuinely used) if:
- It's cited in related work or literature review without being used in the current study
- It's mentioned as an example, comparison, or reference for future work
- It's part of a background discussion without analysis
- It's listed as a potential data source but not actually utilized
- It's mentioned in acknowledgments or funding sections only

YOUR ROLE AS RESEARCH ANALYST:
As a specialized research analyst, you must carefully examine the context of each dataset mention to determine if it represents actual usage in the research methodology, analysis, or results. Consider the publication's structure (methods, results, discussion) and the specific role each dataset plays in the research.

TASK:
1. Identify ALL dataset mentions in the text, including:
   - Datasets that match known datasets in our database
   - Completely new datasets not in our database
   - Datasets mentioned in different formats or variations

2. For each dataset mention, provide:
   - Dataset name (normalized)
   - Confidence score mention (0-10): How confident you are that the dataset is mentioned
   - Confidence score use (0-10): How confident you are that the dataset is actually used
   - Exact text quote showing usage (if found)
   - Context where it was mentioned
   - Brief description if available
   - Whether it matches a known dataset or is new

3. Focus on identifying:
   - Dataset names in quotes or italics
   - Acronyms that might be datasets
   - Dataset names in references or citations
   - Datasets mentioned in methodology sections
   - Datasets in data availability statements

STRICT SCORING RULES:
- Only assign a USE score > 2 when there is DIRECT EVIDENCE in the text_quote or context that explicitly cites the dataset by its exact name or one of its aliases AND indicates it was actually used in methods/analysis/results.
- If the dataset appears only via flag terms (e.g., organization names like "USDA", "NASS") or generic references without explicit dataset name/alias, treat as INDIRECT mention: set confidence_score_use <= 2.
- If no exact text_quote is provided that demonstrates usage, set confidence_score_use <= 2 and explain why.
- Do NOT assign moderate scores (4-6) for indirect mentions. Either it is clearly used (with explicit citation/evidence) or it is a mention/indirect reference (use <= 2).
- For MENTION scoring: only assign a MENTION score > 2 when the dataset's exact name or one of its aliases appears explicitly in the text_quote or context. Mentions via only flag terms or generic references must have confidence_score_mention <= 2.
- If no exact dataset name/alias is present, set confidence_score_mention <= 2 and explain why.

VERY IMPORTANT: 
- Do not add line breaks within the JSON text fields.
- Do not add any other text or comments outside the JSON text fields.

OUTPUT FORMAT (JSON):
{{
  "discovered_datasets": [
    {{
      "name": "Dataset Name",
      "confidence_score": 8.5,
      "confidence_score_mention": 9.0,
      "confidence_score_use": 7.5,
      "text_quote": "Exact quote from paper showing usage",
      "context": "Full context where dataset was mentioned",
      "description": "Brief description if available",
      "source": "Source information if mentioned",
      "domain": "Domain/field if mentioned",
      "size": "Size information if mentioned",
      "format": "Data format if mentioned",
      "access_info": "Access information if mentioned",
      "is_known_dataset": true/false,
      "matched_known_dataset": "Known dataset name if matched",
      "discovery_reasoning": "Brief explanation as research analyst including mention vs usage assessment"
    }}
  ],
  "analysis_summary": {{
    "total_datasets_found": 5,
    "known_datasets_matched": 3,
    "unknown_datasets_found": 2,
    "confidence_distribution": "High confidence: 3, Medium: 1, Low: 1"
  }}
}}

IMPORTANT:
- Be thorough but accurate - only include items that are clearly datasets
- Provide specific context quotes and exact text quotes when possible
- Use confidence scores to indicate certainty for both mention and usage
- Distinguish between dataset mentions and general data references
- Include detailed reasoning for each discovery explaining mention vs usage assessment""",
            variables=["publication_context", "known_datasets_list", "text_content"]
        )
    
    async def discover_datasets(self, state: AnalysisState) -> DiscoveryResult:
        """
        Discover datasets mentioned in the publication.
        
        Args:
            state: Current analysis state containing publication content
            
        Returns:
            DiscoveryResult with discovered datasets
        """
        start_time = datetime.now(UTC)
        logger.info(f"Starting dataset discovery for publication {state.publication_id}")
        
        if not state.is_data_analysis:
            return DiscoveryResult(
                discovered_datasets=[],
                total_datasets_found=0,
                known_datasets_matched=0,
                unknown_datasets_found=0,
                processing_time=0.0,
                llm_calls_made=0,
                errors=["No data analysis found in the publication"]
            )

        try:
            text_content = state.raw_text
            
            if not text_content:
                logger.warning("No text content available for dataset discovery")
                return DiscoveryResult(
                    discovered_datasets=[],
                    total_datasets_found=0,
                    known_datasets_matched=0,
                    unknown_datasets_found=0,
                    processing_time=0.0,
                    llm_calls_made=0,
                    errors=["No text content available"]
                )
            
            # Get known datasets for comparison
            known_datasets = await self.dataset_service.get_dataset_by_publication(state.publication_id)
            
            # Preprocess text
            processed_text = self._preprocess_text(text_content)
            
            # Discover datasets using LLM
            discovered_datasets = await self._discover_datasets_with_llm(
                processed_text, known_datasets, state
            )
            
            # Post-process and validate results
            final_datasets = self._post_process_discoveries(discovered_datasets, known_datasets)
            
            # Calculate statistics
            known_matched = sum(1 for d in final_datasets if d.is_known_dataset)
            unknown_found = sum(1 for d in final_datasets if not d.is_known_dataset)
            
            processing_time = (datetime.now(UTC) - start_time).total_seconds()
            
            result = DiscoveryResult(
                discovered_datasets=final_datasets,
                total_datasets_found=len(final_datasets),
                known_datasets_matched=known_matched,
                unknown_datasets_found=unknown_found,
                processing_time=processing_time,
                llm_calls_made=1  # Single LLM call for discovery
            )
            
            logger.info(f"Dataset discovery completed: {len(final_datasets)} datasets found "
                       f"({known_matched} known, {unknown_found} unknown)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during dataset discovery: {e}")
            processing_time = (datetime.now(UTC) - start_time).total_seconds()
            return DiscoveryResult(
                discovered_datasets=[],
                total_datasets_found=0,
                known_datasets_matched=0,
                unknown_datasets_found=0,
                processing_time=processing_time,
                llm_calls_made=0,
                errors=[f"Discovery error: {str(e)}"]
            )
    
    def _extract_text_content(self, state: AnalysisState) -> str:
        """Extract text content from the analysis state."""
        text_parts = []
        
        if state.grobid_content:
            # Extract from GROBID content (direct structure)
            grobid = state.grobid_content
            
            # Abstract
            if 'abstract' in grobid:
                abstract_text = self._extract_section_text(grobid['abstract'])
                if abstract_text:
                    text_parts.append(f"ABSTRACT: {abstract_text}")

            # Body sections
            if 'body' in grobid:
                body_text = self._extract_section_text(grobid['body'])
                if body_text:
                    text_parts.append(f"BODY: {body_text}")

                # Methodology sections (often contain dataset mentions)
                methodology_text = self._extract_methodology_sections(grobid['body'])
                if methodology_text:
                    text_parts.append(f"METHODOLOGY: {methodology_text}")
            
            # Data availability sections
            if 'availability' in grobid:
                availability_text = self._extract_section_text(grobid['availability'])
                if availability_text:
                    text_parts.append(f"DATA AVAILABILITY: {availability_text}")
        
        return "\n\n".join(text_parts)
    
    def _extract_section_text(self, section: Dict[str, Any]) -> str:
        """Extract text from a GROBID section."""
        if not section:
            return ""
        
        text_parts = []
        
        # Handle sections array
        if 'sections' in section:
            for sec in section['sections']:
                if isinstance(sec, dict):
                    # Extract text from sentences (new structure)
                    if 'sentences' in sec:
                        for sentence in sec['sentences']:
                            if isinstance(sentence, dict) and 'text' in sentence:
                                text_parts.append(sentence['text'])
                    # Extract text from section (legacy structure)
                    elif 'text' in sec:
                        text_parts.append(sec['text'])
                    elif 'content' in sec:
                        text_parts.append(sec['content'])
                    elif 'body' in sec:
                        text_parts.append(self._extract_section_text(sec['body']))
        
        # Handle direct sentences at root level
        elif 'sentences' in section:
            for sentence in section['sentences']:
                if isinstance(sentence, dict) and 'text' in sentence:
                    text_parts.append(sentence['text'])
        
        # Handle direct text (legacy structure)
        elif 'text' in section:
            text_parts.append(section['text'])
        elif 'content' in section:
            text_parts.append(section['content'])
        
        return " ".join(text_parts)
    
    def _extract_methodology_sections(self, body: Dict[str, Any]) -> str:
        """Extract methodology-related sections that often contain dataset mentions."""
        methodology_keywords = [
            'method', 'methodology', 'data', 'dataset', 'materials', 'procedure',
            'experiment', 'analysis', 'statistical', 'sample', 'participants'
        ]
        
        text_parts = []
        
        if 'sections' in body:
            for section in body['sections']:
                if isinstance(section, dict):
                    # Extract title from section
                    section_title = ''
                    if 'title' in section:
                        title_obj = section['title']
                        if isinstance(title_obj, dict) and 'text' in title_obj:
                            section_title = title_obj['text'].lower()
                        elif isinstance(title_obj, str):
                            section_title = title_obj.lower()
                    
                    # Extract text from sentences in section
                    section_text = ''
                    if 'sentences' in section:
                        sentence_texts = []
                        for sentence in section['sentences']:
                            if isinstance(sentence, dict) and 'text' in sentence:
                                sentence_texts.append(sentence['text'])
                        section_text = ' '.join(sentence_texts)
                    
                    # Check if section is methodology-related
                    if any(keyword in section_title for keyword in methodology_keywords):
                        text_parts.append(section_text)
        
        return " ".join(text_parts)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for LLM analysis."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Truncate if too long
        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length] + "..."
        
        return text.strip()
    
    async def _discover_datasets_with_llm(
        self, 
        text: str, 
        known_datasets: List[Dataset], 
        state: AnalysisState
    ) -> List[DiscoveredDataset]:
        """Use LLM to discover datasets in the text."""
        
        # Prepare known datasets list for prompt
        known_datasets_list = []
        for dataset in known_datasets[:50]:  # Limit to first 50 for prompt
            aliases_str = f" (aliases: {', '.join(dataset.aliases)})" if dataset.aliases else ""
            known_datasets_list.append(f"- {dataset.name}{aliases_str}")
        
        known_datasets_str = "\n".join(known_datasets_list) if known_datasets_list else "None"
        
        # Prepare publication context
        publication_context = "Academic publication"
        if state.grobid_content and 'metadata' in state.grobid_content:
            metadata = state.grobid_content['metadata']
            if 'title' in metadata:
                publication_context = f"Publication: {metadata['title']}"
        
        # Create prompt variables
        prompt_vars = {
            "publication_context": publication_context,
            "known_datasets_list": known_datasets_str,
            "text_content": text
        }
        
        try:
            # Call LLM for discovery
            response = await self.llm_service.generate_response(
                prompt_template=self.discovery_prompt,
                variables=prompt_vars,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            discovered_datasets = self._parse_discovery_response(response, text)
            
            logger.info(f"LLM discovered {len(discovered_datasets)} potential datasets")
            return discovered_datasets
            
        except Exception as e:
            logger.error(f"Error in LLM discovery: {e}")
            raise  # Re-raise the exception to be caught by the calling method
    
    def _parse_discovery_response(self, response: str, original_text: str) -> List[DiscoveredDataset]:
        """Parse LLM response to extract discovered datasets."""
        discovered_datasets = []
        
        try:
            import json
            
            # Multiple strategies to extract JSON from response
            json_data = None
            
            # Strategy 2: Look for JSON without code fence
            json_match = re.search(r'\{[^{}]*"discovered_datasets"[^{}]*\[.*?\][^{}]*\}', response, re.DOTALL)
            
            if json_match:
                json_data = json_match.group()
            
            # Strategy 3: Find any JSON-like structure
            if not json_data:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_data = json_match.group()
            
            if not json_data:
                logger.warning("No JSON found in LLM response, using fallback extraction")
                return self._fallback_dataset_extraction(response, original_text)
            
            # Clean up common JSON issues
            json_data = self._clean_json_response(json_data)
            
            # Parse JSON
            data = json.loads(json_data)
            
            if 'discovered_datasets' in data and isinstance(data['discovered_datasets'], list):
                for item in data['discovered_datasets']:
                    try:
                        # Validate required fields
                        if not isinstance(item, dict) or not item.get('name'):
                            continue
                            
                        dataset = DiscoveredDataset(
                            name=item.get('name', '').strip(),
                            confidence_score=float(item.get('confidence_score', 0)),
                            context=item.get('context', ''),
                            confidence_score_mention=float(item.get('confidence_score_mention', 0)) if item.get('confidence_score_mention') is not None else None,
                            confidence_score_use=float(item.get('confidence_score_use', 0)) if item.get('confidence_score_use') is not None else None,
                            text_quote=item.get('text_quote'),
                            description=item.get('description'),
                            source=item.get('source'),
                            domain=item.get('domain'),
                            size=item.get('size'),
                            format=item.get('format'),
                            access_info=item.get('access_info'),
                            discovery_reasoning=item.get('discovery_reasoning'),
                            is_known_dataset=bool(item.get('is_known_dataset', False)),
                            matched_known_dataset=item.get('matched_known_dataset')
                        )

                        # Validate dataset - only check name length and confidence threshold
                        discovered_datasets.append(dataset)
                    
                    except (ValueError, KeyError, TypeError) as e:
                        logger.warning(f"Error parsing dataset item: {e}")
                        continue
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.error(f"Response: {response}")
            # Fallback: try to extract dataset names using regex
            discovered_datasets = self._fallback_dataset_extraction(response, original_text)
        
        return discovered_datasets
    
    def _clean_json_response(self, json_str: str) -> str:
        """Clean up common JSON formatting issues in LLM responses."""
        # Remove any text before first {
        start_idx = json_str.find('{')
        if start_idx > 0:
            json_str = json_str[start_idx:]
        
        # Remove any text after last }
        end_idx = json_str.rfind('}')
        if end_idx >= 0:
            json_str = json_str[:end_idx + 1]
        
        # Fix common issues
        json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas before }
        json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas before ]
        json_str = re.sub(r':\s*,', ': null,', json_str)  # Fix empty values

        json_str = json_str.replace('\r\n', '\n').replace('\r', '\n')
        
        json_str = re.sub(r'\n\s*\n\s*\n+', '\n\n', json_str)
        
        json_str = re.sub(r' +\n', '\n', json_str)
        
        json_str = json_str.strip()
        
        return json_str
    
    def _fallback_dataset_extraction(self, response: str, original_text: str) -> List[DiscoveredDataset]:
        """Fallback method to extract dataset names using regex patterns."""
        datasets = []
        
        # Common dataset name patterns
        patterns = [
            r'"([^"]{3,50})"',  # Quoted names
            r'\b([A-Z][A-Z0-9\s]{2,30})\b',  # Acronyms
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Dataset)\b',  # "X Dataset"
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Database)\b',  # "X Database"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, response)
            for match in matches:
                name = match.group(1).strip()
                if len(name) >= self.config.min_dataset_name_length:
                    # Find context in original text
                    context = self._find_context_for_name(name, original_text)
                    
                    dataset = DiscoveredDataset(
                        name=name,
                        confidence_score=5.0,  # Medium confidence for regex extraction
                        context=context,
                        discovery_reasoning="Extracted using regex pattern matching"
                    )
                    datasets.append(dataset)
        
        return datasets
    
    def _find_context_for_name(self, name: str, text: str) -> str:
        """Find context around a dataset name in the text."""
        # Find the name in the text (case insensitive)
        pattern = re.escape(name)
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            start = max(0, match.start() - self.config.context_window)
            end = min(len(text), match.end() + self.config.context_window)
            return text[start:end].strip()
        
        return ""
    
    def _post_process_discoveries(
        self, 
        discoveries: List[DiscoveredDataset], 
        known_datasets: List[Dataset]
    ) -> List[DiscoveredDataset]:
        """Post-process discovered datasets to validate and enhance results."""
        processed = []
        
        for discovery in discoveries:
            # Check if it matches a known dataset
            matched_known = self._find_matching_known_dataset(discovery.name, known_datasets)
            
            if matched_known:
                discovery.is_known_dataset = True
                discovery.matched_known_dataset = matched_known.name
                discovery.confidence_score = min(discovery.confidence_score + 1, 10)  # Boost confidence
            
            # For unknown datasets, apply additional validation
            processed.append(discovery)
        
        # Remove duplicates based on name similarity
        processed = self._deduplicate_discoveries(processed)
        
        return processed
    
    def _find_matching_known_dataset(self, name: str, known_datasets: List[Dataset]) -> Optional[Dataset]:
        """Find if a discovered dataset matches a known dataset."""
        name_lower = name.lower()
        
        for dataset in known_datasets:
            # Check exact name match
            if dataset.name.lower() == name_lower:
                return dataset
            
            # Check aliases
            if dataset.aliases:
                for alias in dataset.aliases:
                    if alias.lower() == name_lower:
                        return dataset
            
            # Check flag terms
            if dataset.flag_terms:
                for term in dataset.flag_terms:
                    if term.lower() == name_lower:
                        return dataset
        
        return None
    
    def _deduplicate_discoveries(self, discoveries: List[DiscoveredDataset]) -> List[DiscoveredDataset]:
        """Remove duplicate discoveries based on name similarity."""
        if not discoveries:
            return []
        
        # Sort by confidence score (highest first)
        discoveries.sort(key=lambda x: x.confidence_score, reverse=True)
        
        unique_discoveries = []
        seen_names = set()
        
        for discovery in discoveries:
            # Normalize name for comparison
            normalized_name = re.sub(r'\s+', ' ', discovery.name.lower().strip())
            
            # Check if we've seen a similar name
            is_duplicate = False
            for seen_name in seen_names:
                # Simple similarity check
                if (normalized_name in seen_name or 
                    seen_name in normalized_name or
                    self._calculate_similarity(normalized_name, seen_name) > 0.8):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_discoveries.append(discovery)
                seen_names.add(normalized_name)
        
        return unique_discoveries
    
    def _calculate_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two dataset names."""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, name1, name2).ratio()


async def dataset_discovery_agent_step(state: AnalysisState) -> List[DatasetMention]:
    """
    LangGraph step function for dataset discovery agent.
    
    Args:
        state: Current analysis state
        
    Returns:
        List of newly discovered dataset mentions
    """
    logger.info(f"Starting dataset discovery step for publication {state.publication_id}")
    
    try:
        # Initialize services
        from ..services.llm_service import LLMService, LLMModelConfig
        from ..services.dataset_service import DatasetService
        from ..services.mongodb_client import MongoDBClient
        from ..config.settings import DatabaseSettings
        
        # Create model config for LLM service
        model_config = LLMModelConfig(
            name="gpt-oss:120b",
            base_url="http://localhost:1234/v1/",
            temperature=0.7,
            max_tokens=4000
        )
        
        # Create database settings and MongoDB client
        db_settings = DatabaseSettings()
        mongodb_client = MongoDBClient(db_settings)
        
        llm_service = LLMService(model_config)
        dataset_service = DatasetService(mongodb_client)
        
        # Create agent
        agent = DatasetDiscoveryAgent(llm_service, dataset_service)
        
        # Run discovery
        result = await agent.discover_datasets(state)
        
        # Convert to DatasetMention format for state
        dataset_mentions = []
        for discovery in result.discovered_datasets:
            mention = DatasetMention(
                name=discovery.name,
                confidence=discovery.confidence_score,
                context=discovery.context,
                section=discovery.section
            )
            dataset_mentions.append(mention)
            
            # Add to state
            state.add_new_dataset(mention)
        
        # Update state
        state.update_step("dataset_discovery")
        state.mark_step_completed("dataset_discovery")
        
        logger.info(f"Dataset discovery completed: {len(dataset_mentions)} new datasets found")
        
        return dataset_mentions
        
    except Exception as e:
        logger.error(f"Error in dataset discovery step: {e}")
        state.update_step("dataset_discovery", str(e))
        return [] 