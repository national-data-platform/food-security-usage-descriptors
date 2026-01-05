"""
Dataset Mention Validation Agent for Publication Analysis.

This module implements the DatasetValidationAgent that validates if known datasets
from MongoDB are genuinely mentioned and used in publication text using LLM analysis
and fuzzy matching techniques.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
import re
from difflib import SequenceMatcher
from rapidfuzz import fuzz, process
from rich.console import Console

from ..services.llm_service import LLMService, PromptTemplate
from ..services.dataset_service import DatasetService
from ..models.dataset import Dataset, DatasetMatchResult
from ..workflows.state_models import AnalysisState, DatasetMention

console = Console()

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for the DatasetValidationAgent."""
    confidence_threshold: float = 7.0  # Minimum confidence score (0-10) for valid mention
    fuzzy_match_threshold: int = 80  # Minimum fuzzy match score (0-100)
    max_text_length: int = 400000  # Maximum text length to send to LLM
    temperature: float = 0.7  # Low temperature for consistent validation
    max_tokens: int = 4000  # Response length for validation
    context_window: int = 400000  # Characters around match for context
    min_dataset_name_length: int = 3  # Minimum length for dataset name matching
    batch_size: int = 10  # Number of datasets to validate per LLM call


@dataclass
class DatasetEvidence:
    """Evidence for a dataset mention."""
    dataset_id: str
    dataset_name: str
    matched_text: str
    context: str
    confidence_score_mention: float
    confidence_score_use: float
    validation_reasoning: str
    text_position: int
    match_type: str  # 'exact', 'alias', 'flag_term', 'fuzzy'
    fuzzy_score: Optional[int] = None
    dataset_usage_status: str = "none"  # one of {"none", "mentioned", "analyzed"}
    doi: Optional[str] = None  # Dataset DOI if found in the text

    @property
    def confidence_score(self) -> float:
        """Aggregate confidence score used by downstream consumers.

        Prefer usage score when available; otherwise fall back to mention.
        """
        return max(self.confidence_score_use, self.confidence_score_mention)


@dataclass
class ValidationResult:
    """Result from dataset validation analysis."""
    validated_datasets: List[DatasetEvidence]
    total_datasets_checked: int
    total_mentions_found: int
    processing_time: float
    llm_calls_made: int
    errors: List[str] = field(default_factory=list)


class DatasetValidationAgent:
    """
    Agent to validate if known datasets are genuinely mentioned and used in publications.
    
    Queries MongoDB for known datasets, uses fuzzy matching for aliases and flag terms,
    and employs LLM analysis to validate genuine usage with confidence scoring.
    """
    
    def __init__(
        self,
        llm_service: LLMService,
        dataset_service: DatasetService,
        config: Optional[ValidationConfig] = None
    ) -> None:
        """
        Initialize the DatasetValidationAgent.
        
        Args:
            llm_service: LLM service for text analysis
            dataset_service: MongoDB dataset service for querying datasets
            config: Configuration for validation
        """
        self.llm_service = llm_service
        self.dataset_service = dataset_service
        self.config = config or ValidationConfig()
        
        # Cache for datasets to avoid repeated queries
        self._dataset_cache: Optional[List[Dataset]] = None
        self._fuzzy_cache: Dict[str, List[Tuple[str, int]]] = {}
        
        # Initialize prompt templates
        self._setup_prompt_templates()
        
        logger.info("DatasetValidationAgent initialized")
    
    def _setup_prompt_templates(self) -> None:
        """Setup prompt templates for dataset validation."""
        
        # Main validation prompt
        validation_prompt = PromptTemplate(
            name="dataset_validation",
            template="""You are a specialized research analyst and expert at analyzing academic publications to identify genuine dataset usage. Your expertise lies in distinguishing between mere mentions and actual dataset utilization in research papers.

CONTEXT ABOUT DATASET IDENTIFICATION:
The datasets provided to you were initially identified through string search techniques that looked for:
- **Aliases**: Alternative names, abbreviations, or variations of the dataset name (e.g., "NASS Census" might also be called "Agricultural Census", "USDA Census", "AG Census")
- **Flag Terms**: Related keywords, organizations, or terms associated with the dataset (e.g., "USDA", "National Agricultural Statistics Service", "Department of Agriculture")

These string searches found potential matches in the publication text, but now as a specialized research analyst, your critical task is to determine if these datasets are GENUINELY UTILIZED in the research or merely mentioned.

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

DATASETS TO VALIDATE:
{dataset_list}

PAPER TEXT:
{text_content}

INSTRUCTIONS:
For each dataset, provide:
1. A confidence score mention from 0-10 (10 = definitely mentioned in research, 0 = not mentioned)
2. A confidence score use from 0-10 (10 = definitely used in research, 0 = not used)
3. The exact text quote that shows usage (if found)
4. Brief reasoning for your assessment as a research analyst
5. Context around the mention to support your evaluation
. The dataset DOI (Digital Object Identifier) if explicitly mentioned in the text (e.g., "doi:10.1234/dataset.123", "https://doi.org/10.1234/dataset.123", or similar). Return null if no DOI is found. VERY IMPORTANT, ONLY RETURN DOI OF THE DATASET MENTIONED IN THE TEXT, NOT THE DATASET ITSELF.

STRICT SCORING RULES:
- Only assign a USE score > 2 when there is DIRECT EVIDENCE in the text_quote or context that explicitly cites the dataset by its exact name or one of its aliases AND indicates it was actually used in methods/analysis/results.
- If the dataset appears only via flag terms (e.g., organization names like "USDA", "NASS") or generic references without explicit dataset name/alias, treat as INDIRECT mention: set confidence_score_use <= 2.
- If no exact text_quote is provided that demonstrates usage, set confidence_score_use <= 2 and explain why.
- Do NOT assign moderate scores (4-6) for indirect mentions. Either it is clearly used (with explicit citation/evidence) or it is a mention/indirect reference (use <= 2).
- For MENTION scoring: only assign a MENTION score > 2 when the dataset's exact name or one of its aliases appears explicitly in the text_quote or context. Mentions via only flag terms or generic references must have confidence_score_mention <= 2.
- If no exact dataset name/alias is present, set confidence_score_mention <= 2 and explain why.

Format your response as JSON array (include dataset_usage_status per item as one of: "none", "mentioned", "analyzed"):
[
  {{
    "dataset_name": "<dataset_name>",
    "confidence_score_mention": <0-10>,
    "confidence_score_use": <0-10>,
    "text_quote": "<exact quote from paper>",
    "reasoning": "<brief explanation as research analyst>",
    "context": "<surrounding text for context>",
    "dataset_usage_status": "<none | mentioned | analyzed>",
    "doi": "<dataset DOI if found, otherwise null>"
  }}
]""",
            variables=["dataset_list", "text_content"]
        )
        
        self.llm_service.add_prompt_template(validation_prompt)
        
        logger.info("Prompt templates configured for DatasetValidationAgent")
    
    async def validate_datasets(self, state: AnalysisState) -> ValidationResult:
        """
        Validate dataset mentions in the publication.
        
        Args:
            state: Current analysis state with publication content
            
        Returns:
            ValidationResult with validated datasets and evidence
        """
        import time
        start_time = time.time()
        
        if not state.is_data_analysis:
            return ValidationResult(
                validated_datasets=[],
                total_datasets_checked=0,
                total_mentions_found=0,
                processing_time=time.time() - start_time,
                llm_calls_made=0,
                errors=["No data analysis found in the publication"]
            )

        try:
            logger.info(f"Starting dataset validation for publication: {state.publication_id}")
            
            datasets = await self.dataset_service.get_dataset_by_publication(state.publication_id)

            if not datasets:
                logger.warning("No datasets found in MongoDB")
                return ValidationResult(
                    validated_datasets=[],
                    total_datasets_checked=0,
                    total_mentions_found=0,
                    processing_time=time.time() - start_time,
                    llm_calls_made=0,
                    errors=["No datasets available for validation"]
                )
            
            # Extract and preprocess text
            text_content = state.raw_text

            if not text_content:
                logger.warning("No text content available for validation")
                return ValidationResult(
                    validated_datasets=[],
                    total_datasets_checked=len(datasets),
                    total_mentions_found=0,
                    processing_time=time.time() - start_time,
                    llm_calls_made=0,
                    errors=["No text content available"]
                )
            
            # Validate all datasets using LLM to check if they are mentioned in the publication
            validated_evidence = await self._validate_mentions_with_llm(
                datasets, text_content
            )
            
            # Filter by confidence threshold
            final_validated = validated_evidence

            processing_time = time.time() - start_time
            logger.info(
                f"Dataset validation completed - Found {len(final_validated)} valid mentions "
                f"out of {len(datasets)} datasets checked in {processing_time:.2f}s"
            )
            
            return ValidationResult(
                validated_datasets=final_validated,
                total_datasets_checked=len(datasets),
                total_mentions_found=len(final_validated),
                processing_time=processing_time,
                llm_calls_made=len(datasets) // self.config.batch_size + 1
            )
            
        except Exception as e:
            logger.error(f"Error in dataset validation: {e}")
            return ValidationResult(
                validated_datasets=[],
                total_datasets_checked=0,
                total_mentions_found=0,
                processing_time=time.time() - start_time,
                llm_calls_made=0,
                errors=[str(e)]
            )
    
    def _extract_text_content(self, fulltext: Any) -> str:
        """"Extract all text content from fulltext structure and concatenate into a single text field."""
        import numpy as np
        
        def extract_text_recursive(obj: Any) -> List[str]:
            """Recursively extract text from nested structures."""
            texts = []
            
            if isinstance(obj, dict):
                # Look for common text fields first
                text_fields = ['title', 'abstract', 'body']
                processed_fields = set()
                
                for field in text_fields:
                    if field in obj and obj[field]:
                        if isinstance(obj[field], str):
                            texts.append(obj[field])
                            processed_fields.add(field)
                
                # Recursively process remaining values (skip already processed text fields)
                for key, value in obj.items():
                    if key not in processed_fields:
                        texts.extend(extract_text_recursive(value))
                    
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    texts.extend(extract_text_recursive(item))
                    
            elif isinstance(obj, np.ndarray):
                # Convert to list and process
                try:
                    for item in obj.tolist():
                        texts.extend(extract_text_recursive(item))
                except:
                    pass
                    
            elif isinstance(obj, str):
                texts.append(obj)
                
            return texts
        
        try:
            # Extract all text content
            all_texts = extract_text_recursive(fulltext)
            
            # Clean and filter texts, removing duplicates
            cleaned_texts = []
            seen_texts = set()
            
            for text in all_texts:
                if text and isinstance(text, str):
                    # Remove excessive whitespace and empty lines
                    cleaned_text = ' '.join(text.split())
                    if len(cleaned_text) > 3:  # Only include meaningful text
                        # Check for duplicates (case-insensitive)
                        text_lower = cleaned_text.lower()
                        if text_lower not in seen_texts:
                            cleaned_texts.append(cleaned_text)
                            seen_texts.add(text_lower)
            
            # Join all texts with double newlines for readability
            return '\n\n'.join(cleaned_texts)
            
        except Exception as e:
            print(f"Failed to extract text from fulltext: {e}")
            logger.warning(f"Failed to extract text from fulltext: {e}")
            return "" 
        
    def _find_potential_mentions(self, datasets: List[Dataset], text: str) -> List[DatasetEvidence]:
        """
        Find potential dataset mentions using fuzzy matching.
        
        Args:
            datasets: List of known datasets
            text: Text content to search
            
        Returns:
            List of potential dataset mentions with evidence
        """
        potential_mentions = []
        text_lower = text.lower()
        for dataset in datasets:
            # Check main dataset name
            mentions = self._find_mentions_for_terms(
                dataset, [dataset.name], text, text_lower, "exact"
            )
            potential_mentions.extend(mentions)
            
            # Check aliases
            if dataset.aliases:
                alias_mentions = self._find_mentions_for_terms(
                    dataset, dataset.aliases, text, text_lower, "alias"
                )
                potential_mentions.extend(alias_mentions)
            
            # Check flag terms
            if dataset.flag_terms:
                flag_mentions = self._find_mentions_for_terms(
                    dataset, dataset.flag_terms, text, text_lower, "flag_term"
                )
                potential_mentions.extend(flag_mentions)
        
        # Remove duplicates and sort by position
        unique_mentions = self._deduplicate_mentions(potential_mentions)
        unique_mentions.sort(key=lambda x: x.text_position)
        
        logger.info(f"Found {len(unique_mentions)} potential dataset mentions")
        return unique_mentions
    
    def _find_mentions_for_terms(
        self, 
        dataset: Dataset, 
        terms: List[str], 
        text: str, 
        text_lower: str, 
        match_type: str
    ) -> List[DatasetEvidence]:
        """
        Find mentions for a specific set of terms.
        
        Args:
            dataset: Dataset object
            terms: List of terms to search for
            text: Original text
            text_lower: Lowercase text
            match_type: Type of match (exact, alias, flag_term)
            
        Returns:
            List of DatasetEvidence for found mentions
        """
        mentions = []
        
        for term in terms:
            if len(term) < self.config.min_dataset_name_length:
                continue
                
            term_lower = term.lower()
            
            # Exact matching
            for match in re.finditer(re.escape(term_lower), text_lower):
                start_pos = match.start()
                end_pos = match.end()
                
                # Extract context around the match
                context_start = max(0, start_pos - self.config.context_window)
                context_end = min(len(text), end_pos + self.config.context_window)
                context = text[context_start:context_end]
                
                # Get the actual matched text (preserving original case)
                matched_text = text[start_pos:end_pos]
                
                evidence = DatasetEvidence(
                    dataset_id=dataset.id,
                    dataset_name=dataset.name,
                    matched_text=matched_text,
                    context=context,
                    confidence_score_mention=0.0,
                    confidence_score_use=0.0,
                    validation_reasoning="",  # Will be set by LLM
                    text_position=start_pos,
                    match_type=match_type
                )
                mentions.append(evidence)
            
            # Fuzzy matching for approximate matches
            if match_type in ["exact", "alias"]:
                fuzzy_matches = self._find_fuzzy_matches(term, text, text_lower)
                for match_info in fuzzy_matches:
                    evidence = DatasetEvidence(
                        dataset_id=dataset.id,
                        dataset_name=dataset.name,
                        matched_text=match_info["matched_text"],
                        context=match_info["context"],
                        confidence_score_mention=0.0,
                        confidence_score_use=0.0,
                        validation_reasoning="",  # Will be set by LLM
                        text_position=match_info["position"],
                        match_type="fuzzy",
                        fuzzy_score=match_info["fuzzy_score"]
                    )
                    mentions.append(evidence)
        
        return mentions
    
    def _find_fuzzy_matches(self, term: str, text: str, text_lower: str) -> List[Dict[str, Any]]:
        """
        Find fuzzy matches for a term in text.
        
        Args:
            term: Term to search for
            text: Original text
            text_lower: Lowercase text
            
        Returns:
            List of fuzzy match information
        """
        matches = []
        
        # Split text into words for fuzzy matching
        words = re.findall(r'\b\w+\b', text_lower)
        word_positions = []
        
        # Find positions of words
        for word in words:
            for match in re.finditer(re.escape(word), text_lower):
                word_positions.append({
                    "word": word,
                    "start": match.start(),
                    "end": match.end()
                })
        
        # Check fuzzy similarity for each word
        for word_info in word_positions:
            word = word_info["word"]
            if len(word) >= self.config.min_dataset_name_length:
                fuzzy_score = fuzz.ratio(term.lower(), word)
                
                if fuzzy_score >= self.config.fuzzy_match_threshold:
                    start_pos = word_info["start"]
                    end_pos = word_info["end"]
                    
                    # Extract context
                    context_start = max(0, start_pos - self.config.context_window)
                    context_end = min(len(text), end_pos + self.config.context_window)
                    context = text[context_start:context_end]
                    
                    # Get matched text preserving case
                    matched_text = text[start_pos:end_pos]
                    
                    matches.append({
                        "matched_text": matched_text,
                        "context": context,
                        "position": start_pos,
                        "fuzzy_score": fuzzy_score
                    })
        
        return matches
    
    def _deduplicate_mentions(self, mentions: List[DatasetEvidence]) -> List[DatasetEvidence]:
        """
        Remove duplicate mentions based on position and dataset.
        
        Args:
            mentions: List of potential mentions
            
        Returns:
            Deduplicated list of mentions
        """
        seen = set()
        unique_mentions = []
        
        for mention in mentions:
            # Create a key based on dataset_id and approximate position
            # Allow some overlap tolerance (e.g., 10 characters)
            position_key = mention.text_position // 10
            key = (mention.dataset_id, position_key)
            
            if key not in seen:
                seen.add(key)
                unique_mentions.append(mention)
        
        return unique_mentions
    
    async def _validate_mentions_with_llm(
        self, 
        datasets: List[Dataset], 
        text_content: str
    ) -> List[DatasetEvidence]:
        """
        Validate if each dataset from the list is mentioned in the publication using LLM analysis.
        
        Args:
            datasets: List of datasets to validate for mentions
            text_content: Full text content of the publication
            
        Returns:
            List of validated DatasetEvidence with confidence scores
        """
        validated_mentions = []
        # Process datasets in batches
        for i in range(0, len(datasets), self.config.batch_size):
            batch = datasets[i:i + self.config.batch_size]
            try:
                # Prepare dataset list for LLM
                dataset_list = []
                for dataset in batch:
                    # Include dataset name and aliases for comprehensive validation
                    aliases_str = ", ".join(dataset.aliases) if dataset.aliases else "None"
                    flag_terms_str = ", ".join(dataset.flag_terms) if dataset.flag_terms else "None"
                    dataset_list.append(
                        f"- {dataset.name} (aliases: {aliases_str}, flag_terms: {flag_terms_str})"
                    )
                
                dataset_list_str = "\n".join(dataset_list)
                
                # Get validation prompt
                prompt_template = self.llm_service.get_prompt_template("dataset_validation")
                prompt = prompt_template.render(
                    dataset_list=dataset_list_str,
                    text_content=text_content
                )

                # Call LLM
                response = await self.llm_service.generate(
                    prompt=prompt,
                    parameters={
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens
                    }
                )
                
                # Process LLM response and create DatasetEvidence objects
                batch_validated = self._process_dataset_validation_response(response, batch, text_content)
                validated_mentions.extend(batch_validated)
                
            except Exception as e:
                logger.error(f"Error validating batch {i//self.config.batch_size + 1}: {e}")
                # Add datasets with zero confidence on error
                for dataset in batch:
                    evidence = DatasetEvidence(
                        dataset_id=dataset.id or "",
                        dataset_name=dataset.name,
                        matched_text="",
                        context="",
                        confidence_score_mention=0.0,
                        confidence_score_use=0.0,
                        validation_reasoning=f"Validation failed: {str(e)}",
                        text_position=0,
                        match_type="validation_error"
                    )
                    validated_mentions.append(evidence)
        
        return validated_mentions
    
    def _process_dataset_validation_response(
        self, 
        llm_response: Dict[str, Any], 
        datasets: List[Dataset],
        text_content: str
    ) -> List[DatasetEvidence]:
        """
        Process LLM validation response and create DatasetEvidence objects for each dataset.
        
        Args:
            llm_response: Response from LLM
            datasets: Original datasets that were validated
            text_content: Full text content for context extraction
            
        Returns:
            List of DatasetEvidence objects with validation results
        """
        validated_evidence = []
        try:
            # Extract text from LLM response (chat/completions first, fallback to legacy completions)
            response_text = ""
            if "choices" in llm_response and llm_response["choices"]:
                first_choice = llm_response["choices"][0]
                message = first_choice.get("message") or {}
                response_text = (message.get("content") or "").strip()
                if not response_text:
                    response_text = first_choice.get("text", "").strip()
            
            # Try to parse JSON response
            import json
            try:
                # json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                json_match = re.search(r'\[\s*{.*?}\s*\]', response_text, re.DOTALL)
                if not json_match:
                    logger.warning("No JSON found in LLM response")
                    # Fallback: create evidence with low confidence for all datasets
                    for dataset in datasets:
                        evidence = DatasetEvidence(
                            dataset_id=dataset.id or "",
                            dataset_name=dataset.name,
                            matched_text="",
                            context="",
                            confidence_score_mention=0.0,
                            confidence_score_use=0.0,
                            validation_reasoning="Failed to parse LLM response",
                            text_position=0,
                            match_type="llm_parse_error"
                        )
                        validated_evidence.append(evidence)
                    return validated_evidence
                
                validations = json.loads(json_match.group())
                if not isinstance(validations, list):
                    validations = [validations]
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse LLM validation response: {e}")
                # Fallback: create evidence with low confidence for all datasets
                for dataset in datasets:
                    evidence = DatasetEvidence(
                        dataset_id=dataset.id or "",
                        dataset_name=dataset.name,
                        matched_text="",
                        context="",
                        confidence_score_mention=0.0,
                        confidence_score_use=0.0,
                        validation_reasoning="Failed to parse LLM response",
                        text_position=0,
                        match_type="llm_parse_error"
                    )
                    validated_evidence.append(evidence)
                return validated_evidence
            
            # Match validations to datasets
            for dataset in datasets:
                # Find matching validation by dataset name
                matching_validation = None
                for validation in validations:
                    if (isinstance(validation, dict) and 
                        validation.get("dataset_name", "").lower() == dataset.name.lower()):
                        matching_validation = validation
                        break
                
                # Extract text quote and context from LLM response
                text_quote = matching_validation.get("text_quote", "") if matching_validation else ""
                llm_context = matching_validation.get("context", "") if matching_validation else ""
                confidence_score_mention = float(matching_validation.get("confidence_score_mention", 0.0)) if matching_validation else 0.0
                confidence_score_use = float(matching_validation.get("confidence_score_use", 0.0)) if matching_validation else 0.0
                reasoning = matching_validation.get("reasoning", "No reasoning provided") if matching_validation else "No validation found in LLM response"
                usage_status = str(matching_validation.get("dataset_usage_status", "none")).strip().lower() if matching_validation else "none"
                if usage_status not in {"none", "mentioned", "analyzed"}:
                    usage_status = "none"
                
                # Extract DOI if present
                doi = matching_validation.get("doi") if matching_validation else None
                # Normalize DOI: convert empty strings or "null" strings to None
                if doi is not None and isinstance(doi, str):
                    doi = doi.strip()
                    if not doi or doi.lower() in ("null", "none", "n/a"):
                        doi = None

                # # Enforce direct-evidence rule for USE score: cap to 2 if no explicit name/alias present
                # try:
                #     dataset_names = [dataset.name] + (dataset.aliases or [])
                # except Exception:
                #     dataset_names = [dataset.name]
                # lowered_names = {n.lower() for n in dataset_names if isinstance(n, str) and n}
                # text_quote_lower = (text_quote or "").lower()
                # context_lower = (llm_context or "").lower()

                # # Direct if the quoted text contains an exact dataset name/alias
                # direct_in_quote = any(name in text_quote_lower for name in lowered_names)
                # # Heuristic: also consider direct if context contains explicit dataset name/alias and usage verbs
                # usage_verbs = (" use ", " used ", " using ", " utilize ", " utilized ", " employed ", " applied ")
                # direct_in_context = any(name in context_lower for name in lowered_names) and any(v in context_lower for v in usage_verbs)
                # is_direct_usage = direct_in_quote or direct_in_context

                # # Mention must also be direct (dataset name/alias present). If not, cap mention score.
                # is_direct_mention = direct_in_quote or any(name in context_lower for name in lowered_names)

                # if not is_direct_usage:
                #     if confidence_score_use > 2.0:
                #         confidence_score_use = 2.0
                #         if reasoning:
                #             reasoning = f"{reasoning} [Auto-adjusted: indirect mention detected → use score capped ≤2]"

                # if not is_direct_mention:
                #     if confidence_score_mention > 2.0:
                #         confidence_score_mention = 2.0
                #         if reasoning:
                #             reasoning = f"{reasoning} [Auto-adjusted: no explicit dataset name/alias → mention score capped ≤2]"
                
                # Find position of the quote in the text if available
                text_position = 0
                if text_quote and text_content:
                    pos = text_content.find(text_quote)
                    if pos != -1:
                        text_position = pos
                
                # Create DatasetEvidence object
                evidence = DatasetEvidence(
                    dataset_id=dataset.id or "",
                    dataset_name=dataset.name,
                    matched_text=text_quote,
                    context=llm_context,
                    confidence_score_mention=confidence_score_mention,
                    confidence_score_use=confidence_score_use,
                    validation_reasoning=reasoning,
                    text_position=text_position,
                    match_type="llm_validation",
                    dataset_usage_status=usage_status,
                    doi=doi
                )
                validated_evidence.append(evidence)
            
            return validated_evidence
            
        except Exception as e:
            logger.error(f"Error processing validation response: {e}")
            # Create evidence with error for all datasets
            for dataset in datasets:
                evidence = DatasetEvidence(
                    dataset_id=dataset.id or "",
                    dataset_name=dataset.name,
                    matched_text="",
                    context="",
                    confidence_score_mention=0.0,
                    confidence_score_use=0.0,
                    validation_reasoning=f"Processing failed: {str(e)}",
                    text_position=0,
                    match_type="processing_error"
                )
                validated_evidence.append(evidence)
            return validated_evidence


# Workflow integration function
async def dataset_validation_agent_step(state: AnalysisState) -> List[DatasetMention]:
    """
    Workflow step function for dataset validation.
    
    Args:
        state: Current analysis state
        
    Returns:
        List of validated DatasetMention objects
    """
    from ..services.llm_service import LLMService, LLMModelConfig
    from ..services.dataset_service import DatasetService
    from ..services.mongodb_client import MongoDBClient
    
    # Create services (in real implementation, these would be injected)
    model_config = LLMModelConfig(
        name="llama3",
        base_url="http://localhost:11434",
        temperature=0.7,
        max_tokens=4000
    )
    llm_service = LLMService(model_config)
    
    # Create MongoDB client and dataset service
    mongodb_client = MongoDBClient()
    dataset_service = DatasetService(mongodb_client)
    
    # Create and run validation agent
    agent = DatasetValidationAgent(llm_service, dataset_service)
    result = await agent.validate_datasets(state)
    
    # Convert evidence to DatasetMention objects
    dataset_mentions = []
    for evidence in result.validated_datasets:
        mention = DatasetMention(
            name=evidence.dataset_name,
            confidence_score_mention=evidence.confidence_score_mention,
            confidence_score_use=evidence.confidence_score_use,
            text_quote=evidence.matched_text,
            context=evidence.context,
            dataset_usage_status=evidence.dataset_usage_status,
            doi=evidence.doi
        )
        dataset_mentions.append(mention)
    
    logger.info(
        f"Dataset validation result: {len(dataset_mentions)} validated mentions "
        f"from {result.total_mentions_found} potential mentions"
    )
    
    return dataset_mentions 