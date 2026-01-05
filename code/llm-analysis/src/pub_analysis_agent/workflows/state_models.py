"""
State models for workflow management.

This module defines the state models used in the workflow orchestrator.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DatasetMention:
    """Represents a dataset mention found in the publication."""
    name: str
    confidence_score_mention: float
    confidence_score_use: float
    text_quote: str
    context: str
    description: Optional[str] = None
    source: Optional[str] = None
    domain: Optional[str] = None
    is_known_dataset: bool = False
    discovery_reasoning: Optional[str] = None
    dataset_usage_status: str = "none"  # one of {"none", "mentioned", "analyzed"}
    doi: Optional[str] = None  # Dataset DOI if found in the text


@dataclass
class DatasetJoin:
    """Represents a join between datasets."""
    dataset1: str
    dataset2: str
    join_type: str
    confidence_score: float
    methodology: Optional[str] = None
    join_keys: Optional[List[str]] = None
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


@dataclass
class ExtractedGitHubRepository:
    """Serializable representation of GitHub repository information."""
    url: str
    owner: str
    repository: str
    path: Optional[str] = None
    branch: Optional[str] = None
    is_valid: bool = False
    description: Optional[str] = None
    language: Optional[str] = None
    stars: Optional[int] = None


@dataclass
class ExtractedCodeSnippet:
    """Serializable representation of extracted code snippet."""
    content: str
    language: str
    context: str
    relevance_score: float
    description: Optional[str] = None
    purpose: Optional[str] = None
    start_position: int = -1
    end_position: int = -1


@dataclass
class ExtractedExternalLink:
    """Serializable representation of external link information."""
    url: str
    link_type: str
    title: Optional[str] = None
    description: Optional[str] = None
    context: str = ""
    is_accessible: Optional[bool] = None
    relevance_score: float = 0.0


@dataclass
class ExtractionMetadata:
    """Metadata about the code and link extraction process."""
    total_code_blocks: int = 0
    total_links_found: int = 0
    programming_languages: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    extraction_errors: List[str] = field(default_factory=list)
    extraction_timestamp: Optional[str] = None


@dataclass
class RepositoryVerificationResult:
    """Result of GitHub repository verification."""
    url: str
    owner: str
    repository: str
    default_branch: Optional[str] = None
    downloaded: bool = False
    files_analyzed_count: int = 0
    alignment_score: float = 0.0  # 0-10 score indicating alignment with publication
    passed: bool = False  # Whether the repository passed the alignment check
    key_findings: List[str] = field(default_factory=list)
    code_description: str = ""  # LLM-generated description of what the analyzed code does
    data_sanitization: str = ""  # LLM analysis of data cleaning/filtering/preprocessing
    dataset_joins: str = ""  # LLM analysis of dataset joins/integration evidence
    data_sanitization_code: str = ""  # Actual code snippets showing data sanitization
    dataset_joins_code: str = ""  # Actual code snippets showing dataset joins
    error: Optional[str] = None


@dataclass
class AnalysisState:
    """
    Central state dataclass for LangGraph workflow management.
    
    Contains all state information passed between analysis agents during
    sequential execution of the publication analysis pipeline.
    """
    
    # Core identifiers
    publication_id: str
    workflow_id: Optional[str] = None
    
    # Input content
    grobid_content: Optional[Dict[str, Any]] = None
    raw_text: Optional[str] = None
    
    # Analysis flags
    is_data_analysis: Optional[bool] = None
    has_datasets: Optional[bool] = None
    
    # Dataset analysis results
    validated_datasets: List[DatasetMention] = field(default_factory=list)
    newly_discovered_datasets: List[DatasetMention] = field(default_factory=list)
    dataset_joins: List[DatasetJoin] = field(default_factory=list)
    
    # Code and link extraction results
    extracted_code: List[ExtractedCodeSnippet] = field(default_factory=list)
    extracted_links: List[ExtractedExternalLink] = field(default_factory=list)
    extracted_github_repos: List[ExtractedGitHubRepository] = field(default_factory=list)
    extraction_metadata: Optional[ExtractionMetadata] = None
    # GitHub repository verification
    repo_verification_results: List[RepositoryVerificationResult] = field(default_factory=list)
    
    # Final output
    final_json: Optional[Dict[str, Any]] = None
    
    # Workflow metadata
    current_step: Optional[str] = None
    completed_steps: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the state to a dictionary for storage/debugging.
        
        Returns:
            Dictionary representation of the state
        """
        try:
            state_dict = asdict(self)
            # Convert datetime objects to ISO strings for JSON serialization
            state_dict['created_at'] = self.created_at.isoformat()
            state_dict['updated_at'] = self.updated_at.isoformat()
            
            # Convert numpy arrays and other non-serializable types
            return self._convert_for_serialization(state_dict)
        except Exception as e:
            logger.error(f"Failed to serialize state: {e}")
            raise ValueError(f"State serialization failed: {e}")
    
    def _convert_for_serialization(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable types to serializable formats."""
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._convert_for_serialization(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_serialization(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisState':
        """
        Deserialize state from a dictionary.
        
        Args:
            data: Dictionary representation of the state
            
        Returns:
            AnalysisState instance
        """
        try:
            # Convert ISO strings back to datetime objects
            if 'created_at' in data and isinstance(data['created_at'], str):
                data['created_at'] = datetime.fromisoformat(data['created_at'])
            if 'updated_at' in data and isinstance(data['updated_at'], str):
                data['updated_at'] = datetime.fromisoformat(data['updated_at'])
            
            # Convert dataset mentions and joins from dicts if needed
            if 'validated_datasets' in data:
                data['validated_datasets'] = [
                    DatasetMention(**item) if isinstance(item, dict) else item
                    for item in data['validated_datasets']
                ]
            
            if 'newly_discovered_datasets' in data:
                data['newly_discovered_datasets'] = [
                    DatasetMention(**item) if isinstance(item, dict) else item
                    for item in data['newly_discovered_datasets']
                ]
            
            if 'dataset_joins' in data:
                data['dataset_joins'] = [
                    DatasetJoin(**item) if isinstance(item, dict) else item
                    for item in data['dataset_joins']
                ]
            
            # Convert extraction data from dicts if needed
            if 'extracted_code' in data:
                data['extracted_code'] = [
                    ExtractedCodeSnippet(**item) if isinstance(item, dict) else item
                    for item in data['extracted_code']
                ]
            
            if 'extracted_links' in data:
                data['extracted_links'] = [
                    ExtractedExternalLink(**item) if isinstance(item, dict) else item
                    for item in data['extracted_links']
                ]
            
            if 'extracted_github_repos' in data:
                data['extracted_github_repos'] = [
                    ExtractedGitHubRepository(**item) if isinstance(item, dict) else item
                    for item in data['extracted_github_repos']
                ]
            
            if 'extraction_metadata' in data and isinstance(data['extraction_metadata'], dict):
                data['extraction_metadata'] = ExtractionMetadata(**data['extraction_metadata'])
            
            if 'repo_verification_results' in data:
                data['repo_verification_results'] = [
                    RepositoryVerificationResult(**item) if isinstance(item, dict) else item
                    for item in data['repo_verification_results']
                ]
            
            return cls(**data)
        except Exception as e:
            logger.error(f"Failed to deserialize state: {e}")
            raise ValueError(f"State deserialization failed: {e}")
    
    def to_json(self) -> str:
        """
        Serialize state to JSON string.
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AnalysisState':
        """
        Deserialize state from JSON string.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            AnalysisState instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def update_step(self, step_name: str, error: Optional[str] = None) -> None:
        """
        Update the current workflow step and timestamp.
        
        Args:
            step_name: Name of the current step
            error: Optional error message
        """
        if self.current_step and self.current_step not in self.completed_steps:
            self.completed_steps.append(self.current_step)
        
        self.current_step = step_name
        self.error_message = error
        self.updated_at = datetime.now(UTC)
        
        logger.info(f"State updated - Step: {step_name}, Error: {error}")
    
    def mark_step_completed(self, step_name: Optional[str] = None) -> None:
        """
        Mark a step as completed.
        
        Args:
            step_name: Name of the step to mark as completed. If None, uses current_step.
        """
        step_to_complete = step_name or self.current_step
        if step_to_complete and step_to_complete not in self.completed_steps:
            self.completed_steps.append(step_to_complete)
            self.updated_at = datetime.now(UTC)
            logger.info(f"Step marked as completed: {step_to_complete}")
    
    def add_validated_dataset(self, dataset: DatasetMention) -> None:
        """Add a validated dataset mention to the state."""
        self.validated_datasets.append(dataset)
        self.updated_at = datetime.now(UTC)
    
    def add_new_dataset(self, dataset: DatasetMention) -> None:
        """Add a newly discovered dataset to the state."""
        self.newly_discovered_datasets.append(dataset)
        self.updated_at = datetime.now(UTC)
    
    def add_dataset_join(self, join: DatasetJoin) -> None:
        """Add a dataset join to the state."""
        self.dataset_joins.append(join)
        self.updated_at = datetime.now(UTC)
    
    def get_all_datasets(self) -> List[DatasetMention]:
        """Get all datasets (validated + newly discovered)."""
        return self.validated_datasets + self.newly_discovered_datasets
    
    def update_extraction_results(
        self,
        code_snippets: List[ExtractedCodeSnippet] = None,
        external_links: List[ExtractedExternalLink] = None,
        github_repos: List[ExtractedGitHubRepository] = None,
        metadata: Optional[ExtractionMetadata] = None
    ) -> None:
        """
        Update the state with extraction results.
        
        Args:
            code_snippets: List of extracted code snippets
            external_links: List of extracted external links  
            github_repos: List of extracted GitHub repositories
            metadata: Extraction metadata
        """
        if code_snippets is not None:
            self.extracted_code = code_snippets
        
        if external_links is not None:
            self.extracted_links = external_links
        
        if github_repos is not None:
            self.extracted_github_repos = github_repos
        
        if metadata is not None:
            self.extraction_metadata = metadata
        
        self.updated_at = datetime.now(UTC)
        logger.info(f"State updated with extraction results: {len(self.extracted_code)} code snippets, "
                   f"{len(self.extracted_links)} links, {len(self.extracted_github_repos)} GitHub repos")
    
    def add_extracted_code_snippet(self, snippet: ExtractedCodeSnippet) -> None:
        """Add a single extracted code snippet to the state."""
        self.extracted_code.append(snippet)
        self.updated_at = datetime.now(UTC)
    
    def add_extracted_link(self, link: ExtractedExternalLink) -> None:
        """Add a single extracted external link to the state."""
        self.extracted_links.append(link)
        self.updated_at = datetime.now(UTC)
    
    def add_extracted_github_repo(self, repo: ExtractedGitHubRepository) -> None:
        """Add a single extracted GitHub repository to the state."""
        self.extracted_github_repos.append(repo)
        self.updated_at = datetime.now(UTC)
    
    def get_all_extracted_content(self) -> Dict[str, Any]:
        """
        Get all extracted content in a structured format.
        
        Returns:
            Dictionary with all extracted content and metadata
        """
        return {
            "code_snippets": [asdict(snippet) for snippet in self.extracted_code],
            "external_links": [asdict(link) for link in self.extracted_links],
            "github_repositories": [asdict(repo) for repo in self.extracted_github_repos],
            "repo_verification_results": [asdict(r) for r in self.repo_verification_results],
            "metadata": asdict(self.extraction_metadata) if self.extraction_metadata else None
        }
    
    def add_repo_verification_result(self, result: RepositoryVerificationResult) -> None:
        """Add a repository verification result to the state."""
        self.repo_verification_results.append(result)
        self.updated_at = datetime.now(UTC)

    def has_extraction_results(self) -> bool:
        """Check if the state contains any extraction results."""
        return (len(self.extracted_code) > 0 or 
                len(self.extracted_links) > 0 or 
                len(self.extracted_github_repos) > 0)
    
    def get_extraction_summary(self) -> Dict[str, int]:
        """Get a summary of extraction results."""
        return {
            "code_snippets_count": len(self.extracted_code),
            "external_links_count": len(self.extracted_links),
            "github_repos_count": len(self.extracted_github_repos),
            "total_programming_languages": len(self.extraction_metadata.programming_languages) if self.extraction_metadata else 0
        }
    
    def validate_extraction_data_integrity(self) -> List[str]:
        """
        Validate the integrity of extraction data.
        
        Returns:
            List of validation errors (empty if no errors)
        """
        errors = []
        
        # Validate code snippets
        for i, snippet in enumerate(self.extracted_code):
            if not snippet.content or not snippet.content.strip():
                errors.append(f"Code snippet {i} has empty content")
            if not snippet.language:
                errors.append(f"Code snippet {i} has no language specified")
            if snippet.relevance_score < 0 or snippet.relevance_score > 10:
                errors.append(f"Code snippet {i} has invalid relevance score: {snippet.relevance_score}")
        
        # Validate external links
        for i, link in enumerate(self.extracted_links):
            if not link.url or not link.url.strip():
                errors.append(f"External link {i} has empty URL")
            if not link.link_type:
                errors.append(f"External link {i} has no link type specified")
            if link.relevance_score < 0 or link.relevance_score > 10:
                errors.append(f"External link {i} has invalid relevance score: {link.relevance_score}")
        
        # Validate GitHub repositories
        for i, repo in enumerate(self.extracted_github_repos):
            if not repo.url or not repo.url.strip():
                errors.append(f"GitHub repository {i} has empty URL")
            if not repo.owner or not repo.repository:
                errors.append(f"GitHub repository {i} has incomplete owner/repository information")
        
        # Validate metadata consistency
        if self.extraction_metadata:
            total_items = (len(self.extracted_code) + len(self.extracted_links) + 
                          len(self.extracted_github_repos))
            if self.extraction_metadata.total_links_found > 0:
                actual_links = len(self.extracted_links) + len(self.extracted_github_repos)
                if abs(actual_links - self.extraction_metadata.total_links_found) > 2:
                    errors.append(f"Metadata link count mismatch: actual={actual_links}, metadata={self.extraction_metadata.total_links_found}")
        
        return errors
    
    def is_complete(self) -> bool:
        """Check if the workflow is complete (has final_json)."""
        return self.final_json is not None
    
    def has_error(self) -> bool:
        """Check if the workflow has an error."""
        return self.error_message is not None 