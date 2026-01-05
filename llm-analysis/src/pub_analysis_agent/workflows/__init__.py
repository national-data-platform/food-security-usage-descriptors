"""
Workflow management for publication analysis pipeline.

This package contains workflow orchestration, state management, and
state integration utilities for the LangGraph-based analysis pipeline.
"""

from .state_models import (
    AnalysisState,
    DatasetMention,
    DatasetJoin,
    ExtractedCodeSnippet,
    ExtractedExternalLink,
    ExtractedGitHubRepository,
    ExtractionMetadata
)
from .state_converters import (
    convert_code_snippets_to_state,
    convert_external_links_to_state,
    convert_github_repos_to_state,
    convert_extraction_result_to_state,
    validate_state_data_structure,
    convert_state_to_extraction_summary
)
from .workflow_orchestrator import WorkflowOrchestrator

__all__ = [
    "AnalysisState",
    "DatasetMention", 
    "DatasetJoin",
    "ExtractedCodeSnippet",
    "ExtractedExternalLink",
    "ExtractedGitHubRepository",
    "ExtractionMetadata",
    "convert_code_snippets_to_state",
    "convert_external_links_to_state",
    "convert_github_repos_to_state",
    "convert_extraction_result_to_state",
    "validate_state_data_structure",
    "convert_state_to_extraction_summary",
    "WorkflowOrchestrator"
] 