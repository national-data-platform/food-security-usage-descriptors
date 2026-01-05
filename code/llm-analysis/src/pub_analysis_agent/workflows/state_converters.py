"""
Utility functions for converting between extraction data types and state data types.

This module provides conversion functions to transform complex nested objects
from agents into serializable state-compatible data structures.
"""

import logging
from datetime import datetime, UTC
from typing import List, Dict, Any

from .state_models import (
    ExtractedCodeSnippet, 
    ExtractedExternalLink, 
    ExtractedGitHubRepository,
    ExtractionMetadata
)

logger = logging.getLogger(__name__)


def convert_code_snippets_to_state(code_snippets) -> List[ExtractedCodeSnippet]:
    """
    Convert CodeSnippet objects to ExtractedCodeSnippet objects for state storage.
    
    Args:
        code_snippets: List of CodeSnippet objects from extraction
        
    Returns:
        List of ExtractedCodeSnippet objects for state
    """
    extracted_code = []
    
    for snippet in code_snippets:
        try:
            extracted_snippet = ExtractedCodeSnippet(
                content=snippet.content,
                language=snippet.language.value if hasattr(snippet.language, 'value') else str(snippet.language),
                context=snippet.context or "",
                relevance_score=snippet.relevance_score,
                description=snippet.description,
                purpose=snippet.purpose,
                start_position=getattr(snippet, 'start_position', -1),
                end_position=getattr(snippet, 'end_position', -1)
            )
            extracted_code.append(extracted_snippet)
        except Exception as e:
            logger.warning(f"Failed to convert code snippet to state format: {e}")
            # Create a minimal valid snippet
            extracted_snippet = ExtractedCodeSnippet(
                content=snippet.content if hasattr(snippet, 'content') else "",
                language="other",
                context="",
                relevance_score=0.0
            )
            extracted_code.append(extracted_snippet)
    
    return extracted_code


def convert_external_links_to_state(external_links) -> List[ExtractedExternalLink]:
    """
    Convert ExternalLink objects to ExtractedExternalLink objects for state storage.
    
    Args:
        external_links: List of ExternalLink objects from extraction
        
    Returns:
        List of ExtractedExternalLink objects for state
    """
    extracted_links = []
    
    for link in external_links:
        try:
            extracted_link = ExtractedExternalLink(
                url=link.url,
                link_type=link.link_type.value if hasattr(link.link_type, 'value') else str(link.link_type),
                title=link.title,
                description=link.description,
                context=link.context or "",
                is_accessible=getattr(link, 'is_accessible', None),
                relevance_score=link.relevance_score
            )
            extracted_links.append(extracted_link)
        except Exception as e:
            logger.warning(f"Failed to convert external link to state format: {e}")
            # Create a minimal valid link
            extracted_link = ExtractedExternalLink(
                url=link.url if hasattr(link, 'url') else "",
                link_type="other",
                context="",
                relevance_score=0.0
            )
            extracted_links.append(extracted_link)
    
    return extracted_links


def convert_github_repos_to_state(github_repos) -> List[ExtractedGitHubRepository]:
    """
    Convert GitHubInfo objects to ExtractedGitHubRepository objects for state storage.
    
    Args:
        github_repos: List of GitHubInfo objects from extraction
        
    Returns:
        List of ExtractedGitHubRepository objects for state
    """
    extracted_repos = []
    
    for repo in github_repos:
        try:
            extracted_repo = ExtractedGitHubRepository(
                url=repo.url,
                owner=repo.owner,
                repository=repo.repository,
                path=getattr(repo, 'path', None),
                branch=getattr(repo, 'branch', None),
                is_valid=getattr(repo, 'is_valid', False),
                description=getattr(repo, 'description', None),
                language=getattr(repo, 'language', None),
                stars=getattr(repo, 'stars', None)
            )
            extracted_repos.append(extracted_repo)
        except Exception as e:
            logger.warning(f"Failed to convert GitHub repository to state format: {e}")
            # Create a minimal valid repository
            extracted_repo = ExtractedGitHubRepository(
                url=repo.url if hasattr(repo, 'url') else "",
                owner=repo.owner if hasattr(repo, 'owner') else "",
                repository=repo.repository if hasattr(repo, 'repository') else ""
            )
            extracted_repos.append(extracted_repo)
    
    return extracted_repos


def convert_extraction_result_to_state(extraction_result) -> Dict[str, Any]:
    """
    Convert a complete ExtractionResult to state-compatible data structures.
    
    Args:
        extraction_result: ExtractionResult object from CodeExtractionAgent
        
    Returns:
        Dictionary with converted data structures ready for state update
    """
    try:
        # Convert individual components
        extracted_code = convert_code_snippets_to_state(extraction_result.code_snippets)
        extracted_links = convert_external_links_to_state(extraction_result.external_links)
        extracted_repos = convert_github_repos_to_state(extraction_result.github_repositories)
        
        # Create extraction metadata
        extraction_metadata = ExtractionMetadata(
            total_code_blocks=extraction_result.total_code_blocks,
            total_links_found=extraction_result.total_links_found,
            programming_languages=list(extraction_result.programming_languages),
            processing_time=extraction_result.processing_time,
            extraction_errors=extraction_result.errors or [],
            extraction_timestamp=datetime.now(UTC).isoformat()
        )
        
        return {
            "code_snippets": extracted_code,
            "external_links": extracted_links,
            "github_repos": extracted_repos,
            "metadata": extraction_metadata
        }
        
    except Exception as e:
        logger.error(f"Failed to convert extraction result to state format: {e}")
        # Return minimal valid structure
        return {
            "code_snippets": [],
            "external_links": [],
            "github_repos": [],
            "metadata": ExtractionMetadata(
                extraction_errors=[f"Conversion error: {str(e)}"],
                extraction_timestamp=datetime.now(UTC).isoformat()
            )
        }


def validate_state_data_structure(data: Dict[str, Any]) -> List[str]:
    """
    Validate the structure of converted state data.
    
    Args:
        data: Dictionary with converted extraction data
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required fields
    required_fields = ["code_snippets", "external_links", "github_repos", "metadata"]
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    # Validate code snippets
    if "code_snippets" in data:
        if not isinstance(data["code_snippets"], list):
            errors.append("code_snippets must be a list")
        else:
            for i, snippet in enumerate(data["code_snippets"]):
                if not isinstance(snippet, ExtractedCodeSnippet):
                    errors.append(f"Code snippet {i} is not an ExtractedCodeSnippet instance")
                elif not snippet.content:
                    errors.append(f"Code snippet {i} has empty content")
    
    # Validate external links
    if "external_links" in data:
        if not isinstance(data["external_links"], list):
            errors.append("external_links must be a list")
        else:
            for i, link in enumerate(data["external_links"]):
                if not isinstance(link, ExtractedExternalLink):
                    errors.append(f"External link {i} is not an ExtractedExternalLink instance")
                elif not link.url:
                    errors.append(f"External link {i} has empty URL")
    
    # Validate GitHub repositories
    if "github_repos" in data:
        if not isinstance(data["github_repos"], list):
            errors.append("github_repos must be a list")
        else:
            for i, repo in enumerate(data["github_repos"]):
                if not isinstance(repo, ExtractedGitHubRepository):
                    errors.append(f"GitHub repository {i} is not an ExtractedGitHubRepository instance")
                elif not repo.url:
                    errors.append(f"GitHub repository {i} has empty URL")
    
    # Validate metadata
    if "metadata" in data:
        if not isinstance(data["metadata"], ExtractionMetadata):
            errors.append("metadata is not an ExtractionMetadata instance")
    
    return errors


def convert_state_to_extraction_summary(state) -> Dict[str, Any]:
    """
    Convert state extraction data back to a summary format.
    
    Args:
        state: AnalysisState with extraction data
        
    Returns:
        Dictionary summary of extraction results
    """
    try:
        summary = {
            "total_items": (len(state.extracted_code) + 
                          len(state.extracted_links) + 
                          len(state.extracted_github_repos)),
            "code_snippets": {
                "count": len(state.extracted_code),
                "languages": list(set(snippet.language for snippet in state.extracted_code)),
                "avg_relevance": (sum(snippet.relevance_score for snippet in state.extracted_code) / 
                                len(state.extracted_code)) if state.extracted_code else 0.0
            },
            "external_links": {
                "count": len(state.extracted_links),
                "types": list(set(link.link_type for link in state.extracted_links)),
                "avg_relevance": (sum(link.relevance_score for link in state.extracted_links) / 
                                len(state.extracted_links)) if state.extracted_links else 0.0,
                "accessible_count": sum(1 for link in state.extracted_links if link.is_accessible)
            },
            "github_repositories": {
                "count": len(state.extracted_github_repos),
                "valid_count": sum(1 for repo in state.extracted_github_repos if repo.is_valid),
                "languages": list(set(repo.language for repo in state.extracted_github_repos if repo.language)),
                "total_stars": sum(repo.stars for repo in state.extracted_github_repos if repo.stars)
            },
            "metadata": {
                "processing_time": state.extraction_metadata.processing_time if state.extraction_metadata else 0.0,
                "total_errors": len(state.extraction_metadata.extraction_errors) if state.extraction_metadata else 0,
                "timestamp": state.extraction_metadata.extraction_timestamp if state.extraction_metadata else None
            }
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Failed to convert state to extraction summary: {e}")
        return {
            "total_items": 0,
            "error": str(e)
        } 