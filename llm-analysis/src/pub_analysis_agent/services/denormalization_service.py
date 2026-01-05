"""
Denormalization Service for Elasticsearch.

This module provides functionality to transform nested MongoDB structures
into flat Elasticsearch documents optimized for search and retrieval.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Union
from functools import wraps

from ..models.schema_validator import (
    ConsolidatedAnalysisSchema,
    DatasetMentionSchema,
    CodeSnippetSchema,
    ExternalLinkSchema,
    GitHubRepositorySchema,
    DatasetJoinSchema
)
from ..models.dimensions import Author, Institution
from ..utils.circuit_breaker import circuit_breaker

logger = logging.getLogger(__name__)

# Circuit breaker for denormalization operations
denorm_circuit_breaker = circuit_breaker(
    service_name="denormalization",
    failure_threshold=3,
    recovery_timeout=30,
    expected_exceptions=(ValueError, TypeError, KeyError)
)


class DenormalizationService:
    """
    Service for denormalizing nested MongoDB structures into flat Elasticsearch documents.
    
    This service transforms complex nested objects from the MongoDB llm_analyses collection
    into search-optimized flat documents for Elasticsearch indexing.
    """
    
    def __init__(self):
        """Initialize the denormalization service."""
        self.logger = logging.getLogger(__name__)
    
    def denormalize_analysis_result(self, mongo_doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Denormalize a MongoDB analysis result into a flat Elasticsearch document.
        
        Args:
            mongo_doc: MongoDB document from llm_analyses collection
            
        Returns:
            Flattened document optimized for Elasticsearch search
            
        Raises:
            ValueError: If document structure is invalid
            TypeError: If data types are incompatible
        """
        try:
            # Validate the document structure
            validated_doc = ConsolidatedAnalysisSchema(**mongo_doc)
            
            # Start with base fields
            es_doc = {
                "publication_id": validated_doc.publication_id,
                "workflow_id": validated_doc.workflow_id,
                "analysis_timestamp": validated_doc.analysis_timestamp,
                "workflow_status": validated_doc.workflow_status.value,
                "created_at": validated_doc.analysis_timestamp,
                "updated_at": validated_doc.analysis_timestamp
            }
            
            # Denormalize publication metadata
            es_doc.update(self._denormalize_publication_metadata(validated_doc.publication_metadata))
            
            # Denormalize analysis flags
            es_doc.update(self._denormalize_analysis_flags(validated_doc.analysis_flags))
            
            # Denormalize dataset analysis
            es_doc.update(self._denormalize_dataset_analysis(validated_doc.dataset_analysis))
            
            # Denormalize code extraction
            es_doc.update(self._denormalize_code_extraction(validated_doc.code_extraction))
            
            # Denormalize link extraction
            es_doc.update(self._denormalize_link_extraction(validated_doc.link_extraction))
            
            # Denormalize workflow metadata
            es_doc.update(self._denormalize_workflow_metadata(validated_doc.workflow_metadata))
            
            # Denormalize LLM metadata
            if validated_doc.llm_metadata:
                es_doc.update(self._denormalize_llm_metadata(validated_doc.llm_metadata))
            
            # Denormalize error information
            if validated_doc.error_information:
                es_doc.update(self._denormalize_error_information(validated_doc.error_information))
            
            # Add search-optimized fields
            es_doc.update(self._add_search_fields(es_doc))
            
            return es_doc
            
        except Exception as e:
            self.logger.error(f"Failed to denormalize document {mongo_doc.get('publication_id', 'unknown')}: {e}")
            raise
    
    def _denormalize_publication_metadata(self, metadata) -> Dict[str, Any]:
        """Denormalize publication metadata into flat fields."""
        return {
            "publication_title": metadata.title or "",
            "publication_authors": metadata.authors or [],
            "publication_abstract": metadata.abstract or "",
            "publication_date": metadata.publication_date or "",
            "publication_journal": metadata.journal or "",
            "publication_doi": metadata.doi or "",
            "authors_text": " ".join(metadata.authors) if metadata.authors else "",
            "publication_text": f"{metadata.title or ''} {metadata.abstract or ''}".strip()
        }
    
    def _denormalize_analysis_flags(self, flags) -> Dict[str, Any]:
        """Denormalize analysis flags into flat fields."""
        classification = flags.analysis_classification
        return {
            "is_data_analysis": classification.is_data_analysis or False,
            "has_datasets": classification.has_datasets or False,
            "dataset_count": classification.dataset_count,
            "code_snippets_count": classification.code_snippets_count,
            "external_links_count": classification.external_links_count,
            "github_repos_count": classification.github_repos_count,
            "analysis_flags": {
                "is_data_analysis": classification.is_data_analysis,
                "has_datasets": classification.has_datasets,
                "dataset_count": classification.dataset_count,
                "code_snippets_count": classification.code_snippets_count,
                "external_links_count": classification.external_links_count,
                "github_repos_count": classification.github_repos_count
            }
        }
    
    def _denormalize_dataset_analysis(self, dataset_analysis) -> Dict[str, Any]:
        """Denormalize dataset analysis into flat fields with nested structures for search."""
        # Extract all dataset names for search
        all_dataset_names = []
        all_dataset_contexts = []
        
        # Process validated datasets
        validated_datasets = []
        for dataset in dataset_analysis.validated_datasets:
            validated_datasets.append({
                "name": dataset.name,
                "confidence": dataset.confidence,
                "context": dataset.context or "",
                "section": dataset.section or "",
                "page_number": dataset.page_number
            })
            all_dataset_names.append(dataset.name)
            if dataset.context:
                all_dataset_contexts.append(dataset.context)
        
        # Process newly discovered datasets
        new_datasets = []
        for dataset in dataset_analysis.newly_discovered_datasets:
            new_datasets.append({
                "name": dataset.name,
                "confidence": dataset.confidence,
                "context": dataset.context or "",
                "section": dataset.section or "",
                "page_number": dataset.page_number
            })
            all_dataset_names.append(dataset.name)
            if dataset.context:
                all_dataset_contexts.append(dataset.context)
        
        # Process dataset joins
        dataset_joins = []
        for join in dataset_analysis.dataset_joins:
            dataset_joins.append({
                "dataset1": join.dataset1,
                "dataset2": join.dataset2,
                "join_type": join.join_type,
                "confidence": join.confidence,
                "description": join.description or ""
            })
            all_dataset_names.extend([join.dataset1, join.dataset2])
        
        summary = dataset_analysis.summary
        
        return {
            "validated_datasets": validated_datasets,
            "newly_discovered_datasets": new_datasets,
            "dataset_joins": dataset_joins,
            "dataset_analysis_summary": {
                "total_validated_datasets": summary.total_validated_datasets,
                "total_new_datasets": summary.total_new_datasets,
                "total_dataset_joins": summary.total_dataset_joins,
                "total_unique_datasets": summary.total_unique_datasets
            },
            "all_dataset_names": list(set(all_dataset_names)),  # Remove duplicates
            "dataset_names_text": " ".join(set(all_dataset_names)),
            "dataset_contexts_text": " ".join(all_dataset_contexts),
            "total_datasets": len(set(all_dataset_names))
        }
    
    def _denormalize_code_extraction(self, code_extraction) -> Dict[str, Any]:
        """Denormalize code extraction into flat fields."""
        # Extract all programming languages
        all_languages = set()
        all_code_content = []
        
        code_snippets = []
        for snippet in code_extraction.extracted_code_snippets:
            code_snippets.append({
                "content": snippet.content,
                "language": snippet.language,
                "context": snippet.context,
                "relevance_score": snippet.relevance_score,
                "description": snippet.description or "",
                "purpose": snippet.purpose or "",
                "start_position": snippet.start_position,
                "end_position": snippet.end_position
            })
            all_languages.add(snippet.language)
            all_code_content.append(snippet.content)
        
        summary = code_extraction.summary
        metadata = code_extraction.extraction_metadata
        
        return {
            "code_snippets": code_snippets,
            "code_extraction_summary": {
                "total_code_snippets": summary.total_code_snippets,
                "programming_languages": summary.programming_languages,
                "average_relevance_score": summary.average_relevance_score
            },
            "code_extraction_metadata": {
                "total_code_blocks": metadata.total_code_blocks if metadata else 0,
                "total_links_found": metadata.total_links_found if metadata else 0,
                "programming_languages": metadata.programming_languages if metadata else [],
                "processing_time": metadata.processing_time if metadata else 0.0,
                "extraction_errors": metadata.extraction_errors if metadata else [],
                "extraction_timestamp": metadata.extraction_timestamp if metadata else None
            },
            "all_programming_languages": list(all_languages),
            "programming_languages_text": " ".join(all_languages),
            "code_content_text": " ".join(all_code_content),
            "total_code_snippets": len(code_snippets)
        }
    
    def _denormalize_link_extraction(self, link_extraction) -> Dict[str, Any]:
        """Denormalize link extraction into flat fields."""
        # Extract all URLs and domains
        all_urls = []
        all_domains = set()
        all_link_types = set()
        
        external_links = []
        for link in link_extraction.external_links:
            external_links.append({
                "url": link.url,
                "link_type": link.link_type,
                "title": link.title or "",
                "description": link.description or "",
                "context": link.context,
                "is_accessible": link.is_accessible,
                "relevance_score": link.relevance_score
            })
            all_urls.append(link.url)
            all_link_types.add(link.link_type)
            # Extract domain from URL
            try:
                from urllib.parse import urlparse
                domain = urlparse(link.url).netloc
                if domain:
                    all_domains.add(domain)
            except:
                pass
        
        github_repos = []
        for repo in link_extraction.github_repositories:
            github_repos.append({
                "url": repo.url,
                "owner": repo.owner,
                "repository": repo.repository,
                "path": repo.path or "",
                "branch": repo.branch or "",
                "is_valid": repo.is_valid,
                "description": repo.description or "",
                "language": repo.language or "",
                "stars": repo.stars
            })
            all_urls.append(repo.url)
            all_link_types.add("github")
        
        summary = link_extraction.summary
        
        return {
            "external_links": external_links,
            "github_repositories": github_repos,
            "link_extraction_summary": {
                "total_external_links": summary.total_external_links,
                "total_github_repos": summary.total_github_repos,
                "accessible_links": summary.accessible_links,
                "valid_github_repos": summary.valid_github_repos,
                "average_link_relevance": summary.average_link_relevance
            },
            "all_urls": all_urls,
            "all_domains": list(all_domains),
            "all_link_types": list(all_link_types),
            "urls_text": " ".join(all_urls),
            "domains_text": " ".join(all_domains),
            "link_types_text": " ".join(all_link_types),
            "total_links": len(all_urls)
        }
    
    def _denormalize_workflow_metadata(self, metadata) -> Dict[str, Any]:
        """Denormalize workflow metadata into flat fields."""
        return {
            "workflow_current_step": metadata.current_step or "",
            "workflow_completed_steps": metadata.completed_steps,
            "workflow_duration": metadata.workflow_duration,
            "workflow_step_count": metadata.step_count,
            "workflow_total_steps": metadata.total_steps,
            "workflow_completion_percentage": metadata.completion_percentage,
            "completed_steps_text": " ".join(metadata.completed_steps)
        }
    
    def _denormalize_llm_metadata(self, metadata) -> Dict[str, Any]:
        """Denormalize LLM metadata into flat fields."""
        return {
            "llm_model_name": metadata.model_name,
            "llm_model_version": metadata.model_version or "",
            "llm_tokens_used": metadata.tokens_used,
            "llm_response_time": metadata.response_time,
            "llm_temperature": metadata.temperature,
            "llm_max_tokens": metadata.max_tokens
        }
    
    def _denormalize_error_information(self, error_info) -> Dict[str, Any]:
        """Denormalize error information into flat fields."""
        return {
            "error_message": error_info.error_message,
            "error_timestamp": error_info.error_timestamp,
            "error_current_step": error_info.current_step_at_error or "",
            "has_errors": True
        }
    
    def _add_search_fields(self, es_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Add search-optimized fields to the document."""
        # Create a comprehensive search text field
        search_parts = []
        
        # Add publication information
        if es_doc.get("publication_title"):
            search_parts.append(es_doc["publication_title"])
        if es_doc.get("publication_abstract"):
            search_parts.append(es_doc["publication_abstract"])
        if es_doc.get("authors_text"):
            search_parts.append(es_doc["authors_text"])
        
        # Add dataset information
        if es_doc.get("dataset_names_text"):
            search_parts.append(es_doc["dataset_names_text"])
        if es_doc.get("dataset_contexts_text"):
            search_parts.append(es_doc["dataset_contexts_text"])
        
        # Add code information
        if es_doc.get("programming_languages_text"):
            search_parts.append(es_doc["programming_languages_text"])
        if es_doc.get("code_content_text"):
            search_parts.append(es_doc["code_content_text"])
        
        # Add link information
        if es_doc.get("urls_text"):
            search_parts.append(es_doc["urls_text"])
        if es_doc.get("domains_text"):
            search_parts.append(es_doc["domains_text"])
        
        # Create autocomplete fields
        autocomplete_parts = []
        if es_doc.get("publication_title"):
            autocomplete_parts.append(es_doc["publication_title"])
        if es_doc.get("all_dataset_names"):
            autocomplete_parts.extend(es_doc["all_dataset_names"][:10])  # Limit for performance
        if es_doc.get("all_programming_languages"):
            autocomplete_parts.extend(es_doc["all_programming_languages"])
        
        return {
            "search_text": " ".join(search_parts),
            "autocomplete_text": " ".join(autocomplete_parts),
            "content_length": len(" ".join(search_parts)),
            "has_content": len(search_parts) > 0
        }
    
    def batch_denormalize(self, mongo_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Denormalize a batch of MongoDB documents.
        
        Args:
            mongo_docs: List of MongoDB documents
            
        Returns:
            List of denormalized Elasticsearch documents
        """
        denormalized_docs = []
        errors = []
        
        for i, doc in enumerate(mongo_docs):
            try:
                denormalized_doc = self.denormalize_analysis_result(doc)
                denormalized_docs.append(denormalized_doc)
            except Exception as e:
                errors.append({
                    "index": i,
                    "publication_id": doc.get("publication_id", "unknown"),
                    "error": str(e)
                })
                self.logger.warning(f"Failed to denormalize document {i}: {e}")
        
        if errors:
            self.logger.warning(f"Denormalization completed with {len(errors)} errors out of {len(mongo_docs)} documents")
        
        return denormalized_docs
    
    def validate_denormalized_document(self, es_doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a denormalized document structure.
        
        Args:
            es_doc: Denormalized Elasticsearch document
            
        Returns:
            Validation result with success status and any issues
        """
        issues = []
        
        # Check required fields
        required_fields = ["publication_id", "workflow_status", "created_at"]
        for field in required_fields:
            if field not in es_doc or not es_doc[field]:
                issues.append(f"Missing required field: {field}")
        
        # Check data types
        if "dataset_count" in es_doc and not isinstance(es_doc["dataset_count"], int):
            issues.append("dataset_count must be an integer")
        
        if "confidence" in es_doc and not isinstance(es_doc["confidence"], (int, float)):
            issues.append("confidence must be a number")
        
        # Check array fields
        array_fields = ["publication_authors", "all_dataset_names", "all_programming_languages"]
        for field in array_fields:
            if field in es_doc and not isinstance(es_doc[field], list):
                issues.append(f"{field} must be a list")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "document_size": len(str(es_doc))
        } 