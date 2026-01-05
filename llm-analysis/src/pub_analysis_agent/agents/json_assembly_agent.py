"""
JSON Assembly Agent for consolidating LangGraph workflow state.

This module provides the JSONAssemblyAgent class for consolidating all LangGraph
workflow state data into structured JSON format following the llm_analyses schema.
"""

import logging
import asyncio
from datetime import datetime, UTC
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from ..workflows.state_models import AnalysisState
from ..models.analysis_result import AnalysisResult, AnalysisType, AnalysisStatus, LLMMetadata
from ..models.schema_validator import SchemaValidator
from ..services.results_service import ResultsService
from ..services.data_quality_validator import DataQualityValidator, QualityMetrics


logger = logging.getLogger(__name__)


class JSONAssemblyAgent:
    """
    Agent for consolidating LangGraph workflow state into structured JSON format.
    
    This agent takes the complete AnalysisState from the LangGraph workflow and
    consolidates it into a structured JSON format that follows the llm_analyses
    collection schema for MongoDB storage.
    """
    
    def __init__(self, results_service: Optional[ResultsService] = None, dataset_service=None):
        """
        Initialize the JSON Assembly Agent.
        
        Args:
            results_service: Optional ResultsService for storing analysis results
            dataset_service: Optional DatasetService for dataset reference validation
        """
        self.results_service = results_service
        self.schema_validator = SchemaValidator()
        self.quality_validator = DataQualityValidator(dataset_service)
        logger.info("JSONAssemblyAgent initialized")
    
    async def consolidate_state_to_json(
        self, 
        state: AnalysisState,
        llm_metadata: Optional[LLMMetadata] = None
    ) -> Dict[str, Any]:
        """
        Consolidate the complete AnalysisState into structured JSON format.
        
        Args:
            state: Complete AnalysisState from LangGraph workflow
            llm_metadata: Optional LLM metadata for the analysis
            
        Returns:
            Consolidated JSON following llm_analyses schema
            
        Raises:
            Exception: If consolidation fails
        """
        try:
            # Determine workflow status
            workflow_status = self._determine_workflow_status(state)
            
            # Consolidate all state data into structured JSON
            consolidated_json = {
                "publication_id": state.publication_id,
                "workflow_id": state.workflow_id,
                "analysis_timestamp": datetime.now(UTC).isoformat(),
                "workflow_status": workflow_status,
                "publication_metadata": self._extract_publication_metadata(state),
                "analysis_flags": self._extract_analysis_flags(state),
                "dataset_analysis": self._extract_dataset_analysis(state),
                "code_extraction": self._extract_code_analysis(state),
                "link_extraction": self._extract_link_analysis(state),
                "workflow_metadata": self._extract_workflow_metadata(state),
                "llm_metadata": self._extract_llm_metadata(llm_metadata),
                "error_information": self._extract_error_information(state)
            }
            
            # Validate the consolidated JSON against the schema
            validation_result = self.schema_validator.validate_consolidated_json(consolidated_json)
            
            if not validation_result["success"]:
                logger.error(f"Schema validation failed: {validation_result['errors']}")
                # Still return the data but log the validation errors
                state.error_message = f"Schema validation failed: {validation_result['errors']}"
            else:
                logger.info("Schema validation successful")
                # Update state with the validated JSON
                state.final_json = validation_result["validated_data"]
                state.completed_steps.append("json_assembly")
            
            # Perform data quality validation
            quality_metrics = await self.quality_validator.validate_analysis_quality(consolidated_json)
            
            # Add quality metrics to the consolidated JSON
            consolidated_json["quality_metrics"] = self.quality_validator.get_quality_report(quality_metrics)
            
            # Log quality validation results
            logger.info(f"Quality validation completed. Overall quality: {quality_metrics.overall_quality_level.value}")
            if quality_metrics.critical_issues > 0:
                logger.warning(f"Found {quality_metrics.critical_issues} critical quality issues")
            
            # Store analysis result if results service is available
            if self.results_service:
                await self._store_analysis_result(state, consolidated_json, llm_metadata)
            
            return consolidated_json
            
        except Exception as e:
            logger.error(f"Failed to consolidate state: {e}", exc_info=True)
            state.error_message = f"Failed to consolidate state: {e}"
            raise
    
    def _determine_workflow_status(self, state: AnalysisState) -> str:
        """Determine the overall workflow status."""
        if state.error_message:
            return "failed"
        elif state.final_json:
            return "completed"
        elif state.completed_steps:
            return "in_progress"
        else:
            return "pending"
    
    def _extract_llm_metadata(self, llm_metadata: Optional[LLMMetadata]) -> Optional[Dict[str, Any]]:
        """Extract LLM metadata for the analysis."""
        if not llm_metadata:
            return None
        
        return {
            "model_name": llm_metadata.model_name,
            "model_version": llm_metadata.model_version,
            "tokens_used": llm_metadata.tokens_used,
            "response_time": llm_metadata.response_time,
            "temperature": llm_metadata.temperature,
            "max_tokens": llm_metadata.max_tokens
        }
    
    def _extract_publication_metadata(self, state: AnalysisState) -> Dict[str, Any]:
        """Extract publication metadata from GROBID content."""
        metadata = {
            "publication_id": state.publication_id,
            "created_at": state.created_at.isoformat(),
            "updated_at": state.updated_at.isoformat()
        }
        
        if state.grobid_content:
            # Extract basic publication information from GROBID content
            grobid_data = state.grobid_content
            
            # Extract title
            if "title" in grobid_data:
                metadata["title"] = grobid_data["title"]
            
            # Extract authors
            if "authors" in grobid_data:
                metadata["authors"] = grobid_data["authors"]
            
            # Extract abstract
            if "abstract" in grobid_data:
                metadata["abstract"] = grobid_data["abstract"]
            
            # Extract publication date
            if "publication_date" in grobid_data:
                metadata["publication_date"] = grobid_data["publication_date"]
            
            # Extract journal/conference information
            if "journal" in grobid_data:
                metadata["journal"] = grobid_data["journal"]
            
            # Extract DOI if available
            if "doi" in grobid_data:
                metadata["doi"] = grobid_data["doi"]
        
        return metadata
    
    def _extract_analysis_flags(self, state: AnalysisState) -> Dict[str, Any]:
        """Extract analysis flags and classification results."""
        return {
            "is_data_analysis": state.is_data_analysis,
            "has_datasets": state.has_datasets,
            "analysis_classification": {
                "is_data_analysis": state.is_data_analysis,
                "has_datasets": state.has_datasets,
                "dataset_count": len(state.get_all_datasets()),
                "code_snippets_count": len(state.extracted_code),
                "external_links_count": len(state.extracted_links),
                "github_repos_count": len(state.extracted_github_repos)
            }
        }
    
    def _extract_dataset_analysis(self, state: AnalysisState) -> Dict[str, Any]:
        """Extract dataset analysis results."""
        return {
            "validated_datasets": [
                {
                    "name": dataset.name,
                    "confidence": dataset.confidence,
                    "context": dataset.context,
                    "section": dataset.section,
                    "page_number": dataset.page_number
                }
                for dataset in state.validated_datasets
            ],
            "newly_discovered_datasets": [
                {
                    "name": dataset.name,
                    "confidence": dataset.confidence,
                    "context": dataset.context,
                    "section": dataset.section,
                    "page_number": dataset.page_number
                }
                for dataset in state.newly_discovered_datasets
            ],
            "dataset_joins": [
                {
                    "dataset1": join.dataset1,
                    "dataset2": join.dataset2,
                    "join_type": join.join_type,
                    "confidence": join.confidence,
                    "description": join.description
                }
                for join in state.dataset_joins
            ],
            "summary": {
                "total_validated_datasets": len(state.validated_datasets),
                "total_new_datasets": len(state.newly_discovered_datasets),
                "total_dataset_joins": len(state.dataset_joins),
                "total_unique_datasets": len(set(
                    [d.name for d in state.validated_datasets] +
                    [d.name for d in state.newly_discovered_datasets]
                ))
            }
        }
    
    def _extract_code_analysis(self, state: AnalysisState) -> Dict[str, Any]:
        """Extract code analysis results."""
        return {
            "extracted_code_snippets": [
                {
                    "content": snippet.content,
                    "language": snippet.language,
                    "context": snippet.context,
                    "relevance_score": snippet.relevance_score,
                    "description": snippet.description,
                    "purpose": snippet.purpose,
                    "start_position": snippet.start_position,
                    "end_position": snippet.end_position
                }
                for snippet in state.extracted_code
            ],
            "extraction_metadata": asdict(state.extraction_metadata) if state.extraction_metadata else None,
            "summary": {
                "total_code_snippets": len(state.extracted_code),
                "programming_languages": list(set([s.language for s in state.extracted_code if s.language])),
                "average_relevance_score": sum(s.relevance_score for s in state.extracted_code) / len(state.extracted_code) if state.extracted_code else 0
            }
        }
    
    def _extract_link_analysis(self, state: AnalysisState) -> Dict[str, Any]:
        """Extract link analysis results."""
        return {
            "external_links": [
                {
                    "url": link.url,
                    "link_type": link.link_type,
                    "title": link.title,
                    "description": link.description,
                    "context": link.context,
                    "is_accessible": link.is_accessible,
                    "relevance_score": link.relevance_score
                }
                for link in state.extracted_links
            ],
            "github_repositories": [
                {
                    "url": repo.url,
                    "owner": repo.owner,
                    "repository": repo.repository,
                    "path": repo.path,
                    "branch": repo.branch,
                    "is_valid": repo.is_valid,
                    "description": repo.description,
                    "language": repo.language,
                    "stars": repo.stars
                }
                for repo in state.extracted_github_repos
            ],
            "summary": {
                "total_external_links": len(state.extracted_links),
                "total_github_repos": len(state.extracted_github_repos),
                "accessible_links": len([l for l in state.extracted_links if l.is_accessible]),
                "valid_github_repos": len([r for r in state.extracted_github_repos if r.is_valid]),
                "average_link_relevance": sum(l.relevance_score for l in state.extracted_links) / len(state.extracted_links) if state.extracted_links else 0
            }
        }
    
    def _extract_workflow_metadata(self, state: AnalysisState) -> Dict[str, Any]:
        """Extract workflow execution metadata."""
        return {
            "current_step": state.current_step,
            "completed_steps": state.completed_steps,
            "workflow_duration": (state.updated_at - state.created_at).total_seconds(),
            "step_count": len(state.completed_steps),
            "total_steps": 7,  # Total expected steps in workflow
            "completion_percentage": (len(state.completed_steps) / 7) * 100 if state.completed_steps else 0
        }
    
    def _extract_error_information(self, state: AnalysisState) -> Optional[Dict[str, Any]]:
        """Extract error information if any."""
        if not state.error_message:
            return None
        
        return {
            "error_message": state.error_message,
            "error_timestamp": state.updated_at.isoformat(),
            "current_step_at_error": state.current_step
        }
    
    async def _store_analysis_result(
        self, 
        state: AnalysisState, 
        consolidated_json: Dict[str, Any],
        llm_metadata: Optional[LLMMetadata]
    ) -> None:
        """Store the analysis result in the results service."""
        try:
            # Create AnalysisResult object
            analysis_result = AnalysisResult(
                publication_id=state.publication_id,
                analysis_type=AnalysisType.FULL_ANALYSIS,
                analysis_text=str(consolidated_json),
                metadata=consolidated_json,
                confidence_score=self._calculate_overall_confidence(state),
                created_at=state.created_at,
                updated_at=state.updated_at
            )
            
            # Store in results service
            await self.results_service.store_analysis_result(analysis_result)
            logger.info(f"Analysis result stored for publication {state.publication_id}")
            
        except Exception as e:
            logger.error(f"Failed to store analysis result: {e}", exc_info=True)
            # Don't raise the error as this is not critical for the main workflow
    
    def _calculate_overall_confidence(self, state: AnalysisState) -> float:
        """Calculate overall confidence score based on all analysis results."""
        confidence_scores = []
        
        # Dataset confidence scores
        for dataset in state.get_all_datasets():
            confidence_scores.append(dataset.confidence)
        
        # Dataset join confidence scores
        for join in state.dataset_joins:
            confidence_scores.append(join.confidence)
        
        # Code snippet relevance scores (normalize to 0-1)
        for snippet in state.extracted_code:
            confidence_scores.append(snippet.relevance_score / 10.0)
        
        # Link relevance scores (normalize to 0-1)
        for link in state.extracted_links:
            confidence_scores.append(link.relevance_score / 10.0)
        
        # Calculate average confidence
        if confidence_scores:
            return sum(confidence_scores) / len(confidence_scores)
        else:
            return 0.0
    
    async def validate_consolidated_json(self, consolidated_json: Dict[str, Any]) -> List[str]:
        """
        Validate the consolidated JSON structure.
        
        Args:
            consolidated_json: The consolidated JSON to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required fields
        required_fields = ["publication_id", "analysis_timestamp", "workflow_status"]
        for field in required_fields:
            if field not in consolidated_json:
                errors.append(f"Missing required field: {field}")
        
        # Validate publication_id
        if "publication_id" in consolidated_json and not consolidated_json["publication_id"]:
            errors.append("Publication ID cannot be empty")
        
        # Validate workflow_status
        valid_statuses = ["pending", "in_progress", "completed", "failed"]
        if "workflow_status" in consolidated_json:
            if consolidated_json["workflow_status"] not in valid_statuses:
                errors.append(f"Invalid workflow status: {consolidated_json['workflow_status']}")
        
        # Validate dataset analysis structure
        if "dataset_analysis" in consolidated_json:
            dataset_analysis = consolidated_json["dataset_analysis"]
            if not isinstance(dataset_analysis, dict):
                errors.append("Dataset analysis must be a dictionary")
            else:
                required_dataset_fields = ["validated_datasets", "newly_discovered_datasets", "dataset_joins"]
                for field in required_dataset_fields:
                    if field not in dataset_analysis:
                        errors.append(f"Missing required dataset analysis field: {field}")
        
        return errors
    
    def get_consolidation_summary(self, state: AnalysisState) -> Dict[str, Any]:
        """
        Get a summary of the consolidation process.
        
        Args:
            state: AnalysisState to summarize
            
        Returns:
            Summary dictionary
        """
        return {
            "publication_id": state.publication_id,
            "workflow_id": state.workflow_id,
            "total_datasets": len(state.get_all_datasets()),
            "total_code_snippets": len(state.extracted_code),
            "total_links": len(state.extracted_links) + len(state.extracted_github_repos),
            "workflow_completed": state.is_complete(),
            "has_errors": state.has_error(),
            "completed_steps": len(state.completed_steps),
            "processing_time_seconds": (state.updated_at - state.created_at).total_seconds()
        } 