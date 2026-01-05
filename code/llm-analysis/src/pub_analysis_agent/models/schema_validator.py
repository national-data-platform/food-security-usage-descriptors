"""
Schema validation for consolidated JSON against MongoDB requirements.

This module provides comprehensive schema validation to ensure assembled JSON
complies with MongoDB llm_analyses collection requirements.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Valid workflow status values."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class PublicationMetadata(BaseModel):
    """Schema for publication metadata."""
    
    publication_id: str = Field(..., description="Publication identifier")
    created_at: str = Field(..., description="Creation timestamp in ISO format")
    updated_at: str = Field(..., description="Last update timestamp in ISO format")
    title: Optional[str] = Field(None, description="Publication title")
    authors: Optional[List[str]] = Field(None, description="List of authors")
    abstract: Optional[str] = Field(None, description="Publication abstract")
    publication_date: Optional[str] = Field(None, description="Publication date")
    journal: Optional[str] = Field(None, description="Journal or conference name")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    
    @field_validator('created_at', 'updated_at')
    @classmethod
    def validate_iso_timestamp(cls, v):
        """Validate ISO timestamp format."""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError(f"Invalid ISO timestamp format: {v}")


class AnalysisClassification(BaseModel):
    """Schema for analysis classification."""
    
    is_data_analysis: Optional[bool] = Field(None, description="Whether publication is data analysis")
    has_datasets: Optional[bool] = Field(None, description="Whether publication mentions datasets")
    dataset_count: int = Field(0, description="Number of datasets found")
    code_snippets_count: int = Field(0, description="Number of code snippets extracted")
    external_links_count: int = Field(0, description="Number of external links found")
    github_repos_count: int = Field(0, description="Number of GitHub repositories found")


class AnalysisFlags(BaseModel):
    """Schema for analysis flags."""
    
    is_data_analysis: Optional[bool] = Field(None, description="Whether publication is data analysis")
    has_datasets: Optional[bool] = Field(None, description="Whether publication mentions datasets")
    analysis_classification: AnalysisClassification = Field(..., description="Detailed classification")


class DatasetMentionSchema(BaseModel):
    """Schema for dataset mention."""
    
    name: str = Field(..., description="Dataset name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    context: Optional[str] = Field(None, description="Context around the mention")
    section: Optional[str] = Field(None, description="Section where mention was found")
    page_number: Optional[int] = Field(None, ge=1, description="Page number if available")


class DatasetJoinSchema(BaseModel):
    """Schema for dataset join."""
    
    dataset1: str = Field(..., description="First dataset name")
    dataset2: str = Field(..., description="Second dataset name")
    join_type: str = Field(..., description="Type of join")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    description: Optional[str] = Field(None, description="Join description")


class DatasetAnalysisSummary(BaseModel):
    """Schema for dataset analysis summary."""
    
    total_validated_datasets: int = Field(0, ge=0, description="Total validated datasets")
    total_new_datasets: int = Field(0, ge=0, description="Total newly discovered datasets")
    total_dataset_joins: int = Field(0, ge=0, description="Total dataset joins")
    total_unique_datasets: int = Field(0, ge=0, description="Total unique datasets")


class DatasetAnalysis(BaseModel):
    """Schema for dataset analysis results."""
    
    validated_datasets: List[DatasetMentionSchema] = Field(default_factory=list, description="Validated datasets")
    newly_discovered_datasets: List[DatasetMentionSchema] = Field(default_factory=list, description="Newly discovered datasets")
    dataset_joins: List[DatasetJoinSchema] = Field(default_factory=list, description="Dataset joins")
    summary: DatasetAnalysisSummary = Field(..., description="Analysis summary")


class CodeSnippetSchema(BaseModel):
    """Schema for extracted code snippet."""
    
    content: str = Field(..., description="Code content")
    language: str = Field(..., description="Programming language")
    context: str = Field(..., description="Context around the code")
    relevance_score: float = Field(..., ge=0.0, le=10.0, description="Relevance score")
    description: Optional[str] = Field(None, description="Code description")
    purpose: Optional[str] = Field(None, description="Code purpose")
    start_position: int = Field(-1, description="Start position in document")
    end_position: int = Field(-1, description="End position in document")


class ExtractionMetadataSchema(BaseModel):
    """Schema for extraction metadata."""
    
    total_code_blocks: int = Field(0, ge=0, description="Total code blocks found")
    total_links_found: int = Field(0, ge=0, description="Total links found")
    programming_languages: List[str] = Field(default_factory=list, description="Programming languages found")
    processing_time: float = Field(0.0, ge=0.0, description="Processing time in seconds")
    extraction_errors: List[str] = Field(default_factory=list, description="Extraction errors")
    extraction_timestamp: Optional[str] = Field(None, description="Extraction timestamp")


class CodeAnalysisSummary(BaseModel):
    """Schema for code analysis summary."""
    
    total_code_snippets: int = Field(0, ge=0, description="Total code snippets")
    programming_languages: List[str] = Field(default_factory=list, description="Programming languages")
    average_relevance_score: float = Field(0.0, ge=0.0, le=10.0, description="Average relevance score")


class CodeExtraction(BaseModel):
    """Schema for code extraction results."""
    
    extracted_code_snippets: List[CodeSnippetSchema] = Field(default_factory=list, description="Extracted code snippets")
    extraction_metadata: Optional[ExtractionMetadataSchema] = Field(None, description="Extraction metadata")
    summary: CodeAnalysisSummary = Field(..., description="Code analysis summary")


class ExternalLinkSchema(BaseModel):
    """Schema for external link."""
    
    url: str = Field(..., description="Link URL")
    link_type: str = Field(..., description="Type of link")
    title: Optional[str] = Field(None, description="Link title")
    description: Optional[str] = Field(None, description="Link description")
    context: str = Field("", description="Context around the link")
    is_accessible: Optional[bool] = Field(None, description="Whether link is accessible")
    relevance_score: float = Field(0.0, ge=0.0, le=10.0, description="Relevance score")


class GitHubRepositorySchema(BaseModel):
    """Schema for GitHub repository."""
    
    url: str = Field(..., description="Repository URL")
    owner: str = Field(..., description="Repository owner")
    repository: str = Field(..., description="Repository name")
    path: Optional[str] = Field(None, description="Repository path")
    branch: Optional[str] = Field(None, description="Repository branch")
    is_valid: bool = Field(False, description="Whether repository is valid")
    description: Optional[str] = Field(None, description="Repository description")
    language: Optional[str] = Field(None, description="Primary language")
    stars: Optional[int] = Field(None, ge=0, description="Number of stars")


class LinkAnalysisSummary(BaseModel):
    """Schema for link analysis summary."""
    
    total_external_links: int = Field(0, ge=0, description="Total external links")
    total_github_repos: int = Field(0, ge=0, description="Total GitHub repositories")
    accessible_links: int = Field(0, ge=0, description="Number of accessible links")
    valid_github_repos: int = Field(0, ge=0, description="Number of valid GitHub repositories")
    average_link_relevance: float = Field(0.0, ge=0.0, le=10.0, description="Average link relevance")


class LinkExtraction(BaseModel):
    """Schema for link extraction results."""
    
    external_links: List[ExternalLinkSchema] = Field(default_factory=list, description="External links")
    github_repositories: List[GitHubRepositorySchema] = Field(default_factory=list, description="GitHub repositories")
    summary: LinkAnalysisSummary = Field(..., description="Link analysis summary")


class WorkflowMetadata(BaseModel):
    """Schema for workflow metadata."""
    
    current_step: Optional[str] = Field(None, description="Current workflow step")
    completed_steps: List[str] = Field(default_factory=list, description="Completed workflow steps")
    workflow_duration: float = Field(0.0, ge=0.0, description="Workflow duration in seconds")
    step_count: int = Field(0, ge=0, description="Number of completed steps")
    total_steps: int = Field(7, description="Total expected steps")
    completion_percentage: float = Field(0.0, ge=0.0, le=100.0, description="Completion percentage")


class LLMMetadataSchema(BaseModel):
    """Schema for LLM metadata."""
    
    model_name: str = Field(..., description="LLM model name")
    model_version: Optional[str] = Field(None, description="Model version")
    tokens_used: Optional[int] = Field(None, ge=0, description="Number of tokens used")
    response_time: Optional[float] = Field(None, ge=0.0, description="Response time in seconds")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperature setting")
    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum tokens setting")


class ErrorInformation(BaseModel):
    """Schema for error information."""
    
    error_message: str = Field(..., description="Error message")
    error_timestamp: str = Field(..., description="Error timestamp in ISO format")
    current_step_at_error: Optional[str] = Field(None, description="Step where error occurred")
    
    @field_validator('error_timestamp')
    @classmethod
    def validate_iso_timestamp(cls, v):
        """Validate ISO timestamp format."""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError(f"Invalid ISO timestamp format: {v}")


class ConsolidatedAnalysisSchema(BaseModel):
    """
    Schema for consolidated analysis JSON.
    
    This schema validates the complete JSON structure that will be stored
    in the MongoDB llm_analyses collection.
    """
    
    publication_id: str = Field(..., description="Publication identifier")
    workflow_id: Optional[str] = Field(None, description="Workflow identifier")
    analysis_timestamp: str = Field(..., description="Analysis timestamp in ISO format")
    workflow_status: WorkflowStatus = Field(..., description="Workflow status")
    publication_metadata: PublicationMetadata = Field(..., description="Publication metadata")
    analysis_flags: AnalysisFlags = Field(..., description="Analysis flags")
    dataset_analysis: DatasetAnalysis = Field(..., description="Dataset analysis results")
    code_extraction: CodeExtraction = Field(..., description="Code extraction results")
    link_extraction: LinkExtraction = Field(..., description="Link extraction results")
    workflow_metadata: WorkflowMetadata = Field(..., description="Workflow metadata")
    llm_metadata: Optional[LLMMetadataSchema] = Field(None, description="LLM metadata")
    error_information: Optional[ErrorInformation] = Field(None, description="Error information if any")
    
    @field_validator('analysis_timestamp')
    @classmethod
    def validate_iso_timestamp(cls, v):
        """Validate ISO timestamp format."""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError(f"Invalid ISO timestamp format: {v}")
    
    @model_validator(mode='after')
    def validate_consistency(self):
        """Validate consistency between different parts of the schema."""
        # Check that error_information is present if workflow_status is failed
        if self.workflow_status == WorkflowStatus.FAILED:
            if not self.error_information:
                raise ValueError("Error information must be provided when workflow status is failed")
        
        # Check that completion percentage is consistent with step count
        expected_percentage = (self.workflow_metadata.step_count / self.workflow_metadata.total_steps) * 100
        if abs(self.workflow_metadata.completion_percentage - expected_percentage) > 1.0:
            logger.warning(f"Completion percentage {self.workflow_metadata.completion_percentage} "
                         f"doesn't match expected {expected_percentage} based on step count")
        
        return self
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Reject any extra fields
        validate_assignment = True


class SchemaValidator:
    """
    Validator for consolidated JSON against MongoDB requirements.
    
    This class provides comprehensive schema validation to ensure assembled JSON
    complies with MongoDB llm_analyses collection requirements.
    """
    
    def __init__(self, schema_version: str = "1.0"):
        """
        Initialize the schema validator.
        
        Args:
            schema_version: Version of the schema being validated
        """
        self.schema_version = schema_version
        logger.info(f"SchemaValidator initialized with version {schema_version}")
    
    def validate_consolidated_json(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate consolidated JSON against the schema.
        
        Args:
            json_data: JSON data to validate
            
        Returns:
            Validation result with success status and any errors
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            # Validate against the schema
            validated_data = ConsolidatedAnalysisSchema(**json_data)
            
            # Convert back to dict for consistency
            result = validated_data.model_dump()
            
            logger.info(f"Schema validation successful for publication {json_data.get('publication_id', 'unknown')}")
            return {
                "success": True,
                "validated_data": result,
                "errors": [],
                "warnings": []
            }
            
        except ValidationError as e:
            errors = []
            for error in e.errors():
                field_path = " -> ".join(str(loc) for loc in error["loc"])
                errors.append({
                    "field": field_path,
                    "message": error["msg"],
                    "type": error["type"],
                    "value": error.get("input")
                })
            
            logger.error(f"Schema validation failed with {len(errors)} errors")
            return {
                "success": False,
                "validated_data": None,
                "errors": errors,
                "warnings": []
            }
    
    def validate_field_types(self, json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validate field types in the JSON data.
        
        Args:
            json_data: JSON data to validate
            
        Returns:
            List of type validation errors
        """
        errors = []
        
        # Check required string fields
        string_fields = ["publication_id", "analysis_timestamp"]
        for field in string_fields:
            if field in json_data and not isinstance(json_data[field], str):
                errors.append({
                    "field": field,
                    "message": f"Field must be a string, got {type(json_data[field]).__name__}",
                    "type": "type_error",
                    "value": json_data[field]
                })
        
        # Check required dict fields
        dict_fields = ["publication_metadata", "analysis_flags", "dataset_analysis", 
                      "code_extraction", "link_extraction", "workflow_metadata"]
        for field in dict_fields:
            if field in json_data and not isinstance(json_data[field], dict):
                errors.append({
                    "field": field,
                    "message": f"Field must be a dictionary, got {type(json_data[field]).__name__}",
                    "type": "type_error",
                    "value": json_data[field]
                })
        
        return errors
    
    def validate_required_fields(self, json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validate that all required fields are present.
        
        Args:
            json_data: JSON data to validate
            
        Returns:
            List of missing field errors
        """
        errors = []
        
        required_fields = [
            "publication_id", "analysis_timestamp", "workflow_status",
            "publication_metadata", "analysis_flags", "dataset_analysis",
            "code_extraction", "link_extraction", "workflow_metadata"
        ]
        
        for field in required_fields:
            if field not in json_data:
                errors.append({
                    "field": field,
                    "message": f"Required field '{field}' is missing",
                    "type": "missing_field",
                    "value": None
                })
        
        return errors
    
    def validate_data_formats(self, json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validate data formats (timestamps, URLs, etc.).
        
        Args:
            json_data: JSON data to validate
            
        Returns:
            List of format validation errors
        """
        errors = []
        
        # Validate timestamp format
        if "analysis_timestamp" in json_data:
            try:
                datetime.fromisoformat(json_data["analysis_timestamp"].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                errors.append({
                    "field": "analysis_timestamp",
                    "message": "Invalid ISO timestamp format",
                    "type": "format_error",
                    "value": json_data["analysis_timestamp"]
                })
        
        # Validate workflow status
        if "workflow_status" in json_data:
            valid_statuses = ["pending", "in_progress", "completed", "failed"]
            if json_data["workflow_status"] not in valid_statuses:
                errors.append({
                    "field": "workflow_status",
                    "message": f"Invalid workflow status. Must be one of: {valid_statuses}",
                    "type": "format_error",
                    "value": json_data["workflow_status"]
                })
        
        return errors
    
    def get_validation_summary(self, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a summary of validation results.
        
        Args:
            validation_result: Result from validate_consolidated_json
            
        Returns:
            Validation summary
        """
        return {
            "success": validation_result["success"],
            "total_errors": len(validation_result["errors"]),
            "total_warnings": len(validation_result["warnings"]),
            "schema_version": self.schema_version,
            "validation_timestamp": datetime.now().isoformat()
        }
    
    def validate_schema_version_compatibility(self, json_data: Dict[str, Any]) -> bool:
        """
        Validate schema version compatibility.
        
        Args:
            json_data: JSON data to validate
            
        Returns:
            True if compatible, False otherwise
        """
        # For now, assume all versions are compatible
        # This can be extended with version-specific validation logic
        return True 