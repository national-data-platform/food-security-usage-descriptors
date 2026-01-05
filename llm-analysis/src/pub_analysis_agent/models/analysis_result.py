"""
Analysis result models for the publication analysis agent.

This module contains Pydantic models for representing analysis results.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, UTC
from enum import Enum
from pydantic import BaseModel, Field


class DatasetMention(BaseModel):
    """Model representing a dataset mention in a publication."""
    
    dataset_name: str = Field(..., description="Name of the mentioned dataset")
    confidence: float = Field(..., description="Confidence score (0-1)")
    context: Optional[str] = Field(None, description="Context around the mention")
    section: Optional[str] = Field(None, description="Section where mention was found")
    page_number: Optional[int] = Field(None, description="Page number if available")
    doi: Optional[str] = Field(None, description="Dataset DOI if found in the text")


class AnalysisType(str, Enum):
    """Enum for analysis types."""
    
    DATASET_MENTION = "dataset_mention"
    DATASET_VALIDATION = "dataset_validation"
    CODE_EXTRACTION = "code_extraction"
    LINK_EXTRACTION = "link_extraction"
    NEW_DATASET_DISCOVERY = "new_dataset_discovery"
    DATASET_JOIN_ANALYSIS = "dataset_join_analysis"
    FULL_ANALYSIS = "full_analysis"
    TOPIC_EXTRACTION = "topic_extraction"


class AnalysisStatus(str, Enum):
    """Enum for analysis status."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AnalysisResult(BaseModel):
    """Model representing an analysis result."""
    
    id: Optional[str] = Field(None, description="Unique identifier")
    publication_id: str = Field(..., description="Publication ID being analyzed")
    analysis_type: AnalysisType = Field(..., description="Type of analysis performed")
    dataset_mentions: List[DatasetMention] = Field(default_factory=list, description="Dataset mentions found")
    confidence_score: Optional[float] = Field(None, description="Overall confidence score")
    analysis_text: Optional[str] = Field(None, description="Full analysis text")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Analysis timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    def to_mongo_dict(self) -> dict:
        """Convert analysis result to MongoDB document format."""
        data = self.model_dump(by_alias=True)
        if self.id is None:
            data.pop('id', None)
        
        # Convert numpy arrays and other non-serializable types
        return self._convert_for_serialization(data)
    
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
    
    @staticmethod
    def from_mongo_dict(data: dict) -> 'AnalysisResult':
        """Create AnalysisResult from MongoDB document."""
        # Create a copy to avoid modifying the original
        data_copy = data.copy()
        
        # Convert _id to string id if it exists
        if '_id' in data_copy and 'id' not in data_copy:
            data_copy['id'] = str(data_copy.pop('_id'))
        
        return AnalysisResult(**data_copy)
    
    model_config = {
        "populate_by_name": True,
        "validate_assignment": True
    }


class AnalysisQuery(BaseModel):
    """Model for analysis queries."""
    
    publication_id: Optional[str] = Field(None, description="Publication ID to filter by")
    publication_ids: Optional[List[str]] = Field(None, description="List of publication IDs to filter by")
    analysis_type: Optional[AnalysisType] = Field(None, description="Analysis type to filter by")
    analysis_types: Optional[List[AnalysisType]] = Field(None, description="List of analysis types to filter by")
    statuses: Optional[List[AnalysisStatus]] = Field(None, description="List of statuses to filter by")
    tags: Optional[List[str]] = Field(None, description="List of tags to filter by")
    date_from: Optional[datetime] = Field(None, description="Start date for filtering")
    date_to: Optional[datetime] = Field(None, description="End date for filtering")
    created_after: Optional[datetime] = Field(None, description="Created after date")
    created_before: Optional[datetime] = Field(None, description="Created before date")
    has_errors: Optional[bool] = Field(None, description="Filter by presence of errors")
    min_relevance_score: Optional[float] = Field(None, description="Minimum relevance score")
    sort_by: Optional[str] = Field("created_at", description="Field to sort by")
    sort_order: Optional[int] = Field(1, description="Sort order (1 for ascending, -1 for descending)")
    limit: Optional[int] = Field(100, description="Maximum number of results")
    offset: Optional[int] = Field(0, description="Number of results to skip")
    skip: Optional[int] = Field(0, description="Number of results to skip (alias for offset)")


class AnalysisResults(BaseModel):
    """Model representing multiple analysis results."""
    
    results: List[AnalysisResult] = Field(default_factory=list, description="List of analysis results")
    total_count: int = Field(0, description="Total number of results")
    page: int = Field(1, description="Current page number")
    page_size: int = Field(100, description="Results per page")


class LLMMetadata(BaseModel):
    """Model representing LLM metadata."""
    
    model_name: str = Field(..., description="Name of the LLM model used")
    model_version: Optional[str] = Field(None, description="Version of the model")
    tokens_used: Optional[int] = Field(None, description="Number of tokens used")
    response_time: Optional[float] = Field(None, description="Response time in seconds")
    temperature: Optional[float] = Field(None, description="Temperature setting used")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens setting")


class ErrorInfo(BaseModel):
    """Model representing error information."""
    
    error_type: str = Field(..., description="Type of error")
    error_message: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    stack_trace: Optional[str] = Field(None, description="Stack trace if available") 