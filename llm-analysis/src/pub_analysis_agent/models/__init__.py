"""
Data models for the publication analysis agent.

This module contains Pydantic models for representing datasets, publications,
and analysis results.
"""

from .dataset import Dataset, DatasetMatchResult, DatasetQuery, PublicationReference
from .dimensions import (
    Author, Institution, Publication,
    AuthorQuery, InstitutionQuery, PublicationQuery
)
from .analysis_result import (
    AnalysisResult, DatasetMention, AnalysisQuery, AnalysisResults,
    LLMMetadata, ErrorInfo, AnalysisType, AnalysisStatus
)
from .schema_validator import (
    SchemaValidator, ConsolidatedAnalysisSchema, WorkflowStatus,
    PublicationMetadata, AnalysisFlags, DatasetAnalysis, CodeExtraction,
    LinkExtraction, WorkflowMetadata, LLMMetadataSchema, ErrorInformation
)

__all__ = [
    "Dataset",
    "DatasetMatchResult", 
    "DatasetQuery",
    "PublicationReference",
    "Author",
    "Institution", 
    "Publication",
    "AuthorQuery",
    "InstitutionQuery",
    "PublicationQuery",
    "AnalysisResult",
    "DatasetMention",
    "AnalysisQuery",
    "AnalysisResults",
    "LLMMetadata",
    "ErrorInfo",
    "AnalysisType",
    "AnalysisStatus",
    "SchemaValidator",
    "ConsolidatedAnalysisSchema",
    "WorkflowStatus",
    "PublicationMetadata",
    "AnalysisFlags",
    "DatasetAnalysis",
    "CodeExtraction",
    "LinkExtraction",
    "WorkflowMetadata",
    "LLMMetadataSchema",
    "ErrorInformation"
] 