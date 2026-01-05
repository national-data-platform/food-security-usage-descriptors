"""
Service layer for the publication analysis agent.

This module provides service classes for interacting with different data sources
and managing business logic, including MongoDB services, dataset management,
and analysis result storage.
"""

from .mongodb_client import MongoDBClient
from .dataset_service import DatasetService
from .dimensions_service import (
    AuthorService,
    InstitutionService, 
    PublicationService,
    DimensionsService
)
from .results_service import (
    ResultsService,
    StorageError,
    ConnectionError,
    ValidationError,
    TransactionError,
    StorageMetrics
)
from .data_quality_validator import (
    DataQualityValidator,
    QualityMetrics,
    CompletenessScore,
    QualityIssue,
    QualityLevel,
    ValidationSeverity
)
from .grobid_parser import (
    GROBIDParser,
    GROBIDPublication,
    GROBIDFullText,
    GROBIDTextContent,
    GROBIDSection
)
from .elasticsearch_sync_service import (
    ElasticsearchSyncService,
    ElasticsearchConfig,
    IndexMapping
)
from .denormalization_service import (
    DenormalizationService
)
from .incremental_sync_service import (
    IncrementalSyncService,
    SyncState,
    ChangeDetectionResult
)
from .elasticsearch_error_handler import (
    ErrorType,
    ErrorSeverity,
    SyncError,
    DeadLetterQueueItem,
    RetryConfig,
    ErrorClassifier,
    RetryManager,
    DeadLetterQueue,
    SyncStatusTracker,
    with_error_handling
)
from .reindex_service import (
    ReindexService,
    ReindexConfig,
    ReindexState,
    ReindexStatus,
    ConflictInfo,
    ConflictResolutionStrategy
)

__all__ = [
    "MongoDBClient",
    "DatasetService",
    "AuthorService",
    "InstitutionService",
    "PublicationService", 
    "DimensionsService",
    "ResultsService",
    "StorageError",
    "ConnectionError", 
    "ValidationError",
    "TransactionError",
    "StorageMetrics",
    "DataQualityValidator",
    "QualityMetrics",
    "CompletenessScore",
    "QualityIssue",
    "QualityLevel",
    "ValidationSeverity",
    "GROBIDParser",
    "GROBIDPublication",
    "GROBIDFullText",
    "GROBIDTextContent",
    "GROBIDSection",
    "ElasticsearchSyncService",
    "ElasticsearchConfig",
    "IndexMapping",
    "DenormalizationService",
    "IncrementalSyncService",
    "SyncState",
    "ChangeDetectionResult",
    "ErrorType",
    "ErrorSeverity",
    "SyncError",
    "DeadLetterQueueItem",
    "RetryConfig",
    "ErrorClassifier",
    "RetryManager",
    "DeadLetterQueue",
    "SyncStatusTracker",
    "with_error_handling",
    "ReindexService",
    "ReindexConfig",
    "ReindexState",
    "ReindexStatus",
    "ConflictInfo",
    "ConflictResolutionStrategy"
] 