"""
Elasticsearch Error Handling and Retry Mechanisms.

This module provides comprehensive error handling, retry logic, and monitoring
for Elasticsearch synchronization operations.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Callable, Type, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import json

from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import (
    ConnectionError as ESConnectionError,
    ConnectionTimeout,
    RequestError,
    NotFoundError,
    ConflictError,
    AuthenticationException,
    AuthorizationException,
    SerializationError
)
from pymongo.errors import PyMongoError
from pymongo.collection import Collection

from ..utils.circuit_breaker import circuit_breaker

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors that can occur during sync operations."""
    
    CONNECTION = "connection"
    TIMEOUT = "timeout"
    VALIDATION = "validation"
    INDEXING = "indexing"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    SERIALIZATION = "serialization"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SyncError:
    """Represents a sync operation error."""
    
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    timestamp: datetime
    document_id: Optional[str] = None
    operation: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    recoverable: bool = True
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "error_type": self.error_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "document_id": self.document_id,
            "operation": self.operation,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "recoverable": self.recoverable,
            "context": self.context
        }


@dataclass
class DeadLetterQueueItem:
    """Represents an item in the dead letter queue."""
    
    document_id: str
    document_data: Dict[str, Any]
    error: SyncError
    failed_at: datetime
    retry_attempts: int = 0
    max_retry_attempts: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dead letter queue item to dictionary."""
        return {
            "document_id": self.document_id,
            "document_data": self.document_data,
            "error": self.error.to_dict(),
            "failed_at": self.failed_at.isoformat(),
            "retry_attempts": self.retry_attempts,
            "max_retry_attempts": self.max_retry_attempts
        }


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_errors: List[Type[Exception]] = field(default_factory=lambda: [
        ESConnectionError,
        ConnectionTimeout,
        RequestError,
        ConflictError
    ])


class ErrorClassifier:
    """Classifies errors and determines appropriate handling strategies."""
    
    @staticmethod
    def classify_error(error: Exception) -> SyncError:
        """Classify an exception and create a SyncError."""
        error_type = ErrorType.UNKNOWN
        severity = ErrorSeverity.MEDIUM
        recoverable = True
        
        if isinstance(error, ESConnectionError):
            error_type = ErrorType.CONNECTION
            severity = ErrorSeverity.HIGH
            recoverable = True
        elif isinstance(error, ConnectionTimeout):
            error_type = ErrorType.TIMEOUT
            severity = ErrorSeverity.MEDIUM
            recoverable = True
        elif isinstance(error, RequestError):
            error_type = ErrorType.INDEXING
            severity = ErrorSeverity.MEDIUM
            recoverable = True
        elif isinstance(error, ConflictError):
            error_type = ErrorType.INDEXING
            severity = ErrorSeverity.LOW
            recoverable = True
        elif isinstance(error, AuthenticationException):
            error_type = ErrorType.AUTHENTICATION
            severity = ErrorSeverity.CRITICAL
            recoverable = False
        elif isinstance(error, AuthorizationException):
            error_type = ErrorType.AUTHORIZATION
            severity = ErrorSeverity.CRITICAL
            recoverable = False
        elif isinstance(error, SerializationError):
            error_type = ErrorType.SERIALIZATION
            severity = ErrorSeverity.MEDIUM
            recoverable = False
        elif isinstance(error, PyMongoError):
            error_type = ErrorType.CONNECTION
            severity = ErrorSeverity.HIGH
            recoverable = True
        
        return SyncError(
            error_type=error_type,
            severity=severity,
            message=str(error),
            timestamp=datetime.now(timezone.utc),
            recoverable=recoverable
        )


class RetryManager:
    """Manages retry logic with exponential backoff."""
    
    def __init__(self, config: RetryConfig):
        """Initialize retry manager."""
        self.config = config
    
    async def execute_with_retry(
        self,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute an operation with retry logic.
        
        Args:
            operation: Async function to execute
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: If all retry attempts fail
        """
        last_error = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_error = e
                
                # Check if error is retryable
                if not self._is_retryable_error(e):
                    raise e
                
                # If this is the last attempt, raise the error
                if attempt == self.config.max_attempts - 1:
                    raise e
                
                # Calculate delay with exponential backoff
                delay = self._calculate_delay(attempt)
                
                logger.warning(
                    f"Operation failed (attempt {attempt + 1}/{self.config.max_attempts}): {e}. "
                    f"Retrying in {delay:.2f}s"
                )
                
                await asyncio.sleep(delay)
        
        raise last_error
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error is retryable."""
        return any(isinstance(error, error_type) for error_type in self.config.retryable_errors)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        delay = min(
            self.config.base_delay * (self.config.exponential_base ** attempt),
            self.config.max_delay
        )
        
        if self.config.jitter:
            # Add jitter to prevent thundering herd
            jitter = delay * 0.1 * (time.time() % 1.0)
            delay += jitter
        
        return delay


class DeadLetterQueue:
    """Manages failed documents that require manual intervention."""
    
    def __init__(self, collection: Collection):
        """Initialize dead letter queue."""
        self.collection = collection
        self._ensure_indexes()
    
    def _ensure_indexes(self) -> None:
        """Ensure proper indexes for dead letter queue."""
        try:
            self.collection.create_index("document_id", unique=True)
            self.collection.create_index("failed_at")
            self.collection.create_index("error_type")
        except Exception as e:
            logger.warning(f"Failed to create dead letter queue indexes: {e}")
    
    async def add_failed_document(
        self,
        document_id: str,
        document_data: Dict[str, Any],
        error: SyncError
    ) -> None:
        """Add a failed document to the dead letter queue."""
        try:
            dlq_item = DeadLetterQueueItem(
                document_id=document_id,
                document_data=document_data,
                error=error,
                failed_at=datetime.now(timezone.utc)
            )
            
            await self.collection.replace_one(
                {"document_id": document_id},
                dlq_item.to_dict(),
                upsert=True
            )
            
            logger.warning(f"Added document {document_id} to dead letter queue: {error.message}")
            
        except Exception as e:
            logger.error(f"Failed to add document to dead letter queue: {e}")
    
    async def get_failed_documents(
        self,
        limit: int = 100,
        error_type: Optional[ErrorType] = None
    ) -> List[DeadLetterQueueItem]:
        """Get failed documents from the dead letter queue."""
        try:
            query = {}
            if error_type:
                query["error.error_type"] = error_type.value
            
            cursor = self.collection.find(query).sort("failed_at", -1).limit(limit)
            documents = await cursor.to_list(length=limit)
            
            result = []
            for doc in documents:
                try:
                    # Create SyncError from dict
                    error_dict = doc.get("error", {})
                    error = SyncError(
                        error_type=ErrorType(error_dict.get("error_type", "unknown")),
                        severity=ErrorSeverity(error_dict.get("severity", "medium")),
                        message=error_dict.get("message", ""),
                        timestamp=datetime.fromisoformat(error_dict.get("timestamp", datetime.now(timezone.utc).isoformat())),
                        document_id=error_dict.get("document_id"),
                        operation=error_dict.get("operation"),
                        retry_count=error_dict.get("retry_count", 0),
                        max_retries=error_dict.get("max_retries", 3),
                        recoverable=error_dict.get("recoverable", True),
                        context=error_dict.get("context", {})
                    )
                    
                    # Create DeadLetterQueueItem
                    item = DeadLetterQueueItem(
                        document_id=doc["document_id"],
                        document_data=doc["document_data"],
                        error=error,
                        failed_at=datetime.fromisoformat(doc["failed_at"]),
                        retry_attempts=doc.get("retry_attempts", 0),
                        max_retry_attempts=doc.get("max_retry_attempts", 5)
                    )
                    result.append(item)
                except Exception as item_error:
                    logger.warning(f"Failed to create DeadLetterQueueItem from document {doc.get('document_id', 'unknown')}: {item_error}")
                    continue
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get failed documents: {e}")
            return []
    
    async def retry_failed_document(
        self,
        document_id: str,
        operation: Callable
    ) -> bool:
        """Retry processing a failed document."""
        try:
            # Get the failed document
            doc = await self.collection.find_one({"document_id": document_id})
            if not doc:
                logger.warning(f"Failed document {document_id} not found in dead letter queue")
                return False
            
            # Create SyncError from dict
            error_dict = doc.get("error", {})
            error = SyncError(
                error_type=ErrorType(error_dict.get("error_type", "unknown")),
                severity=ErrorSeverity(error_dict.get("severity", "medium")),
                message=error_dict.get("message", ""),
                timestamp=datetime.fromisoformat(error_dict.get("timestamp", datetime.now(timezone.utc).isoformat())),
                document_id=error_dict.get("document_id"),
                operation=error_dict.get("operation"),
                retry_count=error_dict.get("retry_count", 0),
                max_retries=error_dict.get("max_retries", 3),
                recoverable=error_dict.get("recoverable", True),
                context=error_dict.get("context", {})
            )
            
            # Create DeadLetterQueueItem
            dlq_item = DeadLetterQueueItem(
                document_id=doc["document_id"],
                document_data=doc["document_data"],
                error=error,
                failed_at=datetime.fromisoformat(doc["failed_at"]),
                retry_attempts=doc.get("retry_attempts", 0),
                max_retry_attempts=doc.get("max_retry_attempts", 5)
            )
            
            # Try to process the document
            try:
                await operation(dlq_item.document_data)
                
                # Remove from dead letter queue on success
                await self.collection.delete_one({"document_id": document_id})
                logger.info(f"Successfully retried document {document_id}")
                return True
                
            except Exception as e:
                # Update retry count
                dlq_item.retry_attempts += 1
                dlq_item.error.retry_count = dlq_item.retry_attempts
                dlq_item.error.message = str(e)
                dlq_item.error.timestamp = datetime.now(timezone.utc)
                
                await self.collection.replace_one(
                    {"document_id": document_id},
                    dlq_item.to_dict()
                )
                
                logger.warning(f"Retry failed for document {document_id}: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error retrying failed document {document_id}: {e}")
            return False
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about the dead letter queue."""
        try:
            total_count = await self.collection.count_documents({})
            
            # Count by error type
            pipeline = [
                {"$group": {"_id": "$error.error_type", "count": {"$sum": 1}}}
            ]
            error_type_counts = await self.collection.aggregate(pipeline).to_list(length=None)
            
            # Count by severity
            pipeline = [
                {"$group": {"_id": "$error.severity", "count": {"$sum": 1}}}
            ]
            severity_counts = await self.collection.aggregate(pipeline).to_list(length=None)
            
            return {
                "total_failed_documents": total_count,
                "error_type_distribution": {item["_id"]: item["count"] for item in error_type_counts},
                "severity_distribution": {item["_id"]: item["count"] for item in severity_counts}
            }
            
        except Exception as e:
            logger.error(f"Failed to get dead letter queue stats: {e}")
            return {"error": str(e)}


class SyncStatusTracker:
    """Tracks sync operation status and metrics."""
    
    def __init__(self, collection: Collection):
        """Initialize sync status tracker."""
        self.collection = collection
        self._ensure_indexes()
    
    def _ensure_indexes(self) -> None:
        """Ensure proper indexes for sync status tracking."""
        try:
            self.collection.create_index("sync_id")
            self.collection.create_index("timestamp")
            self.collection.create_index("status")
        except Exception as e:
            logger.warning(f"Failed to create sync status indexes: {e}")
    
    async def record_sync_start(self, sync_id: str, metadata: Dict[str, Any]) -> None:
        """Record the start of a sync operation."""
        try:
            status_doc = {
                "sync_id": sync_id,
                "status": "started",
                "timestamp": datetime.now(timezone.utc),
                "metadata": metadata,
                "errors": [],
                "metrics": {
                    "documents_processed": 0,
                    "documents_succeeded": 0,
                    "documents_failed": 0,
                    "start_time": datetime.now(timezone.utc).isoformat()
                }
            }
            
            await self.collection.insert_one(status_doc)
            logger.info(f"Started sync operation: {sync_id}")
            
        except Exception as e:
            logger.error(f"Failed to record sync start: {e}")
    
    async def record_sync_progress(
        self,
        sync_id: str,
        documents_processed: int,
        documents_succeeded: int,
        documents_failed: int
    ) -> None:
        """Record progress of a sync operation."""
        try:
            await self.collection.update_one(
                {"sync_id": sync_id},
                {
                    "$set": {
                        "metrics.documents_processed": documents_processed,
                        "metrics.documents_succeeded": documents_succeeded,
                        "metrics.documents_failed": documents_failed,
                        "last_updated": datetime.now(timezone.utc)
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to record sync progress: {e}")
    
    async def record_sync_error(self, sync_id: str, error: SyncError) -> None:
        """Record an error during sync operation."""
        try:
            await self.collection.update_one(
                {"sync_id": sync_id},
                {
                    "$push": {"errors": error.to_dict()},
                    "$set": {"last_updated": datetime.now(timezone.utc)}
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to record sync error: {e}")
    
    async def record_sync_completion(
        self,
        sync_id: str,
        status: str,
        final_metrics: Dict[str, Any]
    ) -> None:
        """Record the completion of a sync operation."""
        try:
            end_time = datetime.now(timezone.utc)
            
            await self.collection.update_one(
                {"sync_id": sync_id},
                {
                    "$set": {
                        "status": status,
                        "end_time": end_time,
                        "duration": (end_time - datetime.fromisoformat(
                            final_metrics["start_time"]
                        )).total_seconds(),
                        "final_metrics": final_metrics,
                        "last_updated": end_time
                    }
                }
            )
            
            logger.info(f"Completed sync operation: {sync_id} with status: {status}")
            
        except Exception as e:
            logger.error(f"Failed to record sync completion: {e}")
    
    async def get_sync_status(self, sync_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific sync operation."""
        try:
            doc = await self.collection.find_one({"sync_id": sync_id})
            return doc
        except Exception as e:
            logger.error(f"Failed to get sync status: {e}")
            return None
    
    async def get_recent_syncs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent sync operations."""
        try:
            cursor = self.collection.find().sort("timestamp", -1).limit(limit)
            return await cursor.to_list(length=limit)
        except Exception as e:
            logger.error(f"Failed to get recent syncs: {e}")
            return []


# Circuit breaker for error handling operations
error_handling_circuit_breaker = circuit_breaker(
    service_name="error_handling",
    failure_threshold=5,
    recovery_timeout=30,
    expected_exceptions=(ESConnectionError, PyMongoError)
)


def with_error_handling(
    retry_config: Optional[RetryConfig] = None,
    record_errors: bool = True
):
    """
    Decorator to add error handling and retry logic to sync operations.
    
    Args:
        retry_config: Retry configuration
        record_errors: Whether to record errors in status tracker
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract error handling components from self
            self = args[0] if args else None
            error_classifier = getattr(self, 'error_classifier', ErrorClassifier())
            retry_manager = getattr(self, 'retry_manager', RetryManager(retry_config or RetryConfig()))
            status_tracker = getattr(self, 'status_tracker', None)
            
            try:
                # Execute with retry logic
                return await retry_manager.execute_with_retry(func, *args, **kwargs)
            except Exception as e:
                # Classify the error
                sync_error = error_classifier.classify_error(e)
                
                # Log the error
                logger.error(f"Operation {func.__name__} failed: {e}")
                
                # Record error in status tracker if available and enabled
                if record_errors and status_tracker and hasattr(status_tracker, 'record_sync_error'):
                    try:
                        # Generate a sync_id if not available
                        sync_id = getattr(self, '_current_sync_id', f"{func.__name__}_{int(time.time())}")
                        await status_tracker.record_sync_error(sync_id, sync_error)
                    except Exception as record_error:
                        logger.error(f"Failed to record sync error: {record_error}")
                
                # Re-raise the error
                raise
                
        return wrapper
    return decorator 