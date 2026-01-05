"""
Results service for storing and retrieving LLM analysis results.

This module provides the ResultsService class for interacting with the
dimensions.llm_analyses MongoDB collection, including CRUD operations,
schema validation, upserts, and advanced querying capabilities with comprehensive
error handling, transaction management, and performance monitoring.
"""

import logging
import asyncio
import time
from typing import List, Optional, Dict, Any, Union, Tuple
from datetime import datetime, timedelta, UTC
from functools import wraps
from contextlib import asynccontextmanager

from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT
from pymongo.errors import (
    DuplicateKeyError, OperationFailure, ConnectionFailure, 
    ServerSelectionTimeoutError, WriteError, BulkWriteError,
    NetworkTimeout, ExecutionTimeout
)
from pymongo.results import InsertOneResult, UpdateResult, DeleteResult, BulkWriteResult
from pymongo.client_session import ClientSession
from tenacity import (
    retry, stop_after_attempt, wait_exponential, 
    retry_if_exception_type, before_sleep_log
)
from bson import ObjectId

from .mongodb_client import MongoDBClient
from ..models.analysis_result import (
    AnalysisResult, AnalysisResults, LLMMetadata, ErrorInfo,
    AnalysisQuery, AnalysisType, AnalysisStatus
)
from ..config.settings import DatabaseSettings


logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Base exception for storage-related errors."""
    pass


class ConnectionError(StorageError):
    """Exception raised for connection-related errors."""
    pass


class ValidationError(StorageError):
    """Exception raised for data validation errors."""
    pass


class TransactionError(StorageError):
    """Exception raised for transaction-related errors."""
    pass


def ensure_connection(func):
    """
    Decorator to ensure MongoDB connection is available before executing service methods.
    
    Args:
        func: Function to wrap with connection checking
    """
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        try:
            async with self.mongodb_client.ensure_connection():
                return await func(self, *args, **kwargs)
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Connection error in {func.__name__}: {e}")
            raise ConnectionError(f"Database connection failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise
    return wrapper


def retry_on_transient_errors(max_attempts: int = 3):
    """
    Decorator for retrying operations on transient errors.
    
    Args:
        max_attempts: Maximum number of retry attempts
    """
    def decorator(func):
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((
                ConnectionFailure, ServerSelectionTimeoutError,
                NetworkTimeout, ExecutionTimeout
            )),
            before_sleep=before_sleep_log(logger, logging.WARNING)
        )
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            return await func(self, *args, **kwargs)
        return wrapper
    return decorator


class StorageMetrics:
    """Storage performance metrics collector."""
    
    def __init__(self):
        self.operation_times: Dict[str, List[float]] = {}
        self.error_counts: Dict[str, int] = {}
        self.success_counts: Dict[str, int] = {}
        self._lock = asyncio.Lock()
    
    async def record_operation(self, operation: str, duration: float, success: bool):
        """Record operation metrics."""
        async with self._lock:
            if operation not in self.operation_times:
                self.operation_times[operation] = []
                self.error_counts[operation] = 0
                self.success_counts[operation] = 0
            
            self.operation_times[operation].append(duration)
            if success:
                self.success_counts[operation] += 1
            else:
                self.error_counts[operation] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        metrics = {}
        for operation in self.operation_times:
            times = self.operation_times[operation]
            if times:
                metrics[operation] = {
                    "avg_duration": sum(times) / len(times),
                    "min_duration": min(times),
                    "max_duration": max(times),
                    "total_operations": len(times),
                    "success_count": self.success_counts.get(operation, 0),
                    "error_count": self.error_counts.get(operation, 0),
                    "success_rate": (
                        self.success_counts.get(operation, 0) / len(times) * 100
                        if times else 0
                    )
                }
        return metrics


class ResultsService:
    """
    Service for storing and retrieving LLM analysis results.
    
    This service provides comprehensive CRUD operations for analysis results,
    with schema validation, upsert capabilities, advanced querying, and
    robust error handling with transaction management and performance monitoring.
    
    Attributes:
        mongodb_client: MongoDB client instance
        db_settings: Database configuration settings
        metrics: Storage performance metrics collector
    """
    
    def __init__(
        self,
        mongodb_client: MongoDBClient,
        db_settings: Optional[DatabaseSettings] = None
    ):
        """
        Initialize the ResultsService.
        
        Args:
            mongodb_client: MongoDB client instance
            db_settings: Database configuration settings
        """
        self.mongodb_client = mongodb_client
        self.db_settings = db_settings or DatabaseSettings()
        self._initialized = False
        self.metrics = StorageMetrics()
        
        logger.info("ResultsService initialized with enhanced error handling")
    
    async def initialize(self) -> None:
        """
        Initialize the service by ensuring indexes and collection setup.
        
        This method should be called once before using the service to ensure
        optimal query performance and data integrity.
        """
        if self._initialized:
            return
        
        await self._ensure_indexes()
        self._initialized = True
        logger.info("ResultsService initialization completed")
    
    @retry_on_transient_errors(max_attempts=3)
    async def _ensure_indexes(self) -> None:
        """
        Ensure required indexes exist for optimal query performance.
        
        Creates compound and single-field indexes for common query patterns:
        - publication_id and analysis_type (compound)
        - created_at (for time-based queries)
        - status (for filtering by analysis status)
        - tags (for tag-based queries)
        - text indexes for full-text search
        """
        try:
            start_time = time.time()
            
            # Define indexes
            indexes = [
                # Compound index for publication queries
                IndexModel([
                    ("publication_id", ASCENDING),
                    ("analysis_type", ASCENDING)
                ], unique=True),
                
                # Time-based queries
                IndexModel([("created_at", DESCENDING)]),
                IndexModel([("updated_at", DESCENDING)]),
                
                # Status filtering
                IndexModel([("status", ASCENDING)]),
                
                # Tag-based queries
                IndexModel([("tags", ASCENDING)]),
                
                # LLM metadata queries
                IndexModel([("llm_metadata.model_name", ASCENDING)]),
                IndexModel([("llm_metadata.provider", ASCENDING)]),
            ]
            
            # Create indexes individually to handle conflicts
            for index in indexes:
                try:
                    # Extract the key and options from IndexModel
                    key = index.document['key']
                    options = {k: v for k, v in index.document.items() if k != 'key'}
                    self.mongodb_client.results_collection.create_index(key, **options)
                except Exception as index_error:
                    # Log the error but continue with other indexes
                    logger.warning(f"Failed to create index {index}: {index_error}")
                    continue
            
            # Create a single compound text index to avoid conflicts
            # MongoDB only allows one text index per collection
            try:
                compound_text_index = IndexModel([
                    ("results.summary", TEXT),
                    ("results.topics", TEXT)
                ])
                
                # Extract the key and options from IndexModel
                key = compound_text_index.document['key']
                options = {k: v for k, v in compound_text_index.document.items() if k != 'key'}
                self.mongodb_client.results_collection.create_index(key, **options)
                logger.info("Created compound text index for results.summary and results.topics")
                
            except Exception as text_error:
                # Log the error but continue
                logger.warning(f"Failed to create compound text index: {text_error}")
                # Try to create a single text index on summary as fallback
                try:
                    fallback_index = IndexModel([("results.summary", TEXT)])
                    key = fallback_index.document['key']
                    options = {k: v for k, v in fallback_index.document.items() if k != 'key'}
                    self.mongodb_client.results_collection.create_index(key, **options)
                    logger.info("Created fallback text index for results.summary only")
                except Exception as fallback_error:
                    logger.warning(f"Failed to create fallback text index: {fallback_error}")
            
            duration = time.time() - start_time
            await self.metrics.record_operation("ensure_indexes", duration, True)
            
            logger.info("Database indexes creation completed")
            
        except Exception as e:
            duration = time.time() - start_time
            await self.metrics.record_operation("ensure_indexes", duration, False)
            logger.error(f"Failed to create indexes: {e}")
            # Don't raise the exception, just log it and continue
            logger.warning("Continuing without some indexes - this may affect performance")
    
    @asynccontextmanager
    async def transaction(self):
        """
        Context manager for MongoDB transactions.
        
        Provides atomic operations with automatic rollback on failure.
        """
        session: Optional[ClientSession] = None
        try:
            session = await self.mongodb_client.client.start_session()
            await session.start_transaction()
            yield session
            await session.commit_transaction()
        except Exception as e:
            if session:
                await session.abort_transaction()
            logger.error(f"Transaction failed: {e}")
            raise TransactionError(f"Transaction failed: {e}")
        finally:
            if session:
                await session.end_session()
    
    @ensure_connection
    @retry_on_transient_errors(max_attempts=3)
    async def store_analysis_result(
        self,
        analysis_result: AnalysisResult
    ) -> str:
        """
        Store a new analysis result in the database with enhanced error handling.
        
        Args:
            analysis_result: The analysis result to store
            
        Returns:
            The ObjectId of the inserted document as a string
            
        Raises:
            ValidationError: If the analysis result is invalid
            DuplicateKeyError: If analysis for this publication and type already exists
            ConnectionError: If database connection fails
            StorageError: For other storage-related errors
        """
        start_time = time.time()
        success = False
        
        try:
            # Validate the analysis result
            if not analysis_result.publication_id:
                raise ValidationError("Publication ID is required")
            
            # Ensure updated_at is set
            analysis_result.updated_at = datetime.now(UTC)
            
            # Convert to MongoDB document
            doc = analysis_result.to_mongo_dict()
            
            # Insert the document
            result: InsertOneResult = await self.mongodb_client.results_collection.insert_one(doc)
            
            success = True
            duration = time.time() - start_time
            await self.metrics.record_operation("store_analysis_result", duration, True)
            
            logger.info(
                f"Stored analysis result for publication {analysis_result.publication_id}, "
                f"type {analysis_result.analysis_type}, ID: {result.inserted_id}"
            )
            
            return str(result.inserted_id)
            
        except DuplicateKeyError as e:
            duration = time.time() - start_time
            await self.metrics.record_operation("store_analysis_result", duration, False)
            logger.warning(
                f"Analysis already exists for publication {analysis_result.publication_id} "
                f"and type {analysis_result.analysis_type}"
            )
            raise
        except ValidationError:
            duration = time.time() - start_time
            await self.metrics.record_operation("store_analysis_result", duration, False)
            raise
        except Exception as e:
            duration = time.time() - start_time
            await self.metrics.record_operation("store_analysis_result", duration, False)
            logger.error(f"Failed to store analysis result: {e}")
            raise StorageError(f"Failed to store analysis result: {e}")
    
    @ensure_connection
    @retry_on_transient_errors(max_attempts=3)
    async def store_analysis_results_batch(
        self,
        analysis_results: List[AnalysisResult]
    ) -> List[str]:
        """
        Store multiple analysis results in a batch operation.
        
        Args:
            analysis_results: List of analysis results to store
            
        Returns:
            List of ObjectIds of the inserted documents
            
        Raises:
            ValidationError: If any analysis result is invalid
            BulkWriteError: If batch operation fails
            ConnectionError: If database connection fails
        """
        if not analysis_results:
            return []
        
        start_time = time.time()
        success = False
        
        try:
            # Validate all analysis results
            for result in analysis_results:
                if not result.publication_id:
                    raise ValidationError(f"Publication ID is required for analysis result")
            
            # Prepare documents
            docs = []
            for result in analysis_results:
                result.updated_at = datetime.now(UTC)
                docs.append(result.to_mongo_dict())
            
            # Perform bulk insert
            result: BulkWriteResult = await self.mongodb_client.results_collection.bulk_write([
                {"insertOne": {"document": doc}} for doc in docs
            ])
            
            success = True
            duration = time.time() - start_time
            await self.metrics.record_operation("store_analysis_results_batch", duration, True)
            
            inserted_ids = [str(id) for id in result.inserted_ids]
            
            logger.info(
                f"Stored {len(analysis_results)} analysis results in batch. "
                f"Inserted IDs: {inserted_ids}"
            )
            
            return inserted_ids
            
        except BulkWriteError as e:
            duration = time.time() - start_time
            await self.metrics.record_operation("store_analysis_results_batch", duration, False)
            logger.error(f"Bulk write failed: {e}")
            raise
        except ValidationError:
            duration = time.time() - start_time
            await self.metrics.record_operation("store_analysis_results_batch", duration, False)
            raise
        except Exception as e:
            duration = time.time() - start_time
            await self.metrics.record_operation("store_analysis_results_batch", duration, False)
            logger.error(f"Failed to store analysis results batch: {e}")
            raise StorageError(f"Failed to store analysis results batch: {e}")
    
    @ensure_connection
    @retry_on_transient_errors(max_attempts=3)
    async def get_analysis_by_publication_id(
        self,
        publication_id: str,
        analysis_type: Optional[AnalysisType] = None
    ) -> List[AnalysisResult]:
        """
        Retrieve analysis results for a specific publication with enhanced error handling.
        
        Args:
            publication_id: The publication ID to search for
            analysis_type: Optional analysis type filter
            
        Returns:
            List of analysis results for the publication
            
        Raises:
            ValidationError: If publication_id is invalid
            ConnectionError: If database connection fails
        """
        start_time = time.time()
        success = False
        
        try:
            if not publication_id:
                raise ValidationError("Publication ID is required")
            
            # Build query
            query = {"publication_id": publication_id}
            if analysis_type:
                query["analysis_type"] = analysis_type.value
            
            # Execute query with timeout
            cursor = self.mongodb_client.results_collection.find(query)
            cursor = cursor.sort("created_at", DESCENDING)
            
            # Convert to AnalysisResult objects with error handling
            results = []
            async for doc in cursor:
                try:
                    result = AnalysisResult.from_mongo_dict(doc)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to parse analysis result document: {e}")
                    continue
            
            success = True
            duration = time.time() - start_time
            await self.metrics.record_operation("get_analysis_by_publication_id", duration, True)
            
            logger.info(
                f"Retrieved {len(results)} analysis results for publication {publication_id}"
            )
            
            return results
            
        except ValidationError:
            duration = time.time() - start_time
            await self.metrics.record_operation("get_analysis_by_publication_id", duration, False)
            raise
        except Exception as e:
            duration = time.time() - start_time
            await self.metrics.record_operation("get_analysis_by_publication_id", duration, False)
            logger.error(f"Failed to retrieve analysis by publication ID: {e}")
            raise StorageError(f"Failed to retrieve analysis by publication ID: {e}")
    
    @ensure_connection
    @retry_on_transient_errors(max_attempts=3)
    async def get_analysis_by_id(self, analysis_id: str) -> Optional[AnalysisResult]:
        """
        Retrieve a specific analysis result by its ID with enhanced error handling.
        
        Args:
            analysis_id: The analysis result ID
            
        Returns:
            AnalysisResult if found, None otherwise
            
        Raises:
            ValidationError: If analysis_id is invalid
            ConnectionError: If database connection fails
        """
        start_time = time.time()
        success = False
        
        try:
            if not analysis_id:
                raise ValidationError("Analysis ID is required")
            
            # Validate ObjectId format
            try:
                ObjectId(analysis_id)
            except Exception:
                raise ValidationError(f"Invalid ObjectId format: {analysis_id}")
            
            # Execute query
            doc = await self.mongodb_client.results_collection.find_one({"_id": ObjectId(analysis_id)})
            
            if not doc:
                success = True
                duration = time.time() - start_time
                await self.metrics.record_operation("get_analysis_by_id", duration, True)
                return None
            
            # Convert to AnalysisResult
            result = AnalysisResult.from_mongo_dict(doc)
            
            success = True
            duration = time.time() - start_time
            await self.metrics.record_operation("get_analysis_by_id", duration, True)
            
            logger.info(f"Retrieved analysis result with ID: {analysis_id}")
            return result
            
        except ValidationError:
            duration = time.time() - start_time
            await self.metrics.record_operation("get_analysis_by_id", duration, False)
            raise
        except Exception as e:
            duration = time.time() - start_time
            await self.metrics.record_operation("get_analysis_by_id", duration, False)
            logger.error(f"Failed to retrieve analysis by ID: {e}")
            raise StorageError(f"Failed to retrieve analysis by ID: {e}")
    
    @ensure_connection
    @retry_on_transient_errors(max_attempts=3)
    async def update_analysis_result(
        self,
        analysis_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update an analysis result with enhanced error handling and validation.
        
        Args:
            analysis_id: The analysis result ID
            updates: Dictionary of fields to update
            
        Returns:
            True if update was successful, False if not found
            
        Raises:
            ValidationError: If analysis_id or updates are invalid
            ConnectionError: If database connection fails
        """
        start_time = time.time()
        success = False
        
        try:
            if not analysis_id:
                raise ValidationError("Analysis ID is required")
            
            if not updates:
                raise ValidationError("Updates dictionary cannot be empty")
            
            # Validate ObjectId format
            try:
                ObjectId(analysis_id)
            except Exception:
                raise ValidationError(f"Invalid ObjectId format: {analysis_id}")
            
            # Add updated_at timestamp
            updates["updated_at"] = datetime.now(UTC)
            
            # Perform update
            result: UpdateResult = await self.mongodb_client.results_collection.update_one(
                {"_id": ObjectId(analysis_id)},
                {"$set": updates}
            )
            
            success = True
            duration = time.time() - start_time
            await self.metrics.record_operation("update_analysis_result", duration, True)
            
            if result.matched_count == 0:
                logger.warning(f"Analysis result with ID {analysis_id} not found for update")
                return False
            
            logger.info(f"Updated analysis result with ID: {analysis_id}")
            return True
            
        except ValidationError:
            duration = time.time() - start_time
            await self.metrics.record_operation("update_analysis_result", duration, False)
            raise
        except Exception as e:
            duration = time.time() - start_time
            await self.metrics.record_operation("update_analysis_result", duration, False)
            logger.error(f"Failed to update analysis result: {e}")
            raise StorageError(f"Failed to update analysis result: {e}")
    
    @ensure_connection
    @retry_on_transient_errors(max_attempts=3)
    async def upsert_analysis_result(
        self,
        publication_id: str,
        analysis_type: AnalysisType,
        analysis_result: AnalysisResult
    ) -> str:
        """
        Upsert an analysis result with enhanced error handling.
        
        Args:
            publication_id: The publication ID
            analysis_type: The analysis type
            analysis_result: The analysis result to upsert
            
        Returns:
            The ObjectId of the upserted document
            
        Raises:
            ValidationError: If parameters are invalid
            ConnectionError: If database connection fails
        """
        start_time = time.time()
        success = False
        
        try:
            if not publication_id:
                raise ValidationError("Publication ID is required")
            
            if not analysis_type:
                raise ValidationError("Analysis type is required")
            
            # Ensure updated_at is set
            analysis_result.updated_at = datetime.now(UTC)
            
            # Convert to MongoDB document
            doc = analysis_result.to_mongo_dict()
            
            # Perform upsert
            result: UpdateResult = await self.mongodb_client.results_collection.update_one(
                {
                    "publication_id": publication_id,
                    "analysis_type": analysis_type.value
                },
                {"$set": doc},
                upsert=True
            )
            
            success = True
            duration = time.time() - start_time
            await self.metrics.record_operation("upsert_analysis_result", duration, True)
            
            if result.upserted_id:
                logger.info(
                    f"Inserted new analysis result for publication {publication_id}, "
                    f"type {analysis_type}, ID: {result.upserted_id}"
                )
                return str(result.upserted_id)
            else:
                logger.info(
                    f"Updated existing analysis result for publication {publication_id}, "
                    f"type {analysis_type}"
                )
                # For updates, we need to get the existing document ID
                existing_doc = await self.mongodb_client.results_collection.find_one({
                    "publication_id": publication_id,
                    "analysis_type": analysis_type.value
                })
                return str(existing_doc["_id"])
            
        except ValidationError:
            duration = time.time() - start_time
            await self.metrics.record_operation("upsert_analysis_result", duration, False)
            raise
        except Exception as e:
            duration = time.time() - start_time
            await self.metrics.record_operation("upsert_analysis_result", duration, False)
            logger.error(f"Failed to upsert analysis result: {e}")
            raise StorageError(f"Failed to upsert analysis result: {e}")
    
    @ensure_connection
    @retry_on_transient_errors(max_attempts=3)
    async def delete_analysis_result(self, analysis_id: str) -> bool:
        """
        Delete an analysis result with enhanced error handling.
        
        Args:
            analysis_id: The analysis result ID
            
        Returns:
            True if deletion was successful, False if not found
            
        Raises:
            ValidationError: If analysis_id is invalid
            ConnectionError: If database connection fails
        """
        start_time = time.time()
        success = False
        
        try:
            if not analysis_id:
                raise ValidationError("Analysis ID is required")
            
            # Validate ObjectId format
            try:
                ObjectId(analysis_id)
            except Exception:
                raise ValidationError(f"Invalid ObjectId format: {analysis_id}")
            
            # Perform deletion
            result: DeleteResult = await self.mongodb_client.results_collection.delete_one({
                "_id": ObjectId(analysis_id)
            })
            
            success = True
            duration = time.time() - start_time
            await self.metrics.record_operation("delete_analysis_result", duration, True)
            
            if result.deleted_count == 0:
                logger.warning(f"Analysis result with ID {analysis_id} not found for deletion")
                return False
            
            logger.info(f"Deleted analysis result with ID: {analysis_id}")
            return True
            
        except ValidationError:
            duration = time.time() - start_time
            await self.metrics.record_operation("delete_analysis_result", duration, False)
            raise
        except Exception as e:
            duration = time.time() - start_time
            await self.metrics.record_operation("delete_analysis_result", duration, False)
            logger.error(f"Failed to delete analysis result: {e}")
            raise StorageError(f"Failed to delete analysis result: {e}")
    
    @ensure_connection
    @retry_on_transient_errors(max_attempts=3)
    async def query_analysis_results(
        self,
        query: AnalysisQuery
    ) -> List[AnalysisResult]:
        """
        Query analysis results with enhanced error handling and validation.
        
        Args:
            query: AnalysisQuery object containing query parameters
            
        Returns:
            List of matching analysis results
            
        Raises:
            ValidationError: If query parameters are invalid
            ConnectionError: If database connection fails
        """
        start_time = time.time()
        success = False
        
        try:
            if not query:
                raise ValidationError("Query object is required")
            
            # Build MongoDB query
            mongo_query = {}
            
            if query.publication_id:
                mongo_query["publication_id"] = query.publication_id
            
            if query.analysis_type:
                mongo_query["analysis_type"] = query.analysis_type.value
            
            if query.statuses:
                mongo_query["status"] = {"$in": [status.value for status in query.statuses]}
            
            if query.tags:
                mongo_query["tags"] = {"$in": query.tags}
            
            if query.date_from or query.date_to:
                date_filter = {}
                if query.date_from:
                    date_filter["$gte"] = query.date_from
                if query.date_to:
                    date_filter["$lte"] = query.date_to
                mongo_query["created_at"] = date_filter
            
                            # Note: model_name is not available in AnalysisQuery, so we skip this filter
            
            # Execute query
            cursor = self.mongodb_client.results_collection.find(mongo_query)
            
            # Apply sorting
            if query.sort_by:
                sort_direction = DESCENDING if query.sort_order == -1 else ASCENDING
                cursor = cursor.sort(query.sort_by, sort_direction)
            
            # Apply pagination
            if query.skip:
                cursor = cursor.skip(query.skip)
            if query.limit:
                cursor = cursor.limit(query.limit)
            
            # Convert to AnalysisResult objects
            results = []
            async for doc in cursor:
                try:
                    result = AnalysisResult.from_mongo_dict(doc)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to parse analysis result document: {e}")
                    continue
            
            success = True
            duration = time.time() - start_time
            await self.metrics.record_operation("query_analysis_results", duration, True)
            
            logger.info(f"Query returned {len(results)} analysis results")
            return results
            
        except ValidationError:
            duration = time.time() - start_time
            await self.metrics.record_operation("query_analysis_results", duration, False)
            raise
        except Exception as e:
            duration = time.time() - start_time
            await self.metrics.record_operation("query_analysis_results", duration, False)
            logger.error(f"Failed to query analysis results: {e}")
            raise StorageError(f"Failed to query analysis results: {e}")
    
    @ensure_connection
    @retry_on_transient_errors(max_attempts=3)
    async def count_analysis_results(
        self,
        query: Optional[AnalysisQuery] = None
    ) -> int:
        """
        Count analysis results with enhanced error handling.
        
        Args:
            query: Optional AnalysisQuery object for filtering
            
        Returns:
            Number of matching analysis results
            
        Raises:
            ValidationError: If query parameters are invalid
            ConnectionError: If database connection fails
        """
        start_time = time.time()
        success = False
        
        try:
            # Build MongoDB query
            mongo_query = {}
            if query:
                if query.publication_id:
                    mongo_query["publication_id"] = query.publication_id
                if query.analysis_type:
                    mongo_query["analysis_type"] = query.analysis_type.value
                if query.statuses:
                    mongo_query["status"] = {"$in": [status.value for status in query.statuses]}
                if query.tags:
                    mongo_query["tags"] = {"$in": query.tags}
                if query.date_from or query.date_to:
                    date_filter = {}
                    if query.date_from:
                        date_filter["$gte"] = query.date_from
                    if query.date_to:
                        date_filter["$lte"] = query.date_to
                    mongo_query["created_at"] = date_filter
                            # Note: model_name is not available in AnalysisQuery, so we skip this filter
            
            # Execute count
            count = await self.mongodb_client.results_collection.count_documents(mongo_query)
            
            success = True
            duration = time.time() - start_time
            await self.metrics.record_operation("count_analysis_results", duration, True)
            
            logger.info(f"Count query returned {count} analysis results")
            return count
            
        except Exception as e:
            duration = time.time() - start_time
            await self.metrics.record_operation("count_analysis_results", duration, False)
            logger.error(f"Failed to count analysis results: {e}")
            raise StorageError(f"Failed to count analysis results: {e}")
    
    @ensure_connection
    @retry_on_transient_errors(max_attempts=3)
    async def get_analysis_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis statistics with enhanced error handling.
        
        Returns:
            Dictionary containing various statistics
            
        Raises:
            ConnectionError: If database connection fails
        """
        start_time = time.time()
        success = False
        
        try:
            # Get total count
            total_count = await self.mongodb_client.results_collection.count_documents({})
            
            # Get counts by analysis type
            type_pipeline = [
                {"$group": {"_id": "$analysis_type", "count": {"$sum": 1}}}
            ]
            type_stats = {}
            cursor = await self.mongodb_client.results_collection.aggregate(type_pipeline)
            async for doc in cursor:
                type_stats[doc["_id"]] = doc["count"]
            
            # Get counts by status
            status_pipeline = [
                {"$group": {"_id": "$status", "count": {"$sum": 1}}}
            ]
            status_stats = {}
            cursor = await self.mongodb_client.results_collection.aggregate(status_pipeline)
            async for doc in cursor:
                status_stats[doc["_id"]] = doc["count"]
            
            # Get average processing time
            time_pipeline = [
                {"$match": {"llm_metadata.processing_time_seconds": {"$exists": True}}},
                {"$group": {
                    "_id": None,
                    "avg_time": {"$avg": "$llm_metadata.processing_time_seconds"},
                    "min_time": {"$min": "$llm_metadata.processing_time_seconds"},
                    "max_time": {"$max": "$llm_metadata.processing_time_seconds"}
                }}
            ]
            time_stats = {}
            cursor = await self.mongodb_client.results_collection.aggregate(time_pipeline)
            async for doc in cursor:
                time_stats = {
                    "avg_processing_time": doc["avg_time"],
                    "min_processing_time": doc["min_time"],
                    "max_processing_time": doc["max_time"]
                }
            
            # Get recent activity (last 7 days)
            week_ago = datetime.now(UTC) - timedelta(days=7)
            recent_count = await self.mongodb_client.results_collection.count_documents({
                "created_at": {"$gte": week_ago}
            })
            
            success = True
            duration = time.time() - start_time
            await self.metrics.record_operation("get_analysis_statistics", duration, True)
            
            stats = {
                "total_analyses": total_count,
                "analyses_by_type": type_stats,
                "analyses_by_status": status_stats,
                "processing_time_stats": time_stats,
                "recent_analyses_7_days": recent_count,
                "storage_metrics": self.metrics.get_metrics()
            }
            
            logger.info(f"Retrieved analysis statistics: {total_count} total analyses")
            return stats
            
        except Exception as e:
            duration = time.time() - start_time
            await self.metrics.record_operation("get_analysis_statistics", duration, False)
            logger.error(f"Failed to get analysis statistics: {e}")
            raise StorageError(f"Failed to get analysis statistics: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check with enhanced error handling.
        
        Returns:
            Dictionary containing health status and metrics
        """
        start_time = time.time()
        
        try:
            # Check database connection
            db_healthy = await self.mongodb_client.health_check()
            
            # Check collection access
            collection_healthy = False
            try:
                await self.mongodb_client.results_collection.find_one({})
                collection_healthy = True
            except Exception as e:
                logger.warning(f"Collection health check failed: {e}")
            
            # Get storage metrics
            storage_metrics = self.metrics.get_metrics()
            
            # Calculate overall health
            overall_healthy = db_healthy and collection_healthy
            
            duration = time.time() - start_time
            
            health_status = {
                "overall_healthy": overall_healthy,
                "database_healthy": db_healthy,
                "collection_healthy": collection_healthy,
                "response_time_ms": duration * 1000,
                "storage_metrics": storage_metrics,
                "timestamp": datetime.now(UTC).isoformat()
            }
            
            if overall_healthy:
                logger.info("Health check passed")
            else:
                logger.warning("Health check failed")
            
            return health_status
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Health check failed: {e}")
            return {
                "overall_healthy": False,
                "database_healthy": False,
                "collection_healthy": False,
                "response_time_ms": duration * 1000,
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat()
            }
    
    async def cleanup_old_analyses(self, days_old: int = 90) -> int:
        """
        Clean up old analysis results with enhanced error handling.
        
        Args:
            days_old: Number of days after which analyses should be deleted
            
        Returns:
            Number of deleted analyses
            
        Raises:
            ValidationError: If days_old is invalid
            ConnectionError: If database connection fails
        """
        start_time = time.time()
        success = False
        
        try:
            if days_old < 1:
                raise ValidationError("Days old must be at least 1")
            
            cutoff_date = datetime.now(UTC) - timedelta(days=days_old)
            
            # Delete old analyses
            result: DeleteResult = await self.mongodb_client.results_collection.delete_many({
                "created_at": {"$lt": cutoff_date}
            })
            
            success = True
            duration = time.time() - start_time
            await self.metrics.record_operation("cleanup_old_analyses", duration, True)
            
            deleted_count = result.deleted_count
            logger.info(f"Cleaned up {deleted_count} old analysis results (older than {days_old} days)")
            
            return deleted_count
            
        except ValidationError:
            duration = time.time() - start_time
            await self.metrics.record_operation("cleanup_old_analyses", duration, False)
            raise
        except Exception as e:
            duration = time.time() - start_time
            await self.metrics.record_operation("cleanup_old_analyses", duration, False)
            logger.error(f"Failed to cleanup old analyses: {e}")
            raise StorageError(f"Failed to cleanup old analyses: {e}") 