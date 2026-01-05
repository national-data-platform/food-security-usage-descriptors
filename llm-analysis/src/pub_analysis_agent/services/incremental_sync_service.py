"""
Incremental Sync Service for Elasticsearch.

This module provides functionality to detect and sync only changed documents
from MongoDB to Elasticsearch using timestamps and change streams.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from functools import wraps
from dataclasses import dataclass, asdict

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.cursor import Cursor
from pymongo.errors import PyMongoError
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import RequestError, NotFoundError

from ..config.settings import get_settings
from ..services.elasticsearch_sync_service import ElasticsearchSyncService
from ..services.denormalization_service import DenormalizationService
from ..services.results_service import ResultsService
from ..utils.circuit_breaker import circuit_breaker

logger = logging.getLogger(__name__)

# Circuit breaker for incremental sync operations
sync_circuit_breaker = circuit_breaker(
    service_name="incremental_sync",
    failure_threshold=3,
    recovery_timeout=60,
    expected_exceptions=(PyMongoError, RequestError, NotFoundError)
)


@dataclass
class SyncState:
    """Represents the current sync state."""
    
    last_sync_timestamp: datetime
    last_sync_document_count: int
    last_sync_duration: float
    last_sync_status: str  # "success", "partial", "failed"
    last_error_message: Optional[str] = None
    total_synced_documents: int = 0
    total_sync_operations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert sync state to dictionary."""
        return {
            "last_sync_timestamp": self.last_sync_timestamp.isoformat(),
            "last_sync_document_count": self.last_sync_document_count,
            "last_sync_duration": self.last_sync_duration,
            "last_sync_status": self.last_sync_status,
            "last_error_message": self.last_error_message,
            "total_synced_documents": self.total_synced_documents,
            "total_sync_operations": self.total_sync_operations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SyncState':
        """Create sync state from dictionary."""
        return cls(
            last_sync_timestamp=datetime.fromisoformat(data["last_sync_timestamp"]),
            last_sync_document_count=data["last_sync_document_count"],
            last_sync_duration=data["last_sync_duration"],
            last_sync_status=data["last_sync_status"],
            last_error_message=data.get("last_error_message"),
            total_synced_documents=data.get("total_synced_documents", 0),
            total_sync_operations=data.get("total_sync_operations", 0)
        )


@dataclass
class ChangeDetectionResult:
    """Represents the result of change detection."""
    
    changed_documents: List[Dict[str, Any]]
    deleted_document_ids: List[str]
    total_changes: int
    detection_timestamp: datetime
    sync_window_start: datetime
    sync_window_end: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert change detection result to dictionary."""
        return {
            "changed_documents": self.changed_documents,
            "deleted_document_ids": self.deleted_document_ids,
            "total_changes": self.total_changes,
            "detection_timestamp": self.detection_timestamp.isoformat(),
            "sync_window_start": self.sync_window_start.isoformat(),
            "sync_window_end": self.sync_window_end.isoformat()
        }


class IncrementalSyncService:
    """
    Service for incremental synchronization between MongoDB and Elasticsearch.
    
    This service detects changes in MongoDB documents and synchronizes only
    the modified documents to Elasticsearch, improving performance and efficiency.
    """
    
    def __init__(
        self,
        mongo_client: MongoClient,
        es_service: ElasticsearchSyncService,
        denorm_service: DenormalizationService,
        results_service: ResultsService
    ):
        """
        Initialize the incremental sync service.
        
        Args:
            mongo_client: MongoDB client instance
            es_service: Elasticsearch sync service instance
            denorm_service: Denormalization service instance
            results_service: Results service instance
        """
        self.mongo_client = mongo_client
        self.es_service = es_service
        self.denorm_service = denorm_service
        self.results_service = results_service
        self.settings = get_settings()
        
        # Initialize collections
        self.llm_analyses_collection = self.mongo_client[
            self.settings.mongodb.database
        ][self.settings.mongodb.llm_analyses_collection]
        
        self.sync_state_collection = self.mongo_client[
            self.settings.mongodb.database
        ]["elasticsearch_sync_state"]
        
        # Initialize sync state
        self.current_sync_state = self._load_sync_state()
        
        logger.info("IncrementalSyncService initialized")
    
    def _load_sync_state(self) -> SyncState:
        """Load the current sync state from MongoDB."""
        try:
            state_doc = self.sync_state_collection.find_one({"_id": "current"})
            if state_doc:
                return SyncState.from_dict(state_doc["state"])
            else:
                # Initialize with default state
                default_state = SyncState(
                    last_sync_timestamp=datetime.now(timezone.utc) - timedelta(days=1),
                    last_sync_document_count=0,
                    last_sync_duration=0.0,
                    last_sync_status="success"
                )
                self._save_sync_state(default_state)
                return default_state
        except Exception as e:
            logger.error(f"Failed to load sync state: {e}")
            # Return default state on error
            return SyncState(
                last_sync_timestamp=datetime.now(timezone.utc) - timedelta(days=1),
                last_sync_document_count=0,
                last_sync_duration=0.0,
                last_sync_status="failed",
                last_error_message=str(e)
            )
    
    def _save_sync_state(self, state: SyncState) -> None:
        """Save the sync state to MongoDB."""
        try:
            self.sync_state_collection.replace_one(
                {"_id": "current"},
                {"_id": "current", "state": state.to_dict()},
                upsert=True
            )
        except Exception as e:
            logger.error(f"Failed to save sync state: {e}")
    
    def _calculate_document_hash(self, document: Dict[str, Any]) -> str:
        """
        Calculate a hash for document change detection.
        
        Args:
            document: MongoDB document
            
        Returns:
            SHA-256 hash of the document content
        """
        # Create a normalized version of the document for hashing
        # Remove MongoDB-specific fields and sort for consistency
        normalized_doc = {
            k: v for k, v in document.items() 
            if k not in ["_id", "__v", "created_at", "updated_at"]
        }
        
        # Convert to sorted JSON string for consistent hashing
        doc_str = json.dumps(normalized_doc, sort_keys=True, default=str)
        return hashlib.sha256(doc_str.encode()).hexdigest()
    
    @sync_circuit_breaker
    async def detect_changes(
        self,
        since_timestamp: Optional[datetime] = None,
        batch_size: int = 100
    ) -> ChangeDetectionResult:
        """
        Detect changes in MongoDB documents since the last sync.
        
        Args:
            since_timestamp: Timestamp to detect changes since (defaults to last sync)
            batch_size: Number of documents to process in each batch
            
        Returns:
            ChangeDetectionResult with detected changes
        """
        start_time = datetime.now(timezone.utc)
        
        # Use provided timestamp or last sync timestamp
        sync_since = since_timestamp or self.current_sync_state.last_sync_timestamp
        
        logger.info(f"Detecting changes since {sync_since}")
        
        try:
            # Query for modified documents
            query = {
                "updated_at": {"$gte": sync_since}
            }
            
            # Get all modified documents
            cursor = self.llm_analyses_collection.find(
                query,
                sort=[("updated_at", 1)]
            ).batch_size(batch_size)
            
            changed_documents = []
            deleted_document_ids = []
            
            # Process documents in batches
            async for doc in self._async_cursor_iter(cursor):
                try:
                    # Validate document structure
                    if self._is_valid_document(doc):
                        changed_documents.append(doc)
                    else:
                        logger.warning(f"Invalid document structure: {doc.get('publication_id', 'unknown')}")
                except Exception as e:
                    logger.error(f"Error processing document {doc.get('publication_id', 'unknown')}: {e}")
            
            # Note: For deleted documents, we would need to implement a different strategy
            # such as maintaining a separate collection of document IDs or using change streams
            # For now, we focus on modified documents
            
            end_time = datetime.now(timezone.utc)
            
            result = ChangeDetectionResult(
                changed_documents=changed_documents,
                deleted_document_ids=deleted_document_ids,
                total_changes=len(changed_documents) + len(deleted_document_ids),
                detection_timestamp=end_time,
                sync_window_start=sync_since,
                sync_window_end=end_time
            )
            
            logger.info(f"Detected {result.total_changes} changes in {(end_time - start_time).total_seconds():.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error detecting changes: {e}")
            raise
    
    def _is_valid_document(self, doc: Dict[str, Any]) -> bool:
        """Check if document has valid structure for sync."""
        required_fields = ["publication_id", "workflow_status"]
        return all(field in doc for field in required_fields)
    
    async def _async_cursor_iter(self, cursor: Cursor):
        """Convert MongoDB cursor to async iterator."""
        for doc in cursor:
            yield doc
            # Small delay to prevent blocking
            await asyncio.sleep(0.001)
    
    @sync_circuit_breaker
    async def sync_changes(
        self,
        changes: ChangeDetectionResult,
        batch_size: int = 50
    ) -> Dict[str, Any]:
        """
        Synchronize detected changes to Elasticsearch.
        
        Args:
            changes: ChangeDetectionResult with detected changes
            batch_size: Number of documents to process in each batch
            
        Returns:
            Sync result with statistics
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info(f"Starting sync of {changes.total_changes} changes")
        
        sync_stats = {
            "total_documents": changes.total_changes,
            "successful_syncs": 0,
            "failed_syncs": 0,
            "errors": [],
            "sync_duration": 0.0,
            "documents_per_second": 0.0
        }
        
        try:
            # Ensure Elasticsearch connection
            await self.es_service._ensure_connection()
            
            # Process changed documents in batches
            for i in range(0, len(changes.changed_documents), batch_size):
                batch = changes.changed_documents[i:i + batch_size]
                
                # Denormalize batch
                denormalized_docs = self.denorm_service.batch_denormalize(batch)
                
                # Index batch in Elasticsearch
                for doc in denormalized_docs:
                    try:
                        await self._index_document(doc)
                        sync_stats["successful_syncs"] += 1
                    except Exception as e:
                        sync_stats["failed_syncs"] += 1
                        error_info = {
                            "publication_id": doc.get("publication_id", "unknown"),
                            "error": str(e)
                        }
                        sync_stats["errors"].append(error_info)
                        logger.error(f"Failed to sync document {doc.get('publication_id', 'unknown')}: {e}")
                
                # Small delay between batches
                await asyncio.sleep(0.1)
            
            # Process deleted documents
            for doc_id in changes.deleted_document_ids:
                try:
                    await self._delete_document(doc_id)
                    sync_stats["successful_syncs"] += 1
                except Exception as e:
                    sync_stats["failed_syncs"] += 1
                    error_info = {
                        "document_id": doc_id,
                        "error": str(e)
                    }
                    sync_stats["errors"].append(error_info)
                    logger.error(f"Failed to delete document {doc_id}: {e}")
            
            end_time = datetime.now(timezone.utc)
            sync_duration = (end_time - start_time).total_seconds()
            
            sync_stats["sync_duration"] = sync_duration
            if sync_duration > 0:
                sync_stats["documents_per_second"] = sync_stats["successful_syncs"] / sync_duration
            
            # Update sync state
            self._update_sync_state(changes.detection_timestamp, sync_stats)
            
            logger.info(f"Sync completed: {sync_stats['successful_syncs']} successful, "
                       f"{sync_stats['failed_syncs']} failed in {sync_duration:.2f}s")
            
            return sync_stats
            
        except Exception as e:
            logger.error(f"Error during sync: {e}")
            sync_stats["errors"].append({"sync_error": str(e)})
            raise
    
    async def _index_document(self, doc: Dict[str, Any]) -> None:
        """Index a single document in Elasticsearch."""
        try:
            await self.es_service.client.index(
                index=self.es_service.config.index,
                id=doc["publication_id"],
                body=doc
            )
        except Exception as e:
            logger.error(f"Failed to index document {doc.get('publication_id', 'unknown')}: {e}")
            raise
    
    async def _delete_document(self, doc_id: str) -> None:
        """Delete a document from Elasticsearch."""
        try:
            await self.es_service.client.delete(
                index=self.es_service.config.index,
                id=doc_id
            )
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            raise
    
    def _update_sync_state(self, sync_timestamp: datetime, sync_stats: Dict[str, Any]) -> None:
        """Update the current sync state."""
        self.current_sync_state.last_sync_timestamp = sync_timestamp
        self.current_sync_state.last_sync_document_count = sync_stats["successful_syncs"]
        self.current_sync_state.last_sync_duration = sync_stats["sync_duration"]
        self.current_sync_state.total_synced_documents += sync_stats["successful_syncs"]
        self.current_sync_state.total_sync_operations += 1
        
        # Determine sync status
        if sync_stats["failed_syncs"] == 0:
            self.current_sync_state.last_sync_status = "success"
            self.current_sync_state.last_error_message = None
        elif sync_stats["successful_syncs"] > 0:
            self.current_sync_state.last_sync_status = "partial"
            self.current_sync_state.last_error_message = f"{sync_stats['failed_syncs']} documents failed to sync"
        else:
            self.current_sync_state.last_sync_status = "failed"
            self.current_sync_state.last_error_message = "All documents failed to sync"
        
        # Save updated state
        self._save_sync_state(self.current_sync_state)
    
    @sync_circuit_breaker
    async def perform_incremental_sync(
        self,
        since_timestamp: Optional[datetime] = None,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Perform a complete incremental sync operation.
        
        Args:
            since_timestamp: Timestamp to sync since (defaults to last sync)
            batch_size: Batch size for processing
            
        Returns:
            Complete sync result with statistics
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info("Starting incremental sync operation")
        
        try:
            # Step 1: Detect changes
            changes = await self.detect_changes(since_timestamp, batch_size)
            
            if changes.total_changes == 0:
                logger.info("No changes detected, sync complete")
                return {
                    "status": "no_changes",
                    "total_changes": 0,
                    "sync_duration": (datetime.now(timezone.utc) - start_time).total_seconds()
                }
            
            # Step 2: Sync changes
            sync_result = await self.sync_changes(changes, batch_size)
            
            # Step 3: Prepare final result
            total_duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            result = {
                "status": "completed",
                "total_changes": changes.total_changes,
                "sync_result": sync_result,
                "total_duration": total_duration,
                "sync_state": self.current_sync_state.to_dict()
            }
            
            logger.info(f"Incremental sync completed in {total_duration:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Incremental sync failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "total_duration": (datetime.now(timezone.utc) - start_time).total_seconds()
            }
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get current sync status and statistics."""
        return {
            "current_sync_state": self.current_sync_state.to_dict(),
            "last_sync_ago": (datetime.now(timezone.utc) - self.current_sync_state.last_sync_timestamp).total_seconds(),
            "total_documents_synced": self.current_sync_state.total_synced_documents,
            "total_sync_operations": self.current_sync_state.total_sync_operations
        }
    
    async def reset_sync_state(self, timestamp: Optional[datetime] = None) -> None:
        """
        Reset the sync state to a specific timestamp.
        
        Args:
            timestamp: Timestamp to reset to (defaults to 24 hours ago)
        """
        reset_time = timestamp or (datetime.now(timezone.utc) - timedelta(hours=24))
        
        new_state = SyncState(
            last_sync_timestamp=reset_time,
            last_sync_document_count=0,
            last_sync_duration=0.0,
            last_sync_status="reset",
            last_error_message=None,
            total_synced_documents=self.current_sync_state.total_synced_documents,
            total_sync_operations=self.current_sync_state.total_sync_operations
        )
        
        self.current_sync_state = new_state
        self._save_sync_state(new_state)
        
        logger.info(f"Sync state reset to {reset_time}")
    
    async def validate_sync_consistency(self) -> Dict[str, Any]:
        """
        Validate consistency between MongoDB and Elasticsearch.
        
        Returns:
            Consistency validation result
        """
        try:
            # Get document counts
            mongo_count = self.llm_analyses_collection.count_documents({})
            es_count = await self.es_service.client.count(index=self.es_service.config.index)
            
            # Get sample documents for comparison
            mongo_sample = list(self.llm_analyses_collection.find().limit(10))
            es_sample = await self.es_service.client.search(
                index=self.es_service.config.index,
                body={"size": 10}
            )
            
            consistency_result = {
                "mongo_document_count": mongo_count,
                "elasticsearch_document_count": es_count["count"],
                "count_difference": mongo_count - es_count["count"],
                "sample_comparison": {
                    "mongo_sample_size": len(mongo_sample),
                    "es_sample_size": len(es_sample["hits"]["hits"])
                },
                "is_consistent": abs(mongo_count - es_count["count"]) <= 5  # Allow small difference
            }
            
            return consistency_result
            
        except Exception as e:
            logger.error(f"Error validating sync consistency: {e}")
            return {
                "error": str(e),
                "is_consistent": False
            } 