"""
Unit tests for ReindexService.

Tests the reindex service functionality including zero-downtime operations,
conflict resolution, and rollback mechanisms.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import pytest

from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import RequestError, ConflictError
from motor.motor_asyncio import AsyncIOMotorClient

from pub_analysis_agent.services.reindex_service import (
    ReindexService,
    ReindexConfig,
    ReindexState,
    ReindexStatus,
    ConflictInfo,
    ConflictResolutionStrategy,
)


class TestReindexConfig:
    """Test ReindexConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ReindexConfig()
        
        assert config.batch_size == 1000
        assert config.scroll_timeout == "5m"
        assert config.max_concurrent_requests == 5
        assert config.conflict_resolution == ConflictResolutionStrategy.TIMESTAMP_BASED
        assert config.enable_zero_downtime is True
        assert config.validate_after_reindex is True
        assert config.enable_rollback is True
        assert config.rollback_threshold == 0.95
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ReindexConfig(
            batch_size=500,
            conflict_resolution=ConflictResolutionStrategy.VERSION_BASED,
            enable_zero_downtime=False,
            rollback_threshold=0.90
        )
        
        assert config.batch_size == 500
        assert config.conflict_resolution == ConflictResolutionStrategy.VERSION_BASED
        assert config.enable_zero_downtime is False
        assert config.rollback_threshold == 0.90


class TestReindexState:
    """Test ReindexState dataclass."""
    
    def test_reindex_state_creation(self):
        """Test ReindexState creation with required fields."""
        start_time = datetime.now()
        state = ReindexState(
            operation_id="test_op",
            status=ReindexStatus.PENDING,
            source_index="source",
            target_index="target",
            alias_name="alias",
            start_time=start_time
        )
        
        assert state.operation_id == "test_op"
        assert state.status == ReindexStatus.PENDING
        assert state.source_index == "source"
        assert state.target_index == "target"
        assert state.alias_name == "alias"
        assert state.start_time == start_time
        assert state.end_time is None
        assert state.total_documents == 0
        assert state.processed_documents == 0
        assert state.failed_documents == 0
        assert state.conflicts_resolved == 0
        assert state.rollback_count == 0
        assert state.error_message is None
        assert state.metadata == {}


class TestConflictInfo:
    """Test ConflictInfo dataclass."""
    
    def test_conflict_info_creation(self):
        """Test ConflictInfo creation."""
        source_time = datetime.now()
        target_time = source_time + timedelta(hours=1)
        
        conflict = ConflictInfo(
            document_id="doc123",
            source_version=1,
            target_version=2,
            source_timestamp=source_time,
            target_timestamp=target_time,
            conflict_type="version_conflict",
            resolution_strategy=ConflictResolutionStrategy.TIMESTAMP_BASED
        )
        
        assert conflict.document_id == "doc123"
        assert conflict.source_version == 1
        assert conflict.target_version == 2
        assert conflict.source_timestamp == source_time
        assert conflict.target_timestamp == target_time
        assert conflict.conflict_type == "version_conflict"
        assert conflict.resolution_strategy == ConflictResolutionStrategy.TIMESTAMP_BASED
        assert conflict.resolved is False


class TestReindexService:
    """Test ReindexService functionality."""
    
    @pytest.fixture
    def mock_es_client(self):
        """Create mock Elasticsearch client."""
        client = AsyncMock(spec=AsyncElasticsearch)
        client.indices = AsyncMock()
        client.bulk = AsyncMock()
        client.count = AsyncMock()
        return client
    
    @pytest.fixture
    def mock_mongo_client(self):
        """Create mock MongoDB client."""
        client = AsyncMock(spec=AsyncIOMotorClient)
        client.get_database = Mock(return_value=AsyncMock())
        return client
    
    @pytest.fixture
    def reindex_service(self, mock_es_client, mock_mongo_client):
        """Create ReindexService instance with mocked clients."""
        return ReindexService(mock_es_client, mock_mongo_client)
    
    @pytest.mark.asyncio
    async def test_create_zero_downtime_reindex_success(self, reindex_service, mock_es_client):
        """Test successful zero-downtime reindex creation."""
        mock_es_client.indices.create.return_value = {"acknowledged": True}
        
        operation_id = await reindex_service.create_zero_downtime_reindex(
            source_index="source_idx",
            target_index="target_idx",
            alias_name="test_alias"
        )
        
        # Verify operation was created
        assert operation_id.startswith("reindex_source_idx_")
        assert operation_id in reindex_service._active_operations
        
        # Verify target index was created
        mock_es_client.indices.create.assert_called_once_with(
            index="target_idx",
            body={},
            ignore=400
        )
        
        # Verify state
        state = reindex_service._active_operations[operation_id]
        assert state.status == ReindexStatus.PENDING
        assert state.source_index == "source_idx"
        assert state.target_index == "target_idx"
        assert state.alias_name == "test_alias"
    
    @pytest.mark.asyncio
    async def test_create_zero_downtime_reindex_with_mapping(self, reindex_service, mock_es_client):
        """Test reindex creation with custom mapping and settings."""
        mock_es_client.indices.create.return_value = {"acknowledged": True}
        
        mapping = {"properties": {"title": {"type": "text"}}}
        settings = {"number_of_shards": 3}
        
        operation_id = await reindex_service.create_zero_downtime_reindex(
            source_index="source_idx",
            target_index="target_idx",
            alias_name="test_alias",
            mapping=mapping,
            settings=settings
        )
        
        # Verify index creation with mapping and settings
        mock_es_client.indices.create.assert_called_once_with(
            index="target_idx",
            body={"mappings": mapping, "settings": settings},
            ignore=400
        )
    
    @pytest.mark.asyncio
    async def test_create_zero_downtime_reindex_duplicate_operation(self, reindex_service):
        """Test creating duplicate reindex operation."""
        # Create first operation
        operation_id1 = await reindex_service.create_zero_downtime_reindex(
            source_index="source_idx",
            target_index="target_idx",
            alias_name="test_alias"
        )
        
        # Set it to in progress
        reindex_service._active_operations[operation_id1].status = ReindexStatus.IN_PROGRESS
        
        # Try to create duplicate operation
        with pytest.raises(ValueError, match="Reindex operation already exists"):
            await reindex_service.create_zero_downtime_reindex(
                source_index="source_idx",
                target_index="target_idx2",
                alias_name="test_alias2"
            )
    
    @pytest.mark.asyncio
    async def test_execute_reindex_success(self, reindex_service, mock_es_client, mock_mongo_client):
        """Test successful reindex execution."""
        # Setup mocks
        mock_es_client.count.return_value = {"count": 100}
        mock_es_client.bulk.return_value = {"errors": False, "items": []}
        
        # Mock MongoDB collection
        mock_collection = AsyncMock()
        mock_collection.count_documents.return_value = 100
        
        # Configure the find method to return documents directly
        mock_collection.find.return_value = [
            {"_id": "doc1", "title": "Test 1"},
            {"_id": "doc2", "title": "Test 2"}
        ]
        mock_mongo_client.get_database.return_value.__getitem__.return_value = mock_collection
        
        # Create operation
        operation_id = await reindex_service.create_zero_downtime_reindex(
            source_index="source_idx",
            target_index="target_idx",
            alias_name="test_alias"
        )
        
        # Execute reindex
        result = await reindex_service.execute_reindex(operation_id, "test_collection")
        
        # Verify results
        assert result.status == ReindexStatus.COMPLETED
        assert result.total_documents == 100
        assert result.processed_documents == 2
        assert result.failed_documents == 0
        
        # Verify bulk operation was called
        mock_es_client.bulk.assert_called()
        
        # Verify alias swap was performed
        mock_es_client.indices.delete_alias.assert_called_once()
        mock_es_client.indices.put_alias.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_reindex_not_found(self, reindex_service):
        """Test executing non-existent reindex operation."""
        with pytest.raises(ValueError, match="Reindex operation.*not found"):
            await reindex_service.execute_reindex("non_existent", "test_collection")
    
    @pytest.mark.asyncio
    async def test_execute_reindex_with_bulk_errors(self, reindex_service, mock_es_client, mock_mongo_client):
        """Test reindex execution with bulk operation errors."""
        # Setup mocks
        mock_es_client.count.return_value = {"count": 100}
        mock_es_client.bulk.return_value = {
            "errors": True,
            "items": [
                {
                    "index": {
                        "_id": "doc1",
                        "error": {"type": "mapper_parsing_exception", "reason": "Invalid mapping"}
                    }
                }
            ]
        }
        
        # Mock MongoDB collection
        mock_collection = AsyncMock()
        mock_collection.count_documents.return_value = 100
        
        # Configure the find method to return documents directly
        mock_collection.find.return_value = [
            {"_id": "doc1", "title": "Test 1"}
        ]
        mock_mongo_client.get_database.return_value.__getitem__.return_value = mock_collection
        
        # Create operation
        operation_id = await reindex_service.create_zero_downtime_reindex(
            source_index="source_idx",
            target_index="target_idx",
            alias_name="test_alias"
        )
        
        # Execute reindex
        result = await reindex_service.execute_reindex(operation_id, "test_collection")
        
        # Verify error handling
        assert result.status == ReindexStatus.COMPLETED
        assert result.failed_documents == 1
    
    @pytest.mark.asyncio
    async def test_execute_reindex_validation_failure(self, reindex_service, mock_es_client, mock_mongo_client):
        """Test reindex execution with validation failure."""
        # Setup mocks
        mock_es_client.count.return_value = {"count": 50}  # Less than threshold
        mock_es_client.bulk.return_value = {"errors": False, "items": []}
        
        # Mock MongoDB collection
        mock_collection = AsyncMock()
        mock_collection.count_documents.return_value = 100
        
        # Configure the find method to return documents directly
        mock_collection.find.return_value = []
        mock_mongo_client.get_database.return_value.__getitem__.return_value = mock_collection
        
        # Create operation
        operation_id = await reindex_service.create_zero_downtime_reindex(
            source_index="source_idx",
            target_index="target_idx",
            alias_name="test_alias"
        )
        
        # Execute reindex with validation failure
        with pytest.raises(ValueError, match="Reindex validation failed"):
            await reindex_service.execute_reindex(operation_id, "test_collection")
    
    @pytest.mark.asyncio
    async def test_resolve_conflicts_timestamp_based(self, reindex_service):
        """Test conflict resolution using timestamp-based strategy."""
        # Create operation
        operation_id = await reindex_service.create_zero_downtime_reindex(
            source_index="source_idx",
            target_index="target_idx",
            alias_name="test_alias"
        )
        
        # Create conflicts
        source_time = datetime.now()
        target_time = source_time - timedelta(hours=1)  # Source is newer
        
        conflicts = [
            ConflictInfo(
                document_id="doc1",
                source_version=1,
                target_version=2,
                source_timestamp=source_time,
                target_timestamp=target_time,
                conflict_type="version_conflict",
                resolution_strategy=ConflictResolutionStrategy.TIMESTAMP_BASED
            )
        ]
        
        # Resolve conflicts
        resolved = await reindex_service.resolve_conflicts(operation_id, conflicts)
        
        # Verify resolution
        assert len(resolved) == 1
        assert resolved[0].resolved is True
        assert reindex_service._active_operations[operation_id].conflicts_resolved == 1
    
    @pytest.mark.asyncio
    async def test_resolve_conflicts_version_based(self, reindex_service):
        """Test conflict resolution using version-based strategy."""
        # Create operation
        operation_id = await reindex_service.create_zero_downtime_reindex(
            source_index="source_idx",
            target_index="target_idx",
            alias_name="test_alias"
        )
        
        # Create conflicts
        conflicts = [
            ConflictInfo(
                document_id="doc1",
                source_version=3,  # Higher version
                target_version=2,
                source_timestamp=datetime.now(),
                target_timestamp=datetime.now(),
                conflict_type="version_conflict",
                resolution_strategy=ConflictResolutionStrategy.VERSION_BASED
            )
        ]
        
        # Resolve conflicts
        resolved = await reindex_service.resolve_conflicts(
            operation_id, conflicts, ConflictResolutionStrategy.VERSION_BASED
        )
        
        # Verify resolution
        assert len(resolved) == 1
        assert resolved[0].resolved is True
    
    @pytest.mark.asyncio
    async def test_rollback_reindex_success(self, reindex_service, mock_es_client):
        """Test successful reindex rollback."""
        # Create operation
        operation_id = await reindex_service.create_zero_downtime_reindex(
            source_index="source_idx",
            target_index="target_idx",
            alias_name="test_alias"
        )
        
        # Perform rollback
        result = await reindex_service.rollback_reindex(operation_id)
        
        # Verify rollback
        assert result is True
        assert reindex_service._active_operations[operation_id].status == ReindexStatus.ROLLED_BACK
        assert reindex_service._active_operations[operation_id].rollback_count == 1
        
        # Verify index deletion and alias restoration
        mock_es_client.indices.delete.assert_called_once_with(
            index="target_idx", ignore=404
        )
        mock_es_client.indices.put_alias.assert_called_once_with(
            index="source_idx", name="test_alias"
        )
    
    @pytest.mark.asyncio
    async def test_rollback_reindex_not_found(self, reindex_service):
        """Test rollback of non-existent operation."""
        with pytest.raises(ValueError, match="Reindex operation.*not found"):
            await reindex_service.rollback_reindex("non_existent")
    
    @pytest.mark.asyncio
    async def test_get_reindex_status(self, reindex_service):
        """Test getting reindex operation status."""
        # Create operation
        operation_id = await reindex_service.create_zero_downtime_reindex(
            source_index="source_idx",
            target_index="target_idx",
            alias_name="test_alias"
        )
        
        # Get status
        status = await reindex_service.get_reindex_status(operation_id)
        
        # Verify status
        assert status is not None
        assert status.operation_id == operation_id
        assert status.status == ReindexStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_get_reindex_status_not_found(self, reindex_service):
        """Test getting status of non-existent operation."""
        status = await reindex_service.get_reindex_status("non_existent")
        assert status is None
    
    @pytest.mark.asyncio
    async def test_list_active_operations(self, reindex_service):
        """Test listing active reindex operations."""
        # Create multiple operations
        operation_id1 = await reindex_service.create_zero_downtime_reindex(
            source_index="source1",
            target_index="target1",
            alias_name="alias1"
        )
        
        operation_id2 = await reindex_service.create_zero_downtime_reindex(
            source_index="source2",
            target_index="target2",
            alias_name="alias2"
        )
        
        # List operations
        operations = await reindex_service.list_active_operations()
        
        # Verify operations
        assert len(operations) == 2
        operation_ids = [op.operation_id for op in operations]
        assert operation_id1 in operation_ids
        assert operation_id2 in operation_ids
    
    @pytest.mark.asyncio
    async def test_execute_reindex_with_query_filter(self, reindex_service, mock_es_client, mock_mongo_client):
        """Test reindex execution with query filter."""
        # Setup mocks
        mock_es_client.count.return_value = {"count": 50}
        mock_es_client.bulk.return_value = {"errors": False, "items": []}
        
        # Mock MongoDB collection
        mock_collection = AsyncMock()
        mock_collection.count_documents.return_value = 50
        
        # Configure the find method to return documents directly
        mock_collection.find.return_value = [
            {"_id": "doc1", "title": "Test 1", "status": "active"}
        ]
        mock_mongo_client.get_database.return_value.__getitem__.return_value = mock_collection
        
        # Create operation
        operation_id = await reindex_service.create_zero_downtime_reindex(
            source_index="source_idx",
            target_index="target_idx",
            alias_name="test_alias"
        )
        
        # Execute reindex with filter
        query_filter = {"status": "active"}
        result = await reindex_service.execute_reindex(
            operation_id, "test_collection", query_filter
        )
        
        # Verify filter was applied
        mock_collection.count_documents.assert_called_with({})
        mock_collection.find.assert_called_with(query_filter)
        
        # Verify results
        assert result.status == ReindexStatus.COMPLETED
        assert result.total_documents == 50
    
    @pytest.mark.asyncio
    async def test_execute_reindex_zero_downtime_disabled(self, reindex_service, mock_es_client, mock_mongo_client):
        """Test reindex execution with zero-downtime disabled."""
        # Create service with zero-downtime disabled
        config = ReindexConfig(enable_zero_downtime=False)
        service = ReindexService(
            reindex_service.es_client,
            reindex_service.mongo_client,
            config
        )
        
        # Setup mocks
        mock_es_client.count.return_value = {"count": 100}
        mock_es_client.bulk.return_value = {"errors": False, "items": []}
        
        # Mock MongoDB collection
        mock_collection = AsyncMock()
        mock_collection.count_documents.return_value = 100
        
        # Create a proper mock cursor that can be awaited
        mock_cursor = AsyncMock()
        mock_cursor.to_list.return_value = [
            {"_id": "doc1", "title": "Test 1"}
        ]
        
        # Configure the find method to return a cursor with skip and limit methods
        mock_find_result = AsyncMock()
        mock_find_result.skip.return_value = mock_find_result
        mock_find_result.limit.return_value = mock_cursor
        mock_collection.find.return_value = mock_find_result
        mock_mongo_client.get_database.return_value.__getitem__.return_value = mock_collection
        
        # Create operation
        operation_id = await service.create_zero_downtime_reindex(
            source_index="source_idx",
            target_index="target_idx",
            alias_name="test_alias"
        )
        
        # Execute reindex
        result = await service.execute_reindex(operation_id, "test_collection")
        
        # Verify no alias swap was performed
        mock_es_client.indices.delete_alias.assert_not_called()
        mock_es_client.indices.put_alias.assert_not_called()
        
        # Verify operation completed
        assert result.status == ReindexStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_execute_reindex_validation_disabled(self, reindex_service, mock_es_client, mock_mongo_client):
        """Test reindex execution with validation disabled."""
        # Create service with validation disabled
        config = ReindexConfig(validate_after_reindex=False)
        service = ReindexService(
            reindex_service.es_client,
            reindex_service.mongo_client,
            config
        )
        
        # Setup mocks
        mock_es_client.bulk.return_value = {"errors": False, "items": []}
        
        # Mock MongoDB collection
        mock_collection = AsyncMock()
        mock_collection.count_documents.return_value = 100
        
        # Create a proper mock cursor that can be awaited
        mock_cursor = AsyncMock()
        mock_cursor.to_list.return_value = [
            {"_id": "doc1", "title": "Test 1"}
        ]
        
        # Configure the find method to return a cursor with skip and limit methods
        mock_find_result = AsyncMock()
        mock_find_result.skip.return_value = mock_find_result
        mock_find_result.limit.return_value = mock_cursor
        mock_collection.find.return_value = mock_find_result
        mock_mongo_client.get_database.return_value.__getitem__.return_value = mock_collection
        
        # Create operation
        operation_id = await service.create_zero_downtime_reindex(
            source_index="source_idx",
            target_index="target_idx",
            alias_name="test_alias"
        )
        
        # Execute reindex
        result = await service.execute_reindex(operation_id, "test_collection")
        
        # Verify no validation was performed
        mock_es_client.count.assert_not_called()
        
        # Verify operation completed
        assert result.status == ReindexStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_execute_reindex_rollback_disabled(self, reindex_service, mock_es_client, mock_mongo_client):
        """Test reindex execution with rollback disabled."""
        # Create service with rollback disabled
        config = ReindexConfig(enable_rollback=False)
        service = ReindexService(
            reindex_service.es_client,
            reindex_service.mongo_client,
            config
        )
        
        # Setup mocks to cause failure
        from elasticsearch import BadRequestError
        
        # Create a simple object with status attribute
        class MockMeta:
            def __init__(self):
                self.status = 400
        
        mock_es_client.bulk.side_effect = BadRequestError("Bulk operation failed", meta=MockMeta(), body={})
        
        # Mock MongoDB collection
        mock_collection = AsyncMock()
        mock_collection.count_documents.return_value = 100
        
        # Configure the find method to return documents directly
        mock_collection.find.return_value = [
            {"_id": "doc1", "title": "Test 1"}
        ]
        mock_mongo_client.get_database.return_value.__getitem__.return_value = mock_collection
        
        # Create operation
        operation_id = await service.create_zero_downtime_reindex(
            source_index="source_idx",
            target_index="target_idx",
            alias_name="test_alias"
        )
        
        # Execute reindex (should fail)
        with pytest.raises(BadRequestError):
            await service.execute_reindex(operation_id, "test_collection")
        
        # Verify no rollback was attempted
        mock_es_client.indices.delete.assert_not_called() 