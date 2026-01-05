"""
Unit tests for IncrementalSyncService.

This module tests the incremental sync service functionality,
including change detection, synchronization, and state management.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from datetime import datetime, timezone, timedelta
from pymongo.cursor import Cursor
from pymongo.errors import PyMongoError

from pub_analysis_agent.services.incremental_sync_service import (
    IncrementalSyncService,
    SyncState,
    ChangeDetectionResult
)
from pub_analysis_agent.services.elasticsearch_sync_service import ElasticsearchSyncService
from pub_analysis_agent.services.denormalization_service import DenormalizationService
from pub_analysis_agent.services.results_service import ResultsService


class TestSyncState:
    """Test SyncState dataclass."""
    
    def test_sync_state_creation(self):
        """Test creating a SyncState instance."""
        timestamp = datetime.now(timezone.utc)
        state = SyncState(
            last_sync_timestamp=timestamp,
            last_sync_document_count=10,
            last_sync_duration=5.5,
            last_sync_status="success"
        )
        
        assert state.last_sync_timestamp == timestamp
        assert state.last_sync_document_count == 10
        assert state.last_sync_duration == 5.5
        assert state.last_sync_status == "success"
        assert state.last_error_message is None
        assert state.total_synced_documents == 0
        assert state.total_sync_operations == 0
    
    def test_sync_state_to_dict(self):
        """Test converting SyncState to dictionary."""
        timestamp = datetime.now(timezone.utc)
        state = SyncState(
            last_sync_timestamp=timestamp,
            last_sync_document_count=10,
            last_sync_duration=5.5,
            last_sync_status="success",
            last_error_message="Test error",
            total_synced_documents=100,
            total_sync_operations=5
        )
        
        state_dict = state.to_dict()
        
        assert state_dict["last_sync_timestamp"] == timestamp.isoformat()
        assert state_dict["last_sync_document_count"] == 10
        assert state_dict["last_sync_duration"] == 5.5
        assert state_dict["last_sync_status"] == "success"
        assert state_dict["last_error_message"] == "Test error"
        assert state_dict["total_synced_documents"] == 100
        assert state_dict["total_sync_operations"] == 5
    
    def test_sync_state_from_dict(self):
        """Test creating SyncState from dictionary."""
        timestamp = datetime.now(timezone.utc)
        state_dict = {
            "last_sync_timestamp": timestamp.isoformat(),
            "last_sync_document_count": 10,
            "last_sync_duration": 5.5,
            "last_sync_status": "success",
            "last_error_message": "Test error",
            "total_synced_documents": 100,
            "total_sync_operations": 5
        }
        
        state = SyncState.from_dict(state_dict)
        
        assert state.last_sync_timestamp == timestamp
        assert state.last_sync_document_count == 10
        assert state.last_sync_duration == 5.5
        assert state.last_sync_status == "success"
        assert state.last_error_message == "Test error"
        assert state.total_synced_documents == 100
        assert state.total_sync_operations == 5


class TestChangeDetectionResult:
    """Test ChangeDetectionResult dataclass."""
    
    def test_change_detection_result_creation(self):
        """Test creating a ChangeDetectionResult instance."""
        timestamp = datetime.now(timezone.utc)
        result = ChangeDetectionResult(
            changed_documents=[{"id": "1"}, {"id": "2"}],
            deleted_document_ids=["3", "4"],
            total_changes=4,
            detection_timestamp=timestamp,
            sync_window_start=timestamp - timedelta(hours=1),
            sync_window_end=timestamp
        )
        
        assert len(result.changed_documents) == 2
        assert len(result.deleted_document_ids) == 2
        assert result.total_changes == 4
        assert result.detection_timestamp == timestamp
    
    def test_change_detection_result_to_dict(self):
        """Test converting ChangeDetectionResult to dictionary."""
        timestamp = datetime.now(timezone.utc)
        result = ChangeDetectionResult(
            changed_documents=[{"id": "1"}],
            deleted_document_ids=["2"],
            total_changes=2,
            detection_timestamp=timestamp,
            sync_window_start=timestamp - timedelta(hours=1),
            sync_window_end=timestamp
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["changed_documents"] == [{"id": "1"}]
        assert result_dict["deleted_document_ids"] == ["2"]
        assert result_dict["total_changes"] == 2
        assert result_dict["detection_timestamp"] == timestamp.isoformat()


class TestIncrementalSyncService:
    """Test IncrementalSyncService functionality."""
    
    @pytest.fixture
    def mock_mongo_client(self):
        """Create a mock MongoDB client."""
        client = MagicMock()
        client.__getitem__.return_value.__getitem__.return_value = MagicMock()
        return client
    
    @pytest.fixture
    def mock_es_service(self):
        """Create a mock Elasticsearch service."""
        service = MagicMock(spec=ElasticsearchSyncService)
        service.client = AsyncMock()
        service.config = MagicMock()
        service.config.index = "test_index"
        return service
    
    @pytest.fixture
    def mock_denorm_service(self):
        """Create a mock denormalization service."""
        service = MagicMock(spec=DenormalizationService)
        service.batch_denormalize.return_value = [
            {"publication_id": "1", "denormalized": True},
            {"publication_id": "2", "denormalized": True}
        ]
        return service
    
    @pytest.fixture
    def mock_results_service(self):
        """Create a mock results service."""
        return MagicMock(spec=ResultsService)
    
    @pytest.fixture
    def service(self, mock_mongo_client, mock_es_service, mock_denorm_service, mock_results_service):
        """Create a service instance for testing."""
        with patch('pub_analysis_agent.services.incremental_sync_service.get_settings') as mock_settings:
            mock_settings.return_value.mongodb.database = "test_db"
            mock_settings.return_value.mongodb.llm_analyses_collection = "llm_analyses"
            
            # Mock the collections
            mock_mongo_client.__getitem__.return_value.__getitem__.side_effect = lambda x: {
                "llm_analyses": MagicMock(),
                "elasticsearch_sync_state": MagicMock()
            }[x]
            
            service = IncrementalSyncService(
                mock_mongo_client,
                mock_es_service,
                mock_denorm_service,
                mock_results_service
            )
            
            # Mock the sync state loading
            service.sync_state_collection.find_one.return_value = None
            service.sync_state_collection.replace_one = MagicMock()
            
            return service
    
    def test_initialization(self, service):
        """Test service initialization."""
        assert service is not None
        assert hasattr(service, 'mongo_client')
        assert hasattr(service, 'es_service')
        assert hasattr(service, 'denorm_service')
        assert hasattr(service, 'results_service')
        assert hasattr(service, 'current_sync_state')
    
    def test_load_sync_state_new(self, service):
        """Test loading sync state when none exists."""
        # Mock find_one to return None (no existing state)
        service.sync_state_collection.find_one.return_value = None
        
        state = service._load_sync_state()
        
        assert state.last_sync_status == "success"
        assert state.last_sync_document_count == 0
        assert state.last_sync_duration == 0.0
        # Should be initialized to 24 hours ago
        assert (datetime.now(timezone.utc) - state.last_sync_timestamp).total_seconds() > 23 * 3600
    
    def test_load_sync_state_existing(self, service):
        """Test loading existing sync state."""
        timestamp = datetime.now(timezone.utc) - timedelta(hours=1)
        existing_state = {
            "_id": "current",
            "state": {
                "last_sync_timestamp": timestamp.isoformat(),
                "last_sync_document_count": 10,
                "last_sync_duration": 5.5,
                "last_sync_status": "success",
                "last_error_message": None,
                "total_synced_documents": 100,
                "total_sync_operations": 5
            }
        }
        
        service.sync_state_collection.find_one.return_value = existing_state
        
        state = service._load_sync_state()
        
        assert state.last_sync_timestamp == timestamp
        assert state.last_sync_document_count == 10
        assert state.last_sync_duration == 5.5
        assert state.last_sync_status == "success"
        assert state.total_synced_documents == 100
        assert state.total_sync_operations == 5
    
    def test_save_sync_state(self, service):
        """Test saving sync state."""
        timestamp = datetime.now(timezone.utc)
        state = SyncState(
            last_sync_timestamp=timestamp,
            last_sync_document_count=10,
            last_sync_duration=5.5,
            last_sync_status="success"
        )
        
        service._save_sync_state(state)
        
        service.sync_state_collection.replace_one.assert_called_once()
        call_args = service.sync_state_collection.replace_one.call_args
        assert call_args[0][0] == {"_id": "current"}
        assert call_args[0][1]["_id"] == "current"
        assert "state" in call_args[0][1]
        assert call_args[1]["upsert"] is True
    
    def test_calculate_document_hash(self, service):
        """Test document hash calculation."""
        doc = {
            "_id": "123",
            "publication_id": "pub_1",
            "title": "Test Document",
            "content": "Test content",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z"
        }
        
        hash1 = service._calculate_document_hash(doc)
        hash2 = service._calculate_document_hash(doc)
        
        # Same document should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hash length
        
        # Different document should produce different hash
        doc2 = doc.copy()
        doc2["title"] = "Different Title"
        hash3 = service._calculate_document_hash(doc2)
        assert hash1 != hash3
    
    def test_is_valid_document(self, service):
        """Test document validation."""
        # Valid document
        valid_doc = {
            "publication_id": "pub_1",
            "workflow_status": "completed"
        }
        assert service._is_valid_document(valid_doc) is True
        
        # Invalid document - missing publication_id
        invalid_doc1 = {
            "workflow_status": "completed"
        }
        assert service._is_valid_document(invalid_doc1) is False
        
        # Invalid document - missing workflow_status
        invalid_doc2 = {
            "publication_id": "pub_1"
        }
        assert service._is_valid_document(invalid_doc2) is False
    
    @pytest.mark.asyncio
    async def test_detect_changes_no_changes(self, service):
        """Test change detection when no changes exist."""
        # Mock cursor with no documents
        mock_cursor = MagicMock(spec=Cursor)
        mock_cursor.batch_size.return_value = mock_cursor
        
        service.llm_analyses_collection.find.return_value = mock_cursor
        
        # Mock async cursor iteration
        async def mock_cursor_iter(cursor):
            return
            yield  # Empty generator
        
        service._async_cursor_iter = mock_cursor_iter
        
        result = await service.detect_changes()
        
        assert result.total_changes == 0
        assert len(result.changed_documents) == 0
        assert len(result.deleted_document_ids) == 0
    
    @pytest.mark.asyncio
    async def test_detect_changes_with_changes(self, service):
        """Test change detection with existing changes."""
        # Mock documents
        mock_docs = [
            {
                "publication_id": "pub_1",
                "workflow_status": "completed",
                "updated_at": datetime.now(timezone.utc)
            },
            {
                "publication_id": "pub_2",
                "workflow_status": "completed",
                "updated_at": datetime.now(timezone.utc)
            }
        ]
        
        # Mock cursor
        mock_cursor = MagicMock(spec=Cursor)
        mock_cursor.batch_size.return_value = mock_cursor
        
        service.llm_analyses_collection.find.return_value = mock_cursor
        
        # Mock async cursor iteration
        async def mock_cursor_iter(cursor):
            for doc in mock_docs:
                yield doc
        
        service._async_cursor_iter = mock_cursor_iter
        
        result = await service.detect_changes()
        
        assert result.total_changes == 2
        assert len(result.changed_documents) == 2
        assert len(result.deleted_document_ids) == 0
        assert result.changed_documents[0]["publication_id"] == "pub_1"
        assert result.changed_documents[1]["publication_id"] == "pub_2"
    
    @pytest.mark.asyncio
    async def test_sync_changes_success(self, service):
        """Test successful synchronization of changes."""
        # Mock change detection result
        changes = ChangeDetectionResult(
            changed_documents=[
                {"publication_id": "pub_1", "content": "test1"},
                {"publication_id": "pub_2", "content": "test2"}
            ],
            deleted_document_ids=[],
            total_changes=2,
            detection_timestamp=datetime.now(timezone.utc),
            sync_window_start=datetime.now(timezone.utc) - timedelta(hours=1),
            sync_window_end=datetime.now(timezone.utc)
        )
        
        # Mock successful indexing
        service.es_service.client.index = AsyncMock()
        
        result = await service.sync_changes(changes)
        
        assert result["total_documents"] == 2
        assert result["successful_syncs"] == 2
        assert result["failed_syncs"] == 0
        assert len(result["errors"]) == 0
        assert result["sync_duration"] > 0
        assert result["documents_per_second"] > 0
    
    @pytest.mark.asyncio
    async def test_sync_changes_with_failures(self, service):
        """Test synchronization with some failures."""
        # Mock change detection result
        changes = ChangeDetectionResult(
            changed_documents=[
                {"publication_id": "pub_1", "content": "test1"},
                {"publication_id": "pub_2", "content": "test2"}
            ],
            deleted_document_ids=[],
            total_changes=2,
            detection_timestamp=datetime.now(timezone.utc),
            sync_window_start=datetime.now(timezone.utc) - timedelta(hours=1),
            sync_window_end=datetime.now(timezone.utc)
        )
        
        # Mock one success, one failure
        service.es_service.client.index = AsyncMock()
        service.es_service.client.index.side_effect = [None, Exception("Index error")]
        
        result = await service.sync_changes(changes)
        
        assert result["total_documents"] == 2
        assert result["successful_syncs"] == 1
        assert result["failed_syncs"] == 1
        assert len(result["errors"]) == 1
        assert result["errors"][0]["publication_id"] == "2"
    
    @pytest.mark.asyncio
    async def test_perform_incremental_sync_no_changes(self, service):
        """Test incremental sync when no changes are detected."""
        # Mock detect_changes to return no changes
        service.detect_changes = AsyncMock(return_value=ChangeDetectionResult(
            changed_documents=[],
            deleted_document_ids=[],
            total_changes=0,
            detection_timestamp=datetime.now(timezone.utc),
            sync_window_start=datetime.now(timezone.utc) - timedelta(hours=1),
            sync_window_end=datetime.now(timezone.utc)
        ))
        
        result = await service.perform_incremental_sync()
        
        assert result["status"] == "no_changes"
        assert result["total_changes"] == 0
        assert result["sync_duration"] > 0
    
    @pytest.mark.asyncio
    async def test_perform_incremental_sync_with_changes(self, service):
        """Test incremental sync with changes."""
        # Mock change detection result
        changes = ChangeDetectionResult(
            changed_documents=[
                {"publication_id": "pub_1", "content": "test1"}
            ],
            deleted_document_ids=[],
            total_changes=1,
            detection_timestamp=datetime.now(timezone.utc),
            sync_window_start=datetime.now(timezone.utc) - timedelta(hours=1),
            sync_window_end=datetime.now(timezone.utc)
        )
        
        # Mock sync result
        sync_result = {
            "total_documents": 1,
            "successful_syncs": 1,
            "failed_syncs": 0,
            "errors": [],
            "sync_duration": 1.0,
            "documents_per_second": 1.0
        }
        
        service.detect_changes = AsyncMock(return_value=changes)
        service.sync_changes = AsyncMock(return_value=sync_result)
        
        result = await service.perform_incremental_sync()
        
        assert result["status"] == "completed"
        assert result["total_changes"] == 1
        assert result["sync_result"] == sync_result
        assert result["total_duration"] > 0
        assert "sync_state" in result
    
    def test_get_sync_status(self, service):
        """Test getting sync status."""
        status = service.get_sync_status()
        
        assert "current_sync_state" in status
        assert "last_sync_ago" in status
        assert "total_documents_synced" in status
        assert "total_sync_operations" in status
        assert isinstance(status["last_sync_ago"], (int, float))
    
    @pytest.mark.asyncio
    async def test_reset_sync_state(self, service):
        """Test resetting sync state."""
        original_timestamp = service.current_sync_state.last_sync_timestamp
        
        await service.reset_sync_state()
        
        # Should be reset to 24 hours ago
        new_timestamp = service.current_sync_state.last_sync_timestamp
        assert (datetime.now(timezone.utc) - new_timestamp).total_seconds() > 23 * 3600
        assert service.current_sync_state.last_sync_status == "reset"
        
        # Verify state was saved
        service.sync_state_collection.replace_one.assert_called()
    
    @pytest.mark.asyncio
    async def test_validate_sync_consistency(self, service):
        """Test sync consistency validation."""
        # Mock MongoDB count
        service.llm_analyses_collection.count_documents.return_value = 100
        
        # Mock Elasticsearch count
        service.es_service.client.count.return_value = {"count": 98}
        
        # Mock sample documents
        service.llm_analyses_collection.find.return_value.limit.return_value = [
            {"id": "1"}, {"id": "2"}
        ]
        
        service.es_service.client.search.return_value = {
            "hits": {
                "hits": [
                    {"_source": {"id": "1"}},
                    {"_source": {"id": "2"}}
                ]
            }
        }
        
        result = await service.validate_sync_consistency()
        
        assert result["mongo_document_count"] == 100
        assert result["elasticsearch_document_count"] == 98
        assert result["count_difference"] == 2
        assert result["is_consistent"] is True  # Difference <= 5
    
    @pytest.mark.asyncio
    async def test_validate_sync_consistency_large_difference(self, service):
        """Test sync consistency validation with large difference."""
        # Mock large difference
        service.llm_analyses_collection.count_documents.return_value = 100
        service.es_service.client.count.return_value = {"count": 90}
        
        service.llm_analyses_collection.find.return_value.limit.return_value = []
        service.es_service.client.search.return_value = {"hits": {"hits": []}}
        
        result = await service.validate_sync_consistency()
        
        assert result["count_difference"] == 10
        assert result["is_consistent"] is False  # Difference > 5
    
    @pytest.mark.asyncio
    async def test_validate_sync_consistency_error(self, service):
        """Test sync consistency validation with error."""
        # Mock error
        service.llm_analyses_collection.count_documents.side_effect = PyMongoError("Connection error")
        
        result = await service.validate_sync_consistency()
        
        assert result["is_consistent"] is False
        assert "error" in result 