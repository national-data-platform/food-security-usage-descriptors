"""
Unit tests for Elasticsearch Error Handling and Retry Mechanisms.

This module tests the error handling, retry logic, dead letter queue,
and sync status tracking functionality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from datetime import datetime, timezone, timedelta
from elasticsearch.exceptions import (
    ConnectionError as ESConnectionError,
    ConnectionTimeout,
    RequestError,
    ConflictError,
    AuthenticationException,
    AuthorizationException,
    SerializationError
)
from pymongo.errors import PyMongoError

from pub_analysis_agent.services.elasticsearch_error_handler import (
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


class TestErrorType:
    """Test ErrorType enum."""
    
    def test_error_types(self):
        """Test all error types are defined."""
        assert ErrorType.CONNECTION.value == "connection"
        assert ErrorType.TIMEOUT.value == "timeout"
        assert ErrorType.VALIDATION.value == "validation"
        assert ErrorType.INDEXING.value == "indexing"
        assert ErrorType.AUTHENTICATION.value == "authentication"
        assert ErrorType.AUTHORIZATION.value == "authorization"
        assert ErrorType.SERIALIZATION.value == "serialization"
        assert ErrorType.UNKNOWN.value == "unknown"


class TestErrorSeverity:
    """Test ErrorSeverity enum."""
    
    def test_error_severities(self):
        """Test all error severities are defined."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"


class TestSyncError:
    """Test SyncError dataclass."""
    
    def test_sync_error_creation(self):
        """Test creating a SyncError instance."""
        timestamp = datetime.now(timezone.utc)
        error = SyncError(
            error_type=ErrorType.CONNECTION,
            severity=ErrorSeverity.HIGH,
            message="Connection failed",
            timestamp=timestamp,
            document_id="test-doc-123",
            operation="index_document"
        )
        
        assert error.error_type == ErrorType.CONNECTION
        assert error.severity == ErrorSeverity.HIGH
        assert error.message == "Connection failed"
        assert error.timestamp == timestamp
        assert error.document_id == "test-doc-123"
        assert error.operation == "index_document"
        assert error.retry_count == 0
        assert error.max_retries == 3
        assert error.recoverable is True
        assert error.context == {}
    
    def test_sync_error_to_dict(self):
        """Test converting SyncError to dictionary."""
        timestamp = datetime.now(timezone.utc)
        error = SyncError(
            error_type=ErrorType.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            message="Request timeout",
            timestamp=timestamp,
            document_id="test-doc-456",
            operation="bulk_index",
            retry_count=2,
            max_retries=5,
            recoverable=False,
            context={"timeout_seconds": 30}
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["error_type"] == "timeout"
        assert error_dict["severity"] == "medium"
        assert error_dict["message"] == "Request timeout"
        assert error_dict["timestamp"] == timestamp.isoformat()
        assert error_dict["document_id"] == "test-doc-456"
        assert error_dict["operation"] == "bulk_index"
        assert error_dict["retry_count"] == 2
        assert error_dict["max_retries"] == 5
        assert error_dict["recoverable"] is False
        assert error_dict["context"] == {"timeout_seconds": 30}


class TestDeadLetterQueueItem:
    """Test DeadLetterQueueItem dataclass."""
    
    def test_dead_letter_queue_item_creation(self):
        """Test creating a DeadLetterQueueItem instance."""
        timestamp = datetime.now(timezone.utc)
        error = SyncError(
            error_type=ErrorType.INDEXING,
            severity=ErrorSeverity.MEDIUM,
            message="Indexing failed",
            timestamp=timestamp
        )
        
        item = DeadLetterQueueItem(
            document_id="test-doc-789",
            document_data={"publication_id": "test-doc-789", "content": "test"},
            error=error,
            failed_at=timestamp,
            retry_attempts=1,
            max_retry_attempts=3
        )
        
        assert item.document_id == "test-doc-789"
        assert item.document_data == {"publication_id": "test-doc-789", "content": "test"}
        assert item.error == error
        assert item.failed_at == timestamp
        assert item.retry_attempts == 1
        assert item.max_retry_attempts == 3
    
    def test_dead_letter_queue_item_to_dict(self):
        """Test converting DeadLetterQueueItem to dictionary."""
        timestamp = datetime.now(timezone.utc)
        error = SyncError(
            error_type=ErrorType.SERIALIZATION,
            severity=ErrorSeverity.HIGH,
            message="Serialization error",
            timestamp=timestamp
        )
        
        item = DeadLetterQueueItem(
            document_id="test-doc-101",
            document_data={"publication_id": "test-doc-101"},
            error=error,
            failed_at=timestamp,
            retry_attempts=2,
            max_retry_attempts=5
        )
        
        item_dict = item.to_dict()
        
        assert item_dict["document_id"] == "test-doc-101"
        assert item_dict["document_data"] == {"publication_id": "test-doc-101"}
        assert item_dict["error"] == error.to_dict()
        assert item_dict["failed_at"] == timestamp.isoformat()
        assert item_dict["retry_attempts"] == 2
        assert item_dict["max_retry_attempts"] == 5


class TestRetryConfig:
    """Test RetryConfig dataclass."""
    
    def test_retry_config_defaults(self):
        """Test RetryConfig default values."""
        config = RetryConfig()
        
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert len(config.retryable_errors) == 4
    
    def test_retry_config_custom(self):
        """Test RetryConfig with custom values."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            max_delay=120.0,
            exponential_base=3.0,
            jitter=False
        )
        
        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.exponential_base == 3.0
        assert config.jitter is False


class TestErrorClassifier:
    """Test ErrorClassifier class."""
    
    def test_classify_connection_error(self):
        """Test classifying connection errors."""
        error = ESConnectionError("Connection error")
        sync_error = ErrorClassifier.classify_error(error)
        
        assert sync_error.error_type == ErrorType.CONNECTION
        assert sync_error.severity == ErrorSeverity.HIGH
        assert sync_error.recoverable is True
        assert "Connection error" in sync_error.message
    
    def test_classify_timeout_error(self):
        """Test classifying timeout errors."""
        error = ConnectionTimeout("Request timeout")
        sync_error = ErrorClassifier.classify_error(error)
        
        assert sync_error.error_type == ErrorType.TIMEOUT
        assert sync_error.severity == ErrorSeverity.MEDIUM
        assert sync_error.recoverable is True
    
    def test_classify_request_error(self):
        """Test classifying request errors."""
        # Create a mock that inherits from RequestError
        class MockRequestError(RequestError):
            def __init__(self):
                self.body = {}
                self.meta = type('MockMeta', (), {'status': 400})()
                self.message = "Invalid request"
        
        error = MockRequestError()
        sync_error = ErrorClassifier.classify_error(error)
        
        assert sync_error.error_type == ErrorType.INDEXING
        assert sync_error.severity == ErrorSeverity.MEDIUM
        assert sync_error.recoverable is True
    
    def test_classify_conflict_error(self):
        """Test classifying conflict errors."""
        # Create a mock that inherits from ConflictError
        class MockConflictError(ConflictError):
            def __init__(self):
                self.body = {}
                self.meta = type('MockMeta', (), {'status': 409})()
                self.message = "Document conflict"
        
        error = MockConflictError()
        sync_error = ErrorClassifier.classify_error(error)
        
        assert sync_error.error_type == ErrorType.INDEXING
        assert sync_error.severity == ErrorSeverity.LOW
        assert sync_error.recoverable is True
    
    def test_classify_authentication_error(self):
        """Test classifying authentication errors."""
        # Create a mock that inherits from AuthenticationException
        class MockAuthError(AuthenticationException):
            def __init__(self):
                self.body = {}
                self.meta = type('MockMeta', (), {'status': 401})()
                self.message = "Invalid credentials"
        
        error = MockAuthError()
        sync_error = ErrorClassifier.classify_error(error)
        
        assert sync_error.error_type == ErrorType.AUTHENTICATION
        assert sync_error.severity == ErrorSeverity.CRITICAL
        assert sync_error.recoverable is False
    
    def test_classify_authorization_error(self):
        """Test classifying authorization errors."""
        # Create a mock that inherits from AuthorizationException
        class MockAuthzError(AuthorizationException):
            def __init__(self):
                self.body = {}
                self.meta = type('MockMeta', (), {'status': 403})()
                self.message = "Insufficient permissions"
        
        error = MockAuthzError()
        sync_error = ErrorClassifier.classify_error(error)
        
        assert sync_error.error_type == ErrorType.AUTHORIZATION
        assert sync_error.severity == ErrorSeverity.CRITICAL
        assert sync_error.recoverable is False
    
    def test_classify_serialization_error(self):
        """Test classifying serialization errors."""
        error = SerializationError("Serialization failed")
        sync_error = ErrorClassifier.classify_error(error)
        
        assert sync_error.error_type == ErrorType.SERIALIZATION
        assert sync_error.severity == ErrorSeverity.MEDIUM
        assert sync_error.recoverable is False
    
    def test_classify_pymongo_error(self):
        """Test classifying PyMongo errors."""
        error = PyMongoError("MongoDB error")
        sync_error = ErrorClassifier.classify_error(error)
        
        assert sync_error.error_type == ErrorType.CONNECTION
        assert sync_error.severity == ErrorSeverity.HIGH
        assert sync_error.recoverable is True
    
    def test_classify_unknown_error(self):
        """Test classifying unknown errors."""
        error = ValueError("Unknown error")
        sync_error = ErrorClassifier.classify_error(error)
        
        assert sync_error.error_type == ErrorType.UNKNOWN
        assert sync_error.severity == ErrorSeverity.MEDIUM
        assert sync_error.recoverable is True


class TestRetryManager:
    """Test RetryManager class."""
    
    @pytest.fixture
    def retry_config(self):
        """Create a test retry configuration."""
        return RetryConfig(
            max_attempts=3,
            base_delay=0.1,  # Short delay for testing
            max_delay=1.0,
            exponential_base=2.0,
            jitter=False
        )
    
    @pytest.fixture
    def retry_manager(self, retry_config):
        """Create a RetryManager instance."""
        return RetryManager(retry_config)
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self, retry_manager):
        """Test successful execution without retries."""
        async def successful_operation():
            return "success"
        
        result = await retry_manager.execute_with_retry(successful_operation)
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_eventual_success(self, retry_manager):
        """Test execution that succeeds after retries."""
        call_count = 0
        
        async def failing_then_successful_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ESConnectionError("Connection failed")
            return "success"
        
        result = await retry_manager.execute_with_retry(failing_then_successful_operation)
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_all_failures(self, retry_manager):
        """Test execution that fails all attempts."""
        async def always_failing_operation():
            raise ESConnectionError("Connection failed")
        
        with pytest.raises(ESConnectionError):
            await retry_manager.execute_with_retry(always_failing_operation)
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_non_retryable_error(self, retry_manager):
        """Test execution with non-retryable error."""
        async def non_retryable_operation():
            raise ValueError("Non-retryable error")
        
        with pytest.raises(ValueError):
            await retry_manager.execute_with_retry(non_retryable_operation)
    
    def test_is_retryable_error(self, retry_manager):
        """Test retryable error detection."""
        assert retry_manager._is_retryable_error(ESConnectionError("test")) is True
        assert retry_manager._is_retryable_error(ConnectionTimeout("test")) is True
        assert retry_manager._is_retryable_error(Mock(spec=RequestError)) is True
        assert retry_manager._is_retryable_error(Mock(spec=ConflictError)) is True
        assert retry_manager._is_retryable_error(Mock(spec=AuthenticationException)) is False
        assert retry_manager._is_retryable_error(ValueError("test")) is False
    
    def test_calculate_delay(self, retry_manager):
        """Test delay calculation."""
        # Test exponential backoff
        delay1 = retry_manager._calculate_delay(0)
        delay2 = retry_manager._calculate_delay(1)
        delay3 = retry_manager._calculate_delay(2)
        
        assert delay1 == 0.1  # base_delay
        assert delay2 == 0.2  # base_delay * 2^1
        assert delay3 == 0.4  # base_delay * 2^2
    
    def test_calculate_delay_with_jitter(self):
        """Test delay calculation with jitter."""
        config = RetryConfig(jitter=True, base_delay=1.0)
        retry_manager = RetryManager(config)
        
        delay = retry_manager._calculate_delay(0)
        # Should be base_delay + some jitter
        assert 1.0 <= delay <= 1.1


class TestDeadLetterQueue:
    """Test DeadLetterQueue class."""
    
    @pytest.fixture
    def mock_collection(self):
        """Create a mock MongoDB collection."""
        return AsyncMock()
    
    @pytest.fixture
    def dead_letter_queue(self, mock_collection):
        """Create a DeadLetterQueue instance."""
        return DeadLetterQueue(mock_collection)
    
    @pytest.mark.asyncio
    async def test_add_failed_document(self, dead_letter_queue, mock_collection):
        """Test adding a failed document to the queue."""
        error = SyncError(
            error_type=ErrorType.INDEXING,
            severity=ErrorSeverity.MEDIUM,
            message="Indexing failed",
            timestamp=datetime.now(timezone.utc)
        )
        
        await dead_letter_queue.add_failed_document(
            document_id="test-doc-123",
            document_data={"publication_id": "test-doc-123"},
            error=error
        )
        
        mock_collection.replace_one.assert_called_once()
        call_args = mock_collection.replace_one.call_args
        assert call_args[0][0] == {"document_id": "test-doc-123"}
        assert call_args[1]["upsert"] is True
    
    @pytest.mark.asyncio
    async def test_get_failed_documents(self, dead_letter_queue, mock_collection):
        """Test getting failed documents from the queue."""
        # Mock the documents that would be returned
        mock_documents = [
            {
                "document_id": "test-doc-123",
                "document_data": {"publication_id": "test-doc-123"},
                "error": {
                    "error_type": "indexing",
                    "severity": "medium",
                    "message": "Indexing failed",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "retry_count": 0,
                    "max_retries": 3,
                    "recoverable": True,
                    "context": {}
                },
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "retry_attempts": 1,
                "max_retry_attempts": 3
            }
        ]
        
        # Mock the entire chain properly
        mock_collection.find = MagicMock()
        mock_collection.find.return_value.sort.return_value.limit.return_value.to_list = AsyncMock(return_value=mock_documents)
        
        documents = await dead_letter_queue.get_failed_documents(limit=10)
        
        assert len(documents) == 1
        assert documents[0].document_id == "test-doc-123"
        assert documents[0].error.error_type == ErrorType.INDEXING
    
    @pytest.mark.asyncio
    async def test_get_failed_documents_with_error_type_filter(self, dead_letter_queue, mock_collection):
        """Test getting failed documents with error type filter."""
        mock_cursor = AsyncMock()
        mock_cursor.to_list.return_value = []
        mock_collection.find.return_value.sort.return_value.limit.return_value = mock_cursor
        
        await dead_letter_queue.get_failed_documents(
            limit=10,
            error_type=ErrorType.CONNECTION
        )
        
        # Check that the query includes error type filter
        mock_collection.find.assert_called_with({"error.error_type": "connection"})
    
    @pytest.mark.asyncio
    async def test_retry_failed_document_success(self, dead_letter_queue, mock_collection):
        """Test successful retry of a failed document."""
        # Mock document in queue
        mock_collection.find_one.return_value = {
            "document_id": "test-doc-123",
            "document_data": {"publication_id": "test-doc-123"},
            "error": {
                "error_type": "indexing",
                "severity": "medium",
                "message": "Indexing failed",
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "retry_attempts": 1,
            "max_retry_attempts": 3
        }
        
        async def successful_operation(document_data):
            return True
        
        result = await dead_letter_queue.retry_failed_document(
            document_id="test-doc-123",
            operation=successful_operation
        )
        
        assert result is True
        mock_collection.delete_one.assert_called_once_with({"document_id": "test-doc-123"})
    
    @pytest.mark.asyncio
    async def test_retry_failed_document_failure(self, dead_letter_queue, mock_collection):
        """Test failed retry of a failed document."""
        # Mock document in queue
        mock_collection.find_one.return_value = {
            "document_id": "test-doc-123",
            "document_data": {"publication_id": "test-doc-123"},
            "error": {
                "error_type": "indexing",
                "severity": "medium",
                "message": "Indexing failed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "retry_count": 1,
                "max_retries": 3,
                "recoverable": True,
                "context": {}
            },
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "retry_attempts": 1,
            "max_retry_attempts": 3
        }
        
        async def failing_operation(document_data):
            raise ValueError("Operation failed")
        
        result = await dead_letter_queue.retry_failed_document(
            document_id="test-doc-123",
            operation=failing_operation
        )
        
        assert result is False
        mock_collection.replace_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_queue_stats(self, dead_letter_queue, mock_collection):
        """Test getting dead letter queue statistics."""
        # Mock aggregation results
        mock_collection.count_documents = AsyncMock(return_value=5)
        
        # Create proper mock for aggregate
        mock_collection.aggregate = MagicMock()
        mock_collection.aggregate.return_value.to_list = AsyncMock(side_effect=[
            [{"_id": "connection", "count": 2}, {"_id": "indexing", "count": 3}],
            [{"_id": "high", "count": 1}, {"_id": "medium", "count": 4}]
        ])
        
        stats = await dead_letter_queue.get_queue_stats()
        
        assert stats["total_failed_documents"] == 5
        assert stats["error_type_distribution"] == {"connection": 2, "indexing": 3}
        assert stats["severity_distribution"] == {"high": 1, "medium": 4}


class TestSyncStatusTracker:
    """Test SyncStatusTracker class."""
    
    @pytest.fixture
    def mock_collection(self):
        """Create a mock MongoDB collection."""
        return AsyncMock()
    
    @pytest.fixture
    def status_tracker(self, mock_collection):
        """Create a SyncStatusTracker instance."""
        return SyncStatusTracker(mock_collection)
    
    @pytest.mark.asyncio
    async def test_record_sync_start(self, status_tracker, mock_collection):
        """Test recording sync start."""
        metadata = {"batch_size": 100, "source": "mongodb"}
        
        await status_tracker.record_sync_start("sync-123", metadata)
        
        mock_collection.insert_one.assert_called_once()
        call_args = mock_collection.insert_one.call_args[0][0]
        assert call_args["sync_id"] == "sync-123"
        assert call_args["status"] == "started"
        assert call_args["metadata"] == metadata
        assert "metrics" in call_args
    
    @pytest.mark.asyncio
    async def test_record_sync_progress(self, status_tracker, mock_collection):
        """Test recording sync progress."""
        await status_tracker.record_sync_progress(
            sync_id="sync-123",
            documents_processed=50,
            documents_succeeded=45,
            documents_failed=5
        )
        
        mock_collection.update_one.assert_called_once()
        call_args = mock_collection.update_one.call_args
        assert call_args[0][0] == {"sync_id": "sync-123"}
        assert "$set" in call_args[0][1]
    
    @pytest.mark.asyncio
    async def test_record_sync_error(self, status_tracker, mock_collection):
        """Test recording sync error."""
        error = SyncError(
            error_type=ErrorType.CONNECTION,
            severity=ErrorSeverity.HIGH,
            message="Connection failed",
            timestamp=datetime.now(timezone.utc)
        )
        
        await status_tracker.record_sync_error("sync-123", error)
        
        mock_collection.update_one.assert_called_once()
        call_args = mock_collection.update_one.call_args
        assert call_args[0][0] == {"sync_id": "sync-123"}
        assert "$push" in call_args[0][1]
        assert "$set" in call_args[0][1]
    
    @pytest.mark.asyncio
    async def test_record_sync_completion(self, status_tracker, mock_collection):
        """Test recording sync completion."""
        final_metrics = {
            "documents_processed": 100,
            "documents_succeeded": 95,
            "documents_failed": 5,
            "start_time": datetime.now(timezone.utc).isoformat()
        }
        
        await status_tracker.record_sync_completion(
            sync_id="sync-123",
            status="completed",
            final_metrics=final_metrics
        )
        
        mock_collection.update_one.assert_called_once()
        call_args = mock_collection.update_one.call_args
        assert call_args[0][0] == {"sync_id": "sync-123"}
        assert call_args[0][1]["$set"]["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_get_sync_status(self, status_tracker, mock_collection):
        """Test getting sync status."""
        mock_status = {
            "sync_id": "sync-123",
            "status": "in_progress",
            "metrics": {"documents_processed": 50}
        }
        mock_collection.find_one.return_value = mock_status
        
        status = await status_tracker.get_sync_status("sync-123")
        
        assert status == mock_status
        mock_collection.find_one.assert_called_once_with({"sync_id": "sync-123"})
    
    @pytest.mark.asyncio
    async def test_get_recent_syncs(self, status_tracker, mock_collection):
        """Test getting recent syncs."""
        mock_syncs = [
            {"sync_id": "sync-123", "status": "completed"},
            {"sync_id": "sync-124", "status": "failed"}
        ]
        
        # Mock the entire chain properly
        mock_collection.find = MagicMock()
        mock_collection.find.return_value.sort.return_value.limit.return_value.to_list = AsyncMock(return_value=mock_syncs)
        
        syncs = await status_tracker.get_recent_syncs(limit=5)
        
        assert syncs == mock_syncs
        mock_collection.find.assert_called_once()


class TestErrorHandlingDecorator:
    """Test the with_error_handling decorator."""
    
    @pytest.mark.asyncio
    async def test_with_error_handling_success(self):
        """Test decorator with successful operation."""
        @with_error_handling()
        async def test_operation():
            return "success"
        
        result = await test_operation()
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_with_error_handling_retryable_error(self):
        """Test decorator with retryable error."""
        call_count = 0
        
        @with_error_handling(RetryConfig(max_attempts=2, base_delay=0.01))
        async def test_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ESConnectionError("Connection failed")
            return "success"
        
        result = await test_operation()
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_with_error_handling_non_retryable_error(self):
        """Test decorator with non-retryable error."""
        @with_error_handling()
        async def test_operation():
            raise ValueError("Non-retryable error")
        
        with pytest.raises(ValueError):
            await test_operation()
    
    @pytest.mark.asyncio
    async def test_with_error_handling_with_context(self):
        """Test decorator with operation context."""
        class TestService:
            def __init__(self):
                self.error_classifier = ErrorClassifier()
                self.retry_manager = RetryManager(RetryConfig())
                self.status_tracker = AsyncMock()
                self.current_sync_id = "test-sync-123"
        
        service = TestService()
        
        @with_error_handling()
        async def test_operation(self, document_data):
            raise ValueError("Indexing failed")
        
        with pytest.raises(ValueError):
            await test_operation(service, {"publication_id": "test-doc-123"})
        
        # Verify error was recorded
        service.status_tracker.record_sync_error.assert_called_once() 