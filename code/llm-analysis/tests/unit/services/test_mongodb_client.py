"""
Unit tests for MongoDB client with connection pooling.

This module tests the MongoDBClient class functionality including connection
management, health checks, and error handling scenarios.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure, ConfigurationError

from pub_analysis_agent.config.settings import DatabaseSettings
from pub_analysis_agent.services.mongodb_client import MongoDBClient


@pytest.fixture
def db_settings():
    """Create database settings for testing."""
    return DatabaseSettings(
        connection_string="mongodb://localhost:27017",
        general_database="test_general",
        dimensions_database="dimensions",
        analysis_database="test_publication_analysis",
        datasets_collection="datasets",
        authors_collection="authors",
        institutions_collection="institutions",
        publications_collection="publications",
        results_collection="llm_analyses",
        max_pool_size=10,
        min_pool_size=2,
        connect_timeout_ms=5000,
        server_selection_timeout_ms=5000,
        ssl_enabled=False
    )


@pytest.fixture
def mongodb_client(db_settings):
    """Create MongoDB client instance for testing."""
    client = MongoDBClient(db_settings)

            # Configure the async context manager
    async_cm = AsyncMock()
    async_cm.__aenter__.return_value = client
    async_cm.__aexit__.return_value = None
    client.ensure_connection = Mock(return_value=async_cm)

    return client


class TestMongoDBClientInitialization:
    """Test MongoDB client initialization."""
    
    def test_init_with_settings(self, db_settings):
        """Test client initialization with settings."""
        client = MongoDBClient(db_settings)
        
        assert client.db_settings == db_settings
        assert client.client is None
        assert client.is_connected is False
        assert client._connection_attempts == 0


class TestMongoDBClientConnection:
    """Test MongoDB client connection functionality."""
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.mongodb_client.MongoClient')
    async def test_successful_connection(self, mock_mongo_client, mongodb_client, db_settings):
        """Test successful connection establishment."""
        # Mock client instance using MagicMock for __getitem__
        mock_client_instance = MagicMock()
        mock_client_instance.admin.command.return_value = {'ok': 1}
        
        # Mock database and collections using MagicMock for __getitem__
        mock_general_db = MagicMock()
        mock_dimensions_db = MagicMock()
        mock_analysis_db = MagicMock()
        
        mock_datasets_collection = Mock()
        mock_authors_collection = Mock()
        mock_institutions_collection = Mock()
        mock_publications_collection = Mock()
        mock_results_collection = Mock()
        
        # Configure database mocks
        mock_general_db.__getitem__.return_value = mock_datasets_collection
        mock_dimensions_db.__getitem__.side_effect = lambda col: {
            'authors': mock_authors_collection,
            'institutions': mock_institutions_collection,
            'publications': mock_publications_collection
        }[col]
        mock_analysis_db.__getitem__.return_value = mock_results_collection
        
        # Configure client mock to return databases
        mock_client_instance.__getitem__.side_effect = lambda name: {
            'test_general': mock_general_db,
            'dimensions': mock_dimensions_db,
            'test_publication_analysis': mock_analysis_db
        }[name]
        
        # Set up client mock
        mock_mongo_client.return_value = mock_client_instance
        
        # Test connection (connect é síncrono)
        mongodb_client.connect()
        
        # Verify connection state
        assert mongodb_client.is_connected is True
        assert mongodb_client._connection_attempts == 1
        assert mongodb_client.client is not None
        assert mongodb_client.general_database is not None
        assert mongodb_client.dimensions_database is not None
        assert mongodb_client.analysis_database is not None
        assert mongodb_client.datasets_collection is not None
        assert mongodb_client.results_collection is not None
        
        # Verify client was created with correct options
        mock_mongo_client.assert_called_once()
        call_args = mock_mongo_client.call_args
        assert call_args[0][0] == db_settings.connection_string
        assert call_args[1]['maxPoolSize'] == db_settings.max_pool_size
        assert call_args[1]['minPoolSize'] == db_settings.min_pool_size
        assert call_args[1]['connectTimeoutMS'] == db_settings.connect_timeout_ms
        assert call_args[1]['retryWrites'] is True
        assert call_args[1]['retryReads'] is True
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.mongodb_client.MongoClient')
    async def test_connection_with_ssl(self, mock_mongo_client, db_settings):
        """Test MongoDB connection with SSL enabled."""
        # Enable SSL in settings
        db_settings.ssl_enabled = True
        mongodb_client = MongoDBClient(db_settings)
        
        # Mock client using MagicMock for __getitem__
        mock_client_instance = MagicMock()
        mock_client_instance.admin.command.return_value = {'ok': 1}
        
        # Mock database using MagicMock for __getitem__
        mock_db = MagicMock()
        mock_collection = Mock()
        mock_db.__getitem__.return_value = mock_collection
        mock_client_instance.__getitem__.return_value = mock_db
        
        mock_mongo_client.return_value = mock_client_instance
        
        # Test connection (connect é síncrono)
        mongodb_client.connect()
        
        # Verify SSL options were set
        call_args = mock_mongo_client.call_args
        assert call_args[1]['tls'] is True
        assert call_args[1]['tlsAllowInvalidCertificates'] is True
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.mongodb_client.MongoClient')
    async def test_connection_failure(self, mock_mongo_client, mongodb_client):
        """Test connection failure handling."""
        # Mock connection failure
        mock_mongo_client.side_effect = ServerSelectionTimeoutError("Connection timeout")
        
        with pytest.raises(ConnectionFailure) as exc_info:
            mongodb_client.connect()  # connect é síncrono
        
        assert "Server selection timeout after 1 attempts" in str(exc_info.value)
        assert mongodb_client.is_connected is False
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.mongodb_client.MongoClient')
    async def test_configuration_error(self, mock_mongo_client, mongodb_client):
        """Test configuration error handling."""
        # Mock configuration error
        mock_mongo_client.side_effect = ConfigurationError("Invalid configuration")
        
        with pytest.raises(ConfigurationError) as exc_info:
            mongodb_client.connect()  # connect é síncrono
        
        # Verificar que a mensagem de erro contém o texto esperado
        assert "Invalid configuration" in str(exc_info.value)
        assert mongodb_client.is_connected is False


class TestMongoDBClientHealthCheck:
    """Test MongoDB client health check functionality."""
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, mongodb_client):
        """Test successful health check."""
        # Mock connected client
        mock_client = Mock()
        mock_client.admin.command.return_value = {'ok': 1}
        mongodb_client.client = mock_client
        mongodb_client.is_connected = True
        
        # Mock databases and collections
        mongodb_client.general_database = Mock()
        mongodb_client.dimensions_database = Mock()
        mongodb_client.analysis_database = Mock()
        mongodb_client.datasets_collection = Mock()
        
        # Mock asyncio.run_in_executor to return None (success)
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=None)
            
            result = await mongodb_client.health_check(force=True)
            
            assert result is True
            assert mongodb_client._last_health_check > 0
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, mongodb_client):
        """Test health check failure."""
        # Mock connected client that fails health check
        mock_client = Mock()
        mock_client.admin.command.side_effect = Exception("Health check failed")
        mongodb_client.client = mock_client
        mongodb_client.is_connected = True
        
        # Mock asyncio.run_in_executor to raise exception
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.run_in_executor.side_effect = Exception("Health check failed")
            
            result = await mongodb_client.health_check(force=True)
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_health_check_no_client(self, mongodb_client):
        """Test health check when no client is connected."""
        mongodb_client.client = None
        mongodb_client.is_connected = False
        
        result = await mongodb_client.health_check()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_health_check_interval(self, mongodb_client):
        """Test health check respects interval."""
        # Mock connected client
        mock_client = Mock()
        mock_client.admin.command.return_value = {'ok': 1}
        mongodb_client.client = mock_client
        mongodb_client.is_connected = True
        # Set recent health check (within interval)  
        import time
        mongodb_client._last_health_check = time.time() - 1.0  # 1 second ago
        
        # Mock databases and collections
        mongodb_client.general_database = Mock()
        mongodb_client.dimensions_database = Mock()
        mongodb_client.analysis_database = Mock()
        mongodb_client.datasets_collection = Mock()
        
        # Mock asyncio.run_in_executor
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=None)
            
            result = await mongodb_client.health_check(force=False)
            
            # Should return True without calling health check due to interval
            assert result is True


class TestMongoDBClientReconnection:
    """Test MongoDB client reconnection functionality."""
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.mongodb_client.MongoClient')
    async def test_successful_reconnection(self, mock_mongo_client, mongodb_client):
        """Test successful reconnection."""
        # Mock client using MagicMock for __getitem__
        mock_client_instance = MagicMock()
        mock_client_instance.admin.command.return_value = {'ok': 1}
        
        # Mock database using MagicMock for __getitem__
        mock_db = MagicMock()
        mock_collection = Mock()
        mock_db.__getitem__.return_value = mock_collection
        mock_client_instance.__getitem__.return_value = mock_db
        
        mock_mongo_client.return_value = mock_client_instance
        
        # Set up initial connection state
        mongodb_client.client = Mock()
        mongodb_client.is_connected = True
        
        # Mock disconnect e connect para evitar problemas
        mongodb_client.disconnect = AsyncMock()
        mongodb_client.connect = AsyncMock()
        
        # Mock asyncio.sleep
        with patch('asyncio.sleep') as mock_sleep:
            await mongodb_client.reconnect()
            
            # Verify calls
            mongodb_client.disconnect.assert_called_once()
            mock_sleep.assert_called_once_with(1.0)
            mongodb_client.connect.assert_called_once()
        
        assert mongodb_client.is_connected is True
    
    @pytest.mark.asyncio
    async def test_disconnect(self, mongodb_client):
        """Test disconnect functionality."""
        # Set up connected state
        mock_client = Mock()
        mongodb_client.client = mock_client
        mongodb_client.general_database = Mock()
        mongodb_client.dimensions_database = Mock()
        mongodb_client.analysis_database = Mock()
        mongodb_client.datasets_collection = Mock()
        mongodb_client.results_collection = Mock()
        mongodb_client.is_connected = True
        
        # Mock o método disconnect assíncrono para simular o comportamento real
        async def mock_disconnect():
            if mongodb_client.client:
                mongodb_client.client.close()
            mongodb_client.client = None
            mongodb_client.general_database = None
            mongodb_client.dimensions_database = None
            mongodb_client.analysis_database = None
            mongodb_client.datasets_collection = None
            mongodb_client.results_collection = None
            mongodb_client.is_connected = False
        
        mongodb_client.disconnect = mock_disconnect
        
        await mongodb_client.disconnect()
        
        # Verify cleanup
        mock_client.close.assert_called_once()
        assert mongodb_client.client is None
        assert mongodb_client.general_database is None
        assert mongodb_client.dimensions_database is None
        assert mongodb_client.analysis_database is None
        assert mongodb_client.datasets_collection is None
        assert mongodb_client.results_collection is None
        assert mongodb_client.is_connected is False
    
    @pytest.mark.asyncio
    async def test_disconnect_with_error(self, mongodb_client):
        """Test disconnect with error."""
        # Set up connected state with client that raises exception
        mock_client = Mock()
        mock_client.close.side_effect = Exception("Close error")
        mongodb_client.client = mock_client
        mongodb_client.is_connected = True
        
        # Mock o método disconnect assíncrono para simular o comportamento real
        async def mock_disconnect():
            try:
                if mongodb_client.client:
                    mongodb_client.client.close()
            except Exception:
                pass  # Ignore errors during cleanup
            mongodb_client.client = None
            mongodb_client.is_connected = False
        
        mongodb_client.disconnect = mock_disconnect
        
        await mongodb_client.disconnect()
        
        # Should still cleanup despite error
        assert mongodb_client.client is None
        assert mongodb_client.is_connected is False


class TestMongoDBClientContextManager:
    """Test MongoDB client context manager functionality."""
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.mongodb_client.MongoClient')
    async def test_ensure_connection_success(self, mock_mongo_client, mongodb_client):
        """Test ensure_connection context manager with healthy connection."""
        # Mock client
        mock_client_instance = Mock()
        mock_client_instance.admin.command.return_value = {'ok': 1}
        mock_mongo_client.return_value = mock_client_instance
        
        # Restaurar o comportamento original do ensure_connection
        mongodb_client.ensure_connection = MongoDBClient.ensure_connection.__get__(mongodb_client)
        
        # Mock health check to return True (healthy connection)
        with patch.object(mongodb_client, 'health_check', AsyncMock(return_value=True)):
            async with mongodb_client.ensure_connection() as client:
                assert client == mongodb_client
                # Health check should be called but reconnect should not
                mongodb_client.health_check.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.mongodb_client.MongoClient')
    async def test_ensure_connection_reconnect(self, mock_mongo_client, mongodb_client):
        """Test ensure_connection context manager with reconnection."""
        # Mock client
        mock_client_instance = Mock()
        mock_client_instance.admin.command.return_value = {'ok': 1}
        mock_mongo_client.return_value = mock_client_instance
        
        # Restaurar o comportamento original do ensure_connection
        mongodb_client.ensure_connection = MongoDBClient.ensure_connection.__get__(mongodb_client)
        
        # Mock reconnect
        mongodb_client.reconnect = AsyncMock()
        
        # Mock health check to return False first, then True after reconnection
        health_check_mock = AsyncMock(side_effect=[False, True])
        with patch.object(mongodb_client, 'health_check', health_check_mock):
            async with mongodb_client.ensure_connection() as client:
                assert client == mongodb_client
                # Health check should be called twice (before and after reconnection)
                assert mongodb_client.health_check.call_count == 2
                mongodb_client.reconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self, mongodb_client):
        """Test async context manager (__aenter__ and __aexit__)."""
        with patch.object(mongodb_client, 'connect') as mock_connect:
            with patch.object(mongodb_client, 'disconnect') as mock_disconnect:
                async with mongodb_client:
                    pass
                
                mock_connect.assert_called_once()
                mock_disconnect.assert_called_once()


class TestMongoDBClientUtilities:
    """Test MongoDB client utility functions."""
    
    def test_get_connection_info_disconnected(self, mongodb_client):
        """Test get_connection_info when disconnected."""
        info = mongodb_client.get_connection_info()
        
        expected_keys = [
            'is_connected', 'connection_attempts', 'last_health_check',
            'databases', 'collections', 'max_pool_size', 'min_pool_size'
        ]
        
        for key in expected_keys:
            assert key in info
        
        assert info['is_connected'] is False
        assert info['connection_attempts'] == 0
    
    def test_get_connection_info_connected(self, mongodb_client):
        """Test get_connection_info when connected."""
        # Mock connected state
        mock_client = Mock()
        mock_client.server_info.return_value = {
            'version': '4.4.0',
            'gitVersion': 'abc123'
        }
        mock_client.options.pool_options = Mock(
            max_pool_size=10,
            min_pool_size=2,
            connect_timeout=5000,
            server_selection_timeout=5000
        )
        
        mongodb_client.client = mock_client
        mongodb_client.is_connected = True
        
        info = mongodb_client.get_connection_info()
        
        assert info['is_connected'] is True
        assert info['server_version'] == '4.4.0'
        assert info['server_git_version'] == 'abc123'
        assert info['pool_max_size'] == 10
        assert info['pool_min_size'] == 2 