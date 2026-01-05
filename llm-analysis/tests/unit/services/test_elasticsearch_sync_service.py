"""
Unit tests for ElasticsearchSyncService.

This module tests the Elasticsearch synchronization service functionality,
including client setup, index mapping, connection management, and health checks.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from elasticsearch import AsyncElasticsearch, ConnectionTimeout, ConnectionError as ESConnectionError
from elasticsearch.exceptions import RequestError, NotFoundError

from pub_analysis_agent.services.elasticsearch_sync_service import (
    ElasticsearchSyncService,
    ElasticsearchConfig,
    IndexMapping,
    ensure_connection
)


class TestElasticsearchConfig:
    """Test Elasticsearch configuration."""
    
    def test_config_creation_with_defaults(self):
        """Test creating config with default values."""
        config = ElasticsearchConfig(url="http://localhost:9200")
        
        assert config.url == "http://localhost:9200"
        assert config.index == "llm_analyses"
        assert config.username is None
        assert config.password is None
        assert config.ssl_enabled is False
        assert config.verify_certs is True
        assert config.request_timeout == 30
        assert config.max_retries == 3
        assert config.connection_pool_size == 10
    
    def test_config_creation_with_custom_values(self):
        """Test creating config with custom values."""
        config = ElasticsearchConfig(
            url="https://elasticsearch.example.com:9200",
            index="custom_index",
            username="user",
            password="pass",
            api_key="test_api_key",
            ssl_enabled=True,
            verify_certs=False,
            ca_cert_path="/path/to/ca.crt",
            request_timeout=60,
            max_retries=5,
            connection_pool_size=20
        )
        
        assert config.url == "https://elasticsearch.example.com:9200"
        assert config.index == "custom_index"
        assert config.username == "user"
        assert config.password == "pass"
        assert config.api_key == "test_api_key"
        assert config.ssl_enabled is True
        assert config.verify_certs is False
        assert config.ca_cert_path == "/path/to/ca.crt"
        assert config.request_timeout == 60
        assert config.max_retries == 5
        assert config.connection_pool_size == 20


class TestIndexMapping:
    """Test Elasticsearch index mapping configuration."""
    
    def test_index_mapping_creation(self):
        """Test creating index mapping with default configuration."""
        mapping = IndexMapping()
        
        # Test settings
        assert "number_of_shards" in mapping.settings
        assert "number_of_replicas" in mapping.settings
        assert "refresh_interval" in mapping.settings
        assert "analysis" in mapping.settings
        
        # Test analyzers
        analyzers = mapping.settings["analysis"]["analyzer"]
        assert "text_analyzer" in analyzers
        assert "autocomplete_analyzer" in analyzers
        assert "keyword_analyzer" in analyzers
        
        # Test filters
        filters = mapping.settings["analysis"]["filter"]
        assert "autocomplete_filter" in filters
        
        # Test mappings
        assert "properties" in mapping.mappings
        properties = mapping.mappings["properties"]
        
        # Test core fields
        assert "publication_id" in properties
        assert "analysis_id" in properties
        assert "workflow_status" in properties
        assert "overall_confidence" in properties
        assert "created_at" in properties
        assert "updated_at" in properties
        
        # Test nested structures
        assert "publication_metadata" in properties
        assert "dataset_analysis" in properties
        assert "code_extraction" in properties
        assert "link_extraction" in properties
        assert "dataset_joins" in properties
        assert "llm_metadata" in properties
        assert "error_information" in properties
    
    def test_index_mapping_custom_settings(self):
        """Test creating index mapping with custom settings."""
        custom_settings = {
            "number_of_shards": 3,
            "number_of_replicas": 1,
            "refresh_interval": "5s"
        }
        
        mapping = IndexMapping(settings=custom_settings)
        
        assert mapping.settings["number_of_shards"] == 3
        assert mapping.settings["number_of_replicas"] == 1
        assert mapping.settings["refresh_interval"] == "5s"
        # Custom settings replace defaults completely, so analysis won't be present


class TestElasticsearchSyncServiceInitialization:
    """Test ElasticsearchSyncService initialization."""
    
    @patch('pub_analysis_agent.services.elasticsearch_sync_service.get_settings')
    def test_initialization_with_default_config(self, mock_get_settings):
        """Test service initialization with default configuration."""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.elasticsearch.url = "http://localhost:9200"
        mock_settings.elasticsearch.index = "test_index"
        mock_settings.elasticsearch.username = None
        mock_settings.elasticsearch.password = None
        mock_settings.elasticsearch.api_key = None
        mock_settings.elasticsearch.ssl_enabled = False
        mock_settings.elasticsearch.verify_certs = True
        mock_settings.elasticsearch.ca_cert_path = None
        mock_settings.elasticsearch.request_timeout = 30
        mock_settings.elasticsearch.max_retries = 3
        mock_get_settings.return_value = mock_settings
        
        service = ElasticsearchSyncService()
        
        assert service.config.url == "http://localhost:9200"
        assert service.config.index == "test_index"
        assert service.client is None
        assert service._connection_established is False
        assert isinstance(service.index_mapping, IndexMapping)
    
    def test_initialization_with_custom_config(self):
        """Test service initialization with custom configuration."""
        config = ElasticsearchConfig(
            url="https://elasticsearch.example.com:9200",
            index="custom_index",
            username="user",
            password="pass"
        )
        
        service = ElasticsearchSyncService(config=config)
        
        assert service.config.url == "https://elasticsearch.example.com:9200"
        assert service.config.index == "custom_index"
        assert service.config.username == "user"
        assert service.config.password == "pass"
        assert service.client is None
        assert service._connection_established is False


class TestElasticsearchSyncServiceConnection:
    """Test ElasticsearchSyncService connection management."""
    
    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        config = ElasticsearchConfig(url="http://localhost:9200")
        return ElasticsearchSyncService(config=config)
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.elasticsearch_sync_service.AsyncElasticsearch')
    async def test_ensure_connection_success(self, mock_elasticsearch_class, service):
        """Test successful connection establishment."""
        # Mock client
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_elasticsearch_class.return_value = mock_client
        
        await service._ensure_connection()
        
        assert service.client is not None
        assert service._connection_established is True
        mock_client.ping.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.elasticsearch_sync_service.AsyncElasticsearch')
    async def test_ensure_connection_with_auth(self, mock_elasticsearch_class, service):
        """Test connection establishment with authentication."""
        service.config.username = "user"
        service.config.password = "pass"
        
        # Mock client
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_elasticsearch_class.return_value = mock_client
        
        await service._ensure_connection()
        
        # Verify http_auth was passed
        mock_elasticsearch_class.assert_called_once()
        call_args = mock_elasticsearch_class.call_args[1]
        assert call_args["http_auth"] == ("user", "pass")
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.elasticsearch_sync_service.AsyncElasticsearch')
    async def test_ensure_connection_with_api_key(self, mock_elasticsearch_class, service):
        """Test connection establishment with API key authentication."""
        service.config.api_key = "test_api_key"
        
        # Mock client
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_elasticsearch_class.return_value = mock_client
        
        await service._ensure_connection()
        
        # Verify api_key was passed
        mock_elasticsearch_class.assert_called_once()
        call_args = mock_elasticsearch_class.call_args[1]
        assert call_args["api_key"] == "test_api_key"
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.elasticsearch_sync_service.AsyncElasticsearch')
    async def test_ensure_connection_with_ca_cert(self, mock_elasticsearch_class, service):
        """Test connection establishment with CA certificate."""
        service.config.ssl_enabled = True
        service.config.verify_certs = True
        service.config.ca_cert_path = "localhost_http_ca.crt"
        
        # Mock client
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_elasticsearch_class.return_value = mock_client
        
        await service._ensure_connection()
        
        # Verify SSL parameters were passed
        mock_elasticsearch_class.assert_called_once()
        call_args = mock_elasticsearch_class.call_args[1]
        assert call_args["use_ssl"] is True
        assert call_args["verify_certs"] is True
        assert "ca_certs" in call_args
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.elasticsearch_sync_service.AsyncElasticsearch')
    async def test_ensure_connection_with_ssl(self, mock_elasticsearch_class, service):
        """Test connection establishment with SSL."""
        service.config.ssl_enabled = True
        service.config.verify_certs = False
        
        # Mock client
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_elasticsearch_class.return_value = mock_client
        
        await service._ensure_connection()
        
        # Verify SSL parameters were passed
        mock_elasticsearch_class.assert_called_once()
        call_args = mock_elasticsearch_class.call_args[1]
        assert call_args["use_ssl"] is True
        assert call_args["verify_certs"] is False
        assert call_args["ssl_show_warn"] is False
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.elasticsearch_sync_service.AsyncElasticsearch')
    async def test_ensure_connection_ping_failure(self, mock_elasticsearch_class, service):
        """Test connection establishment when ping fails."""
        # Mock client
        mock_client = AsyncMock()
        mock_client.ping.return_value = False
        mock_elasticsearch_class.return_value = mock_client
        
        with pytest.raises(ESConnectionError):
            await service._ensure_connection()
        
        assert service._connection_established is False
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.elasticsearch_sync_service.AsyncElasticsearch')
    async def test_ensure_connection_exception(self, mock_elasticsearch_class, service):
        """Test connection establishment when exception occurs."""
        # Mock client to raise exception
        mock_elasticsearch_class.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception, match="Connection failed"):
            await service._ensure_connection()
        
        assert service._connection_established is False
    
    @pytest.mark.asyncio
    async def test_ensure_connection_already_connected(self, service):
        """Test connection when already established."""
        # Mock existing connection
        service.client = AsyncMock()
        service.client.ping.return_value = True
        service._connection_established = True
        
        await service._ensure_connection()
        
        # Should not create new connection
        assert service.client is not None


class TestElasticsearchSyncServiceIndexOperations:
    """Test ElasticsearchSyncService index operations."""
    
    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        config = ElasticsearchConfig(url="http://localhost:9200")
        return ElasticsearchSyncService(config=config)
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.elasticsearch_sync_service.AsyncElasticsearch')
    async def test_create_index_success(self, mock_elasticsearch_class, service):
        """Test successful index creation."""
        # Mock client
        mock_client = AsyncMock()
        mock_client.indices.exists.return_value = False
        mock_client.indices.create.return_value = {"acknowledged": True}
        service.client = mock_client
        service._connection_established = True
        
        result = await service.create_index()
        
        assert result is True
        mock_client.indices.exists.assert_called_once_with(index=service.config.index)
        mock_client.indices.create.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.elasticsearch_sync_service.AsyncElasticsearch')
    async def test_create_index_already_exists(self, mock_elasticsearch_class, service):
        """Test index creation when index already exists."""
        # Mock client
        mock_client = AsyncMock()
        mock_client.indices.exists.return_value = True
        service.client = mock_client
        service._connection_established = True
        
        result = await service.create_index()
        
        assert result is True
        mock_client.indices.exists.assert_called_once_with(index=service.config.index)
        mock_client.indices.create.assert_not_called()
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.elasticsearch_sync_service.AsyncElasticsearch')
    async def test_create_index_force(self, mock_elasticsearch_class, service):
        """Test index creation with force flag."""
        # Mock client
        mock_client = AsyncMock()
        mock_client.indices.exists.return_value = True
        mock_client.indices.delete.return_value = {"acknowledged": True}
        mock_client.indices.create.return_value = {"acknowledged": True}
        service.client = mock_client
        service._connection_established = True
        
        result = await service.create_index(force=True)
        
        assert result is True
        mock_client.indices.exists.assert_called()
        mock_client.indices.delete.assert_called_once_with(index=service.config.index)
        mock_client.indices.create.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.elasticsearch_sync_service.AsyncElasticsearch')
    async def test_create_index_failure(self, mock_elasticsearch_class, service):
        """Test index creation failure."""
        # Mock client
        mock_client = AsyncMock()
        mock_client.indices.exists.return_value = False
        mock_client.indices.create.side_effect = Exception("Index creation failed")
        service.client = mock_client
        service._connection_established = True
        
        result = await service.create_index()
        
        assert result is False
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.elasticsearch_sync_service.AsyncElasticsearch')
    async def test_get_index_info_exists(self, mock_elasticsearch_class, service):
        """Test getting index info when index exists."""
        # Mock client
        mock_client = AsyncMock()
        mock_client.indices.exists.return_value = True
        mock_client.indices.get_settings.return_value = {
            service.config.index: {"settings": {"index": {"number_of_shards": "1"}}}
        }
        mock_client.indices.get_mapping.return_value = {
            service.config.index: {"mappings": {"properties": {}}}
        }
        mock_client.indices.stats.return_value = {
            "indices": {service.config.index: {"total": {"docs": {"count": 100}}}}
        }
        service.client = mock_client
        service._connection_established = True
        
        result = await service.get_index_info()
        
        assert result["exists"] is True
        assert "settings" in result
        assert "mappings" in result
        assert "stats" in result
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.elasticsearch_sync_service.AsyncElasticsearch')
    async def test_get_index_info_not_exists(self, mock_elasticsearch_class, service):
        """Test getting index info when index doesn't exist."""
        # Mock client
        mock_client = AsyncMock()
        mock_client.indices.exists.return_value = False
        service.client = mock_client
        service._connection_established = True
        
        result = await service.get_index_info()
        
        assert result["exists"] is False
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.elasticsearch_sync_service.AsyncElasticsearch')
    async def test_get_index_info_error(self, mock_elasticsearch_class, service):
        """Test getting index info when error occurs."""
        # Mock client
        mock_client = AsyncMock()
        mock_client.indices.exists.side_effect = Exception("Connection error")
        service.client = mock_client
        service._connection_established = True
        
        result = await service.get_index_info()
        
        assert result["exists"] is False
        assert "error" in result


class TestElasticsearchSyncServiceHealthCheck:
    """Test ElasticsearchSyncService health check functionality."""
    
    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        config = ElasticsearchConfig(url="http://localhost:9200")
        return ElasticsearchSyncService(config=config)
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.elasticsearch_sync_service.AsyncElasticsearch')
    async def test_health_check_healthy(self, mock_elasticsearch_class, service):
        """Test health check when service is healthy."""
        # Mock client
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.cluster.health.return_value = {"status": "green"}
        mock_client.indices.health.return_value = {"status": "green"}
        service.client = mock_client
        service._connection_established = True
        
        result = await service.health_check()
        
        assert result["status"] == "healthy"
        assert result["ping_success"] is True
        assert result["connection_established"] is True
        assert "cluster_health" in result
        assert "index_health" in result
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.elasticsearch_sync_service.AsyncElasticsearch')
    async def test_health_check_unhealthy(self, mock_elasticsearch_class, service):
        """Test health check when service is unhealthy."""
        # Mock client
        mock_client = AsyncMock()
        mock_client.ping.return_value = False
        mock_client.cluster.health.return_value = {"status": "red"}
        mock_client.indices.health.return_value = {"status": "red"}
        service.client = mock_client
        service._connection_established = True
        
        # Call health check directly without decorator
        result = await service._health_check_internal()
        
        assert result["status"] == "unhealthy"
        assert result["ping_success"] is False
        assert result["connection_established"] is True
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.elasticsearch_sync_service.AsyncElasticsearch')
    async def test_health_check_exception(self, mock_elasticsearch_class, service):
        """Test health check when exception occurs."""
        # Mock client
        mock_client = AsyncMock()
        mock_client.ping.side_effect = ESConnectionError("Connection failed")
        service.client = mock_client
        service._connection_established = True
        
        # Call health check directly without decorator
        result = await service._health_check_internal()
        
        assert result["status"] == "unhealthy"
        assert "error" in result
        # When exception occurs, connection_established should be set to False
        assert result["connection_established"] is False


class TestElasticsearchSyncServiceContextManager:
    """Test ElasticsearchSyncService context manager functionality."""
    
    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        config = ElasticsearchConfig(url="http://localhost:9200")
        return ElasticsearchSyncService(config=config)
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.elasticsearch_sync_service.AsyncElasticsearch')
    async def test_context_manager_enter_exit(self, mock_elasticsearch_class, service):
        """Test context manager enter and exit."""
        # Mock client
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_elasticsearch_class.return_value = mock_client
        
        async with service as es_service:
            assert es_service is service
            assert service._connection_established is True
        
        # Verify close was called
        mock_client.close.assert_called_once()
        assert service._connection_established is False
    
    @pytest.mark.asyncio
    async def test_close_connection(self, service):
        """Test closing connection."""
        # Mock client
        mock_client = AsyncMock()
        service.client = mock_client
        service._connection_established = True
        
        await service.close()
        
        mock_client.close.assert_called_once()
        assert service._connection_established is False


class TestEnsureConnectionDecorator:
    """Test the ensure_connection decorator."""
    
    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        config = ElasticsearchConfig(url="http://localhost:9200")
        return ElasticsearchSyncService(config=config)
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.elasticsearch_sync_service.AsyncElasticsearch')
    async def test_ensure_connection_decorator_success(self, mock_elasticsearch_class, service):
        """Test ensure_connection decorator with successful connection."""
        # Mock client
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_elasticsearch_class.return_value = mock_client
        
        # Create a test method with the decorator
        @ensure_connection
        async def test_method(self):
            return "success"
        
        result = await test_method(service)
        
        assert result == "success"
        assert service._connection_established is True
    
    @pytest.mark.asyncio
    @patch('pub_analysis_agent.services.elasticsearch_sync_service.AsyncElasticsearch')
    async def test_ensure_connection_decorator_already_connected(self, mock_elasticsearch_class, service):
        """Test ensure_connection decorator when already connected."""
        # Mock existing connection
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        service.client = mock_client
        service._connection_established = True
        
        # Create a test method with the decorator
        @ensure_connection
        async def test_method(self):
            return "success"
        
        result = await test_method(service)
        
        assert result == "success"
        # Should not create new connection
        mock_elasticsearch_class.assert_not_called() 