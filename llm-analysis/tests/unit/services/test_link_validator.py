"""
Unit tests for the LinkValidator system.

Tests comprehensive link validation and metadata extraction functionality including
GitHub repository validation, general web link validation, caching, rate limiting,
and error handling.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from typing import Dict, Any, List

from src.pub_analysis_agent.services.link_validator import (
    LinkValidator,
    LinkValidationConfig,
    LinkValidationResult,
    GitHubMetadata,
    WebLinkMetadata,
    LinkCache
)


class TestLinkValidationConfig:
    """Test cases for LinkValidationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = LinkValidationConfig()
        
        assert config.timeout == 10.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.rate_limit_delay == 0.5
        assert config.cache_ttl == 3600
        assert config.max_cache_size == 1000
        assert config.github_api_timeout == 15.0
        assert config.web_content_max_size == 1024 * 1024
        assert "PubAnalysisAgent" in config.user_agent
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = LinkValidationConfig(
            timeout=5.0,
            max_retries=5,
            cache_ttl=7200,
            user_agent="CustomAgent/1.0"
        )
        
        assert config.timeout == 5.0
        assert config.max_retries == 5
        assert config.cache_ttl == 7200
        assert config.user_agent == "CustomAgent/1.0"


class TestLinkCache:
    """Test cases for LinkCache."""
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = LinkCache(max_size=100, ttl=300)
        
        assert cache.max_size == 100
        assert cache.ttl == 300
        assert cache.size() == 0
    
    def test_cache_set_and_get(self):
        """Test cache set and get operations."""
        cache = LinkCache()
        
        result = LinkValidationResult(
            url="https://example.com",
            is_valid=True,
            is_accessible=True,
            validation_time=1.0
        )
        
        # Set item in cache
        cache.set("https://example.com", result)
        assert cache.size() == 1
        
        # Get item from cache
        cached_result = cache.get("https://example.com")
        assert cached_result is not None
        assert cached_result.url == "https://example.com"
        assert cached_result.cached is True
    
    def test_cache_expiration(self):
        """Test cache expiration based on TTL."""
        cache = LinkCache(ttl=1)  # 1 second TTL
        
        result = LinkValidationResult(
            url="https://example.com",
            is_valid=True,
            is_accessible=True,
            validation_time=1.0
        )
        
        cache.set("https://example.com", result)
        
        # Should be available immediately
        cached_result = cache.get("https://example.com")
        assert cached_result is not None
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired now
        cached_result = cache.get("https://example.com")
        assert cached_result is None
        assert cache.size() == 0
    
    def test_cache_size_limit(self):
        """Test cache size limitation."""
        cache = LinkCache(max_size=2)
        
        result1 = LinkValidationResult("https://example1.com", True, True, 1.0)
        result2 = LinkValidationResult("https://example2.com", True, True, 1.0)
        result3 = LinkValidationResult("https://example3.com", True, True, 1.0)
        
        cache.set("url1", result1)
        cache.set("url2", result2)
        assert cache.size() == 2
        
        # Adding third item should remove oldest
        cache.set("url3", result3)
        assert cache.size() == 2
        
        # First item should be evicted
        assert cache.get("url1") is None
        assert cache.get("url2") is not None
        assert cache.get("url3") is not None
    
    def test_cache_clear(self):
        """Test cache clear operation."""
        cache = LinkCache()
        
        result = LinkValidationResult("https://example.com", True, True, 1.0)
        cache.set("url", result)
        assert cache.size() == 1
        
        cache.clear()
        assert cache.size() == 0
        assert cache.get("url") is None


class TestLinkValidator:
    """Test cases for LinkValidator functionality."""
    
    @pytest.fixture
    def validation_config(self):
        """Create test validation configuration."""
        return LinkValidationConfig(
            timeout=5.0,
            max_retries=2,
            rate_limit_delay=0.1,
            cache_ttl=300
        )
    
    @pytest.fixture
    def link_validator(self, validation_config):
        """Create LinkValidator instance for testing."""
        validator = LinkValidator(config=validation_config, github_token="test_token")
        # Mock the HTTP client to avoid real network calls
        validator._client = AsyncMock()
        return validator
    
    def test_validator_initialization(self, validation_config):
        """Test LinkValidator initialization."""
        validator = LinkValidator(config=validation_config, github_token="test_token")
        
        assert validator.config.timeout == 5.0
        assert validator.github_token == "test_token"
        assert validator.cache.max_size == 1000  # Default from config
        assert isinstance(validator._client, MagicMock) or hasattr(validator._client, 'get')
    
    def test_is_github_url(self, link_validator):
        """Test GitHub URL detection."""
        assert link_validator._is_github_url("https://github.com/user/repo") is True
        assert link_validator._is_github_url("https://www.github.com/user/repo") is True
        assert link_validator._is_github_url("http://github.com/user/repo") is True
        assert link_validator._is_github_url("https://example.com") is False
        assert link_validator._is_github_url("https://gitlab.com/user/repo") is False
    
    def test_parse_github_url(self, link_validator):
        """Test GitHub URL parsing."""
        # Valid GitHub URLs
        result = link_validator._parse_github_url("https://github.com/owner/repo")
        assert result == ("owner", "repo")
        
        result = link_validator._parse_github_url("https://github.com/owner/repo.git")
        assert result == ("owner", "repo")
        
        result = link_validator._parse_github_url("https://github.com/owner/repo/tree/main")
        assert result == ("owner", "repo")
        
        # Invalid URLs
        result = link_validator._parse_github_url("https://github.com/user")
        assert result is None
        
        result = link_validator._parse_github_url("https://example.com/user/repo")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_validate_github_link_success(self, link_validator):
        """Test successful GitHub link validation."""
        with patch.object(link_validator, '_get_github_metadata') as mock_github_meta:
            # Mock HEAD request success
            mock_response = Mock()
            mock_response.status_code = 200
            link_validator._client.head = AsyncMock(return_value=mock_response)
            
            # Mock GitHub metadata
            github_metadata = GitHubMetadata(
                url="https://github.com/user/repo",
                name="repo",
                full_name="user/repo",
                description="Test repository",
                language="Python",
                stars=150
            )
            mock_github_meta.return_value = github_metadata
            
            result = await link_validator._validate_github_link("https://github.com/user/repo")
            
            assert result.url == "https://github.com/user/repo"
            assert result.is_valid is True
            assert result.is_accessible is True
            assert result.github_metadata is not None
            assert result.github_metadata.stars == 150
            assert result.github_metadata.language == "Python"
    
    @pytest.mark.asyncio
    async def test_validate_github_link_not_found(self, link_validator):
        """Test GitHub link validation for non-existent repository."""
        # Mock HEAD request 404
        mock_response = Mock()
        mock_response.status_code = 404
        link_validator._client.head = AsyncMock(return_value=mock_response)
        
        result = await link_validator._validate_github_link("https://github.com/user/nonexistent")
        
        assert result.url == "https://github.com/user/nonexistent"
        assert result.is_valid is True
        assert result.is_accessible is False
        assert result.github_metadata is None
    
    @pytest.mark.asyncio
    async def test_validate_web_link_success(self, link_validator):
        """Test successful web link validation."""
        with patch.object(link_validator, '_extract_html_metadata') as mock_extract_html:
            # Mock successful HTTP response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {
                "content-type": "text/html; charset=utf-8",
                "content-length": "1024",
                "server": "nginx"
            }
            mock_response.url = "https://example.com"
            link_validator._client.get = AsyncMock(return_value=mock_response)
            
            result = await link_validator._validate_web_link("https://example.com")
            
            assert result.url == "https://example.com"
            assert result.is_valid is True
            assert result.is_accessible is True
            assert result.web_metadata is not None
            assert result.web_metadata.status_code == 200
            assert result.web_metadata.content_type == "text/html; charset=utf-8"
            assert result.web_metadata.server == "nginx"
    
    @pytest.mark.asyncio
    async def test_validate_web_link_http_error(self, link_validator):
        """Test web link validation with HTTP error."""
        with patch.object(link_validator, '_client') as mock_client:
            import httpx
            
            # Mock HTTP 404 error
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.reason_phrase = "Not Found"
            
            mock_client.get.side_effect = httpx.HTTPStatusError(
                "404 Not Found", request=Mock(), response=mock_response
            )
            
            result = await link_validator._validate_web_link("https://example.com/notfound")
            
            assert result.url == "https://example.com/notfound"
            assert result.is_valid is True
            assert result.is_accessible is False
            assert "404" in result.error_message
    
    @pytest.mark.asyncio
    async def test_validate_link_with_cache(self, link_validator):
        """Test link validation with caching."""
        # First validation
        with patch.object(link_validator, '_validate_web_link') as mock_validate:
            mock_result = LinkValidationResult(
                url="https://example.com",
                is_valid=True,
                is_accessible=True,
                validation_time=1.0
            )
            mock_validate.return_value = mock_result
            
            result1 = await link_validator.validate_link("https://example.com")
            assert result1.cached is False
            assert mock_validate.call_count == 1
        
        # Second validation should use cache
        with patch.object(link_validator, '_validate_web_link') as mock_validate:
            result2 = await link_validator.validate_link("https://example.com")
            assert result2.cached is True
            assert mock_validate.call_count == 0  # Should not be called due to cache
    
    @pytest.mark.asyncio
    async def test_validate_links_batch(self, link_validator):
        """Test batch link validation."""
        urls = [
            "https://github.com/user/repo",
            "https://example.com",
            "https://another-site.org"
        ]
        
        with patch.object(link_validator, 'validate_link') as mock_validate:
            # Mock individual validation results
            mock_results = [
                LinkValidationResult(url, True, True, 1.0) for url in urls
            ]
            mock_validate.side_effect = mock_results
            
            results = await link_validator.validate_links_batch(urls, max_concurrent=2)
            
            assert len(results) == 3
            assert all(result.is_valid for result in results)
            assert mock_validate.call_count == 3
    
    @pytest.mark.asyncio
    async def test_validate_links_batch_with_exception(self, link_validator):
        """Test batch validation handling exceptions."""
        urls = ["https://example.com", "https://error-site.com"]
        
        with patch.object(link_validator, 'validate_link') as mock_validate:
            # First URL succeeds, second raises exception
            mock_validate.side_effect = [
                LinkValidationResult("https://example.com", True, True, 1.0),
                Exception("Network error")
            ]
            
            results = await link_validator.validate_links_batch(urls)
            
            assert len(results) == 2
            assert results[0].is_valid is True
            assert results[1].is_valid is False
            assert "Network error" in results[1].error_message
    
    @pytest.mark.asyncio
    async def test_get_github_metadata_success(self, link_validator):
        """Test successful GitHub metadata retrieval."""
        # Mock GitHub API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "html_url": "https://github.com/user/repo",
            "name": "repo",
            "full_name": "user/repo",
            "description": "A test repository",
            "language": "Python",
            "stargazers_count": 250,
            "forks_count": 50,
            "watchers_count": 30,
            "open_issues_count": 5,
            "created_at": "2020-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "topics": ["python", "testing"],
            "license": {"name": "MIT License"},
            "fork": False,
            "private": False,
            "archived": False
        }
        link_validator._github_client.get = AsyncMock(return_value=mock_response)
        
        metadata = await link_validator._get_github_metadata("user", "repo")
        
        assert metadata.name == "repo"
        assert metadata.full_name == "user/repo"
        assert metadata.description == "A test repository"
        assert metadata.language == "Python"
        assert metadata.stars == 250
        assert metadata.forks == 50
        assert metadata.topics == ["python", "testing"]
        assert metadata.license_name == "MIT License"
        assert metadata.is_fork is False
    
    @pytest.mark.asyncio
    async def test_get_github_metadata_api_error(self, link_validator):
        """Test GitHub metadata retrieval with API error."""
        with patch.object(link_validator, '_github_client') as mock_client:
            # Mock GitHub API 404 response
            mock_response = Mock()
            mock_response.status_code = 404
            mock_client.get.return_value = mock_response
            
            metadata = await link_validator._get_github_metadata("user", "nonexistent")
            
            # Should return basic metadata even when API fails
            assert metadata.name == "nonexistent"
            assert metadata.full_name == "user/nonexistent"
            assert metadata.url == "https://github.com/user/nonexistent"
            assert metadata.stars is None
    
    @pytest.mark.asyncio
    async def test_extract_html_metadata(self, link_validator):
        """Test HTML metadata extraction."""
        html_content = b"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Page Title</title>
            <meta name="description" content="This is a test page description">
            <meta name="keywords" content="test, page, html">
            <meta name="author" content="Test Author">
            <link rel="canonical" href="https://example.com/canonical">
            <meta property="og:title" content="OG Test Title">
            <meta property="og:description" content="OG Test Description">
            <meta property="og:type" content="website">
        </head>
        <body>
            <h1>Test Content</h1>
        </body>
        </html>
        """
        
        mock_response = Mock()
        mock_response.content = html_content
        
        metadata = WebLinkMetadata(url="https://example.com")
        
        await link_validator._extract_html_metadata(mock_response, metadata)
        
        assert metadata.title == "Test Page Title"
        assert metadata.description == "This is a test page description"
        assert metadata.meta_keywords == ["test", "page", "html"]
        assert metadata.meta_author == "Test Author"
        assert metadata.canonical_url == "https://example.com/canonical"
        assert metadata.og_title == "OG Test Title"
        assert metadata.og_description == "OG Test Description"
        assert metadata.og_type == "website"
    
    def test_get_cache_stats(self, link_validator):
        """Test cache statistics retrieval."""
        stats = link_validator.get_cache_stats()
        
        assert "cache_size" in stats
        assert "max_cache_size" in stats
        assert "cache_ttl" in stats
        assert isinstance(stats["cache_size"], int)
        assert stats["max_cache_size"] == link_validator.config.max_cache_size
    
    def test_clear_cache(self, link_validator):
        """Test cache clearing."""
        # Add something to cache first
        result = LinkValidationResult("https://example.com", True, True, 1.0)
        link_validator.cache.set("test_url", result)
        assert link_validator.cache.size() > 0
        
        link_validator.clear_cache()
        assert link_validator.cache.size() == 0
    
    @pytest.mark.asyncio
    async def test_health_check(self, link_validator):
        """Test health check functionality."""
        with patch.object(link_validator, 'validate_links_batch') as mock_batch:
            # Mock successful validation results
            mock_results = [
                LinkValidationResult("https://github.com/python/cpython", True, True, 1.0),
                LinkValidationResult("https://httpbin.org/get", True, True, 0.5)
            ]
            mock_batch.return_value = mock_results
            
            health = await link_validator.health_check()
            
            assert health["status"] == "healthy"
            assert health["test_urls_count"] == 2
            assert health["successful_validations"] == 2
            assert health["total_validation_time"] > 0
            assert health["github_token_configured"] is True
    
    @pytest.mark.asyncio
    async def test_health_check_degraded(self, link_validator):
        """Test health check with degraded status."""
        with patch.object(link_validator, 'validate_links_batch') as mock_batch:
            # Mock failed validation results
            mock_results = [
                LinkValidationResult("https://github.com/python/cpython", False, False, 1.0),
                LinkValidationResult("https://httpbin.org/get", False, False, 0.5)
            ]
            mock_batch.return_value = mock_results
            
            health = await link_validator.health_check()
            
            assert health["status"] == "degraded"
            assert health["successful_validations"] == 0
    
    @pytest.mark.asyncio
    async def test_close(self, link_validator):
        """Test proper cleanup of resources."""
        with patch.object(link_validator._client, 'aclose') as mock_client_close:
            with patch.object(link_validator._github_client, 'aclose') as mock_github_close:
                await link_validator.close()
                
                mock_client_close.assert_called_once()
                mock_github_close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__]) 