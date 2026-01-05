"""
Link Validation and Metadata Extraction System.

This module provides comprehensive link validation and metadata extraction capabilities
for GitHub repositories, external links, and other web resources. Features include
async HTTP requests, retry mechanisms, rate limiting, caching, and detailed metadata
extraction for academic publication analysis.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import urlparse, urljoin
from pathlib import Path
import re

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class LinkValidationConfig:
    """Configuration for link validation."""
    timeout: float = 10.0
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_delay: float = 0.5
    cache_ttl: int = 3600  # Cache TTL in seconds (1 hour)
    max_cache_size: int = 1000
    github_api_timeout: float = 15.0
    web_content_max_size: int = 1024 * 1024  # 1MB limit for web content
    user_agent: str = "PubAnalysisAgent/1.0 (Academic Research Tool)"


@dataclass
class GitHubMetadata:
    """GitHub repository metadata."""
    url: str
    name: str
    full_name: str
    description: Optional[str] = None
    language: Optional[str] = None
    stars: Optional[int] = None
    forks: Optional[int] = None
    watchers: Optional[int] = None
    open_issues: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    pushed_at: Optional[str] = None
    size: Optional[int] = None
    default_branch: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    license_name: Optional[str] = None
    is_fork: bool = False
    is_private: bool = False
    archived: bool = False
    disabled: bool = False
    has_wiki: bool = False
    has_pages: bool = False
    has_downloads: bool = False
    has_issues: bool = False


@dataclass
class WebLinkMetadata:
    """Web link metadata."""
    url: str
    title: Optional[str] = None
    description: Optional[str] = None
    domain: str = ""
    content_type: str = ""
    status_code: int = 0
    is_accessible: bool = False
    final_url: str = ""  # After redirects
    response_time: float = 0.0
    content_length: Optional[int] = None
    last_modified: Optional[str] = None
    server: Optional[str] = None
    meta_keywords: List[str] = field(default_factory=list)
    meta_author: Optional[str] = None
    canonical_url: Optional[str] = None
    og_title: Optional[str] = None
    og_description: Optional[str] = None
    og_type: Optional[str] = None


@dataclass
class LinkValidationResult:
    """Result of link validation and metadata extraction."""
    url: str
    is_valid: bool
    is_accessible: bool
    validation_time: float
    error_message: Optional[str] = None
    github_metadata: Optional[GitHubMetadata] = None
    web_metadata: Optional[WebLinkMetadata] = None
    cached: bool = False


class LinkCache:
    """Simple in-memory cache for link validation results."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of cached items
            ttl: Time to live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, Tuple[LinkValidationResult, float]] = {}
    
    def get(self, url: str) -> Optional[LinkValidationResult]:
        """Get cached result if valid."""
        if url not in self._cache:
            return None
        
        result, timestamp = self._cache[url]
        if time.time() - timestamp > self.ttl:
            del self._cache[url]
            return None
        
        # Mark as cached
        result.cached = True
        return result
    
    def set(self, url: str, result: LinkValidationResult) -> None:
        """Cache validation result."""
        # Remove oldest entries if at capacity
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        
        result.cached = False
        self._cache[url] = (result, time.time())
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


class LinkValidator:
    """
    Comprehensive link validation and metadata extraction system.
    
    Provides async validation of links with metadata extraction for:
    - GitHub repositories (using GitHub API)
    - General web links (using HTTP requests and HTML parsing)
    - Support for timeout handling, retries, rate limiting, and caching
    """
    
    def __init__(
        self,
        config: Optional[LinkValidationConfig] = None,
        github_token: Optional[str] = None
    ):
        """
        Initialize LinkValidator.
        
        Args:
            config: Validation configuration
            github_token: Optional GitHub API token for higher rate limits
        """
        self.config = config or LinkValidationConfig()
        self.github_token = github_token
        self.cache = LinkCache(self.config.max_cache_size, self.config.cache_ttl)
        self._rate_limiter = asyncio.Semaphore(10)  # Max 10 concurrent requests
        self._github_rate_limiter = asyncio.Semaphore(5)  # Max 5 concurrent GitHub API calls
        
        # Initialize HTTP client with appropriate headers
        self._client = httpx.AsyncClient(
            timeout=self.config.timeout,
            headers={
                "User-Agent": self.config.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            },
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        
        # GitHub API client
        github_headers = {"Accept": "application/vnd.github.v3+json"}
        if self.github_token:
            github_headers["Authorization"] = f"token {self.github_token}"
        
        self._github_client = httpx.AsyncClient(
            timeout=self.config.github_api_timeout,
            headers=github_headers,
            base_url="https://api.github.com"
        )
        
        logger.info("LinkValidator initialized with caching and rate limiting")
    
    async def close(self) -> None:
        """Close HTTP clients."""
        await self._client.aclose()
        await self._github_client.aclose()
    
    async def validate_link(self, url: str, force_refresh: bool = False) -> LinkValidationResult:
        """
        Validate a single link and extract metadata.
        
        Args:
            url: URL to validate
            force_refresh: Skip cache and force fresh validation
            
        Returns:
            LinkValidationResult with validation status and metadata
        """
        # Check cache first (unless forced refresh)
        if not force_refresh:
            cached_result = self.cache.get(url)
            if cached_result:
                logger.debug(f"Cache hit for URL: {url}")
                return cached_result
        
        start_time = time.time()
        
        try:
            # Determine if this is a GitHub URL
            if self._is_github_url(url):
                result = await self._validate_github_link(url)
            else:
                result = await self._validate_web_link(url)
            
            result.validation_time = time.time() - start_time
            
            # Cache the result
            self.cache.set(url, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating link {url}: {e}")
            result = LinkValidationResult(
                url=url,
                is_valid=False,
                is_accessible=False,
                validation_time=time.time() - start_time,
                error_message=str(e)
            )
            self.cache.set(url, result)
            return result
    
    async def validate_links_batch(
        self,
        urls: List[str],
        max_concurrent: int = 5
    ) -> List[LinkValidationResult]:
        """
        Validate multiple links concurrently.
        
        Args:
            urls: List of URLs to validate
            max_concurrent: Maximum concurrent validations
            
        Returns:
            List of LinkValidationResult objects
        """
        if not urls:
            return []
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def validate_with_semaphore(url: str) -> LinkValidationResult:
            async with semaphore:
                await asyncio.sleep(self.config.rate_limit_delay)
                return await self.validate_link(url)
        
        try:
            results = await asyncio.gather(
                *[validate_with_semaphore(url) for url in urls],
                return_exceptions=True
            )
            
            # Convert exceptions to error results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(LinkValidationResult(
                        url=urls[i],
                        is_valid=False,
                        is_accessible=False,
                        validation_time=0.0,
                        error_message=str(result)
                    ))
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in batch validation: {e}")
            return [
                LinkValidationResult(
                    url=url,
                    is_valid=False,
                    is_accessible=False,
                    validation_time=0.0,
                    error_message=str(e)
                )
                for url in urls
            ]
    
    def _is_github_url(self, url: str) -> bool:
        """Check if URL is a GitHub repository URL."""
        parsed = urlparse(url.lower())
        return parsed.netloc in ["github.com", "www.github.com"]
    
    async def _validate_github_link(self, url: str) -> LinkValidationResult:
        """
        Validate GitHub repository and extract metadata using GitHub API.
        
        Args:
            url: GitHub repository URL
            
        Returns:
            LinkValidationResult with GitHub metadata
        """
        async with self._github_rate_limiter:
            # Extract owner and repo from URL
            github_info = self._parse_github_url(url)
            if not github_info:
                return LinkValidationResult(
                    url=url,
                    is_valid=False,
                    is_accessible=False,
                    validation_time=0.0,
                    error_message="Invalid GitHub URL format"
                )
            
            owner, repo = github_info
            
            try:
                # First, check if repository exists with a simple HEAD request
                head_response = await self._client.head(url)
                is_accessible = head_response.status_code == 200
                
                # If accessible, get detailed metadata from GitHub API
                github_metadata = None
                if is_accessible:
                    github_metadata = await self._get_github_metadata(owner, repo)
                
                return LinkValidationResult(
                    url=url,
                    is_valid=True,
                    is_accessible=is_accessible,
                    validation_time=0.0,  # Will be set by caller
                    github_metadata=github_metadata
                )
                
            except httpx.HTTPStatusError as e:
                return LinkValidationResult(
                    url=url,
                    is_valid=True,
                    is_accessible=False,
                    validation_time=0.0,
                    error_message=f"HTTP {e.response.status_code}: {e.response.reason_phrase}"
                )
            except Exception as e:
                return LinkValidationResult(
                    url=url,
                    is_valid=False,
                    is_accessible=False,
                    validation_time=0.0,
                    error_message=str(e)
                )
    
    async def _validate_web_link(self, url: str) -> LinkValidationResult:
        """
        Validate general web link and extract metadata.
        
        Args:
            url: Web URL to validate
            
        Returns:
            LinkValidationResult with web metadata
        """
        async with self._rate_limiter:
            retries = 0
            last_exception = None
            
            while retries <= self.config.max_retries:
                try:
                    response = await self._client.get(url)
                    
                    # Extract basic metadata
                    web_metadata = WebLinkMetadata(
                        url=url,
                        domain=urlparse(url).netloc,
                        content_type=response.headers.get("content-type", ""),
                        status_code=response.status_code,
                        is_accessible=response.status_code == 200,
                        final_url=str(response.url),
                        content_length=int(response.headers.get("content-length", 0)) or None,
                        last_modified=response.headers.get("last-modified"),
                        server=response.headers.get("server")
                    )
                    
                    # If it's HTML content, extract additional metadata
                    if (response.status_code == 200 and 
                        "text/html" in web_metadata.content_type.lower()):
                        await self._extract_html_metadata(response, web_metadata)
                    
                    return LinkValidationResult(
                        url=url,
                        is_valid=True,
                        is_accessible=response.status_code == 200,
                        validation_time=0.0,  # Will be set by caller
                        web_metadata=web_metadata
                    )
                    
                except httpx.HTTPStatusError as e:
                    if retries == self.config.max_retries:
                        return LinkValidationResult(
                            url=url,
                            is_valid=True,
                            is_accessible=False,
                            validation_time=0.0,
                            error_message=f"HTTP {e.response.status_code}: {e.response.reason_phrase}",
                            web_metadata=WebLinkMetadata(
                                url=url,
                                domain=urlparse(url).netloc,
                                status_code=e.response.status_code,
                                is_accessible=False
                            )
                        )
                    last_exception = e
                    
                except Exception as e:
                    if retries == self.config.max_retries:
                        return LinkValidationResult(
                            url=url,
                            is_valid=False,
                            is_accessible=False,
                            validation_time=0.0,
                            error_message=str(e)
                        )
                    last_exception = e
                
                retries += 1
                if retries <= self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * retries)
            
            # If we get here, all retries failed
            return LinkValidationResult(
                url=url,
                is_valid=False,
                is_accessible=False,
                validation_time=0.0,
                error_message=f"Failed after {self.config.max_retries} retries: {str(last_exception)}"
            )
    
    def _parse_github_url(self, url: str) -> Optional[Tuple[str, str]]:
        """
        Parse GitHub URL to extract owner and repository name.
        
        Args:
            url: GitHub URL
            
        Returns:
            Tuple of (owner, repo) or None if invalid
        """
        try:
            # First validate it's a GitHub URL
            if not self._is_github_url(url):
                return None
                
            parsed = urlparse(url)
            path_parts = [p for p in parsed.path.split("/") if p]
            
            if len(path_parts) >= 2:
                owner = path_parts[0]
                repo = path_parts[1]
                # Clean up repo name (remove .git suffix, etc.)
                repo = repo.rstrip(".git")
                return owner, repo
            
            return None
        except Exception:
            return None
    
    async def _get_github_metadata(self, owner: str, repo: str) -> GitHubMetadata:
        """
        Get detailed metadata from GitHub API.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            GitHubMetadata object
        """
        try:
            response = await self._github_client.get(f"/repos/{owner}/{repo}")
            
            if response.status_code == 200:
                data = response.json()
                
                return GitHubMetadata(
                    url=data.get("html_url", f"https://github.com/{owner}/{repo}"),
                    name=data.get("name", repo),
                    full_name=data.get("full_name", f"{owner}/{repo}"),
                    description=data.get("description"),
                    language=data.get("language"),
                    stars=data.get("stargazers_count"),
                    forks=data.get("forks_count"),
                    watchers=data.get("watchers_count"),
                    open_issues=data.get("open_issues_count"),
                    created_at=data.get("created_at"),
                    updated_at=data.get("updated_at"),
                    pushed_at=data.get("pushed_at"),
                    size=data.get("size"),
                    default_branch=data.get("default_branch"),
                    topics=data.get("topics", []),
                    license_name=data.get("license", {}).get("name") if data.get("license") else None,
                    is_fork=data.get("fork", False),
                    is_private=data.get("private", False),
                    archived=data.get("archived", False),
                    disabled=data.get("disabled", False),
                    has_wiki=data.get("has_wiki", False),
                    has_pages=data.get("has_pages", False),
                    has_downloads=data.get("has_downloads", False),
                    has_issues=data.get("has_issues", False)
                )
            else:
                # Return basic metadata if API call fails
                return GitHubMetadata(
                    url=f"https://github.com/{owner}/{repo}",
                    name=repo,
                    full_name=f"{owner}/{repo}"
                )
                
        except Exception as e:
            logger.warning(f"Failed to get GitHub metadata for {owner}/{repo}: {e}")
            return GitHubMetadata(
                url=f"https://github.com/{owner}/{repo}",
                name=repo,
                full_name=f"{owner}/{repo}"
            )
    
    async def _extract_html_metadata(self, response: httpx.Response, metadata: WebLinkMetadata) -> None:
        """
        Extract metadata from HTML content.
        
        Args:
            response: HTTP response object
            metadata: WebLinkMetadata object to populate
        """
        try:
            # Limit content size to prevent memory issues
            content = response.content[:self.config.web_content_max_size]
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                metadata.title = title_tag.get_text().strip()
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                metadata.description = meta_desc.get('content', '').strip()
            
            # Extract meta keywords
            meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
            if meta_keywords:
                keywords = meta_keywords.get('content', '')
                metadata.meta_keywords = [k.strip() for k in keywords.split(',') if k.strip()]
            
            # Extract meta author
            meta_author = soup.find('meta', attrs={'name': 'author'})
            if meta_author:
                metadata.meta_author = meta_author.get('content', '').strip()
            
            # Extract canonical URL
            canonical = soup.find('link', attrs={'rel': 'canonical'})
            if canonical:
                metadata.canonical_url = canonical.get('href', '').strip()
            
            # Extract Open Graph metadata
            og_title = soup.find('meta', attrs={'property': 'og:title'})
            if og_title:
                metadata.og_title = og_title.get('content', '').strip()
            
            og_desc = soup.find('meta', attrs={'property': 'og:description'})
            if og_desc:
                metadata.og_description = og_desc.get('content', '').strip()
            
            og_type = soup.find('meta', attrs={'property': 'og:type'})
            if og_type:
                metadata.og_type = og_type.get('content', '').strip()
            
        except Exception as e:
            logger.warning(f"Failed to extract HTML metadata: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": self.cache.size(),
            "max_cache_size": self.config.max_cache_size,
            "cache_ttl": self.config.cache_ttl
        }
    
    def clear_cache(self) -> None:
        """Clear the validation cache."""
        self.cache.clear()
        logger.info("Link validation cache cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the link validator."""
        test_urls = [
            "https://github.com/python/cpython",  # Test GitHub API
            "https://httpbin.org/get"  # Test general web validation
        ]
        
        start_time = time.time()
        results = await self.validate_links_batch(test_urls, max_concurrent=2)
        total_time = time.time() - start_time
        
        successful_validations = sum(1 for r in results if r.is_valid)
        
        return {
            "status": "healthy" if successful_validations > 0 else "degraded",
            "test_urls_count": len(test_urls),
            "successful_validations": successful_validations,
            "total_validation_time": total_time,
            "average_validation_time": total_time / len(test_urls) if test_urls else 0,
            "cache_stats": self.get_cache_stats(),
            "github_token_configured": bool(self.github_token)
        } 