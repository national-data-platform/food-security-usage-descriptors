"""
Elasticsearch Synchronization Service.

This module provides functionality to synchronize MongoDB analysis results
into Elasticsearch for efficient search and retrieval operations.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from functools import wraps

from elasticsearch import AsyncElasticsearch, ConnectionTimeout, ConnectionError as ESConnectionError
from elasticsearch.exceptions import RequestError, NotFoundError, ConflictError
from pydantic import BaseModel, Field

from ..config.settings import get_settings
from ..models.analysis_result import AnalysisResult
from ..utils.circuit_breaker import circuit_breaker

logger = logging.getLogger(__name__)

# Circuit breaker for Elasticsearch operations
es_circuit_breaker = circuit_breaker(
    service_name="elasticsearch",
    failure_threshold=5,
    recovery_timeout=60,
    expected_exceptions=(ESConnectionError, ConnectionTimeout, RequestError)
)


class ElasticsearchConfig(BaseModel):
    """Elasticsearch configuration."""
    
    url: str = Field(..., description="Elasticsearch cluster URL")
    index: str = Field(default="llm_analyses", description="Index name for analysis results")
    username: Optional[str] = Field(None, description="Elasticsearch username")
    password: Optional[str] = Field(None, description="Elasticsearch password")
    api_key: Optional[str] = Field(None, description="Elasticsearch API key for authentication")
    ssl_enabled: bool = Field(default=False, description="Enable SSL for connections")
    verify_certs: bool = Field(default=True, description="Verify SSL certificates")
    ca_cert_path: Optional[str] = Field(None, description="Path to CA certificate file for SSL verification")
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    connection_pool_size: int = Field(default=10, description="Connection pool size")


class IndexMapping(BaseModel):
    """Elasticsearch index mapping configuration."""
    
    settings: Dict[str, Any] = Field(
        default_factory=lambda: {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "1s",
            "analysis": {
                "analyzer": {
                    "text_analyzer": {
                        "type": "standard",
                        "stopwords": "_english_"
                    },
                    "autocomplete_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "autocomplete_filter"]
                    },
                    "keyword_analyzer": {
                        "type": "custom",
                        "tokenizer": "keyword",
                        "filter": ["lowercase"]
                    }
                },
                "filter": {
                    "autocomplete_filter": {
                        "type": "edge_ngram",
                        "min_gram": 1,
                        "max_gram": 20
                    }
                }
            }
        },
        description="Index settings"
    )
    
    mappings: Dict[str, Any] = Field(
        default_factory=lambda: {
            "properties": {
                "publication_id": {
                    "type": "keyword",
                    "index": True
                },
                "analysis_id": {
                    "type": "keyword",
                    "index": True
                },
                "workflow_status": {
                    "type": "keyword",
                    "index": True
                },
                "overall_confidence": {
                    "type": "float",
                    "index": True
                },
                "created_at": {
                    "type": "date",
                    "index": True
                },
                "updated_at": {
                    "type": "date",
                    "index": True
                },
                "publication_metadata": {
                    "properties": {
                        "title": {
                            "type": "text",
                            "analyzer": "text_analyzer",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "analyzer": "keyword_analyzer"
                                },
                                "autocomplete": {
                                    "type": "text",
                                    "analyzer": "autocomplete_analyzer"
                                }
                            }
                        },
                        "authors": {
                            "type": "nested",
                            "properties": {
                                "name": {
                                    "type": "text",
                                    "analyzer": "text_analyzer",
                                    "fields": {
                                        "keyword": {"type": "keyword"},
                                        "autocomplete": {
                                            "type": "text",
                                            "analyzer": "autocomplete_analyzer"
                                        }
                                    }
                                },
                                "institution": {
                                    "type": "text",
                                    "analyzer": "text_analyzer",
                                    "fields": {
                                        "keyword": {"type": "keyword"},
                                        "autocomplete": {
                                            "type": "text",
                                            "analyzer": "autocomplete_analyzer"
                                        }
                                    }
                                },
                                "email": {"type": "keyword"}
                            }
                        },
                        "abstract": {
                            "type": "text",
                            "analyzer": "text_analyzer"
                        },
                        "keywords": {
                            "type": "keyword",
                            "analyzer": "keyword_analyzer"
                        },
                        "doi": {"type": "keyword"},
                        "journal": {
                            "type": "text",
                            "analyzer": "text_analyzer",
                            "fields": {
                                "keyword": {"type": "keyword"}
                            }
                        },
                        "publication_date": {"type": "date"}
                    }
                },
                "dataset_analysis": {
                    "properties": {
                        "validated_datasets": {
                            "type": "nested",
                            "properties": {
                                "name": {
                                    "type": "text",
                                    "analyzer": "text_analyzer",
                                    "fields": {
                                        "keyword": {"type": "keyword"},
                                        "autocomplete": {
                                            "type": "text",
                                            "analyzer": "autocomplete_analyzer"
                                        }
                                    }
                                },
                                "aliases": {
                                    "type": "keyword",
                                    "analyzer": "keyword_analyzer"
                                },
                                "domain": {"type": "keyword"},
                                "confidence": {"type": "float"},
                                "evidence": {
                                    "type": "text",
                                    "analyzer": "text_analyzer"
                                }
                            }
                        },
                        "new_datasets": {
                            "type": "nested",
                            "properties": {
                                "name": {
                                    "type": "text",
                                    "analyzer": "text_analyzer",
                                    "fields": {
                                        "keyword": {"type": "keyword"},
                                        "autocomplete": {
                                            "type": "text",
                                            "analyzer": "autocomplete_analyzer"
                                        }
                                    }
                                },
                                "description": {
                                    "type": "text",
                                    "analyzer": "text_analyzer"
                                },
                                "confidence": {"type": "float"}
                            }
                        }
                    }
                },
                "code_extraction": {
                    "properties": {
                        "code_snippets": {
                            "type": "nested",
                            "properties": {
                                "language": {"type": "keyword"},
                                "content": {
                                    "type": "text",
                                    "analyzer": "text_analyzer"
                                },
                                "file_path": {"type": "keyword"},
                                "line_numbers": {
                                    "type": "object",
                                    "properties": {
                                        "start": {"type": "integer"},
                                        "end": {"type": "integer"}
                                    }
                                },
                                "confidence": {"type": "float"}
                            }
                        },
                        "github_repositories": {
                            "type": "nested",
                            "properties": {
                                "url": {"type": "keyword"},
                                "name": {
                                    "type": "text",
                                    "analyzer": "text_analyzer",
                                    "fields": {
                                        "keyword": {"type": "keyword"}
                                    }
                                },
                                "description": {
                                    "type": "text",
                                    "analyzer": "text_analyzer"
                                },
                                "stars": {"type": "integer"},
                                "language": {"type": "keyword"},
                                "confidence": {"type": "float"}
                            }
                        }
                    }
                },
                "link_extraction": {
                    "properties": {
                        "external_links": {
                            "type": "nested",
                            "properties": {
                                "url": {"type": "keyword"},
                                "title": {
                                    "type": "text",
                                    "analyzer": "text_analyzer"
                                },
                                "description": {
                                    "type": "text",
                                    "analyzer": "text_analyzer"
                                },
                                "link_type": {"type": "keyword"},
                                "status": {"type": "keyword"},
                                "confidence": {"type": "float"}
                            }
                        }
                    }
                },
                "dataset_joins": {
                    "type": "nested",
                    "properties": {
                        "source_datasets": {
                            "type": "keyword",
                            "analyzer": "keyword_analyzer"
                        },
                        "target_datasets": {
                            "type": "keyword",
                            "analyzer": "keyword_analyzer"
                        },
                        "join_type": {"type": "keyword"},
                        "methodology": {
                            "type": "text",
                            "analyzer": "text_analyzer"
                        },
                        "challenges": {
                            "type": "text",
                            "analyzer": "text_analyzer"
                        },
                        "confidence": {"type": "float"}
                    }
                },
                "llm_metadata": {
                    "properties": {
                        "model_name": {"type": "keyword"},
                        "provider": {"type": "keyword"},
                        "temperature": {"type": "float"},
                        "max_tokens": {"type": "integer"},
                        "response_time": {"type": "float"}
                    }
                },
                "error_information": {
                    "properties": {
                        "error_type": {"type": "keyword"},
                        "error_message": {
                            "type": "text",
                            "analyzer": "text_analyzer"
                        },
                        "error_timestamp": {"type": "date"},
                        "step_failed": {"type": "keyword"}
                    }
                }
            }
        },
        description="Index mappings"
    )


def ensure_connection(func):
    """Decorator to ensure Elasticsearch connection is available."""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        if not self.client or not await self.client.ping():
            await self._ensure_connection()
        return await func(self, *args, **kwargs)
    return wrapper


class ElasticsearchSyncService:
    """
    Service for synchronizing MongoDB analysis results to Elasticsearch.
    
    This service provides functionality to:
    - Connect to Elasticsearch with proper authentication and SSL
    - Create and manage index mappings
    - Denormalize MongoDB documents for search optimization
    - Perform batch synchronization operations
    - Handle errors and retries
    """
    
    def __init__(self, config: Optional[ElasticsearchConfig] = None):
        """
        Initialize the Elasticsearch sync service.
        
        Args:
            config: Elasticsearch configuration. If None, uses settings from environment.
        """
        self.config = config or self._get_config_from_settings()
        self.client: Optional[AsyncElasticsearch] = None
        self.index_mapping = IndexMapping()
        self._connection_established = False
        
        logger.info(f"Initialized ElasticsearchSyncService for index: {self.config.index}")
    
    def _get_config_from_settings(self) -> ElasticsearchConfig:
        """Get Elasticsearch configuration from application settings."""
        settings = get_settings()
        return ElasticsearchConfig(
            url=settings.elasticsearch.url,
            index=settings.elasticsearch.index,
            username=settings.elasticsearch.username,
            password=settings.elasticsearch.password,
            api_key=settings.elasticsearch.api_key,
            ssl_enabled=settings.elasticsearch.ssl_enabled,
            verify_certs=settings.elasticsearch.verify_certs,
            ca_cert_path=settings.elasticsearch.ca_cert_path,
            request_timeout=settings.elasticsearch.request_timeout,
            max_retries=settings.elasticsearch.max_retries
        )
    
    async def _ensure_connection(self) -> None:
        """Ensure Elasticsearch connection is established."""
        if self._connection_established and self.client and await self.client.ping():
            return
        
        try:
            # Build connection parameters
            connection_params = {
                "hosts": [self.config.url],
                "timeout": self.config.request_timeout,
                "max_retries": self.config.max_retries,
                "retry_on_timeout": True,
                "maxsize": self.config.connection_pool_size
            }
            
            # Add authentication if provided
            if self.config.api_key:
                connection_params["api_key"] = self.config.api_key
            elif self.config.username and self.config.password:
                connection_params["http_auth"] = (self.config.username, self.config.password)
            
            # Add SSL configuration
            if self.config.ssl_enabled:
                connection_params["use_ssl"] = True
                connection_params["verify_certs"] = self.config.verify_certs
                
                # Add CA certificate if provided
                if self.config.ca_cert_path:
                    from pathlib import Path
                    ca_cert_path = Path(self.config.ca_cert_path)
                    if not ca_cert_path.is_absolute():
                        # If relative path, make it relative to project root
                        project_root = Path(__file__).parent.parent.parent.parent
                        ca_cert_path = project_root / self.config.ca_cert_path
                    connection_params["ca_certs"] = str(ca_cert_path)
                
                if not self.config.verify_certs:
                    connection_params["ssl_show_warn"] = False
            
            self.client = AsyncElasticsearch(**connection_params)
            
            # Test connection
            if await self.client.ping():
                self._connection_established = True
                logger.info("Successfully connected to Elasticsearch")
            else:
                raise ESConnectionError("Failed to ping Elasticsearch")
                
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            self._connection_established = False
            raise
    
    @ensure_connection
    async def create_index(self, force: bool = False) -> bool:
        """
        Create the Elasticsearch index with proper mapping.
        
        Args:
            force: If True, delete existing index before creating new one.
            
        Returns:
            True if index was created successfully, False otherwise.
        """
        try:
            index_exists = await self.client.indices.exists(index=self.config.index)
            
            if index_exists and force:
                logger.info(f"Deleting existing index: {self.config.index}")
                await self.client.indices.delete(index=self.config.index)
                index_exists = False
            
            if not index_exists:
                logger.info(f"Creating index: {self.config.index}")
                await self.client.indices.create(
                    index=self.config.index,
                    body={
                        "settings": self.index_mapping.settings,
                        "mappings": self.index_mapping.mappings
                    }
                )
                logger.info(f"Successfully created index: {self.config.index}")
                return True
            else:
                logger.info(f"Index {self.config.index} already exists")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create index {self.config.index}: {e}")
            return False
    
    @ensure_connection
    async def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the Elasticsearch index.
        
        Returns:
            Dictionary containing index information.
        """
        try:
            index_exists = await self.client.indices.exists(index=self.config.index)
            if not index_exists:
                return {"exists": False}
            
            # Get index settings and mappings
            settings = await self.client.indices.get_settings(index=self.config.index)
            mappings = await self.client.indices.get_mapping(index=self.config.index)
            
            # Get index stats
            stats = await self.client.indices.stats(index=self.config.index)
            
            return {
                "exists": True,
                "settings": settings[self.config.index]["settings"],
                "mappings": mappings[self.config.index]["mappings"],
                "stats": stats["indices"][self.config.index]
            }
            
        except Exception as e:
            logger.error(f"Failed to get index info: {e}")
            return {"exists": False, "error": str(e)}
    
    async def _health_check_internal(self) -> Dict[str, Any]:
        """
        Internal health check method without connection decorator.
        
        Returns:
            Dictionary containing health status information.
        """
        try:
            # Check cluster health
            cluster_health = await self.client.cluster.health()
            
            # Check index health
            index_health = await self.client.indices.health(index=self.config.index)
            
            # Check connection
            ping_success = await self.client.ping()
            
            return {
                "status": "healthy" if ping_success else "unhealthy",
                "cluster_health": cluster_health,
                "index_health": index_health,
                "ping_success": ping_success,
                "connection_established": self._connection_established
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "connection_established": False
            }
    
    @ensure_connection
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Elasticsearch connection and index.
        
        Returns:
            Dictionary containing health status information.
        """
        return await self._health_check_internal()
    
    async def close(self) -> None:
        """Close the Elasticsearch connection."""
        if self.client:
            await self.client.close()
            self._connection_established = False
            logger.info("Elasticsearch connection closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_connection()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close() 