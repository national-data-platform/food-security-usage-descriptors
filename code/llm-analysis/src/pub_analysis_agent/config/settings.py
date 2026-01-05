"""
Configuration settings for pub-analysis-agent.

This module defines all configuration settings using Pydantic for validation
and environment variable loading.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings
from pydantic.networks import AnyHttpUrl


class DatabaseSettings(BaseSettings):
    """MongoDB database configuration."""
    
    # Connection settings
    connection_string: str = Field(
        default="mongodb://localhost:27016/",
        env="MONGODB_CONNECTION_STRING",
        description="MongoDB connection string"
    )
    
    # Database names
    general_database: str = Field(
        default="general",
        env="MONGODB_GENERAL_DATABASE",
        description="General database name"
    )
    dimensions_database: str = Field(
        default="dimensions",
        env="MONGODB_DIMENSIONS_DATABASE",
        description="Dimensions database name"
    )
    analysis_database: str = Field(
        default="publication_analysis",
        env="MONGODB_ANALYSIS_DATABASE",
        description="Analysis database name for storing LLM analysis results"
    )
    
    # Collection settings in general database
    datasets_collection: str = Field(
        default="datasets",
        env="MONGODB_DATASETS_COLLECTION",
        description="Collection containing known datasets"
    )
    
    # Collection settings in dimensions database
    authors_collection: str = Field(
        default="authors",
        env="MONGODB_AUTHORS_COLLECTION",
        description="Collection containing authors"
    )
    institutions_collection: str = Field(
        default="institutions",
        env="MONGODB_INSTITUTIONS_COLLECTION",
        description="Collection containing institutions"
    )
    publications_collection: str = Field(
        default="publications",
        env="MONGODB_PUBLICATIONS_COLLECTION",
        description="Collection containing publications"
    )
    
    # Collection settings in analysis database
    results_collection: str = Field(
        default="llm_analyses",
        env="MONGODB_RESULTS_COLLECTION",
        description="Collection for storing analysis results"
    )
    
    # Connection pool settings
    max_pool_size: int = Field(
        default=50,
        env="MONGODB_MAX_POOL_SIZE",
        description="Maximum connection pool size"
    )
    min_pool_size: int = Field(
        default=5,
        env="MONGODB_MIN_POOL_SIZE",
        description="Minimum connection pool size"
    )
    
    # Timeout settings
    connect_timeout_ms: int = Field(
        default=5000,
        env="MONGODB_CONNECT_TIMEOUT_MS",
        description="Connection timeout in milliseconds"
    )
    server_selection_timeout_ms: int = Field(
        default=5000,
        env="MONGODB_SERVER_SELECTION_TIMEOUT_MS",
        description="Server selection timeout in milliseconds"
    )
    
    # SSL settings
    ssl_enabled: bool = Field(
        default=False,
        env="MONGODB_SSL_ENABLED",
        description="Enable SSL for MongoDB connections"
    )

    model_config = {
        "env_prefix": "MONGODB_",
        "extra": "forbid"  # Explicitly forbid extra fields for validation
    }


class ElasticsearchSettings(BaseSettings):
    """Elasticsearch configuration."""
    
    # Connection settings
    url: AnyHttpUrl = Field(
        default="http://localhost:9200",
        env="ELASTICSEARCH_URL",
        description="Elasticsearch endpoint URL"
    )
    index: str = Field(
        default="publications",
        env="ELASTICSEARCH_INDEX",
        description="Elasticsearch index name"
    )
    
    # Authentication
    username: Optional[str] = Field(
        default=None,
        env="ELASTICSEARCH_USERNAME",
        description="Elasticsearch username"
    )
    password: Optional[str] = Field(
        default=None,
        env="ELASTICSEARCH_PASSWORD",
        description="Elasticsearch password"
    )
    api_key: Optional[str] = Field(
        default=None,
        env="ELASTICSEARCH_API_KEY",
        description="Elasticsearch API key for authentication"
    )
    
    # SSL settings
    ssl_enabled: bool = Field(
        default=False,
        env="ELASTICSEARCH_SSL_ENABLED",
        description="Enable SSL for Elasticsearch connections"
    )
    verify_certs: bool = Field(
        default=False,
        env="ELASTICSEARCH_VERIFY_CERTS",
        description="Verify SSL certificates"
    )
    ca_cert_path: Optional[str] = Field(
        default=None,
        env="ELASTICSEARCH_CA_CERT_PATH",
        description="Path to CA certificate file for SSL verification"
    )
    
    # Performance settings
    request_timeout: int = Field(
        default=30,
        env="ELASTICSEARCH_REQUEST_TIMEOUT",
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        env="ELASTICSEARCH_MAX_RETRIES",
        description="Maximum number of retries"
    )

    model_config = {
        "env_prefix": "ELASTICSEARCH_"
    }


class LLMSettings(BaseSettings):
    """LLM service configuration."""
    
    # Provider selection
    provider: str = Field(
        default="ollama",
        env="LLM_PROVIDER",
        description="LLM provider: 'ollama' or 'lmstudio'"
    )
    
    # Service endpoints
    ollama_base_url: AnyHttpUrl = Field(
        default="http://localhost:11434/api",
        env="OLLAMA_BASE_URL",
        description="Ollama API base URL"
    )
    lmstudio_base_url: AnyHttpUrl = Field(
        default="http://localhost:1234/v1/",
        env="LMSTUDIO_BASE_URL",
        description="LMStudio API base URL"
    )
    
    # Model configuration
    default_model: str = Field(
        default="gpt-oss:120b",
        env="DEFAULT_LLM_MODEL",
        description="Default LLM model name"
    )
    temperature: float = Field(
        default=0.1,
        env="LLM_TEMPERATURE",
        ge=0.0,
        le=2.0,
        description="LLM temperature parameter"
    )
    max_tokens: int = Field(
        default=4000,
        env="LLM_MAX_TOKENS",
        gt=0,
        description="Maximum tokens per request"
    )
    top_p: float = Field(
        default=0.9,
        env="LLM_TOP_P",
        ge=0.0,
        le=1.0,
        description="LLM top-p parameter"
    )
    
    # Request settings
    request_timeout: int = Field(
        default=60,
        env="LLM_REQUEST_TIMEOUT",
        gt=0,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        env="LLM_MAX_RETRIES",
        ge=0,
        description="Maximum number of retries"
    )
    retry_delay: float = Field(
        default=1.0,
        env="LLM_RETRY_DELAY",
        ge=0.0,
        description="Delay between retries in seconds"
    )
    
    # API Keys (optional for cloud services)
    openai_api_key: Optional[str] = Field(
        default=None,
        env="OPENAI_API_KEY",
        description="OpenAI API key"
    )
    anthropic_api_key: Optional[str] = Field(
        default=None,
        env="ANTHROPIC_API_KEY",
        description="Anthropic API key"
    )

    model_config = {
        "env_prefix": "LLM_"
    }
    
    @property
    def base_url(self) -> str:
        """Get the appropriate base URL based on the selected provider."""
        if self.provider.lower() == "lmstudio":
            return str(self.lmstudio_base_url)
        else:  # Default to ollama
            return str(self.ollama_base_url)


class ProcessingSettings(BaseSettings):
    """Processing and workflow configuration."""
    
    # Batch processing
    batch_size: int = Field(
        default=10,
        env="PROCESSING_BATCH_SIZE",
        gt=0,
        description="Number of publications to process in a batch"
    )
    max_concurrent: int = Field(
        default=5,
        env="PROCESSING_MAX_CONCURRENT",
        gt=0,
        description="Maximum concurrent processing tasks"
    )
    chunk_size: int = Field(
        default=1000,
        env="PROCESSING_CHUNK_SIZE",
        gt=0,
        description="Text chunk size for processing"
    )
    
    # Memory management
    max_memory_usage_mb: int = Field(
        default=2048,
        env="MAX_MEMORY_USAGE_MB",
        gt=0,
        description="Maximum memory usage in MB"
    )
    enable_memory_monitoring: bool = Field(
        default=True,
        env="ENABLE_MEMORY_MONITORING",
        description="Enable memory usage monitoring"
    )
    
    # Workflow settings
    workflow_max_retries: int = Field(
        default=3,
        env="WORKFLOW_MAX_RETRIES",
        ge=0,
        description="Maximum workflow retries"
    )
    workflow_timeout_minutes: int = Field(
        default=30,
        env="WORKFLOW_TIMEOUT_MINUTES",
        gt=0,
        description="Workflow timeout in minutes"
    )
    workflow_checkpoint_enabled: bool = Field(
        default=True,
        env="WORKFLOW_CHECKPOINT_ENABLED",
        description="Enable workflow checkpointing"
    )
    
    # Agent confidence thresholds
    triage_confidence_threshold: float = Field(
        default=0.7,
        env="TRIAGE_CONFIDENCE_THRESHOLD",
        ge=0.0,
        le=1.0,
        description="Minimum confidence for triage agent"
    )
    dataset_validation_confidence_threshold: float = Field(
        default=0.6,
        env="DATASET_VALIDATION_CONFIDENCE_THRESHOLD",
        ge=0.0,
        le=1.0,
        description="Minimum confidence for dataset validation"
    )
    code_extraction_confidence_threshold: float = Field(
        default=0.5,
        env="CODE_EXTRACTION_CONFIDENCE_THRESHOLD",
        ge=0.0,
        le=1.0,
        description="Minimum confidence for code extraction"
    )

    model_config = {
        "env_prefix": "PROCESSING_"
    }


class LoggingSettings(BaseSettings):
    """Logging configuration."""
    
    # Basic logging settings
    level: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="Log level"
    )
    format: str = Field(
        default="json",
        env="LOG_FORMAT",
        description="Log format (json or text)"
    )
    
    # File logging
    file_path: str = Field(
        default="logs/pub_analysis.log",
        env="LOG_FILE_PATH",
        description="Log file path"
    )
    max_file_size_mb: int = Field(
        default=100,
        env="LOG_MAX_FILE_SIZE_MB",
        gt=0,
        description="Maximum log file size in MB"
    )
    backup_count: int = Field(
        default=5,
        env="LOG_BACKUP_COUNT",
        ge=0,
        description="Number of backup log files"
    )
    
    # Component-specific logging
    log_llm_requests: bool = Field(
        default=True,
        env="LOG_LLM_REQUESTS",
        description="Log LLM requests and responses"
    )
    log_database_queries: bool = Field(
        default=False,
        env="LOG_DATABASE_QUERIES",
        description="Log database queries"
    )
    log_workflow_state: bool = Field(
        default=True,
        env="LOG_WORKFLOW_STATE",
        description="Log workflow state changes"
    )

    @field_validator('level')
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Invalid log level: {v}. Must be one of: {valid_levels}')
        return v.upper()

    @field_validator('format')
    @classmethod
    def validate_log_format(cls, v):
        valid_formats = ['json', 'text']
        if v not in valid_formats:
            raise ValueError(f'Invalid log format: {v}. Must be one of: {valid_formats}')
        return v

    model_config = {
        "env_prefix": "LOG_"
    }


class SecuritySettings(BaseSettings):
    """Security configuration."""
    
    # Rate limiting
    enable_rate_limiting: bool = Field(
        default=True,
        env="ENABLE_RATE_LIMITING",
        description="Enable rate limiting"
    )
    requests_per_minute: int = Field(
        default=60,
        env="REQUESTS_PER_MINUTE",
        gt=0,
        description="Maximum requests per minute"
    )

    model_config = {
        "env_prefix": "SECURITY_"
    }


class DevelopmentSettings(BaseSettings):
    """Development and debugging configuration."""
    
    # Development mode
    development_mode: bool = Field(
        default=True,
        env="DEVELOPMENT_MODE",
        description="Enable development mode"
    )
    
    # Profiling
    enable_performance_profiling: bool = Field(
        default=False,
        env="ENABLE_PERFORMANCE_PROFILING",
        description="Enable performance profiling"
    )
    enable_memory_profiling: bool = Field(
        default=False,
        env="ENABLE_MEMORY_PROFILING",
        description="Enable memory profiling"
    )
    
    # Test settings
    test_mongodb_database: str = Field(
        default="test_dimensions",
        env="TEST_MONGODB_DATABASE",
        description="Test MongoDB database name"
    )
    test_elasticsearch_index: str = Field(
        default="test_publications",
        env="TEST_ELASTICSEARCH_INDEX",
        description="Test Elasticsearch index name"
    )

    model_config = {
        "env_prefix": "DEV_"
    }


class Settings(BaseSettings):
    """Main configuration settings."""
    
    # Environment settings
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    elasticsearch: ElasticsearchSettings = Field(default_factory=ElasticsearchSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    development: DevelopmentSettings = Field(default_factory=DevelopmentSettings)
    
    # Additional API keys that might be present in environment
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    perplexity_api_key: Optional[str] = Field(default=None, env="PERPLEXITY_API_KEY") 
    openrouter_api_key: Optional[str] = Field(default=None, env="OPENROUTER_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")

    @model_validator(mode='after')
    def validate_configuration(self):
        """Validate cross-field dependencies."""
        
        # Validate security settings for production
        if self.environment == "production":
            if self.security.requests_per_minute <= 0:
                raise ValueError("Rate limiting must be enabled in production")
        
        # Ensure processing settings are reasonable
        if self.processing.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.processing.max_concurrent <= 0:
            raise ValueError("Max concurrent requests must be positive")
            
        return self

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "allow"  # Allow extra fields from environment
    }
        
    def is_local_service(self, service: str) -> bool:
        """Check if a service is configured for local deployment."""
        if service == "mongodb":
            return "localhost" in str(self.database.connection_string)
        elif service == "elasticsearch":
            return "localhost" in str(self.elasticsearch.url)
        elif service == "ollama":
            return "localhost" in str(self.llm.ollama_base_url)
        elif service == "lmstudio":
            return "localhost" in str(self.llm.lmstudio_base_url)
        return False
    
    def get_service_health_check_url(self, service: str) -> Optional[str]:
        """Get health check URL for a service."""
        if service == "elasticsearch":
            return f"{self.elasticsearch.url}/_cluster/health"
        elif service == "ollama":
            return f"{self.llm.ollama_base_url}/tags"
        elif service == "lmstudio":
            return f"{self.llm.lmstudio_base_url}/models"
        return None


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Global settings instance
settings = get_settings() 