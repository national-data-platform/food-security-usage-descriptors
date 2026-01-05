"""
Configuration loader for pub-analysis-agent.

This module provides utilities for loading configuration from multiple sources
with proper precedence: environment variables > YAML files > defaults.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from functools import lru_cache

from .settings import Settings


class ConfigurationLoader:
    """
    Configuration loader with support for multiple sources.
    
    Precedence order (highest to lowest):
    1. Environment variables
    2. YAML configuration files
    3. Default values in Pydantic models
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to YAML configuration file. If None, looks for
                        config.yaml in project root or config/config.yaml.
        """
        self.config_path = self._resolve_config_path(config_path)
        self._yaml_config: Optional[Dict[str, Any]] = None

    def _resolve_config_path(self, config_path: Optional[Union[str, Path]]) -> Optional[Path]:
        """Resolve configuration file path."""
        if config_path:
            path = Path(config_path)
            if path.exists():
                return path
            else:
                raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Look for config files in standard locations
        possible_paths = [
            Path("config.yaml"),
            Path("config/config.yaml"),
            Path(".config.yaml"),
        ]

        for path in possible_paths:
            if path.exists():
                return path

        return None

    def load_yaml_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if self._yaml_config is not None:
            return self._yaml_config

        if not self.config_path:
            self._yaml_config = {}
            return self._yaml_config

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            
            self._yaml_config = config
            return config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file {self.config_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading config file {self.config_path}: {e}")

    def merge_config_sources(self) -> Dict[str, Any]:
        """
        Merge configuration from all sources with proper precedence.
        
        Returns:
            Merged configuration dictionary.
        """
        # Start with YAML config
        config = self.load_yaml_config().copy()

        # Override with environment variables
        env_overrides = self._extract_env_overrides()
        config = self._deep_merge(config, env_overrides)

        return config

    def _extract_env_overrides(self) -> Dict[str, Any]:
        """Extract environment variable overrides."""
        overrides = {}
        
        # Define environment variable mappings
        env_mappings = {
            # Database settings
            'MONGODB_CONNECTION_STRING': ['database', 'connection_string'],
            'MONGODB_DATASETS_COLLECTION': ['database', 'datasets_collection'],
            'MONGODB_RESULTS_COLLECTION': ['database', 'results_collection'],
            'MONGODB_MAX_POOL_SIZE': ['database', 'max_pool_size'],
            'MONGODB_MIN_POOL_SIZE': ['database', 'min_pool_size'],
            'MONGODB_CONNECT_TIMEOUT_MS': ['database', 'connect_timeout_ms'],
            'MONGODB_SERVER_SELECTION_TIMEOUT_MS': ['database', 'server_selection_timeout_ms'],
            'MONGODB_SSL_ENABLED': ['database', 'ssl_enabled'],

            # Elasticsearch settings
            'ELASTICSEARCH_URL': ['elasticsearch', 'url'],
            'ELASTICSEARCH_INDEX': ['elasticsearch', 'index'],
            'ELASTICSEARCH_USERNAME': ['elasticsearch', 'username'],
            'ELASTICSEARCH_PASSWORD': ['elasticsearch', 'password'],
            'ELASTICSEARCH_SSL_ENABLED': ['elasticsearch', 'ssl_enabled'],
            'ELASTICSEARCH_VERIFY_CERTS': ['elasticsearch', 'verify_certs'],
            'ELASTICSEARCH_REQUEST_TIMEOUT': ['elasticsearch', 'request_timeout'],
            'ELASTICSEARCH_MAX_RETRIES': ['elasticsearch', 'max_retries'],

            # LLM settings
            'OLLAMA_BASE_URL': ['llm', 'ollama_base_url'],
            'LMSTUDIO_BASE_URL': ['llm', 'lmstudio_base_url'],
            'DEFAULT_LLM_MODEL': ['llm', 'default_model'],
            'LLM_TEMPERATURE': ['llm', 'temperature'],
            'LLM_MAX_TOKENS': ['llm', 'max_tokens'],
            'LLM_TOP_P': ['llm', 'top_p'],
            'LLM_REQUEST_TIMEOUT': ['llm', 'request_timeout'],
            'LLM_MAX_RETRIES': ['llm', 'max_retries'],
            'LLM_RETRY_DELAY': ['llm', 'retry_delay'],
            'OPENAI_API_KEY': ['llm', 'openai_api_key'],
            'ANTHROPIC_API_KEY': ['llm', 'anthropic_api_key'],

            # Processing settings
            'PROCESSING_BATCH_SIZE': ['processing', 'batch_size'],
            'PROCESSING_MAX_CONCURRENT': ['processing', 'max_concurrent'],
            'PROCESSING_CHUNK_SIZE': ['processing', 'chunk_size'],
            'MAX_MEMORY_USAGE_MB': ['processing', 'max_memory_usage_mb'],
            'ENABLE_MEMORY_MONITORING': ['processing', 'enable_memory_monitoring'],
            'WORKFLOW_MAX_RETRIES': ['processing', 'workflow_max_retries'],
            'WORKFLOW_TIMEOUT_MINUTES': ['processing', 'workflow_timeout_minutes'],
            'WORKFLOW_CHECKPOINT_ENABLED': ['processing', 'workflow_checkpoint_enabled'],
            'TRIAGE_CONFIDENCE_THRESHOLD': ['processing', 'triage_confidence_threshold'],
            'DATASET_VALIDATION_CONFIDENCE_THRESHOLD': ['processing', 'dataset_validation_confidence_threshold'],
            'CODE_EXTRACTION_CONFIDENCE_THRESHOLD': ['processing', 'code_extraction_confidence_threshold'],

            # Logging settings
            'LOG_LEVEL': ['logging', 'level'],
            'LOG_FORMAT': ['logging', 'format'],
            'LOG_FILE_PATH': ['logging', 'file_path'],
            'LOG_MAX_FILE_SIZE_MB': ['logging', 'max_file_size_mb'],
            'LOG_BACKUP_COUNT': ['logging', 'backup_count'],
            'LOG_LLM_REQUESTS': ['logging', 'log_llm_requests'],
            'LOG_DATABASE_QUERIES': ['logging', 'log_database_queries'],
            'LOG_WORKFLOW_STATE': ['logging', 'log_workflow_state'],

            # Security settings
            'ENABLE_RATE_LIMITING': ['security', 'enable_rate_limiting'],
            'REQUESTS_PER_MINUTE': ['security', 'requests_per_minute'],

            # Development settings
            'DEVELOPMENT_MODE': ['development', 'development_mode'],
            'ENABLE_PERFORMANCE_PROFILING': ['development', 'enable_performance_profiling'],
            'ENABLE_MEMORY_PROFILING': ['development', 'enable_memory_profiling'],
            'TEST_MONGODB_DATABASE': ['development', 'test_mongodb_database'],
            'TEST_ELASTICSEARCH_INDEX': ['development', 'test_elasticsearch_index'],

            # Global settings
            'CUSTOM_CONFIG_PATH': ['custom_config_path'],
            'PROMPT_TEMPLATES_PATH': ['prompt_templates_path'],
            'DATASET_SCHEMAS_PATH': ['dataset_schemas_path'],
        }

        for env_var, path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert value to appropriate type
                converted_value = self._convert_env_value(value)
                self._set_nested_value(overrides, path, converted_value)

        return overrides

    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # None/null conversion
        if value.lower() in ('none', 'null', ''):
            return None
        
        # Integer conversion
        try:
            if '.' not in value:
                return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value

    def _set_nested_value(self, dictionary: Dict[str, Any], path: List[str], value: Any) -> None:
        """Set a nested value in a dictionary using a path list."""
        current = dictionary
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result

    def create_settings(self) -> Settings:
        """
        Create Settings instance with merged configuration.
        
        Returns:
            Configured Settings instance.
        """
        # Load and merge all configuration sources
        merged_config = self.merge_config_sources()
        
        # Create settings with the merged configuration
        # Note: Pydantic will still apply its own validation and environment variable handling
        return Settings(**merged_config)

    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate the current configuration and return validation results.
        
        Returns:
            Dictionary with validation results.
        """
        try:
            settings = self.create_settings()
            return {
                'valid': True,
                'settings': settings,
                'config_file': str(self.config_path) if self.config_path else None,
                'message': 'Configuration is valid'
            }
        except Exception as e:
            return {
                'valid': False,
                'settings': None,
                'config_file': str(self.config_path) if self.config_path else None,
                'error': str(e),
                'message': f'Configuration validation failed: {e}'
            }


@lru_cache()
def get_configuration_loader(config_path: Optional[str] = None) -> ConfigurationLoader:
    """Get cached configuration loader instance."""
    return ConfigurationLoader(config_path)


def load_settings_from_file(config_path: Optional[Union[str, Path]] = None) -> Settings:
    """
    Load settings from configuration file with environment variable overrides.
    
    Args:
        config_path: Path to YAML configuration file.
        
    Returns:
        Configured Settings instance.
    """
    loader = ConfigurationLoader(config_path)
    return loader.create_settings()


def validate_configuration_file(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Validate a configuration file.
    
    Args:
        config_path: Path to YAML configuration file.
        
    Returns:
        Validation results dictionary.
    """
    loader = ConfigurationLoader(config_path)
    return loader.validate_configuration() 