"""
Environment variable management system for pub-analysis-agent.

This module provides comprehensive environment variable management with validation,
security features, and support for multiple deployment environments.
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field, validator


class EnvironmentType(str, Enum):
    """Supported environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing" 
    STAGING = "staging"
    PRODUCTION = "production"


class ValidationLevel(str, Enum):
    """Environment variable validation levels."""
    STRICT = "strict"      # All required variables must be present
    LENIENT = "lenient"    # Warning for missing non-critical variables
    DISABLED = "disabled"  # No validation


@dataclass
class EnvironmentVariable:
    """Definition of an environment variable with validation rules."""
    
    name: str
    description: str
    required: bool = True
    default: Optional[str] = None
    sensitive: bool = False
    validator_func: Optional[Callable[[str], bool]] = None
    allowed_values: Optional[List[str]] = None
    environment_specific: Dict[EnvironmentType, str] = field(default_factory=dict)
    
    def validate(self, value: str) -> bool:
        """Validate environment variable value."""
        if self.allowed_values and value not in self.allowed_values:
            return False
            
        if self.validator_func:
            return self.validator_func(value)
            
        return True


class EnvironmentSchema:
    """Schema definition for all environment variables."""
    
    # Database configuration
    MONGODB_CONNECTION_STRING = EnvironmentVariable(
        name="MONGODB_CONNECTION_STRING",
        description="MongoDB connection string (mongodb://... or mongodb+srv://...)",
        required=True,
        sensitive=True,
        default="mongodb://localhost:27016/",
        validator_func=lambda x: x.startswith(('mongodb://', 'mongodb+srv://'))
    )
    
    MONGODB_DATABASE = EnvironmentVariable(
        name="MONGODB_DATABASE",
        description="MongoDB database name",
        required=True,
        default="dimensions"
    )
    
    # Elasticsearch configuration
    ELASTICSEARCH_URL = EnvironmentVariable(
        name="ELASTICSEARCH_URL",
        description="Elasticsearch cluster URL",
        required=True,
        default="http://localhost:9200",
        validator_func=lambda x: x.startswith(('http://', 'https://'))
    )
    
    ELASTICSEARCH_USERNAME = EnvironmentVariable(
        name="ELASTICSEARCH_USERNAME",
        description="Elasticsearch username (optional for local)",
        required=False,
        sensitive=True
    )
    
    ELASTICSEARCH_PASSWORD = EnvironmentVariable(
        name="ELASTICSEARCH_PASSWORD", 
        description="Elasticsearch password (optional for local)",
        required=False,
        sensitive=True
    )
    
    # LLM service configuration
    OLLAMA_ENDPOINT = EnvironmentVariable(
        name="OLLAMA_ENDPOINT",
        description="Ollama API endpoint URL",
        required=False,
        default="http://localhost:11434/api",
        validator_func=lambda x: x.startswith(('http://', 'https://'))
    )
    
    LMSTUDIO_ENDPOINT = EnvironmentVariable(
        name="LMSTUDIO_ENDPOINT",
        description="LMStudio API endpoint URL", 
        required=False,
        default="http://localhost:1234/v1",
        validator_func=lambda x: x.startswith(('http://', 'https://'))
    )
    
    # API Keys (all sensitive)
    OPENAI_API_KEY = EnvironmentVariable(
        name="OPENAI_API_KEY",
        description="OpenAI API key for GPT models",
        required=False,
        sensitive=True
    )
    
    ANTHROPIC_API_KEY = EnvironmentVariable(
        name="ANTHROPIC_API_KEY", 
        description="Anthropic API key for Claude models",
        required=False,
        sensitive=True
    )
    
    # Application configuration
    ENVIRONMENT = EnvironmentVariable(
        name="ENVIRONMENT",
        description="Deployment environment",
        required=True,
        default="development",
        allowed_values=[e.value for e in EnvironmentType]
    )
    
    LOG_LEVEL = EnvironmentVariable(
        name="LOG_LEVEL",
        description="Logging level",
        required=False,
        default="INFO",
        allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
    
    LOG_DIRECTORY = EnvironmentVariable(
        name="LOG_DIRECTORY",
        description="Directory for log files",
        required=False,
        default="logs"
    )
    
    # Security configuration
    SECRET_KEY = EnvironmentVariable(
        name="SECRET_KEY",
        description="Secret key for encryption and signing",
        required=True,
        sensitive=True,
        environment_specific={
            EnvironmentType.DEVELOPMENT: "dev-secret-key-change-in-production",
            EnvironmentType.PRODUCTION: None  # Must be provided
        }
    )
    
    # Processing configuration
    MAX_CONCURRENT_REQUESTS = EnvironmentVariable(
        name="MAX_CONCURRENT_REQUESTS",
        description="Maximum concurrent LLM requests",
        required=False,
        default="10",
        validator_func=lambda x: x.isdigit() and int(x) > 0
    )
    
    BATCH_SIZE = EnvironmentVariable(
        name="BATCH_SIZE",
        description="Default batch size for processing",
        required=False,
        default="50",
        validator_func=lambda x: x.isdigit() and int(x) > 0
    )
    
    @classmethod
    def get_all_variables(cls) -> List[EnvironmentVariable]:
        """Get all defined environment variables."""
        variables = []
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if isinstance(attr, EnvironmentVariable):
                variables.append(attr)
        return variables


class EnvironmentManager:
    """Centralized environment variable management with validation and security."""
    
    def __init__(
        self,
        env_file: Optional[Union[str, Path]] = None,
        environment: Optional[EnvironmentType] = None,
        validation_level: ValidationLevel = ValidationLevel.STRICT,
        auto_load: bool = True
    ):
        """
        Initialize environment manager.
        
        Args:
            env_file: Path to .env file. If None, searches for .env files
            environment: Target environment type
            validation_level: Level of validation to perform
            auto_load: Whether to automatically load environment variables
        """
        self.env_file = env_file
        self.environment = environment or self._detect_environment()
        self.validation_level = validation_level
        self.schema = EnvironmentSchema()
        self._loaded_variables: Dict[str, str] = {}
        self._validation_errors: List[str] = []
        self._validation_warnings: List[str] = []
        
        if auto_load:
            self.load_environment()
    
    def _detect_environment(self) -> EnvironmentType:
        """Detect current environment from environment variables."""
        env_name = os.getenv("ENVIRONMENT", "development").lower()
        try:
            return EnvironmentType(env_name)
        except ValueError:
            warnings.warn(f"Unknown environment '{env_name}', defaulting to development")
            return EnvironmentType.DEVELOPMENT
    
    def load_environment(self) -> bool:
        """
        Load environment variables from .env file and system environment.
        
        Returns:
            True if loading was successful, False otherwise
        """
        try:
            # Load from .env file
            if self.env_file:
                env_path = Path(self.env_file)
                if env_path.exists():
                    load_dotenv(env_path, override=True)
                else:
                    self._validation_errors.append(f"Environment file not found: {env_path}")
            else:
                # Search for .env files
                env_path = find_dotenv()
                if env_path:
                    load_dotenv(env_path, override=True)
            
            # Load environment-specific file if it exists
            env_specific_file = f".env.{self.environment.value}"
            env_specific_path = Path(env_specific_file)
            if env_specific_path.exists():
                load_dotenv(env_specific_path, override=True)
            
            # Validate all variables
            self._validate_environment()
            
            # Report validation results
            if self.validation_level != ValidationLevel.DISABLED:
                self._report_validation_results()
            
            return len(self._validation_errors) == 0
            
        except Exception as e:
            self._validation_errors.append(f"Failed to load environment: {e}")
            return False
    
    def _validate_environment(self) -> None:
        """Validate all environment variables according to schema."""
        self._validation_errors.clear()
        self._validation_warnings.clear()
        
        for var in self.schema.get_all_variables():
            value = os.getenv(var.name)
            
            # Check if required variable is missing
            if var.required and not value:
                # Check for environment-specific default
                env_default = var.environment_specific.get(self.environment)
                if env_default:
                    os.environ[var.name] = env_default
                    value = env_default
                elif var.default:
                    os.environ[var.name] = var.default
                    value = var.default
                else:
                    self._validation_errors.append(
                        f"Required environment variable '{var.name}' is not set. "
                        f"Description: {var.description}"
                    )
                    continue
            
            # Set default if variable is missing but not required
            if not value and var.default:
                os.environ[var.name] = var.default
                value = var.default
            
            # Validate value if present
            if value:
                if not var.validate(value):
                    error_msg = f"Invalid value for '{var.name}': {value}"
                    if var.allowed_values:
                        error_msg += f" (allowed: {', '.join(var.allowed_values)})"
                    self._validation_errors.append(error_msg)
                else:
                    self._loaded_variables[var.name] = value
    
    def _report_validation_results(self) -> None:
        """Report validation results based on validation level."""
        if self._validation_errors:
            error_msg = "Environment validation failed:\n" + "\n".join(
                f"  - {error}" for error in self._validation_errors
            )
            
            if self.validation_level == ValidationLevel.STRICT:
                raise EnvironmentError(error_msg)
            else:
                warnings.warn(error_msg)
        
        if self._validation_warnings:
            warning_msg = "Environment validation warnings:\n" + "\n".join(
                f"  - {warning}" for warning in self._validation_warnings
            )
            warnings.warn(warning_msg)
    
    def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get environment variable value.
        
        Args:
            name: Environment variable name
            default: Default value if not found
            
        Returns:
            Environment variable value or default
        """
        return os.getenv(name, default)
    
    def get_required(self, name: str) -> str:
        """
        Get required environment variable value.
        
        Args:
            name: Environment variable name
            
        Returns:
            Environment variable value
            
        Raises:
            EnvironmentError: If variable is not set
        """
        value = os.getenv(name)
        if value is None:
            raise EnvironmentError(f"Required environment variable '{name}' is not set")
        return value
    
    def set(self, name: str, value: str, override: bool = True) -> None:
        """
        Set environment variable.
        
        Args:
            name: Environment variable name
            value: Environment variable value
            override: Whether to override existing value
        """
        if override or name not in os.environ:
            os.environ[name] = value
            self._loaded_variables[name] = value
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == EnvironmentType.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == EnvironmentType.DEVELOPMENT
    
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == EnvironmentType.TESTING
    
    def get_database_url(self) -> str:
        """Get MongoDB connection string."""
        return self.get_required("MONGODB_CONNECTION_STRING")
    
    def get_elasticsearch_config(self) -> Dict[str, Any]:
        """Get Elasticsearch configuration."""
        config = {
            "url": self.get_required("ELASTICSEARCH_URL"),
            "username": self.get("ELASTICSEARCH_USERNAME"),
            "password": self.get("ELASTICSEARCH_PASSWORD"),
        }
        return {k: v for k, v in config.items() if v is not None}
    
    def get_llm_endpoints(self) -> Dict[str, str]:
        """Get LLM service endpoints."""
        endpoints = {}
        
        ollama_endpoint = self.get("OLLAMA_ENDPOINT")
        if ollama_endpoint:
            endpoints["ollama"] = ollama_endpoint
            
        lmstudio_endpoint = self.get("LMSTUDIO_ENDPOINT")
        if lmstudio_endpoint:
            endpoints["lmstudio"] = lmstudio_endpoint
            
        return endpoints
    
    def get_api_keys(self) -> Dict[str, str]:
        """Get API keys for external services."""
        keys = {}
        
        openai_key = self.get("OPENAI_API_KEY")
        if openai_key:
            keys["openai"] = openai_key
            
        anthropic_key = self.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            keys["anthropic"] = anthropic_key
            
        return keys
    
    def export_to_dict(self, include_sensitive: bool = False) -> Dict[str, str]:
        """
        Export environment variables to dictionary.
        
        Args:
            include_sensitive: Whether to include sensitive variables
            
        Returns:
            Dictionary of environment variables
        """
        result = {}
        
        for var in self.schema.get_all_variables():
            value = self.get(var.name)
            if value is not None:
                if var.sensitive and not include_sensitive:
                    result[var.name] = "*" * min(len(value), 8)
                else:
                    result[var.name] = value
                    
        return result
    
    def generate_env_template(self, output_path: Optional[Path] = None) -> str:
        """
        Generate .env template file with all variables and descriptions.
        
        Args:
            output_path: Optional path to write template file
            
        Returns:
            Template content as string
        """
        lines = [
            "# Environment Configuration for pub-analysis-agent",
            "# Copy this file to .env and fill in your values",
            "",
            "# =============================================================================",
            "# DEPLOYMENT ENVIRONMENT",
            "# =============================================================================",
            "",
        ]
        
        # Group variables by category
        categories = {
            "Database Configuration": ["MONGODB_"],
            "Elasticsearch Configuration": ["ELASTICSEARCH_"],
            "LLM Service Configuration": ["OLLAMA_", "LMSTUDIO_"],
            "API Keys": ["_API_KEY"],
            "Application Configuration": ["ENVIRONMENT", "LOG_", "SECRET_KEY"],
            "Processing Configuration": ["MAX_CONCURRENT_", "BATCH_SIZE"],
        }
        
        for category, prefixes in categories.items():
            lines.extend([
                f"# {category}",
                "# " + "=" * (len(category) + 2),
                "",
            ])
            
            for var in self.schema.get_all_variables():
                if any(prefix in var.name for prefix in prefixes):
                    # Add description as comment
                    lines.append(f"# {var.description}")
                    
                    # Add required/optional indicator
                    if var.required:
                        lines.append("# Required")
                    else:
                        lines.append("# Optional")
                    
                    # Add allowed values if any
                    if var.allowed_values:
                        lines.append(f"# Allowed values: {', '.join(var.allowed_values)}")
                    
                    # Add environment variable line
                    if var.default:
                        if var.sensitive:
                            lines.append(f"{var.name}=# {var.default} (CHANGE THIS)")
                        else:
                            lines.append(f"{var.name}={var.default}")
                    else:
                        lines.append(f"{var.name}=")
                    
                    lines.append("")
        
        template_content = "\n".join(lines)
        
        if output_path:
            output_path.write_text(template_content)
            
        return template_content


# Global environment manager instance
_environment_manager: Optional[EnvironmentManager] = None


def get_environment_manager() -> EnvironmentManager:
    """Get the global environment manager instance."""
    global _environment_manager
    if _environment_manager is None:
        _environment_manager = EnvironmentManager()
    return _environment_manager


def setup_environment(
    env_file: Optional[Union[str, Path]] = None,
    environment: Optional[EnvironmentType] = None,
    validation_level: ValidationLevel = ValidationLevel.STRICT
) -> EnvironmentManager:
    """
    Setup and configure the global environment manager.
    
    Args:
        env_file: Path to .env file
        environment: Target environment type
        validation_level: Level of validation to perform
        
    Returns:
        Configured environment manager
    """
    global _environment_manager
    _environment_manager = EnvironmentManager(
        env_file=env_file,
        environment=environment,
        validation_level=validation_level
    )
    return _environment_manager 