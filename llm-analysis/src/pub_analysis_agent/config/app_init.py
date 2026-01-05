"""
Application initialization module for pub-analysis-agent.

This module provides functions to initialize the complete application
configuration and logging systems in the correct order.
"""

import sys
from pathlib import Path
from typing import Optional, Union

import structlog

from .settings import Settings, get_settings
from .loader import ConfigurationLoader, load_settings_from_file
from .logging_config import setup_logging, get_logger, set_correlation_id


def initialize_application(
    config_path: Optional[Union[str, Path]] = None,
    correlation_id: Optional[str] = None
) -> Settings:
    """
    Initialize the complete application with configuration and logging.
    
    This function:
    1. Loads configuration from file and environment variables
    2. Sets up the logging system
    3. Validates all configurations
    4. Sets initial correlation ID if provided
    
    Args:
        config_path: Optional path to YAML configuration file
        correlation_id: Optional initial correlation ID
        
    Returns:
        Initialized Settings object
        
    Raises:
        SystemExit: If initialization fails critically
    """
    try:
        # Load configuration
        if config_path:
            settings = load_settings_from_file(config_path)
        else:
            settings = get_settings()
        
        # Setup logging system
        setup_logging(settings.logging)
        
        # Get logger after logging is set up
        logger = get_logger("app_init")
        
        # Set correlation ID for initialization tracking
        if correlation_id:
            set_correlation_id(correlation_id)
        
        logger.info(
            "Application initialization completed successfully",
            config_source="file" if config_path else "environment",
            config_path=str(config_path) if config_path else None,
            log_level=settings.logging.level,
            environment=settings.environment,
            debug_mode=settings.development.debug_enabled,
        )
        
        return settings
        
    except Exception as e:
        # If logging isn't set up yet, fall back to print
        error_msg = f"Failed to initialize application: {e}"
        
        try:
            logger = get_logger("app_init")
            logger.critical(error_msg, exception=str(e))
        except:
            print(f"CRITICAL: {error_msg}", file=sys.stderr)
        
        sys.exit(1)


def initialize_minimal(log_level: str = "INFO") -> Settings:
    """
    Initialize application with minimal configuration for testing/development.
    
    This is useful for unit tests or development scenarios where you need
    basic logging without full configuration files.
    
    Args:
        log_level: Logging level to use
        
    Returns:
        Settings object with default values
    """
    try:
        # Get default settings
        settings = get_settings()
        
        # Override logging level
        settings.logging.level = log_level
        settings.logging.console_enabled = True
        settings.logging.file_enabled = False
        settings.logging.json_format = False
        
        # Setup logging
        setup_logging(settings.logging)
        
        logger = get_logger("app_init")
        logger.info(
            "Minimal application initialization completed",
            log_level=log_level,
            mode="minimal"
        )
        
        return settings
        
    except Exception as e:
        print(f"CRITICAL: Failed to initialize minimal application: {e}", file=sys.stderr)
        sys.exit(1)


def validate_runtime_environment(settings: Settings) -> bool:
    """
    Validate that the runtime environment meets requirements.
    
    Args:
        settings: Application settings to validate against
        
    Returns:
        True if environment is valid, False otherwise
    """
    logger = get_logger("app_init.validation")
    
    validation_errors = []
    
    # Check required directories exist or can be created
    try:
        log_dir = Path(settings.logging.log_directory)
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Log directory validated", path=str(log_dir))
    except Exception as e:
        validation_errors.append(f"Cannot create log directory: {e}")
    
    # Validate database connection string format
    if not settings.database.connection_string.startswith(('mongodb://', 'mongodb+srv://')):
        validation_errors.append("Invalid MongoDB connection string format")
    
    # Validate Elasticsearch URL format
    if not settings.elasticsearch.url.startswith(('http://', 'https://')):
        validation_errors.append("Invalid Elasticsearch URL format")
    
    # Validate Elasticsearch CA certificate path if provided
    if settings.elasticsearch.ca_cert_path:
        from pathlib import Path
        ca_cert_path = Path(settings.elasticsearch.ca_cert_path)
        if not ca_cert_path.is_absolute():
            # If relative path, make it relative to project root
            project_root = Path(__file__).parent.parent.parent.parent
            ca_cert_path = project_root / settings.elasticsearch.ca_cert_path
        
        if not ca_cert_path.exists():
            validation_errors.append(f"Elasticsearch CA certificate file not found: {ca_cert_path}")
    
    # Check LLM service endpoints if configured
    for service_name, endpoint in settings.llm.local_endpoints.items():
        if endpoint and not endpoint.startswith(('http://', 'https://')):
            validation_errors.append(f"Invalid {service_name} endpoint format: {endpoint}")
    
    # Log validation results
    if validation_errors:
        logger.error(
            "Runtime environment validation failed",
            errors=validation_errors,
            error_count=len(validation_errors)
        )
        for error in validation_errors:
            logger.error("Validation error", error=error)
        return False
    else:
        logger.info("Runtime environment validation passed")
        return True


def setup_development_environment() -> Settings:
    """
    Setup application for development with appropriate defaults.
    
    Returns:
        Settings configured for development
    """
    # Initialize with development defaults
    settings = initialize_minimal(log_level="DEBUG")
    
    # Override development-specific settings
    settings.development.debug_enabled = True
    settings.development.profiling_enabled = True
    settings.development.test_database_enabled = True
    
    # Use local services
    settings.llm.local_endpoints = {
        "ollama": "http://localhost:11434/api",
        "lmstudio": "http://localhost:1234/v1",
    }
    
    # Setup logging for development
    settings.logging.console_enabled = True
    settings.logging.file_enabled = True
    settings.logging.json_format = False  # Human-readable for development
    settings.logging.level = "DEBUG"
    
    # Re-setup logging with new settings
    setup_logging(settings.logging)
    
    logger = get_logger("app_init.dev")
    logger.info(
        "Development environment initialized",
        debug_enabled=settings.development.debug_enabled,
        profiling_enabled=settings.development.profiling_enabled,
        local_llm_endpoints=list(settings.llm.local_endpoints.keys())
    )
    
    return settings


def setup_production_environment(config_path: Union[str, Path]) -> Settings:
    """
    Setup application for production with validation and security checks.
    
    Args:
        config_path: Path to production configuration file
        
    Returns:
        Settings configured for production
    """
    settings = initialize_application(config_path)
    
    # Validate production environment
    if not validate_runtime_environment(settings):
        logger = get_logger("app_init.production")
        logger.critical("Production environment validation failed")
        sys.exit(1)
    
    # Ensure production settings
    settings.development.debug_enabled = False
    settings.development.profiling_enabled = False
    settings.development.test_database_enabled = False
    
    # Ensure secure logging
    if settings.environment == "production":
        settings.logging.json_format = True  # Structured logging for production
    
    logger = get_logger("app_init.production")
    logger.info(
        "Production environment initialized successfully",
        environment=settings.environment,
        validation_passed=True
    )
    
    return settings 