"""
Configuration management for the publication analysis system.

This package handles loading and validation of configuration from
environment variables, YAML files, and other sources.
"""

from .settings import (
    Settings,
    DatabaseSettings,
    ElasticsearchSettings,
    LLMSettings,
    ProcessingSettings,
    LoggingSettings,
    SecuritySettings,
    DevelopmentSettings,
    get_settings,
    settings,
)

from .loader import (
    ConfigurationLoader,
    get_configuration_loader,
    load_settings_from_file,
    validate_configuration_file,
)

from .logging_config import (
    setup_logging,
    get_logger,
    get_correlation_id,
    set_correlation_id,
    clear_correlation_id,
    bind_context,
    clear_context,
    get_agent_logger,
    get_service_logger,
    get_workflow_logger,
    get_llm_logger,
    get_database_logger,
    LoggingManager,
    JSONFormatter,
    ColoredConsoleFormatter,
)

from .environment import (
    EnvironmentManager,
    EnvironmentType,
    ValidationLevel,
    EnvironmentSchema,
    EnvironmentVariable,
    get_environment_manager,
    setup_environment,
)

from .app_init import (
    initialize_application,
    initialize_minimal,
    validate_runtime_environment,
    setup_development_environment,
    setup_production_environment,
)

__all__ = [
    # Settings
    "Settings",
    "DatabaseSettings",
    "ElasticsearchSettings",
    "LLMSettings",
    "ProcessingSettings",
    "LoggingSettings",
    "SecuritySettings",
    "DevelopmentSettings",
    "get_settings",
    "settings",
    
    # Loader
    "ConfigurationLoader",
    "get_configuration_loader",
    "load_settings_from_file",
    "validate_configuration_file",
    
    # Logging
    "setup_logging",
    "get_logger",
    "get_correlation_id",
    "set_correlation_id",
    "clear_correlation_id",
    "bind_context",
    "clear_context",
    "get_agent_logger",
    "get_service_logger",
    "get_workflow_logger",
    "get_llm_logger",
    "get_database_logger",
    "LoggingManager",
    "JSONFormatter",
    "ColoredConsoleFormatter",
    
    # Environment Management
    "EnvironmentManager",
    "EnvironmentType",
    "ValidationLevel",
    "EnvironmentSchema",
    "EnvironmentVariable", 
    "get_environment_manager",
    "setup_environment",
    
    # App Initialization
    "initialize_application",
    "initialize_minimal",
    "validate_runtime_environment",
    "setup_development_environment",
    "setup_production_environment",
    
] 