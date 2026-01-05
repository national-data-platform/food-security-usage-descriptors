"""
pub-analysis-agent: AI-powered academic publication analysis system.

This package provides a comprehensive system for analyzing academic publications
using LLM agents, extracting dataset information, and performing data analysis
classification through LangGraph workflows.

Quick Start:
    from pub_analysis_agent import setup_development_environment, get_logger
    
    # Initialize the system
    settings = setup_development_environment()
    
    # Get a logger
    logger = get_logger("my_component")
    logger.info("System initialized successfully")

Version: 0.1.0
Author: Publication Analysis Team
"""

from .config import (
    Settings,
    get_settings,
    settings,
    
    # Configuration management
    setup_development_environment,
    setup_production_environment,
    initialize_application,
    initialize_minimal,
    
    # Logging system
    setup_logging,
    get_logger,
    get_agent_logger,
    get_service_logger,
    get_workflow_logger,
    get_llm_logger,
    get_database_logger,
    set_correlation_id,
    get_correlation_id,
    bind_context,
    clear_context,
)

__version__ = "0.1.0"
__author__ = "Publication Analysis Team"

# Core exports for easy access
__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # Configuration
    "Settings",
    "get_settings", 
    "settings",
    
    # Application initialization
    "setup_development_environment",
    "setup_production_environment", 
    "initialize_application",
    "initialize_minimal",
    
    # Logging system
    "setup_logging",
    "get_logger",
    "get_agent_logger",
    "get_service_logger", 
    "get_workflow_logger",
    "get_llm_logger",
    "get_database_logger",
    "set_correlation_id",
    "get_correlation_id",
    "bind_context",
    "clear_context",
    
    # Models (when implemented)
    # "PublicationAnalysisState",
] 