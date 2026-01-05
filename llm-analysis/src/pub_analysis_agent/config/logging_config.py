"""
Logging configuration system for pub-analysis-agent.

This module provides structured logging with JSON formatting, correlation IDs,
multiple handlers, and proper configuration for all system components.
"""

import json
import logging
import logging.config
import logging.handlers
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
from contextvars import ContextVar

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars
from rich.logging import RichHandler
from rich.console import Console

from .settings import LoggingSettings


# Context variable for correlation IDs across async tasks
correlation_id: ContextVar[str] = ContextVar('correlation_id', default='')


class CorrelationIdProcessor:
    """Structlog processor that adds correlation IDs to log entries."""
    
    def __call__(self, logger, method_name, event_dict):
        """Add correlation ID to log entry if available."""
        cid = correlation_id.get()
        if cid:
            event_dict['correlation_id'] = cid
        return event_dict


class TimestampProcessor:
    """Structlog processor that adds ISO timestamps."""
    
    def __call__(self, logger, method_name, event_dict):
        """Add ISO timestamp to log entry."""
        event_dict['timestamp'] = datetime.utcnow().isoformat() + 'Z'
        return event_dict


class ComponentProcessor:
    """Structlog processor that adds component information."""
    
    def __call__(self, logger, method_name, event_dict):
        """Add component information to log entry."""
        logger_name = getattr(logger, 'name', 'unknown')
        event_dict['component'] = logger_name
        event_dict['level'] = method_name.upper()
        return event_dict


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'component': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add correlation ID if available
        cid = correlation_id.get()
        if cid:
            log_entry['correlation_id'] = cid
            
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 
                          'pathname', 'filename', 'module', 'lineno', 
                          'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process',
                          'getMessage', 'exc_info', 'exc_text', 'stack_info'):
                log_entry[key] = value
                
        return json.dumps(log_entry, default=str, ensure_ascii=False)


class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter for better readability in development."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors for console output."""
        color = self.COLORS.get(record.levelname, self.RESET)
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        
        # Get correlation ID
        cid = correlation_id.get()
        cid_str = f" [{cid[:8]}]" if cid else ""
        
        # Format the message
        formatted = (
            f"{color}[{timestamp}] {record.levelname:8} "
            f"{record.name}:{record.funcName}:{record.lineno}{cid_str} - "
            f"{record.getMessage()}{self.RESET}"
        )
        
        # Add exception information if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
            
        return formatted


class LoggingManager:
    """Centralized logging configuration and management."""
    
    def __init__(self, settings: LoggingSettings):
        """Initialize logging manager with settings."""
        self.settings = settings
        self.log_dir = Path(settings.log_directory)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_logging(self) -> None:
        """Setup comprehensive logging configuration."""
        # Setup standard library logging
        self._setup_stdlib_logging()
        
        # Setup structlog
        self._setup_structlog()
        
        # Log startup message
        logger = structlog.get_logger("logging_manager")
        logger.info(
            "Logging system initialized",
            log_level=self.settings.level,
            log_directory=str(self.log_dir),
            console_enabled=self.settings.console_enabled,
            file_enabled=self.settings.file_enabled,
            json_format=self.settings.json_format
        )
    
    def _setup_stdlib_logging(self) -> None:
        """Setup standard library logging configuration."""
        # Clear any existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        # Set root logger level
        root_logger.setLevel(getattr(logging, self.settings.level.upper()))
        
        handlers = []
        
        # Console handler
        if self.settings.console_enabled:
            if self.settings.json_format:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(JSONFormatter())
            else:
                # Use Rich handler for better console output in development
                console_handler = RichHandler(
                    console=Console(stderr=False),
                    show_time=True,
                    show_level=True,
                    show_path=True,
                    rich_tracebacks=True,
                    markup=True
                )
                console_handler.setFormatter(
                    logging.Formatter(
                        "%(name)s:%(funcName)s:%(lineno)d - %(message)s"
                    )
                )
            
            console_handler.setLevel(getattr(logging, self.settings.console_level.upper()))
            handlers.append(console_handler)
        
        # File handlers
        if self.settings.file_enabled:
            # Main log file with rotation
            main_log_path = self.log_dir / self.settings.main_log_file
            main_handler = logging.handlers.RotatingFileHandler(
                main_log_path,
                maxBytes=self.settings.max_file_size,
                backupCount=self.settings.backup_count,
                encoding='utf-8'
            )
            main_handler.setFormatter(JSONFormatter())
            main_handler.setLevel(getattr(logging, self.settings.file_level.upper()))
            handlers.append(main_handler)
            
            # Error log file (ERROR and CRITICAL only)
            error_log_path = self.log_dir / self.settings.error_log_file
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_path,
                maxBytes=self.settings.max_file_size,
                backupCount=self.settings.backup_count,
                encoding='utf-8'
            )
            error_handler.setFormatter(JSONFormatter())
            error_handler.setLevel(logging.ERROR)
            handlers.append(error_handler)
        
        # Add all handlers to root logger
        for handler in handlers:
            root_logger.addHandler(handler)
            
        # Configure specific loggers with different levels if specified
        for logger_name, level in self.settings.component_levels.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(getattr(logging, level.upper()))
    
    def _setup_structlog(self) -> None:
        """Setup structlog configuration."""
        processors = [
            # Add correlation ID and component info
            CorrelationIdProcessor(),
            ComponentProcessor(),
            TimestampProcessor(),
            
            # Add structlog's built-in processors
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_logger_name,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
        ]
        
        if self.settings.json_format:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.extend([
                structlog.dev.ConsoleRenderer(colors=True),
            ])
        
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, self.settings.level.upper())
            ),
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )


# Global logging manager instance
_logging_manager: Optional[LoggingManager] = None


def setup_logging(settings: LoggingSettings) -> None:
    """Setup logging system with the provided settings."""
    global _logging_manager
    _logging_manager = LoggingManager(settings)
    _logging_manager.setup_logging()


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance for the specified component.
    
    Args:
        name: Logger name (typically the module or component name)
        
    Returns:
        Configured structlog logger instance
    """
    return structlog.get_logger(name)


def get_correlation_id() -> str:
    """Get the current correlation ID."""
    return correlation_id.get()


def set_correlation_id(cid: Optional[str] = None) -> str:
    """
    Set correlation ID for the current context.
    
    Args:
        cid: Correlation ID to set. If None, generates a new UUID.
        
    Returns:
        The correlation ID that was set
    """
    if cid is None:
        cid = str(uuid.uuid4())
    correlation_id.set(cid)
    return cid


def clear_correlation_id() -> None:
    """Clear the current correlation ID."""
    correlation_id.set('')


def bind_context(**kwargs) -> None:
    """
    Bind additional context to all log messages in the current context.
    
    Args:
        **kwargs: Key-value pairs to bind to log context
    """
    bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all bound context variables."""
    clear_contextvars()


# Component-specific logger getters
def get_agent_logger(agent_name: str) -> structlog.BoundLogger:
    """Get logger for an LLM agent."""
    return get_logger(f"agents.{agent_name}")


def get_service_logger(service_name: str) -> structlog.BoundLogger:
    """Get logger for a service component."""
    return get_logger(f"services.{service_name}")


def get_workflow_logger() -> structlog.BoundLogger:
    """Get logger for LangGraph workflow operations."""
    return get_logger("workflow")


def get_llm_logger() -> structlog.BoundLogger:
    """Get logger for LLM interactions."""
    return get_logger("llm")


def get_database_logger() -> structlog.BoundLogger:
    """Get logger for database operations."""
    return get_logger("database") 