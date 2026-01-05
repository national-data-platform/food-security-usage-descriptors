"""
Utility modules for the publication analysis agent.

This module provides utilities for circuit breaking, concurrent operations,
error handling, regex pattern matching, and other support functionality.
"""

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerError,
    CircuitBreakerManager,
    circuit_breaker,
    circuit_breaker_manager
)

from .concurrent_operations import (
    OperationResult,
    RateLimiter,
    ConcurrentOperationManager,
    BatchProcessor,
    ConnectionPool
)

from .regex_pattern_engine import (
    RegexPatternEngine,
    GitHubURLInfo,
    CodeBlockInfo,
    ExternalLinkInfo,
    PatternType,
    PatternMatch
)

__all__ = [
    # Circuit breaker utilities
    "CircuitBreaker",
    "CircuitBreakerState", 
    "CircuitBreakerError",
    "CircuitBreakerManager",
    "circuit_breaker",
    "circuit_breaker_manager",
    
    # Concurrent operations utilities
    "OperationResult",
    "RateLimiter",
    "ConcurrentOperationManager", 
    "BatchProcessor",
    "ConnectionPool",
    
    # Regex pattern engine utilities
    "RegexPatternEngine",
    "GitHubURLInfo",
    "CodeBlockInfo", 
    "ExternalLinkInfo",
    "PatternType",
    "PatternMatch"
] 