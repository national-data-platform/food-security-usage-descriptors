"""
Circuit breaker pattern implementation for service resilience.

This module provides circuit breaker functionality to prevent cascading failures
when external services (like MongoDB) become unavailable or degraded.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Optional, Callable, Any, Dict, Union, List
from dataclasses import dataclass, field
from functools import wraps
import threading


logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Service unavailable, failing fast
    HALF_OPEN = "half_open" # Testing if service is recovered


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changed_time: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    
    def __init__(self, service_name: str, state: CircuitBreakerState):
        self.service_name = service_name
        self.state = state
        super().__init__(f"Circuit breaker for {service_name} is {state.value}")


class CircuitBreaker:
    """
    Circuit breaker implementation for service resilience.
    
    The circuit breaker monitors service failures and opens the circuit when
    failure thresholds are exceeded, preventing further requests until the
    service recovers.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service unavailable, requests fail fast
    - HALF_OPEN: Testing recovery, limited requests allowed
    
    Attributes:
        service_name: Name of the service being protected
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time to wait before attempting recovery (seconds)
        success_threshold: Successes needed in half-open state to close circuit
        request_timeout: Maximum time to wait for requests (seconds)
    """
    
    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3,
        request_timeout: float = 30.0,
        expected_exceptions: tuple = (Exception,)
    ):
        """
        Initialize circuit breaker.
        
        Args:
            service_name: Name of the service being protected
            failure_threshold: Number of consecutive failures before opening
            recovery_timeout: Seconds to wait before testing recovery
            success_threshold: Consecutive successes needed to close circuit
            request_timeout: Maximum request timeout in seconds
            expected_exceptions: Exceptions that count as failures
        """
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.request_timeout = request_timeout
        self.expected_exceptions = expected_exceptions
        
        self._state = CircuitBreakerState.CLOSED
        self._stats = CircuitBreakerStats()
        self._lock = threading.RLock()
        
        logger.info(
            f"Circuit breaker initialized for {service_name}: "
            f"failure_threshold={failure_threshold}, "
            f"recovery_timeout={recovery_timeout}s, "
            f"success_threshold={success_threshold}"
        )
    
    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        with self._lock:
            return self._state
    
    @property
    def stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics."""
        with self._lock:
            return CircuitBreakerStats(
                total_requests=self._stats.total_requests,
                successful_requests=self._stats.successful_requests,
                failed_requests=self._stats.failed_requests,
                last_failure_time=self._stats.last_failure_time,
                last_success_time=self._stats.last_success_time,
                state_changed_time=self._stats.state_changed_time,
                consecutive_failures=self._stats.consecutive_failures,
                consecutive_successes=self._stats.consecutive_successes
            )
    
    def _change_state(self, new_state: CircuitBreakerState) -> None:
        """Change circuit breaker state and log the transition."""
        old_state = self._state
        self._state = new_state
        self._stats.state_changed_time = time.time()
        
        logger.warning(
            f"Circuit breaker for {self.service_name} changed state: "
            f"{old_state.value} -> {new_state.value}"
        )
    
    def _should_attempt_recovery(self) -> bool:
        """Check if circuit breaker should attempt recovery."""
        if self._state != CircuitBreakerState.OPEN:
            return False
        
        time_since_failure = time.time() - (self._stats.last_failure_time or 0)
        return time_since_failure >= self.recovery_timeout
    
    def _record_success(self) -> None:
        """Record a successful request."""
        current_time = time.time()
        
        self._stats.total_requests += 1
        self._stats.successful_requests += 1
        self._stats.last_success_time = current_time
        self._stats.consecutive_failures = 0
        self._stats.consecutive_successes += 1
        
        # Transition from HALF_OPEN to CLOSED if enough successes
        if (self._state == CircuitBreakerState.HALF_OPEN and 
            self._stats.consecutive_successes >= self.success_threshold):
            self._change_state(CircuitBreakerState.CLOSED)
            self._stats.consecutive_successes = 0
    
    def _record_failure(self, exception: Exception) -> None:
        """Record a failed request."""
        current_time = time.time()
        
        self._stats.total_requests += 1
        self._stats.failed_requests += 1
        self._stats.last_failure_time = current_time
        self._stats.consecutive_successes = 0
        self._stats.consecutive_failures += 1
        
        logger.warning(
            f"Circuit breaker for {self.service_name} recorded failure "
            f"({self._stats.consecutive_failures}/{self.failure_threshold}): {exception}"
        )
        
        # Transition to OPEN if failure threshold exceeded
        if (self._state == CircuitBreakerState.CLOSED and 
            self._stats.consecutive_failures >= self.failure_threshold):
            self._change_state(CircuitBreakerState.OPEN)
        elif self._state == CircuitBreakerState.HALF_OPEN:
            self._change_state(CircuitBreakerState.OPEN)
            self._stats.consecutive_failures = 0
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Original exception from function
        """
        with self._lock:
            # Check if circuit is open and recovery should be attempted
            if self._state == CircuitBreakerState.OPEN:
                if self._should_attempt_recovery():
                    self._change_state(CircuitBreakerState.HALF_OPEN)
                    logger.info(f"Circuit breaker for {self.service_name} attempting recovery")
                else:
                    raise CircuitBreakerError(self.service_name, self._state)
        
        # Execute function with timeout
        try:
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs), 
                    timeout=self.request_timeout
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, func, *args, **kwargs),
                    timeout=self.request_timeout
                )
            
            with self._lock:
                self._record_success()
            
            return result
            
        except self.expected_exceptions as e:
            with self._lock:
                self._record_failure(e)
            raise
        except asyncio.TimeoutError as e:
            with self._lock:
                self._record_failure(e)
            logger.error(f"Request timeout in circuit breaker for {self.service_name}")
            raise
    
    def reset(self) -> None:
        """Reset circuit breaker to closed state and clear statistics."""
        with self._lock:
            old_state = self._state
            self._state = CircuitBreakerState.CLOSED
            self._stats = CircuitBreakerStats()
            
            logger.info(f"Circuit breaker for {self.service_name} reset from {old_state.value}")
    
    def force_open(self) -> None:
        """Force circuit breaker to open state."""
        with self._lock:
            self._change_state(CircuitBreakerState.OPEN)
            self._stats.last_failure_time = time.time()
    
    def force_close(self) -> None:
        """Force circuit breaker to closed state."""
        with self._lock:
            self._change_state(CircuitBreakerState.CLOSED)
            self._stats.consecutive_failures = 0
            self._stats.consecutive_successes = 0
    
    def get_health_info(self) -> Dict[str, Any]:
        """
        Get health information for monitoring.
        
        Returns:
            Dictionary with health and statistics information
        """
        with self._lock:
            stats = self.stats
            current_time = time.time()
            
            return {
                "service_name": self.service_name,
                "state": self._state.value,
                "healthy": self._state == CircuitBreakerState.CLOSED,
                "stats": {
                    "total_requests": stats.total_requests,
                    "successful_requests": stats.successful_requests,
                    "failed_requests": stats.failed_requests,
                    "success_rate": (
                        stats.successful_requests / stats.total_requests 
                        if stats.total_requests > 0 else 0.0
                    ),
                    "consecutive_failures": stats.consecutive_failures,
                    "consecutive_successes": stats.consecutive_successes,
                    "last_failure_time": stats.last_failure_time,
                    "last_success_time": stats.last_success_time,
                    "time_since_last_failure": (
                        current_time - stats.last_failure_time 
                        if stats.last_failure_time else None
                    ),
                    "state_changed_time": stats.state_changed_time,
                    "time_in_current_state": current_time - stats.state_changed_time
                },
                "config": {
                    "failure_threshold": self.failure_threshold,
                    "recovery_timeout": self.recovery_timeout,
                    "success_threshold": self.success_threshold,
                    "request_timeout": self.request_timeout
                }
            }


def circuit_breaker(
    service_name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    success_threshold: int = 3,
    request_timeout: float = 30.0,
    expected_exceptions: tuple = (Exception,)
):
    """
    Decorator to wrap functions with circuit breaker protection.
    
    Args:
        service_name: Name of the service being protected
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time to wait before attempting recovery (seconds)
        success_threshold: Successes needed in half-open state to close circuit
        request_timeout: Maximum time to wait for requests (seconds)
        expected_exceptions: Exceptions that count as failures
    """
    def decorator(func):
        breaker = CircuitBreaker(
            service_name=service_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
            request_timeout=request_timeout,
            expected_exceptions=expected_exceptions
        )
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)
        
        # Attach circuit breaker to wrapper for external access
        wrapper.circuit_breaker = breaker
        return wrapper
    
    return decorator


class CircuitBreakerManager:
    """
    Centralized manager for multiple circuit breakers.
    
    This class provides a registry for circuit breakers and methods for
    monitoring and controlling multiple services.
    """
    
    def __init__(self):
        """Initialize circuit breaker manager."""
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
    
    def register(self, circuit_breaker: CircuitBreaker) -> None:
        """
        Register a circuit breaker.
        
        Args:
            circuit_breaker: Circuit breaker instance to register
        """
        with self._lock:
            self._breakers[circuit_breaker.service_name] = circuit_breaker
            logger.info(f"Registered circuit breaker for {circuit_breaker.service_name}")
    
    def get(self, service_name: str) -> Optional[CircuitBreaker]:
        """
        Get circuit breaker by service name.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Circuit breaker instance or None if not found
        """
        with self._lock:
            return self._breakers.get(service_name)
    
    def get_all_health_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get health information for all registered circuit breakers.
        
        Returns:
            Dictionary mapping service names to health information
        """
        with self._lock:
            return {
                name: breaker.get_health_info()
                for name, breaker in self._breakers.items()
            }
    
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
            logger.info("Reset all circuit breakers")
    
    def get_unhealthy_services(self) -> List[str]:
        """
        Get list of services with open circuit breakers.
        
        Returns:
            List of service names with open circuits
        """
        with self._lock:
            return [
                name for name, breaker in self._breakers.items()
                if breaker.state != CircuitBreakerState.CLOSED
            ]


# Global circuit breaker manager instance
circuit_breaker_manager = CircuitBreakerManager() 