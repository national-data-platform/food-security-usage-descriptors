"""
Enhanced MongoDB client with circuit breaker, graceful degradation, and comprehensive error handling.

This module extends the basic MongoDB client with advanced error handling patterns,
circuit breaker protection, concurrent operation support, and graceful degradation.
"""

import logging
import time
import asyncio
from typing import Optional, Dict, Any, List, Callable, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum

import pymongo
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import (
    ServerSelectionTimeoutError,
    ConnectionFailure,
    OperationFailure,
    ConfigurationError,
    NetworkTimeout,
    PyMongoError,
    AutoReconnect,
    ExecutionTimeout
)

from .mongodb_client import MongoDBClient
from ..config.settings import DatabaseSettings
from ..utils.circuit_breaker import CircuitBreaker, CircuitBreakerError, circuit_breaker_manager
from ..utils.concurrent_operations import ConcurrentOperationManager, OperationResult


logger = logging.getLogger(__name__)


class ServiceDegradationLevel(Enum):
    """Levels of service degradation."""
    NORMAL = "normal"           # Full functionality
    DEGRADED = "degraded"       # Reduced functionality, slower operations
    READ_ONLY = "read_only"     # Only read operations allowed
    CIRCUIT_OPEN = "circuit_open"  # Circuit breaker open, no operations


@dataclass
class DatabaseHealthStatus:
    """Database health status information."""
    database_name: str
    healthy: bool
    degradation_level: ServiceDegradationLevel
    last_error: Optional[str] = None
    last_success_time: Optional[float] = None
    last_failure_time: Optional[float] = None
    connection_count: int = 0
    response_time_ms: Optional[float] = None


class EnhancedMongoDBClient(MongoDBClient):
    """
    Enhanced MongoDB client with circuit breaker, graceful degradation, and comprehensive error handling.
    
    This client extends the basic MongoDBClient with:
    - Circuit breaker pattern for each database
    - Graceful degradation modes
    - Comprehensive error classification
    - Enhanced concurrent operation support
    - Detailed health monitoring
    - Fallback mechanisms
    """
    
    def __init__(
        self, 
        db_settings: DatabaseSettings,
        circuit_breaker_config: Optional[Dict[str, Any]] = None,
        degradation_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize enhanced MongoDB client.
        
        Args:
            db_settings: Database configuration settings
            circuit_breaker_config: Circuit breaker configuration options
            degradation_config: Service degradation configuration options
        """
        super().__init__(db_settings)
        
        # Circuit breaker configuration
        cb_config = circuit_breaker_config or {}
        self.circuit_breaker_config = {
            'failure_threshold': cb_config.get('failure_threshold', 5),
            'recovery_timeout': cb_config.get('recovery_timeout', 60.0),
            'success_threshold': cb_config.get('success_threshold', 3),
            'request_timeout': cb_config.get('request_timeout', 30.0)
        }
        
        # Service degradation configuration
        self.degradation_config = degradation_config or {
            'read_only_threshold': 3,  # Failures before going read-only
            'degraded_threshold': 2,   # Failures before degraded mode
            'recovery_check_interval': 30.0  # Seconds between recovery checks
        }
        
        # Circuit breakers for each database
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._initialize_circuit_breakers()
        
        # Concurrent operation manager
        self.operation_manager = ConcurrentOperationManager(
            max_concurrent=db_settings.max_pool_size,
            rate_limit=None,  # No rate limiting by default
            timeout=self.circuit_breaker_config['request_timeout'],
            name="mongodb_operations"
        )
        
        # Health status tracking
        self.database_health: Dict[str, DatabaseHealthStatus] = {}
        self._initialize_health_tracking()
        
        # Degradation state
        self.service_degradation_level = ServiceDegradationLevel.NORMAL
        self._last_degradation_check = 0.0
        
        logger.info("Enhanced MongoDB client initialized with circuit breakers and degradation support")
    
    def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breakers for each database."""
        databases = ['general', 'dimensions', 'analysis']
        
        for db_name in databases:
            breaker = CircuitBreaker(
                service_name=f"mongodb_{db_name}",
                failure_threshold=self.circuit_breaker_config['failure_threshold'],
                recovery_timeout=self.circuit_breaker_config['recovery_timeout'],
                success_threshold=self.circuit_breaker_config['success_threshold'],
                request_timeout=self.circuit_breaker_config['request_timeout'],
                expected_exceptions=(PyMongoError, ConnectionFailure, OperationFailure, TimeoutError)
            )
            
            self.circuit_breakers[db_name] = breaker
            circuit_breaker_manager.register(breaker)
            
            logger.info(f"Circuit breaker initialized for {db_name} database")
    
    def _initialize_health_tracking(self) -> None:
        """Initialize health status tracking for databases."""
        databases = ['general', 'dimensions', 'analysis']
        
        for db_name in databases:
            self.database_health[db_name] = DatabaseHealthStatus(
                database_name=db_name,
                healthy=True,
                degradation_level=ServiceDegradationLevel.NORMAL
            )
    
    async def _execute_with_circuit_breaker(
        self,
        database_name: str,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute database operation through circuit breaker.
        
        Args:
            database_name: Name of the database
            operation: Operation to execute
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Operation result
            
        Raises:
            CircuitBreakerError: If circuit breaker is open
            Exception: Original operation exception
        """
        breaker = self.circuit_breakers.get(database_name)
        if not breaker:
            # No circuit breaker, execute directly
            return await operation(*args, **kwargs)
        
        try:
            start_time = time.time()
            result = await breaker.call(operation, *args, **kwargs)
            
            # Update health status on success
            response_time = (time.time() - start_time) * 1000  # ms
            self._update_health_status(database_name, success=True, response_time=response_time)
            
            return result
            
        except CircuitBreakerError as e:
            self._update_health_status(database_name, success=False, error=str(e))
            await self._handle_circuit_breaker_open(database_name)
            raise
        except Exception as e:
            self._update_health_status(database_name, success=False, error=str(e))
            raise
    
    def _update_health_status(
        self,
        database_name: str,
        success: bool,
        response_time: Optional[float] = None,
        error: Optional[str] = None
    ) -> None:
        """Update health status for a database."""
        if database_name not in self.database_health:
            return
        
        health = self.database_health[database_name]
        current_time = time.time()
        
        if success:
            health.healthy = True
            health.last_success_time = current_time
            health.response_time_ms = response_time
            
            # Check if we can improve degradation level
            if health.degradation_level != ServiceDegradationLevel.NORMAL:
                self._check_recovery(database_name)
        else:
            health.healthy = False
            health.last_failure_time = current_time
            health.last_error = error
            
            # Check if we need to degrade service
            self._check_degradation(database_name)
    
    def _check_degradation(self, database_name: str) -> None:
        """Check if service should be degraded for a database."""
        breaker = self.circuit_breakers.get(database_name)
        if not breaker:
            return
        
        health = self.database_health[database_name]
        consecutive_failures = breaker.stats.consecutive_failures
        
        # Determine degradation level
        if consecutive_failures >= self.degradation_config['read_only_threshold']:
            health.degradation_level = ServiceDegradationLevel.READ_ONLY
            logger.warning(f"Database {database_name} degraded to READ_ONLY mode")
        elif consecutive_failures >= self.degradation_config['degraded_threshold']:
            health.degradation_level = ServiceDegradationLevel.DEGRADED
            logger.warning(f"Database {database_name} degraded to DEGRADED mode")
        
        # Update overall service degradation level
        self._update_overall_degradation_level()
    
    def _check_recovery(self, database_name: str) -> None:
        """Check if service can be recovered for a database."""
        breaker = self.circuit_breakers.get(database_name)
        if not breaker:
            return
        
        health = self.database_health[database_name]
        
        # If circuit breaker is closed and we have recent successes, improve degradation
        if breaker.state.value == "closed" and breaker.stats.consecutive_successes >= 2:
            health.degradation_level = ServiceDegradationLevel.NORMAL
            logger.info(f"Database {database_name} recovered to NORMAL mode")
            
            # Update overall service degradation level
            self._update_overall_degradation_level()
    
    def _update_overall_degradation_level(self) -> None:
        """Update overall service degradation level based on individual databases."""
        # Find worst degradation level across all databases
        worst_level = ServiceDegradationLevel.NORMAL
        
        for health in self.database_health.values():
            if health.degradation_level == ServiceDegradationLevel.READ_ONLY:
                worst_level = ServiceDegradationLevel.READ_ONLY
                break
            elif health.degradation_level == ServiceDegradationLevel.DEGRADED:
                worst_level = ServiceDegradationLevel.DEGRADED
        
        if worst_level != self.service_degradation_level:
            old_level = self.service_degradation_level
            self.service_degradation_level = worst_level
            logger.warning(f"Overall service degradation changed: {old_level.value} -> {worst_level.value}")
    
    async def _handle_circuit_breaker_open(self, database_name: str) -> None:
        """Handle circuit breaker opening for a database."""
        health = self.database_health[database_name]
        health.degradation_level = ServiceDegradationLevel.CIRCUIT_OPEN
        
        logger.error(f"Circuit breaker OPEN for database {database_name}")
        
        # Update overall degradation level
        self._update_overall_degradation_level()
        
        # Could implement fallback mechanisms here, such as:
        # - Switch to cached data
        # - Use read-only replicas
        # - Return degraded responses
    
    async def connect_with_circuit_breaker(self) -> None:
        """Connect to MongoDB with circuit breaker protection."""
        async def _connect():
            return await super().connect()
        
        try:
            await self._execute_with_circuit_breaker('general', _connect)
            logger.info("Enhanced MongoDB client connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect enhanced MongoDB client: {e}")
            raise
    
    async def health_check_enhanced(self, force: bool = False) -> Dict[str, Any]:
        """
        Enhanced health check with circuit breaker status and degradation info.
        
        Args:
            force: Force health check even if within interval
            
        Returns:
            Comprehensive health information
        """
        current_time = time.time()
        
        # Check if we should perform health check
        if not force and (current_time - self._last_degradation_check) < 30.0:
            return self._get_cached_health_info()
        
        self._last_degradation_check = current_time
        
        # Perform health checks for each database
        health_results = {}
        
        for db_name in ['general', 'dimensions', 'analysis']:
            try:
                # Test database connectivity
                start_time = time.time()
                database = getattr(self, f"{db_name}_database", None)
                
                if database:
                    await self._execute_with_circuit_breaker(
                        db_name,
                        lambda: asyncio.get_event_loop().run_in_executor(
                            None, lambda: database.command('ping')
                        )
                    )
                    
                    response_time = (time.time() - start_time) * 1000
                    self._update_health_status(db_name, success=True, response_time=response_time)
                
            except Exception as e:
                logger.warning(f"Health check failed for {db_name}: {e}")
                self._update_health_status(db_name, success=False, error=str(e))
        
        return self._get_comprehensive_health_info()
    
    def _get_cached_health_info(self) -> Dict[str, Any]:
        """Get cached health information."""
        return {
            'status': 'cached',
            'service_degradation_level': self.service_degradation_level.value,
            'databases': {name: self._health_status_to_dict(health) for name, health in self.database_health.items()},
            'circuit_breakers': circuit_breaker_manager.get_all_health_info(),
            'timestamp': time.time()
        }
    
    def _get_comprehensive_health_info(self) -> Dict[str, Any]:
        """Get comprehensive health information."""
        return {
            'status': 'healthy' if self.service_degradation_level == ServiceDegradationLevel.NORMAL else 'degraded',
            'service_degradation_level': self.service_degradation_level.value,
            'databases': {name: self._health_status_to_dict(health) for name, health in self.database_health.items()},
            'circuit_breakers': circuit_breaker_manager.get_all_health_info(),
            'operation_stats': self.operation_manager.get_stats(),
            'connection_info': super().get_connection_info(),
            'timestamp': time.time()
        }
    
    def _health_status_to_dict(self, health: DatabaseHealthStatus) -> Dict[str, Any]:
        """Convert health status to dictionary."""
        return {
            'healthy': health.healthy,
            'degradation_level': health.degradation_level.value,
            'last_error': health.last_error,
            'last_success_time': health.last_success_time,
            'last_failure_time': health.last_failure_time,
            'response_time_ms': health.response_time_ms
        }
    
    @asynccontextmanager
    async def get_connection_context_enhanced(self, database_name: str = 'general'):
        """
        Enhanced connection context manager with circuit breaker protection.
        
        Args:
            database_name: Name of the database to use
        """
        if self.service_degradation_level == ServiceDegradationLevel.CIRCUIT_OPEN:
            raise CircuitBreakerError(f"mongodb_{database_name}", "circuit_open")
        
        try:
            # Use base class connection context
            async with super().ensure_connection():
                yield self
        except Exception as e:
            # Update health status on connection failure
            self._update_health_status(database_name, success=False, error=str(e))
            raise
    
    async def execute_read_operation(
        self,
        database_name: str,
        collection_name: str,
        operation: Callable,
        *args,
        fallback_data: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """
        Execute read operation with degradation handling.
        
        Args:
            database_name: Name of the database
            collection_name: Name of the collection
            operation: Operation to execute
            *args: Operation arguments
            fallback_data: Data to return if operation fails
            **kwargs: Operation keyword arguments
            
        Returns:
            Operation result or fallback data
        """
        health = self.database_health.get(database_name)
        
        # Check if we can perform the operation
        if health and health.degradation_level == ServiceDegradationLevel.CIRCUIT_OPEN:
            if fallback_data is not None:
                logger.warning(f"Circuit open for {database_name}, returning fallback data")
                return fallback_data
            else:
                raise CircuitBreakerError(f"mongodb_{database_name}", "circuit_open")
        
        try:
            # Get collection through circuit breaker
            async def get_collection_and_execute():
                database = getattr(self, f"{database_name}_database")
                collection = database[collection_name]
                return await operation(collection, *args, **kwargs)
            
            return await self._execute_with_circuit_breaker(
                database_name,
                get_collection_and_execute
            )
            
        except Exception as e:
            logger.warning(f"Read operation failed for {database_name}.{collection_name}: {e}")
            
            if fallback_data is not None:
                logger.info("Returning fallback data due to operation failure")
                return fallback_data
            
            raise
    
    async def execute_write_operation(
        self,
        database_name: str,
        collection_name: str,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute write operation with degradation handling.
        
        Args:
            database_name: Name of the database
            collection_name: Name of the collection
            operation: Operation to execute
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Operation result
            
        Raises:
            OperationFailure: If write operations are not allowed in current degradation mode
        """
        health = self.database_health.get(database_name)
        
        # Check if write operations are allowed
        if health:
            if health.degradation_level in [ServiceDegradationLevel.READ_ONLY, ServiceDegradationLevel.CIRCUIT_OPEN]:
                raise OperationFailure(
                    f"Write operations not allowed in {health.degradation_level.value} mode "
                    f"for database {database_name}"
                )
        
        try:
            # Get collection through circuit breaker
            async def get_collection_and_execute():
                database = getattr(self, f"{database_name}_database")
                collection = database[collection_name]
                return await operation(collection, *args, **kwargs)
            
            return await self._execute_with_circuit_breaker(
                database_name,
                get_collection_and_execute
            )
            
        except Exception as e:
            logger.error(f"Write operation failed for {database_name}.{collection_name}: {e}")
            raise
    
    async def reset_circuit_breakers(self) -> None:
        """Reset all circuit breakers to closed state."""
        for breaker in self.circuit_breakers.values():
            breaker.reset()
        
        # Reset health status
        for health in self.database_health.values():
            health.healthy = True
            health.degradation_level = ServiceDegradationLevel.NORMAL
            health.last_error = None
        
        self.service_degradation_level = ServiceDegradationLevel.NORMAL
        logger.info("All circuit breakers reset to closed state")
    
    async def graceful_shutdown(self, timeout: float = 30.0) -> None:
        """
        Gracefully shutdown the enhanced MongoDB client.
        
        Args:
            timeout: Maximum time to wait for operations to complete
        """
        logger.info("Starting graceful shutdown of enhanced MongoDB client")
        
        # Wait for active operations to complete
        completed = await self.operation_manager.wait_for_completion(timeout)
        if not completed:
            # Cancel remaining operations
            cancelled = await self.operation_manager.cancel_all_operations()
            logger.warning(f"Cancelled {cancelled} operations during shutdown")
        
        # Disconnect from MongoDB
        await self.disconnect()
        
        logger.info("Enhanced MongoDB client shutdown completed")
    
    def get_degradation_recommendations(self) -> List[str]:
        """
        Get recommendations for dealing with current degradation.
        
        Returns:
            List of recommended actions
        """
        recommendations = []
        
        if self.service_degradation_level == ServiceDegradationLevel.NORMAL:
            recommendations.append("Service operating normally")
            return recommendations
        
        unhealthy_databases = [
            name for name, health in self.database_health.items()
            if not health.healthy
        ]
        
        if unhealthy_databases:
            recommendations.append(f"Check connectivity to databases: {', '.join(unhealthy_databases)}")
        
        if self.service_degradation_level == ServiceDegradationLevel.READ_ONLY:
            recommendations.extend([
                "Service in READ_ONLY mode - write operations disabled",
                "Consider implementing caching for read operations",
                "Monitor for database recovery"
            ])
        elif self.service_degradation_level == ServiceDegradationLevel.DEGRADED:
            recommendations.extend([
                "Service in DEGRADED mode - reduced performance expected",
                "Consider reducing operation frequency",
                "Monitor error rates and response times"
            ])
        elif self.service_degradation_level == ServiceDegradationLevel.CIRCUIT_OPEN:
            recommendations.extend([
                "Circuit breakers OPEN - operations failing fast",
                "Check database connectivity and health",
                "Consider manual circuit breaker reset if issue resolved"
            ])
        
        return recommendations 