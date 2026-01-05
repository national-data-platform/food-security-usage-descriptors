"""
Unit tests for the circuit breaker utility.

Tests cover circuit breaker states, error handling, recovery mechanisms,
statistics tracking, and centralized management functionality.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from src.pub_analysis_agent.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerError,
    CircuitBreakerManager,
    circuit_breaker,
    circuit_breaker_manager
)


class TestCircuitBreaker:
    """Test suite for CircuitBreaker class."""
    
    @pytest.fixture
    def circuit_breaker_instance(self):
        """Create a circuit breaker instance for testing."""
        return CircuitBreaker(
            service_name="test_service",
            failure_threshold=3,
            recovery_timeout=5.0,
            success_threshold=2,
            request_timeout=1.0
        )
    
    def test_circuit_breaker_initialization(self, circuit_breaker_instance):
        """Test circuit breaker proper initialization."""
        cb = circuit_breaker_instance
        
        assert cb.service_name == "test_service"
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 5.0
        assert cb.success_threshold == 2
        assert cb.request_timeout == 1.0
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.stats.total_requests == 0
        assert cb.stats.successful_requests == 0
        assert cb.stats.failed_requests == 0
    
    @pytest.mark.asyncio
    async def test_successful_operation(self, circuit_breaker_instance):
        """Test successful operation execution."""
        cb = circuit_breaker_instance
        
        async def successful_operation():
            return "success"
        
        result = await cb.call(successful_operation)
        
        assert result == "success"
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.stats.total_requests == 1
        assert cb.stats.successful_requests == 1
        assert cb.stats.failed_requests == 0
        assert cb.stats.consecutive_successes == 1
        assert cb.stats.consecutive_failures == 0
    
    @pytest.mark.asyncio
    async def test_failing_operation(self, circuit_breaker_instance):
        """Test failing operation execution."""
        cb = circuit_breaker_instance
        
        async def failing_operation():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            await cb.call(failing_operation)
        
        assert cb.state == CircuitBreakerState.CLOSED  # Still closed, under threshold
        assert cb.stats.total_requests == 1
        assert cb.stats.successful_requests == 0
        assert cb.stats.failed_requests == 1
        assert cb.stats.consecutive_failures == 1
    
    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self, circuit_breaker_instance):
        """Test circuit breaker opens after failure threshold."""
        cb = circuit_breaker_instance
        
        async def failing_operation():
            raise ValueError("Test error")
        
        # Execute enough failures to open circuit
        for i in range(cb.failure_threshold):
            with pytest.raises(ValueError):
                await cb.call(failing_operation)
            
            if i < cb.failure_threshold - 1:
                assert cb.state == CircuitBreakerState.CLOSED
            else:
                assert cb.state == CircuitBreakerState.OPEN
        
        assert cb.stats.consecutive_failures == cb.failure_threshold
    
    @pytest.mark.asyncio
    async def test_circuit_open_fails_fast(self, circuit_breaker_instance):
        """Test circuit breaker fails fast when open."""
        cb = circuit_breaker_instance
        
        # Force circuit open
        cb.force_open()
        
        async def any_operation():
            return "should not execute"
        
        with pytest.raises(CircuitBreakerError) as exc_info:
            await cb.call(any_operation)
        
        assert exc_info.value.service_name == "test_service"
        assert exc_info.value.state == CircuitBreakerState.OPEN
    
    @pytest.mark.asyncio
    async def test_circuit_half_open_recovery(self, circuit_breaker_instance):
        """Test circuit breaker half-open state and recovery."""
        cb = circuit_breaker_instance
        
        # Force circuit open and simulate recovery timeout
        cb.force_open()
        cb._stats.last_failure_time = time.time() - cb.recovery_timeout - 1
        
        async def successful_operation():
            return "success"
        
        # First call should transition to half-open
        result = await cb.call(successful_operation)
        assert result == "success"
        assert cb.state == CircuitBreakerState.HALF_OPEN
        
        # Second successful call should close the circuit
        result = await cb.call(successful_operation)
        assert result == "success"
        assert cb.state == CircuitBreakerState.CLOSED
    
    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self, circuit_breaker_instance):
        """Test that failure in half-open state reopens circuit."""
        cb = circuit_breaker_instance
        
        # Set to half-open state
        cb._state = CircuitBreakerState.HALF_OPEN
        
        async def failing_operation():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            await cb.call(failing_operation)
        
        assert cb.state == CircuitBreakerState.OPEN
    
    @pytest.mark.asyncio
    async def test_operation_timeout(self, circuit_breaker_instance):
        """Test operation timeout handling."""
        cb = circuit_breaker_instance
        
        async def slow_operation():
            await asyncio.sleep(2.0)  # Longer than timeout
            return "should timeout"
        
        with pytest.raises(asyncio.TimeoutError):
            await cb.call(slow_operation)
        
        assert cb.stats.failed_requests == 1
    
    def test_reset_circuit_breaker(self, circuit_breaker_instance):
        """Test circuit breaker reset functionality."""
        cb = circuit_breaker_instance
        
        # Simulate some activity
        cb._state = CircuitBreakerState.OPEN
        cb._stats.total_requests = 10
        cb._stats.failed_requests = 5
        cb._stats.consecutive_failures = 3
        
        cb.reset()
        
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.stats.total_requests == 0
        assert cb.stats.failed_requests == 0
        assert cb.stats.consecutive_failures == 0
    
    def test_get_health_info(self, circuit_breaker_instance):
        """Test health information retrieval."""
        cb = circuit_breaker_instance
        
        health_info = cb.get_health_info()
        
        assert health_info["service_name"] == "test_service"
        assert health_info["state"] == "closed"
        assert health_info["healthy"] is True
        assert "stats" in health_info
        assert "config" in health_info
        assert health_info["stats"]["total_requests"] == 0
        assert health_info["config"]["failure_threshold"] == 3


class TestCircuitBreakerDecorator:
    """Test suite for circuit breaker decorator."""
    
    @pytest.mark.asyncio
    async def test_decorator_successful_operation(self):
        """Test circuit breaker decorator with successful operation."""
        
        @circuit_breaker("test_decorated_service", failure_threshold=2)
        async def decorated_function(value):
            return f"processed: {value}"
        
        result = await decorated_function("test")
        assert result == "processed: test"
        
        # Check circuit breaker was attached
        assert hasattr(decorated_function, 'circuit_breaker')
        assert decorated_function.circuit_breaker.service_name == "test_decorated_service"
    
    @pytest.mark.asyncio
    async def test_decorator_failing_operation(self):
        """Test circuit breaker decorator with failing operation."""
        
        @circuit_breaker("test_failing_service", failure_threshold=2)
        async def failing_function():
            raise ValueError("Decorated failure")
        
        # First failure
        with pytest.raises(ValueError, match="Decorated failure"):
            await failing_function()
        
        # Second failure should open circuit
        with pytest.raises(ValueError, match="Decorated failure"):
            await failing_function()
        
        # Third call should fail fast
        with pytest.raises(CircuitBreakerError):
            await failing_function()


class TestCircuitBreakerManager:
    """Test suite for CircuitBreakerManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh circuit breaker manager."""
        return CircuitBreakerManager()
    
    @pytest.fixture
    def sample_breakers(self):
        """Create sample circuit breakers for testing."""
        return [
            CircuitBreaker("service_1", failure_threshold=3),
            CircuitBreaker("service_2", failure_threshold=5),
            CircuitBreaker("service_3", failure_threshold=2)
        ]
    
    def test_register_circuit_breakers(self, manager, sample_breakers):
        """Test registering circuit breakers."""
        for breaker in sample_breakers:
            manager.register(breaker)
        
        assert manager.get("service_1") == sample_breakers[0]
        assert manager.get("service_2") == sample_breakers[1]
        assert manager.get("service_3") == sample_breakers[2]
        assert manager.get("nonexistent") is None
    
    def test_get_all_health_info(self, manager, sample_breakers):
        """Test getting health info for all registered breakers."""
        for breaker in sample_breakers:
            manager.register(breaker)
        
        health_info = manager.get_all_health_info()
        
        assert len(health_info) == 3
        assert "service_1" in health_info
        assert "service_2" in health_info
        assert "service_3" in health_info
        
        for service_name, info in health_info.items():
            assert info["service_name"] == service_name
            assert "state" in info
            assert "healthy" in info
    
    def test_reset_all_breakers(self, manager, sample_breakers):
        """Test resetting all circuit breakers."""
        for breaker in sample_breakers:
            # Set some state to reset
            breaker._state = CircuitBreakerState.OPEN
            breaker._stats.total_requests = 10
            manager.register(breaker)
        
        manager.reset_all()
        
        for breaker in sample_breakers:
            assert breaker.state == CircuitBreakerState.CLOSED
            assert breaker.stats.total_requests == 0
    
    def test_get_unhealthy_services(self, manager, sample_breakers):
        """Test getting list of unhealthy services."""
        for breaker in sample_breakers:
            manager.register(breaker)
        
        # Initially all should be healthy
        unhealthy = manager.get_unhealthy_services()
        assert len(unhealthy) == 0
        
        # Force some breakers to open
        sample_breakers[0].force_open()
        sample_breakers[2].force_open()
        
        unhealthy = manager.get_unhealthy_services()
        assert len(unhealthy) == 2
        assert "service_1" in unhealthy
        assert "service_3" in unhealthy
        assert "service_2" not in unhealthy


class TestGlobalCircuitBreakerManager:
    """Test suite for global circuit breaker manager instance."""
    
    def test_global_manager_exists(self):
        """Test that global circuit breaker manager exists."""
        assert circuit_breaker_manager is not None
        assert isinstance(circuit_breaker_manager, CircuitBreakerManager)
    
    def test_global_manager_registration(self):
        """Test registration with global manager."""
        # Clean up any existing registrations
        circuit_breaker_manager._breakers.clear()
        
        breaker = CircuitBreaker("global_test_service")
        circuit_breaker_manager.register(breaker)
        
        retrieved = circuit_breaker_manager.get("global_test_service")
        assert retrieved == breaker
        
        # Clean up
        circuit_breaker_manager._breakers.clear()


@pytest.mark.asyncio
async def test_integration_with_actual_failures():
    """Integration test with realistic failure scenarios."""
    cb = CircuitBreaker(
        service_name="integration_test",
        failure_threshold=3,
        recovery_timeout=0.1,  # Short timeout for testing
        success_threshold=2
    )
    
    call_count = 0
    
    async def unreliable_service():
        nonlocal call_count
        call_count += 1
        
        # Fail first 3 calls, then succeed
        if call_count <= 3:
            raise ConnectionError(f"Service unavailable (call {call_count})")
        return f"Success on call {call_count}"
    
    # First 3 calls should fail and open circuit
    for i in range(3):
        with pytest.raises(ConnectionError):
            await cb.call(unreliable_service)
    
    assert cb.state == CircuitBreakerState.OPEN
    
    # Immediate call should fail fast
    with pytest.raises(CircuitBreakerError):
        await cb.call(unreliable_service)
    
    # Wait for recovery timeout
    await asyncio.sleep(0.2)
    
    # Next call should transition to half-open and succeed
    result = await cb.call(unreliable_service)
    assert result == "Success on call 4"
    assert cb.state == CircuitBreakerState.HALF_OPEN
    
    # Another success should close the circuit
    result = await cb.call(unreliable_service)
    assert result == "Success on call 5"
    assert cb.state == CircuitBreakerState.CLOSED


@pytest.mark.asyncio
async def test_concurrent_operations():
    """Test circuit breaker with concurrent operations."""
    cb = CircuitBreaker("concurrent_test", failure_threshold=5)
    
    success_count = 0
    
    async def concurrent_operation(delay: float):
        nonlocal success_count
        await asyncio.sleep(delay)
        success_count += 1
        return f"Success {success_count}"
    
    # Execute multiple operations concurrently
    tasks = [
        cb.call(concurrent_operation, 0.1),
        cb.call(concurrent_operation, 0.05),
        cb.call(concurrent_operation, 0.15),
        cb.call(concurrent_operation, 0.08)
    ]
    
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 4
    assert cb.stats.total_requests == 4
    assert cb.stats.successful_requests == 4
    assert cb.stats.failed_requests == 0
    assert cb.state == CircuitBreakerState.CLOSED


def test_circuit_breaker_stats_accuracy():
    """Test accuracy of circuit breaker statistics."""
    cb = CircuitBreaker("stats_test")
    
    # Test initial state
    initial_stats = cb.stats
    assert initial_stats.total_requests == 0
    assert initial_stats.successful_requests == 0
    assert initial_stats.failed_requests == 0
    assert initial_stats.consecutive_failures == 0
    assert initial_stats.consecutive_successes == 0
    
    # Simulate some activity manually
    cb._stats.total_requests = 10
    cb._stats.successful_requests = 7
    cb._stats.failed_requests = 3
    cb._stats.consecutive_successes = 2
    cb._stats.last_success_time = time.time()
    
    current_stats = cb.stats
    assert current_stats.total_requests == 10
    assert current_stats.successful_requests == 7
    assert current_stats.failed_requests == 3
    assert current_stats.consecutive_successes == 2
    assert current_stats.last_success_time is not None 