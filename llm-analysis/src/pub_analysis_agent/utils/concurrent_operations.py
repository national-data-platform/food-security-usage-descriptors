"""
Concurrent operations utilities for safe parallel processing.

This module provides utilities for managing concurrent operations with
rate limiting, connection sharing, semaphores, and batch processing.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Callable, TypeVar, Generic, Union
from dataclasses import dataclass
from collections import deque
import threading
from contextlib import asynccontextmanager


logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class OperationResult:
    """Result of a concurrent operation."""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    duration: float = 0.0
    operation_id: Optional[str] = None


class RateLimiter:
    """
    Rate limiter for controlling operation frequency.
    
    Uses token bucket algorithm to limit operations per time window.
    """
    
    def __init__(self, max_operations: int, time_window: float = 1.0):
        """
        Initialize rate limiter.
        
        Args:
            max_operations: Maximum operations allowed in time window
            time_window: Time window in seconds
        """
        self.max_operations = max_operations
        self.time_window = time_window
        self.tokens = max_operations
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens were acquired, False otherwise
        """
        async with self._lock:
            now = time.time()
            
            # Refill tokens based on elapsed time
            elapsed = now - self.last_refill
            self.tokens = min(
                self.max_operations,
                self.tokens + (elapsed * self.max_operations / self.time_window)
            )
            self.last_refill = now
            
            # Check if enough tokens available
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    async def wait_for_tokens(self, tokens: int = 1) -> None:
        """
        Wait until tokens are available.
        
        Args:
            tokens: Number of tokens needed
        """
        while not await self.acquire(tokens):
            # Calculate wait time based on token refill rate
            wait_time = tokens * self.time_window / self.max_operations
            await asyncio.sleep(min(wait_time, 0.1))


class ConcurrentOperationManager:
    """
    Manager for coordinating concurrent operations with limits and monitoring.
    
    This class provides:
    - Concurrency limiting with semaphores
    - Rate limiting
    - Operation monitoring and statistics
    - Graceful error handling
    - Resource management
    """
    
    def __init__(
        self,
        max_concurrent: int = 10,
        rate_limit: Optional[int] = None,
        rate_window: float = 1.0,
        timeout: float = 30.0,
        name: str = "concurrent_operations"
    ):
        """
        Initialize concurrent operation manager.
        
        Args:
            max_concurrent: Maximum concurrent operations
            rate_limit: Maximum operations per rate_window (None for no limit)
            rate_window: Rate limiting time window in seconds
            timeout: Default timeout for operations
            name: Name for logging and identification
        """
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.name = name
        
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._rate_limiter = RateLimiter(rate_limit, rate_window) if rate_limit else None
        self._active_operations: Dict[str, asyncio.Task] = {}
        self._operation_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_duration': 0.0,
            'max_duration': 0.0,
            'min_duration': float('inf'),
            'concurrent_peak': 0
        }
        self._lock = threading.Lock()
        
        logger.info(
            f"ConcurrentOperationManager '{name}' initialized: "
            f"max_concurrent={max_concurrent}, rate_limit={rate_limit}, timeout={timeout}s"
        )
    
    @asynccontextmanager
    async def operation_context(self, operation_id: Optional[str] = None):
        """
        Context manager for individual operations.
        
        Handles semaphore acquisition, rate limiting, and cleanup.
        
        Args:
            operation_id: Optional operation identifier
        """
        if operation_id is None:
            operation_id = f"op_{int(time.time() * 1000000)}"
        
        start_time = time.time()
        
        try:
            # Acquire semaphore
            await self._semaphore.acquire()
            
            # Apply rate limiting if configured
            if self._rate_limiter:
                await self._rate_limiter.wait_for_tokens()
            
            # Track active operation
            current_task = asyncio.current_task()
            if current_task:
                self._active_operations[operation_id] = current_task
            
            # Update stats
            with self._lock:
                self._operation_stats['total_operations'] += 1
                current_concurrent = len(self._active_operations)
                if current_concurrent > self._operation_stats['concurrent_peak']:
                    self._operation_stats['concurrent_peak'] = current_concurrent
            
            logger.debug(f"Operation {operation_id} started in {self.name}")
            yield operation_id
            
        finally:
            # Cleanup
            self._active_operations.pop(operation_id, None)
            self._semaphore.release()
            
            # Update duration stats
            duration = time.time() - start_time
            with self._lock:
                self._operation_stats['total_duration'] += duration
                if duration > self._operation_stats['max_duration']:
                    self._operation_stats['max_duration'] = duration
                if duration < self._operation_stats['min_duration']:
                    self._operation_stats['min_duration'] = duration
            
            logger.debug(f"Operation {operation_id} completed in {duration:.3f}s")
    
    async def execute_single(
        self,
        operation: Callable,
        *args,
        operation_id: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> OperationResult:
        """
        Execute a single operation with concurrency control.
        
        Args:
            operation: Async function to execute
            *args: Operation arguments
            operation_id: Optional operation identifier
            timeout: Operation timeout (uses default if None)
            **kwargs: Operation keyword arguments
            
        Returns:
            OperationResult with execution details
        """
        operation_timeout = timeout or self.timeout
        start_time = time.time()
        
        async with self.operation_context(operation_id) as op_id:
            try:
                if asyncio.iscoroutinefunction(operation):
                    result = await asyncio.wait_for(
                        operation(*args, **kwargs),
                        timeout=operation_timeout
                    )
                else:
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, operation, *args, **kwargs
                        ),
                        timeout=operation_timeout
                    )
                
                duration = time.time() - start_time
                
                with self._lock:
                    self._operation_stats['successful_operations'] += 1
                
                return OperationResult(
                    success=True,
                    result=result,
                    duration=duration,
                    operation_id=op_id
                )
                
            except Exception as e:
                duration = time.time() - start_time
                
                with self._lock:
                    self._operation_stats['failed_operations'] += 1
                
                logger.warning(f"Operation {op_id} failed after {duration:.3f}s: {e}")
                
                return OperationResult(
                    success=False,
                    error=e,
                    duration=duration,
                    operation_id=op_id
                )
    
    async def execute_batch(
        self,
        operations: List[tuple],
        timeout: Optional[float] = None,
        fail_fast: bool = False
    ) -> List[OperationResult]:
        """
        Execute multiple operations concurrently.
        
        Args:
            operations: List of (operation, args, kwargs) tuples
            timeout: Timeout for each operation
            fail_fast: Stop on first failure if True
            
        Returns:
            List of OperationResult objects
        """
        tasks = []
        results = []
        
        for i, (operation, args, kwargs) in enumerate(operations):
            operation_id = f"batch_{int(time.time() * 1000)}_{i}"
            task = asyncio.create_task(
                self.execute_single(
                    operation, *args, 
                    operation_id=operation_id,
                    timeout=timeout,
                    **kwargs
                )
            )
            tasks.append(task)
        
        if fail_fast:
            # Stop on first failure
            for task in asyncio.as_completed(tasks):
                result = await task
                results.append(result)
                if not result.success:
                    # Cancel remaining tasks
                    for remaining_task in tasks:
                        if not remaining_task.done():
                            remaining_task.cancel()
                    break
        else:
            # Wait for all operations to complete
            results = await asyncio.gather(*tasks, return_exceptions=False)
        
        successful = sum(1 for r in results if r.success)
        logger.info(
            f"Batch execution completed: {successful}/{len(results)} successful"
        )
        
        return results
    
    async def execute_with_retry(
        self,
        operation: Callable,
        *args,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
        operation_id: Optional[str] = None,
        **kwargs
    ) -> OperationResult:
        """
        Execute operation with retry logic.
        
        Args:
            operation: Function to execute
            *args: Operation arguments
            max_retries: Maximum number of retries
            retry_delay: Initial delay between retries
            backoff_factor: Multiplier for retry delay
            operation_id: Optional operation identifier
            **kwargs: Operation keyword arguments
            
        Returns:
            OperationResult from the final attempt
        """
        last_result = None
        delay = retry_delay
        
        for attempt in range(max_retries + 1):
            op_id = f"{operation_id}_attempt_{attempt}" if operation_id else None
            result = await self.execute_single(operation, *args, operation_id=op_id, **kwargs)
            
            if result.success:
                return result
            
            last_result = result
            
            if attempt < max_retries:
                logger.warning(
                    f"Operation attempt {attempt + 1} failed, retrying in {delay:.1f}s: "
                    f"{result.error}"
                )
                await asyncio.sleep(delay)
                delay *= backoff_factor
        
        return last_result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get operation statistics.
        
        Returns:
            Dictionary with operation statistics
        """
        with self._lock:
            stats = self._operation_stats.copy()
            stats['active_operations'] = len(self._active_operations)
            stats['available_slots'] = self._semaphore._value
            
            if stats['total_operations'] > 0:
                stats['success_rate'] = stats['successful_operations'] / stats['total_operations']
                stats['average_duration'] = stats['total_duration'] / stats['total_operations']
            else:
                stats['success_rate'] = 0.0
                stats['average_duration'] = 0.0
            
            if stats['min_duration'] == float('inf'):
                stats['min_duration'] = 0.0
        
        return stats
    
    async def cancel_all_operations(self) -> int:
        """
        Cancel all active operations.
        
        Returns:
            Number of operations cancelled
        """
        active_tasks = list(self._active_operations.values())
        cancelled_count = 0
        
        for task in active_tasks:
            if not task.done():
                task.cancel()
                cancelled_count += 1
        
        if cancelled_count > 0:
            logger.warning(f"Cancelled {cancelled_count} active operations in {self.name}")
        
        return cancelled_count
    
    async def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all active operations to complete.
        
        Args:
            timeout: Maximum time to wait (None for no timeout)
            
        Returns:
            True if all operations completed, False if timeout
        """
        if not self._active_operations:
            return True
        
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._active_operations.values(), return_exceptions=True),
                timeout=timeout
            )
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for operations to complete in {self.name}")
            return False


class BatchProcessor(Generic[T, R]):
    """
    Generic batch processor for handling large datasets efficiently.
    
    Processes items in batches with configurable size, concurrency, and error handling.
    """
    
    def __init__(
        self,
        batch_size: int = 100,
        max_concurrent_batches: int = 5,
        processor_name: str = "batch_processor"
    ):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Number of items per batch
            max_concurrent_batches: Maximum concurrent batches
            processor_name: Name for logging
        """
        self.batch_size = batch_size
        self.processor_name = processor_name
        self.operation_manager = ConcurrentOperationManager(
            max_concurrent=max_concurrent_batches,
            name=f"{processor_name}_batches"
        )
    
    def _create_batches(self, items: List[T]) -> List[List[T]]:
        """Create batches from list of items."""
        batches = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batches.append(batch)
        return batches
    
    async def process_batch(
        self,
        batch: List[T],
        processor_func: Callable[[List[T]], R],
        batch_id: str
    ) -> OperationResult:
        """
        Process a single batch.
        
        Args:
            batch: Items to process
            processor_func: Function to process the batch
            batch_id: Batch identifier
            
        Returns:
            OperationResult with batch processing details
        """
        return await self.operation_manager.execute_single(
            processor_func,
            batch,
            operation_id=batch_id
        )
    
    async def process_all(
        self,
        items: List[T],
        processor_func: Callable[[List[T]], R],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[OperationResult]:
        """
        Process all items in batches.
        
        Args:
            items: Items to process
            processor_func: Function to process each batch
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of OperationResult objects for each batch
        """
        if not items:
            return []
        
        batches = self._create_batches(items)
        logger.info(
            f"Processing {len(items)} items in {len(batches)} batches "
            f"of size {self.batch_size}"
        )
        
        # Create batch processing operations
        operations = []
        for i, batch in enumerate(batches):
            batch_id = f"{self.processor_name}_batch_{i}"
            operations.append((
                self.process_batch,
                (batch, processor_func, batch_id),
                {}
            ))
        
        # Process batches concurrently
        results = await self.operation_manager.execute_batch(operations)
        
        # Update progress
        if progress_callback:
            completed_batches = sum(1 for r in results if r.success)
            progress_callback(completed_batches, len(batches))
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processor statistics."""
        stats = self.operation_manager.get_stats()
        stats['batch_size'] = self.batch_size
        stats['processor_name'] = self.processor_name
        return stats


class ConnectionPool:
    """
    Generic connection pool for managing shared resources.
    
    Useful for sharing database connections or other expensive resources
    across multiple concurrent operations.
    """
    
    def __init__(
        self,
        create_connection: Callable,
        max_connections: int = 10,
        min_connections: int = 2,
        connection_timeout: float = 30.0,
        idle_timeout: float = 300.0,
        pool_name: str = "connection_pool"
    ):
        """
        Initialize connection pool.
        
        Args:
            create_connection: Function to create new connections
            max_connections: Maximum connections in pool
            min_connections: Minimum connections to maintain
            connection_timeout: Timeout for acquiring connections
            idle_timeout: Timeout for idle connections
            pool_name: Name for logging
        """
        self.create_connection = create_connection
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.connection_timeout = connection_timeout
        self.idle_timeout = idle_timeout
        self.pool_name = pool_name
        
        self._connections: deque = deque()
        self._in_use_connections: set = set()
        self._lock = asyncio.Lock()
        self._connection_count = 0
        
        logger.info(f"Connection pool '{pool_name}' initialized")
    
    async def _create_new_connection(self):
        """Create a new connection."""
        try:
            connection = await self.create_connection()
            self._connection_count += 1
            logger.debug(f"Created new connection in pool '{self.pool_name}'")
            return connection
        except Exception as e:
            logger.error(f"Failed to create connection in pool '{self.pool_name}': {e}")
            raise
    
    async def acquire(self) -> Any:
        """
        Acquire a connection from the pool.
        
        Returns:
            Connection object
            
        Raises:
            asyncio.TimeoutError: If connection cannot be acquired within timeout
        """
        async with self._lock:
            # Try to get existing connection
            if self._connections:
                connection = self._connections.popleft()
                self._in_use_connections.add(connection)
                return connection
            
            # Create new connection if under limit
            if self._connection_count < self.max_connections:
                connection = await self._create_new_connection()
                self._in_use_connections.add(connection)
                return connection
        
        # Wait for connection to become available
        start_time = time.time()
        while time.time() - start_time < self.connection_timeout:
            await asyncio.sleep(0.1)
            async with self._lock:
                if self._connections:
                    connection = self._connections.popleft()
                    self._in_use_connections.add(connection)
                    return connection
        
        raise asyncio.TimeoutError(f"Timeout acquiring connection from pool '{self.pool_name}'")
    
    async def release(self, connection: Any) -> None:
        """
        Release a connection back to the pool.
        
        Args:
            connection: Connection to release
        """
        async with self._lock:
            if connection in self._in_use_connections:
                self._in_use_connections.remove(connection)
                self._connections.append(connection)
    
    @asynccontextmanager
    async def connection(self):
        """
        Context manager for acquiring and releasing connections.
        
        Yields:
            Connection object
        """
        conn = await self.acquire()
        try:
            yield conn
        finally:
            await self.release(conn)
    
    async def close_all(self) -> None:
        """Close all connections in the pool."""
        async with self._lock:
            total_connections = len(self._connections) + len(self._in_use_connections)
            
            # Close idle connections
            while self._connections:
                connection = self._connections.popleft()
                try:
                    if hasattr(connection, 'close'):
                        await connection.close()
                except Exception as e:
                    logger.warning(f"Error closing connection: {e}")
            
            self._connection_count = 0
            logger.info(f"Closed {total_connections} connections in pool '{self.pool_name}'")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            'pool_name': self.pool_name,
            'total_connections': self._connection_count,
            'idle_connections': len(self._connections),
            'in_use_connections': len(self._in_use_connections),
            'max_connections': self.max_connections,
            'min_connections': self.min_connections
        } 