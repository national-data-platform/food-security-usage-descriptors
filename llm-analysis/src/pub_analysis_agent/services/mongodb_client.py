"""
MongoDB client with connection pooling and health monitoring.

This module provides a MongoDB client with connection pooling, health checks,
automatic reconnection logic, and proper logging for the pub-analysis-agent.
"""

import logging
import time
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import asyncio

import pymongo
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import (
    ServerSelectionTimeoutError,
    ConnectionFailure,
    OperationFailure,
    ConfigurationError,
)

from ..config.settings import DatabaseSettings


logger = logging.getLogger(__name__)


class MongoDBClient:
    """
    MongoDB client with connection pooling and health monitoring.
    
    This class manages MongoDB connections with proper pooling, health checks,
    automatic reconnection logic, and comprehensive logging.
    
    Args:
        db_settings: Database configuration settings
        
    Attributes:
        client: PyMongo client instance
        database: Database instance
        datasets_collection: Collection for known datasets
        results_collection: Collection for LLM analysis results
        is_connected: Connection status flag
    """
    
    def __init__(self, db_settings: DatabaseSettings) -> None:
        """
        Initialize MongoDB client with configuration.
        
        Args:
            db_settings: Database configuration settings from settings.py
        """
        self.db_settings = db_settings
        self.client: Optional[MongoClient] = None
        
        # Databases
        self.general_database: Optional[Database] = None
        self.dimensions_database: Optional[Database] = None
        
        # Collections
        self.datasets_collection: Optional[Collection] = None
        self.authors_collection: Optional[Collection] = None
        self.institutions_collection: Optional[Collection] = None
        self.publications_collection: Optional[Collection] = None
        self.results_collection: Optional[Collection] = None
        
        self.is_connected: bool = False
        self._connection_attempts: int = 0
        self._last_health_check: float = 0.0
        self._health_check_interval: float = 30.0  # 30 seconds
        
        logger.info("MongoDB client initialized with multi-database settings")

    def connect(self) -> None:
        """
        Establish connection to MongoDB with connection pooling.
        
        Raises:
            ConnectionFailure: If connection cannot be established
            ConfigurationError: If configuration is invalid
        """
        try:
            logger.info("Attempting to connect to MongoDB...")
            self._connection_attempts += 1
            
            # Create client with connection pooling configuration
            client_options = {
                'maxPoolSize': self.db_settings.max_pool_size,
                'minPoolSize': self.db_settings.min_pool_size,
                'connectTimeoutMS': self.db_settings.connect_timeout_ms,
                'serverSelectionTimeoutMS': self.db_settings.server_selection_timeout_ms,
                'socketTimeoutMS': 20000,  # 20 second socket timeout
                'retryWrites': True,
                'retryReads': True,
                'waitQueueTimeoutMS': 5000,  # 5 second wait queue timeout
            }
            
            # Add SSL configuration if enabled
            if self.db_settings.ssl_enabled:
                client_options['tls'] = True
                client_options['tlsAllowInvalidCertificates'] = True
                logger.info("SSL enabled for MongoDB connection")
            
            # Create MongoDB client
            self.client = MongoClient(
                self.db_settings.connection_string,
                **client_options
            )
            
            # Test connection
            self.client.admin.command('ismaster')
            
            # Set up database references
            self.general_database = self.client[self.db_settings.general_database]
            self.dimensions_database = self.client[self.db_settings.dimensions_database]
            
            # Set up collection references
            self.datasets_collection = self.general_database[self.db_settings.datasets_collection]
            self.authors_collection = self.dimensions_database[self.db_settings.authors_collection]
            self.institutions_collection = self.dimensions_database[self.db_settings.institutions_collection]
            self.publications_collection = self.dimensions_database[self.db_settings.publications_collection]
            self.results_collection = self.dimensions_database[self.db_settings.results_collection]
            
            self.is_connected = True
            self._last_health_check = time.time()
            
            logger.info(
                f"Successfully connected to MongoDB with databases: "
                f"{self.db_settings.general_database}, "
                f"{self.db_settings.dimensions_database}"
            )
            logger.info(
                f"Collections: datasets={self.db_settings.datasets_collection}, "
                f"authors={self.db_settings.authors_collection}, "
                f"institutions={self.db_settings.institutions_collection}, "
                f"publications={self.db_settings.publications_collection}, "
                f"results={self.db_settings.results_collection}"
            )
            logger.info(
                f"Connection pool: {self.db_settings.min_pool_size}-{self.db_settings.max_pool_size} connections"
            )
            
        except ServerSelectionTimeoutError as e:
            self.is_connected = False
            error_msg = f"Server selection timeout after {self._connection_attempts} attempts: {e}"
            logger.error(error_msg)
            raise ConnectionFailure(error_msg) from e
            
        except ConnectionFailure as e:
            self.is_connected = False
            error_msg = f"Connection failure after {self._connection_attempts} attempts: {e}"
            logger.error(error_msg)
            raise
            
        except ConfigurationError as e:
            self.is_connected = False
            error_msg = f"MongoDB configuration error: {e}"
            logger.error(error_msg)
            raise
            
        except Exception as e:
            self.is_connected = False
            error_msg = f"Unexpected error connecting to MongoDB: {e}"
            logger.error(error_msg)
            raise ConnectionFailure(error_msg) from e

    async def _test_connection(self) -> None:
        """
        Test MongoDB connection with admin command.
        
        Raises:
            ConnectionFailure: If connection test fails
        """
        try:
            # Use asyncio to run the blocking operation
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, 
                lambda: self.client.admin.command('ping')
            )
            logger.debug("MongoDB connection test successful")
            
        except Exception as e:
            logger.error(f"MongoDB connection test failed: {e}")
            raise ConnectionFailure(f"Connection test failed: {e}") from e

    async def health_check(self, force: bool = False) -> bool:
        """
        Perform health check on MongoDB connection.
        
        Args:
            force: Force health check even if within interval
            
        Returns:
            True if connection is healthy, False otherwise
        """
        current_time = time.time()
        
        # Skip health check if within interval (unless forced)
        if not force and (current_time - self._last_health_check) < self._health_check_interval:
            return self.is_connected
        
        try:
            if not self.client:
                logger.warning("MongoDB client not initialized")
                self.is_connected = False
                return False
            
            # Check if databases are initialized
            if (self.general_database is None or 
                self.dimensions_database is None):
                logger.warning("MongoDB databases not initialized")
                self.is_connected = False
                return False
            
            # Check if collections are initialized
            if self.datasets_collection is None:
                logger.warning("MongoDB collections not initialized")
                self.is_connected = False
                return False
            
            # Test connection with ping
            await self._test_connection()
            
            # Test database access
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self.general_database.command('ping')
            )
            await loop.run_in_executor(
                None,
                lambda: self.dimensions_database.command('ping')
            )         
            
            # Test collection access
            await loop.run_in_executor(
                None,
                lambda: self.datasets_collection.estimated_document_count()
            )
            
            self.is_connected = True
            self._last_health_check = current_time
            logger.debug("MongoDB health check passed")
            return True
            
        except Exception as e:
            logger.warning(f"MongoDB health check failed: {e}")
            self.is_connected = False
            return False

    async def reconnect(self) -> None:
        """
        Attempt to reconnect to MongoDB.
        
        Raises:
            ConnectionFailure: If reconnection fails
        """
        try:
            logger.info("Attempting to reconnect to MongoDB...")
            
            # Close existing connection
            self.disconnect()
            
            # Wait a short time before reconnecting
            await asyncio.sleep(1.0)
            
            # Attempt to reconnect
            self.connect()
            
            logger.info("Successfully reconnected to MongoDB")
            
        except Exception as e:
            print(f"Failed to reconnect to MongoDB: {e}")
            logger.error(f"Failed to reconnect to MongoDB: {e}")
            raise ConnectionFailure(f"Reconnection failed: {e}") from e

    def disconnect(self) -> None:
        """
        Disconnect from MongoDB and cleanup resources.
        """
        try:
            if self.client:
                logger.info("Disconnecting from MongoDB...")
                self.client.close()
                self.client = None
                self.general_database = None
                self.dimensions_database = None
                self.datasets_collection = None
                self.authors_collection = None
                self.institutions_collection = None
                self.publications_collection = None
                self.results_collection = None
                self.is_connected = False
                logger.info("Successfully disconnected from MongoDB")
                
        except Exception as e:
            logger.error(f"Error during MongoDB disconnection: {e}")
            # Force cleanup even if errors occur
            self.client = None
            self.general_database = None
            self.dimensions_database = None
            self.datasets_collection = None
            self.authors_collection = None
            self.institutions_collection = None
            self.publications_collection = None
            self.results_collection = None
            self.is_connected = False

    @asynccontextmanager
    async def ensure_connection(self):
        """
        Context manager to ensure MongoDB connection is available.
        
        Automatically handles reconnection if needed.
        
        Yields:
            MongoDBClient: Self reference for chaining
            
        Raises:
            ConnectionFailure: If connection cannot be established
        """
        connection_established = False
        
        try:
            # Check if connection is healthy
            health_status = await self.health_check()
            logger.debug(f"Health check result: {health_status}")
            
            if not health_status:
                logger.info("MongoDB connection unhealthy, attempting reconnection...")
                await self.reconnect()
                
                # Verify reconnection was successful
                health_status = await self.health_check(force=True)
                if not health_status:
                    raise ConnectionFailure("Failed to establish healthy connection after reconnection")
            
            connection_established = True
            logger.debug("MongoDB connection ensured, yielding client")
            
            yield self
            
        except ConnectionFailure:
            # Re-raise connection failures without wrapping
            raise
        except StopIteration as e:
            logger.error(f"StopIteration in connection context manager: {e}")
            if connection_established:
                # If connection was established but StopIteration occurred, it's likely a client usage issue
                raise ConnectionFailure("Context manager generator terminated unexpectedly - check client usage") from e
            else:
                # If connection wasn't established, it's likely a setup issue
                raise ConnectionFailure("Context manager setup failed due to generator termination") from e
        except GeneratorExit as e:
            logger.error(f"GeneratorExit in connection context manager: {e}")
            raise ConnectionFailure("Context manager closed unexpectedly") from e
        except Exception as e:
            logger.error(f"Unexpected error in connection context manager: {e}")
            # Log the full traceback for debugging
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise ConnectionFailure(f"Context manager error: {e}") from e
        finally:
            logger.debug("Exiting ensure_connection context manager")

    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get current connection information and statistics.
        
        Returns:
            Dictionary containing connection information
        """
        info = {
            'is_connected': self.is_connected,
            'connection_attempts': self._connection_attempts,
            'last_health_check': self._last_health_check,
            'databases': {
                'general': self.db_settings.general_database,
                'dimensions': self.db_settings.dimensions_database,
            },
            'collections': {
                'datasets': self.db_settings.datasets_collection,
                'authors': self.db_settings.authors_collection,
                'institutions': self.db_settings.institutions_collection,
                'publications': self.db_settings.publications_collection,
                'results': self.db_settings.results_collection,
            },
            'max_pool_size': self.db_settings.max_pool_size,
            'min_pool_size': self.db_settings.min_pool_size,
        }
        
        if self.client and self.is_connected:
            try:
                # Get server info
                server_info = self.client.server_info()
                info.update({
                    'server_version': server_info.get('version'),
                    'server_git_version': server_info.get('gitVersion'),
                })
                
                # Get connection pool info if available
                pool_options = self.client.options.pool_options
                if pool_options:
                    info.update({
                        'pool_max_size': pool_options.max_pool_size,
                        'pool_min_size': pool_options.min_pool_size,
                        'connect_timeout_ms': pool_options.connect_timeout,
                        'server_selection_timeout_ms': pool_options.server_selection_timeout,
                    })
                    
            except Exception as e:
                logger.warning(f"Could not retrieve server info: {e}")
        
        return info
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.disconnect() 