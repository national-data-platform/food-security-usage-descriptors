"""
Repository module for dataset persistence operations.

This module provides a repository class for managing dataset documents
in MongoDB, implementing basic CRUD operations.
"""

import pymongo
from typing import Dict, List, Optional, Any, Union

from src.infra.configuration import Configuration


class DatasetRepository:
    """
    Repository class for managing dataset documents in MongoDB.
    
    This class provides methods for saving, retrieving, and managing
    dataset documents in a MongoDB collection. It handles connection
    management and provides a clean interface for dataset persistence.
    
    Attributes:
        client: MongoDB client instance for database connections.
        db: Reference to the specific database for the given origin.
        collection: Reference to the datasets collection.
    """
    
    def __init__(self, origin: str) -> None:
        """
        Initialize the dataset repository with a specific database origin.
        
        Args:
            origin: The database name to connect to. This allows the repository
                   to work with different databases based on the data source.
        """
        config = Configuration().get()
        self.client = pymongo.MongoClient(config['mongodb_conn'])
        self.db = self.client[origin]
        self.collection = self.db[config['mongodb_datasets_table']]

    def save(self, task_id: str, dataset: Dict[str, Any]) -> str:
        """
        Save or update a dataset document in the collection.
        
        This method uses upsert logic to either insert a new document
        or replace an existing one based on the task_id.
        
        Args:
            task_id: Unique identifier for the dataset/task.
            dataset: Dictionary containing the dataset data to persist.
            
        Returns:
            The string representation of the upserted document ID,
            or None if the document was updated rather than inserted.
        """
        # Remove None _id to avoid MongoDB insertion errors
        if '_id' in dataset and dataset['_id'] is None:
            dataset.pop('_id')

        # Replace existing document or insert new one (upsert)
        result = self.collection.replace_one(
            {
                "id": task_id
            }, dataset, upsert=True)
        return str(result.upserted_id)

    def find_by_id(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a dataset document by its ID.
        
        Args:
            dataset_id: The unique identifier of the dataset to retrieve.
            
        Returns:
            The dataset document as a dictionary if found, None otherwise.
        """
        return self.collection.find_one({"id": dataset_id})

    def close(self) -> None:
        """
        Close the MongoDB client connection.
        
        This method should be called when the repository is no longer needed
        to properly release database connection resources.
        """
        self.client.close()