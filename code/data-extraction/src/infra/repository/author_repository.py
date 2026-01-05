import pymongo
from typing import Dict, List, Optional, Any

from src.infra.configuration import Configuration


class AuthorRepository:
    """
    Repository for persisting and retrieving scholarly author data in MongoDB.
    
    This class provides a data access layer for author entities, encapsulating
    MongoDB-specific operations and offering a clean interface for managing
    author records. It supports basic CRUD operations with a focus on efficient
    storage and retrieval of author metadata collected from academic data sources.
    """

    def __init__(self, origin):
        """
        Initialize a repository instance for authors from a specific data source.
        
        Creates a MongoDB connection and configures the appropriate database and
        collection based on the data origin parameter. This allows for segregating
        author data by source (e.g., 'dimensions', 'openalex') while using consistent
        access patterns.
        
        Args:
            origin (str): Data source identifier used to select the appropriate database
        """
        config = Configuration().get()
        self.client = pymongo.MongoClient(config['mongodb_conn'])
        self.db = self.client[origin]
        self.collection = self.db[config['mongodb_authors_table']]

    def save(self, author: Dict[str, Any]) -> str:
        """
        Persist a single author document to the database.
        
        Stores an author record in MongoDB, handling the case where an ID field
        might be present but set to None. This method is primarily used when
        processing individual author records during incremental data ingestion.
        
        Args:
            author (Dict[str, Any]): Author data dictionary containing metadata
                such as name, affiliations, and external identifiers
            
        Returns:
            str: String representation of the MongoDB ObjectId assigned to the document
        """
        if 'id' in author and author['id'] is None:
            author.pop('id')

        result = self.collection.insert_one(author)
        return str(result.inserted_id)

    def save_many(self, authors: List[Dict[str, Any]]) -> List[str]:
        """
        Efficiently store multiple author documents in a single database operation.
        
        Performs bulk insertion of author records, optimizing database performance
        when processing batches of authors. This method is particularly useful during
        initial data loading or when processing publication batches with multiple authors.
        
        Args:
            authors (List[Dict[str, Any]]): List of author data dictionaries to be stored
            
        Returns:
            List[str]: List of string representations of MongoDB ObjectIds for all inserted documents
        """
        for author in authors:
            if 'id' in author and author['id'] is None:
                author.pop('id')

        result = self.collection.insert_many(authors)
        return [str(id) for id in result.inserted_ids]

    def find_by_id(self, author_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an author record by its external identifier.
        
        Performs a lookup using the author's source-specific identifier rather than
        the MongoDB internal _id field. This allows for deduplication and referential
        integrity when processing authors across multiple publications.
        
        Args:
            author_id (str): External identifier for the author (typically from the original data source)
            
        Returns:
            Optional[Dict[str, Any]]: Complete author document if found, None if no matching record exists
        """
        return self.collection.find_one({"id": author_id})

    def close(self):
        """
        Release database connection resources.
        
        Properly terminates the MongoDB connection to prevent resource leaks.
        This method should be called when the repository instance is no longer needed,
        especially in long-running applications or when connection pooling is important
        for resource management.
        """
        self.client.close()