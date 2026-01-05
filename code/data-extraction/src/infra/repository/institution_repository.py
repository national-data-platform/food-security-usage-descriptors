import pymongo
from typing import Dict, List, Optional, Any

from src.infra.configuration import Configuration


class InstitutionRepository:
    """
    Repository for managing research institution data persistence in MongoDB.
    
    This class implements the data access layer for academic and research institutions,
    providing methods to store, retrieve, and manage institution records from various
    scholarly data sources. It encapsulates MongoDB-specific operations behind a
    domain-oriented interface aligned with the application's data model.
    """

    def __init__(self, origin: str):
        """
        Initialize a repository instance for a specific data origin.
        
        Creates a MongoDB connection and configures the appropriate database and
        collection based on the provided origin parameter. This allows for segregating
        institution data by source (e.g., 'dimensions', 'openalex', 'scopus') while
        maintaining consistent access patterns.
        
        Args:
            origin (str): Data source identifier used to select the appropriate database
        """
        config = Configuration().get()
        self.client = pymongo.MongoClient(config['mongodb_conn'])
        self.db = self.client[origin]
        self.collection = self.db[config['mongodb_institutions_table']]

    def save(self, institution: Dict[str, Any]) -> str:
        """
        Persist an institution document with upsert semantics.
        
        Stores an institution record in MongoDB, updating it if a matching record
        already exists (based on the institution's external ID) or inserting it as 
        a new document otherwise. This approach ensures data consistency when
        processing institutions that may appear in multiple publications.
        
        Args:
            institution (Dict[str, Any]): Institution data dictionary containing fields
                such as name, country, external identifiers, and classification metadata
            
        Returns:
            str: String representation of the MongoDB ObjectId for the upserted document,
                 which may be None if the document was updated rather than inserted
        """
        if '_id' in institution and institution['_id'] is None:
            institution.pop('_id')

        result = self.collection.replace_one({"id": institution['id']}, institution, upsert=True)
        return str(result.upserted_id)

    def save_many(self, institutions: List[Dict[str, Any]]) -> List[str]:
        """
        Batch insert multiple institution documents for efficient data loading.
        
        Performs bulk insertion of institution records, optimizing database performance
        for large-scale data ingestion scenarios. This method is particularly useful
        during initial data loading or when processing large batches of new institutions.
        
        Note: Unlike the single-document save method, this performs an insert operation
        rather than upsert, so it should be used primarily for new institution records.
        
        Args:
            institutions (List[Dict[str, Any]]): List of institution data dictionaries
            
        Returns:
            List[str]: List of string representations of MongoDB ObjectIds for all inserted documents
        """
        for institution in institutions:
            if '_id' in institution and institution['_id'] is None:
                institution.pop('_id')

        result = self.collection.insert_many(institutions)
        return [str(id) for id in result.inserted_ids]

    def find_by_id(self, institution_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an institution record by its external identifier.
        
        Performs a lookup using the institution's source-specific identifier rather than
        the MongoDB internal _id field. This allows for consistent referencing and
        deduplication when processing institution data across multiple sources.
        
        Args:
            institution_id (str): External identifier for the institution (typically from
                                  the original data source like OpenAlex or Dimensions)
            
        Returns:
            Optional[Dict[str, Any]]: Complete institution document if found, None if no
                                      matching record exists in the database
        """
        return self.collection.find_one({"id": institution_id})

    def close(self):
        """
        Release database connection resources properly.
        
        Terminates the MongoDB connection to prevent resource leaks and connection pool
        exhaustion. This method should be called when the repository instance is no longer
        needed, especially in long-running applications or when connection management is
        important for resource optimization.
        """
        self.client.close()