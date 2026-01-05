import pymongo
from typing import Dict, List, Optional, Any, Union

from src.infra.configuration import Configuration


class PublicationRepository:
    """
    Repository for managing academic publication data persistence in MongoDB.
    
    This class implements the data access layer for scholarly publications, providing
    a comprehensive interface for storing, retrieving, and analyzing publication metadata
    from heterogeneous academic data sources. It encapsulates MongoDB-specific operations
    while exposing domain-oriented methods that align with the application's use cases.
    
    The repository supports multi-tenant data organization by origin, enabling isolated
    data management for different scholarly databases (OpenAlex, OpenAIRE, Dimensions)
    while maintaining consistent access patterns across all data sources.
    
    Key responsibilities:
    - Publication document persistence with upsert semantics
    - Complex citation metric aggregations across scholarly entities
    - Dataset-specific queries for analytics and reporting
    - Connection lifecycle management for optimal resource utilization
    """

    def __init__(self, origin):
        """
        Initialize a repository instance for a specific scholarly data origin.
        
        Establishes MongoDB connections and configures the appropriate database schema
        based on the data origin identifier. This design enables data source segregation,
        allowing different scholarly databases to maintain separate collections while
        sharing the same repository interface and operations.
        
        The origin-based database selection supports:
        - Independent schema evolution per data source
        - Isolated data quality management
        - Source-specific performance optimization
        - Simplified data governance and compliance
        
        Args:
            origin (str): Data source identifier that determines the target database
                         (e.g., 'dimensions', 'openalex', 'openaire')
        """
        config = Configuration().get()
        self.client = pymongo.MongoClient(config['mongodb_conn'])
        self.db = self.client[origin]
        self.collection = self.db[config['mongodb_publications_table']]

    def save(self, publication: Dict[str, Any]) -> str:
        """
        Persist a publication document with intelligent upsert semantics.
        
        Implements atomic document replacement using a composite business key consisting
        of publication ID, organization, and dataset. This approach ensures data consistency
        while supporting incremental updates and reprocessing scenarios common in ETL pipelines.
        
        The method handles MongoDB ObjectId cleanup to prevent serialization conflicts
        and uses replace_one with upsert=True to maintain referential integrity across
        multiple processing runs.
        
        Business key composition:
        - Publication ID: External identifier from the source system
        - Organization: Organizational context for the research
        - Dataset: Specific dataset or collection the publication relates to
        
        Args:
            publication (Dict[str, Any]): Complete publication document with all metadata,
                                        including authors, institutions, citations, and topics
            
        Returns:
            str: String representation of the MongoDB ObjectId for the upserted document,
                 enabling downstream tracking and referencing
        """
        if '_id' in publication and publication['_id'] is None:
            publication.pop('_id')

        result = self.collection.replace_one(
            {
                "id": publication['id']
            }, publication, upsert=True)
        return str(result.upserted_id)

    def save_many(self, publications: List[Dict[str, Any]]) -> List[str]:
        """
        Perform high-performance bulk insertion of multiple publication documents.
        
        Optimizes data loading for large-scale ETL operations by leveraging MongoDB's
        bulk insert capabilities. This method significantly reduces network round-trips
        and improves throughput when processing large datasets from scholarly APIs.
        
        The implementation includes automatic ObjectId cleanup to prevent document
        conflicts and ensures transactional consistency across the entire batch.
        Ideal for initial data loads and bulk synchronization scenarios.
        
        Performance considerations:
        - Single network round-trip for the entire batch
        - Reduced connection overhead and latency
        - Optimized for write-heavy ETL workloads
        - Memory-efficient processing of large publication sets
        
        Args:
            publications (List[Dict[str, Any]]): Collection of publication documents
                                                ready for persistence
            
        Returns:
            List[str]: Ordered list of string ObjectIds corresponding to each inserted document,
                      maintaining the same sequence as the input list for correlation
        """
        for pub in publications:
            if '_id' in pub and pub['_id'] is None:
                pub.pop('_id')

        result = self.collection.insert_many(publications)
        return [str(id) for id in result.inserted_ids]

    def find_by_id(self, publication_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a publication document using its primary external identifier.
        
        Performs a direct lookup using the publication's external ID as assigned by
        the originating scholarly database (e.g., OpenAlex ID, DOI, Dimensions ID).
        This method provides the fastest access pattern for single document retrieval
        when the external identifier is known.
        
        The external ID represents the canonical identifier from the source system
        and serves as the primary reference for cross-system integration and citation
        tracking across different scholarly databases.
        
        Args:
            publication_id (str): External publication identifier from the source database
            
        Returns:
            Optional[Dict[str, Any]]: Complete publication document with all related entities
                                    if found, None if no matching publication exists
        """
        return self.collection.find_one({"id": publication_id})

    def find_by_publication_id_and_dataset(self, publication_id: str, task_id: str) -> Optional[
        Dict[str, Any]]:
        """
        Locate a publication using a comprehensive composite key for dataset-specific retrieval.
        
        Implements precise publication lookup using the complete business context that uniquely
        identifies a publication within a specific research dataset. This method is essential
        for preventing duplicate processing and enabling incremental ETL operations.
        
        The composite key approach supports:
        - Dataset-specific publication tracking
        - Organization-scoped research management
        - Incremental processing and deduplication
        - Multi-tenant research environments
        
        This query pattern is particularly valuable for checking processing status
        and implementing idempotent ETL operations that can safely restart without
        creating duplicate records.
        
        Args:
            task_id:
            publication_id (str): External publication identifier from the scholarly database
        
        Returns:
            Optional[Dict[str, Any]]: Publication document with complete metadata if found,
                                    None if no publication matches the composite criteria
        """
        return self.collection.find_one(
            {
                'task_id': task_id,
                'id': publication_id
            }
        )

    def count_publications_by_dataset(self) -> List[Dict[str, Union[str, int]]]:
        """
        Generate comprehensive dataset coverage analytics through aggregation.
        
        Executes a MongoDB aggregation pipeline to compute publication distribution
        across different datasets, providing critical insights for research impact
        analysis and dataset representation assessment. The results enable stakeholders
        to understand the scholarly coverage and influence of various research datasets.
        
        The aggregation pipeline:
        1. Groups publications by dataset name to eliminate duplicates
        2. Counts total publications per dataset
        3. Projects results into a standardized analytics format
        4. Sorts by publication count to highlight most influential datasets
        
        This analysis supports:
        - Research portfolio assessment
        - Dataset impact measurement
        - Resource allocation decisions
        - Scholarly influence tracking
        
        Returns:
            List[Dict[str, Union[str, int]]]: Ordered list of dataset analytics objects,
                each containing:
                - dataset (str): The name of the research dataset
                - count (int): Total number of publications referencing the dataset
                Sorted in descending order by publication count for impact prioritization.
        """
        pipeline = [
            {"$group": {"_id": "$dataset.name", "count": {"$sum": 1}}},
            {"$project": {"dataset": "$_id", "count": 1, "_id": 0}},
            {"$sort": {"count": -1}}
        ]
        return list(self.collection.aggregate(pipeline))

    def get_citations_count(self, task_id):
        """
        Execute comprehensive citation impact analysis across all scholarly entity dimensions.
        
        Performs sophisticated MongoDB aggregation pipelines to calculate total citation
        metrics for four critical categories of scholarly entities: journals, research topics,
        authors, and institutions. Each pipeline implements a carefully designed aggregation
        strategy to handle the complex many-to-many relationships inherent in scholarly data.
        
        Aggregation methodology for each entity type:
        1. **Array Expansion**: Unwinds nested entity arrays to create individual records
        2. **Deduplication**: Groups by composite keys (entity + publication) to prevent
           double-counting citations when publications have multiple entities
        3. **Citation Accumulation**: Aggregates citation counts across all publications
           for each unique entity
        4. **Impact Ranking**: Sorts results by total citations to identify most influential entities
        
        The citation metrics enable:
        - Journal impact factor analysis and ranking
        - Research topic trending and influence measurement
        - Author h-index and citation impact assessment
        - Institutional research output and influence evaluation
        
        Performance considerations:
        - Computationally intensive operation requiring significant MongoDB resources
        - Memory-intensive for large publication datasets
        - Results suitable for caching due to relatively stable citation patterns
        - Consider running during off-peak hours for production systems
        
        Returns:
            tuple: Comprehensive citation analytics containing five ordered lists:
                1. **Journal Citations**: Journal name and total citation count, ordered by impact
                2. **Topic Citations**: Research topic and aggregated citation metrics
                3. **Author Citations**: Author name and cumulative citation influence
                4. **Institution Citations**: Institution name and total research impact
                5. **Category Citations**: Research category and citation distribution
                
        Note:
            This method executes five separate aggregation pipelines sequentially,
            making it suitable for analytical reporting rather than real-time operations.
            Consider implementing result caching for frequently accessed citation metrics.
        """
        journals_citation_count = self.collection.aggregate([
            {"$match": {"mentioned_datasets.id": task_id}},
            {
                "$unwind": "$publication.journals"
            },
            {
                "$group": {
                    "_id": {
                        "journalId": "$publication.journals.name",
                        "publicationId": "$id"
                    },
                    "journalName": {
                        "$first": "$publication.journals.name"
                    },
                    "citationCount": {
                        "$first": "$publication.citation_count"
                    }
                }
            },
            {
                "$group": {
                    "_id": "$_id.journalId",
                    "journalName": {
                        "$first": "$journalName"
                    },
                    "totalCitations": {
                        "$sum": "$citationCount"
                    }
                }
            },
            {
                "$sort": {
                    "totalCitations": -1
                }
            }
        ])

        journals_citations_count = [a for a in journals_citation_count]

        topics_citation_count = self.collection.aggregate([
            {"$match": {"mentioned_datasets.id": task_id}},
            {"$unwind": "$publication.topics"},
            {
                "$group": {
                    "_id": {
                        "topicId": "$publication.topics.name",
                        "publicationId": "$id"
                    },
                    "topicName": {"$first": "$publication.topics.name"},
                    "citationCount": {"$first": "$publication.citation_count"}
                }
            },
            {
                "$group": {
                    "_id": "$_id.topicId",
                    "topicName": {"$first": "$topicName"},
                    "totalCitations": {"$sum": "$citationCount"}
                }
            },
            {"$sort": {"totalCitations": -1}},
        ])

        topics_citations_count = [a for a in topics_citation_count]

        authors_citation_count = self.collection.aggregate([
            {"$match": {"mentioned_datasets.id": task_id}},
            {"$unwind": "$publication.authors"},
            {
                "$group": {
                    "_id": {
                        "authorId": "$publication.authors.name",
                        "publicationId": "$id"
                    },
                    "authorName": {"$first": "$publication.authors.name"},
                    "citationCount": {"$first": "$publication.citation_count"}
                }
            },
            {
                "$group": {
                    "_id": "$_id.authorId",
                    "authorName": {"$first": "$authorName"},
                    "totalCitations": {"$sum": "$citationCount"}
                }
            },
            {"$sort": {"totalCitations": -1}},
        ])

        authors_citations_count = [a for a in authors_citation_count]

        institutions_citation_count = self.collection.aggregate([
            {"$match": {"mentioned_datasets.id": task_id}},
            {"$unwind": "$publication.institutions"},
            {
                "$group": {
                    "_id": {
                        "institutionId": "$publication.institutions.name",
                        "publicationId": "$id"
                    },
                    "institutionName": {
                        "$first": "$publication.institutions.name"
                    },
                    "citationCount": {
                        "$first": "$publication.citation_count"
                    }
                }
            },
            {
                "$group": {
                    "_id": "$_id.institutionId",
                    "institutionName": {"$first": "$institutionName"},
                    "totalCitations": {"$sum": "$citationCount"}
                }
            },
            {"$sort": {"totalCitations": -1}},
        ])

        institutions_citations_count = [a for a in institutions_citation_count]

        category_for_citation_count = self.collection.aggregate([
            {"$match": {"mentioned_datasets.id": task_id}},
            {"$unwind": "$publication.category_for"},
            {
                "$group": {
                    "_id": {
                        "categoryId": "$publication.category_for.name",
                        "publicationId": "$id"
                    },
                    "categoryName": {
                        "$first": "$publication.category_for.name"
                    },
                    "citationCount": {
                        "$first": "$publication.citation_count"
                    }
                }
            },
            {
                "$group": {
                    "_id": "$_id.categoryId",
                    "categoryName": {"$first": "$categoryName"},
                    "totalCitations": {"$sum": "$citationCount"}
                }
            },
            {"$sort": {"totalCitations": -1}},
        ])

        category_for_citation_count = [a for a in category_for_citation_count]

        return (journals_citations_count,
                topics_citations_count,
                authors_citations_count,
                institutions_citations_count,
                category_for_citation_count)

    def count(self, task_id):
        return self.collection.count_documents({"mentioned_datasets.id": task_id})

    def count_processed(self, task_id):
        return self.collection.count_documents({"mentioned_datasets.id": task_id, "sync_elastic": True})

    def count_failed(self, task_id):
        return self.collection.count_documents({"mentioned_datasets.id": task_id, "sync_elastic": False})

    def close(self):
        """
        Gracefully release MongoDB connection resources and cleanup repository state.
        
        Properly terminates the MongoDB client connection to prevent resource leaks
        and ensure clean shutdown. This method implements the connection lifecycle
        management pattern essential for production applications that manage database
        connections explicitly.
        
        Connection cleanup is critical for:
        - Preventing connection pool exhaustion
        - Ensuring proper resource deallocation
        - Supporting graceful application shutdown
        - Avoiding database connection limits violations
        
        Best practices:
        - Call this method in finally blocks or context managers
        - Invoke during application shutdown sequences
        - Use when transitioning between different data sources
        - Essential for long-running ETL processes with connection pooling
        """
        self.client.close()
