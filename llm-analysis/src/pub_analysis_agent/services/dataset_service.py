"""
Dataset service for querying and managing academic datasets.

This module provides the DatasetService class for interacting with the
general.datasets MongoDB collection, including fuzzy matching capabilities,
caching, and batch operations.
"""

import logging
import asyncio
from typing import List, Optional, Dict, Any, Set, Union
from datetime import datetime, timedelta
from functools import wraps
import re

from rapidfuzz import fuzz, process
from pymongo import IndexModel, ASCENDING, TEXT
from pymongo.errors import DuplicateKeyError, OperationFailure
from tenacity import retry, stop_after_attempt, wait_exponential

from .mongodb_client import MongoDBClient
from ..models.dataset import Dataset, DatasetMatchResult, DatasetQuery, PublicationReference
from ..config.settings import DatabaseSettings


logger = logging.getLogger(__name__)


def cached_method(ttl_seconds: int = 300):
    """
    Decorator for caching method results with TTL.
    
    Args:
        ttl_seconds: Time-to-live for cached results in seconds
    """
    def decorator(func):
        cache = {}
        cache_timestamps = {}
        
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Create cache key from method name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            current_time = datetime.utcnow()
            
            # Check if cached result exists and is not expired
            if (cache_key in cache and 
                cache_key in cache_timestamps and
                current_time - cache_timestamps[cache_key] < timedelta(seconds=ttl_seconds)):
                logger.debug(f"Cache hit for {func.__name__}")
                return cache[cache_key]
            
            # Execute function and cache result
            result = await func(self, *args, **kwargs)
            cache[cache_key] = result
            cache_timestamps[cache_key] = current_time
            logger.debug(f"Cached result for {func.__name__}")
            
            # Clean expired entries (simple cleanup)
            expired_keys = [
                key for key, timestamp in cache_timestamps.items()
                if current_time - timestamp >= timedelta(seconds=ttl_seconds)
            ]
            for key in expired_keys:
                cache.pop(key, None)
                cache_timestamps.pop(key, None)
            
            return result
        
        return wrapper
    return decorator


class DatasetService:
    """
    Service for querying and managing academic datasets.
    
    This service provides methods for searching datasets by aliases and flag terms,
    with fuzzy matching capabilities, caching, and batch operations.
    
    Attributes:
        mongodb_client: MongoDB client instance
        fuzzy_threshold: Default threshold for fuzzy matching (0.0-1.0)
        cache_ttl: Cache time-to-live in seconds
    """
    
    def __init__(
        self, 
        mongodb_client: MongoDBClient,
        fuzzy_threshold: float = 0.8,
        cache_ttl: int = 300
    ) -> None:
        """
        Initialize DatasetService.
        
        Args:
            mongodb_client: MongoDB client instance
            fuzzy_threshold: Default threshold for fuzzy matching
            cache_ttl: Cache time-to-live in seconds
        """
        self.mongodb_client = mongodb_client
        self.fuzzy_threshold = fuzzy_threshold
        self.cache_ttl = cache_ttl
        self._indexes_created = False
        
        logger.info(f"DatasetService initialized with fuzzy_threshold={fuzzy_threshold}")

    async def ensure_indexes(self) -> None:
        """
        Ensure proper indexes exist on the datasets collection for optimal query performance.
        
        Creates indexes for:
        - Text search on name, aliases, flag_terms, description
        - Individual field indexes for exact matching
        - Compound indexes for common query patterns
        """
        if self._indexes_created:
            return
            
        try:
            async with self.mongodb_client:
                collection = self.mongodb_client.datasets_collection
                
                # Create text search index
                text_index = IndexModel([
                    ("name", TEXT),
                    ("aliases", TEXT),
                    ("flag_terms", TEXT), 
                    ("description", TEXT)
                ], name="text_search_index")
                
                # Create individual field indexes
                name_index = IndexModel([("name", ASCENDING)], name="name_index")
                aliases_index = IndexModel([("aliases", ASCENDING)], name="aliases_index")
                flag_terms_index = IndexModel([("flag_terms", ASCENDING)], name="flag_terms_index")
                domain_index = IndexModel([("domain", ASCENDING)], name="domain_index")
                
                # Create compound indexes for common queries
                domain_name_index = IndexModel([
                    ("domain", ASCENDING),
                    ("name", ASCENDING)
                ], name="domain_name_index")
                
                indexes = [
                    text_index,
                    name_index,
                    aliases_index,
                    flag_terms_index,
                    domain_index,
                    domain_name_index
                ]
                
                # Use asyncio to run the blocking operation
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: collection.create_indexes(indexes)
                )
                
                self._indexes_created = True
                logger.info("Successfully created dataset collection indexes")
                
        except DuplicateKeyError:
            # Indexes already exist
            self._indexes_created = True
            logger.debug("Dataset collection indexes already exist")
        except Exception as e:
            logger.error(f"Failed to create dataset collection indexes: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def get_datasets_by_aliases(
        self, 
        aliases: List[str],
        fuzzy_threshold: Optional[float] = None,
        exact_match: bool = False
    ) -> List[DatasetMatchResult]:
        """
        Get datasets by matching against aliases with fuzzy matching support.
        
        Args:
            aliases: List of dataset aliases to search for
            fuzzy_threshold: Threshold for fuzzy matching (overrides default)
            exact_match: If True, only return exact matches
            
        Returns:
            List of dataset match results with similarity scores
        """
        if not aliases:
            return []
            
        threshold = fuzzy_threshold or self.fuzzy_threshold
        
        try:
            await self.ensure_indexes()
            
            async with self.mongodb_client.ensure_connection():
                collection = self.mongodb_client.datasets_collection
                
                # First try exact matches for performance
                exact_results = []
                if not exact_match:
                    query = {
                        "$or": [
                            {"name": {"$in": aliases}},
                            {"aliases": {"$in": aliases}}
                        ]
                    }
                    
                    cursor = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: collection.find(query)
                    )
                    
                    async for doc in self._async_cursor(cursor):
                        dataset = Dataset.from_mongo_dict(doc)
                        # Find which alias matched
                        for alias in aliases:
                            if alias.lower() == dataset.name.lower():
                                exact_results.append(DatasetMatchResult(
                                    dataset=dataset,
                                    match_score=1.0,
                                    matched_field="name",
                                    matched_value=dataset.name,
                                    query=alias
                                ))
                            elif alias.lower() in [a.lower() for a in dataset.aliases]:
                                matched_alias = next(a for a in dataset.aliases if a.lower() == alias.lower())
                                exact_results.append(DatasetMatchResult(
                                    dataset=dataset,
                                    match_score=1.0,
                                    matched_field="alias",
                                    matched_value=matched_alias,
                                    query=alias
                                ))
                
                if exact_match:
                    return exact_results
                
                # For fuzzy matching, get all datasets and perform in-memory fuzzy matching
                all_datasets = await self._get_all_datasets_cached()
                fuzzy_results = []
                
                for alias in aliases:
                    # Skip if we already found exact match
                    exact_matches = [r for r in exact_results if r.query.lower() == alias.lower()]
                    if exact_matches:
                        continue
                        
                    # Perform fuzzy matching
                    for dataset in all_datasets:
                        identifiers = dataset.get_all_identifiers()
                        
                        # Find best match among all identifiers
                        best_match = process.extractOne(
                            alias,
                            identifiers,
                            scorer=fuzz.ratio
                        )
                        
                        if best_match and best_match[1] >= (threshold * 100):
                            match_score = best_match[1] / 100.0
                            matched_value = best_match[0]
                            matched_field = "name" if matched_value == dataset.name else "alias"
                            
                            fuzzy_results.append(DatasetMatchResult(
                                dataset=dataset,
                                match_score=match_score,
                                matched_field=matched_field,
                                matched_value=matched_value,
                                query=alias
                            ))
                
                # Combine and sort results by match score
                all_results = exact_results + fuzzy_results
                all_results.sort(key=lambda x: x.match_score, reverse=True)
                
                # Remove duplicates (keep highest scoring match for each dataset)
                seen_datasets = set()
                unique_results = []
                for result in all_results:
                    dataset_id = str(result.dataset.id) if result.dataset.id else result.dataset.name
                    if dataset_id not in seen_datasets:
                        seen_datasets.add(dataset_id)
                        unique_results.append(result)
                
                logger.info(f"Found {len(unique_results)} datasets matching {len(aliases)} aliases")
                return unique_results
                
        except Exception as e:
            logger.error(f"Error getting datasets by aliases: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def get_datasets_by_flag_terms(
        self, 
        flag_terms: List[str],
        match_all: bool = False,
        fuzzy_threshold: Optional[float] = None
    ) -> List[DatasetMatchResult]:
        """
        Get datasets by matching against flag terms.
        
        Args:
            flag_terms: List of flag terms to search for
            match_all: If True, dataset must contain all flag terms
            fuzzy_threshold: Threshold for fuzzy matching (overrides default)
            
        Returns:
            List of dataset match results
        """
        if not flag_terms:
            return []
            
        threshold = fuzzy_threshold or self.fuzzy_threshold
        
        try:
            await self.ensure_indexes()
            
            # Normalize flag terms
            normalized_terms = [term.strip().lower() for term in flag_terms if term.strip()]
            
            async with self.mongodb_client.ensure_connection():
                collection = self.mongodb_client.datasets_collection
                
                # Build query based on match_all parameter
                if match_all:
                    query = {"flag_terms": {"$all": normalized_terms}}
                else:
                    query = {"flag_terms": {"$in": normalized_terms}}
                
                cursor = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: collection.find(query)
                )
                
                results = []
                async for doc in self._async_cursor(cursor):
                    dataset = Dataset.from_mongo_dict(doc)
                    
                    # Calculate match score based on flag term overlap
                    dataset_terms = set(dataset.flag_terms)
                    query_terms = set(normalized_terms)
                    
                    if match_all:
                        # All terms must be present
                        if query_terms.issubset(dataset_terms):
                            match_score = 1.0
                            matched_terms = list(query_terms)
                        else:
                            continue
                    else:
                        # At least one term must be present
                        matched_terms = list(query_terms.intersection(dataset_terms))
                        if matched_terms:
                            match_score = len(matched_terms) / len(query_terms)
                        else:
                            continue
                    
                    results.append(DatasetMatchResult(
                        dataset=dataset,
                        match_score=match_score,
                        matched_field="flag_terms",
                        matched_value=", ".join(matched_terms),
                        query=", ".join(flag_terms)
                    ))
                
                # Sort by match score
                results.sort(key=lambda x: x.match_score, reverse=True)
                
                logger.info(f"Found {len(results)} datasets matching flag terms")
                return results
                
        except Exception as e:
            logger.error(f"Error getting datasets by flag terms: {e}")
            raise

    @cached_method(ttl_seconds=600)  # Cache for 10 minutes
    async def get_all_known_datasets(
        self, 
        include_references: bool = True,
        domains: Optional[List[str]] = None
    ) -> List[Dataset]:
        """
        Get all known datasets from the collection.
        
        Args:
            include_references: Whether to include publication references
            domains: Optional list of domains to filter by
            
        Returns:
            List of all datasets
        """
        try:
            await self.ensure_indexes()
            
            async with self.mongodb_client.ensure_connection():
                collection = self.mongodb_client.datasets_collection
                
                # Build query
                query = {}
                if domains:
                    query["domain"] = {"$in": domains}
                
                # Projection to exclude references if not needed
                projection = None
                if not include_references:
                    projection = {"publication_references": 0}
                
                cursor = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: collection.find(query, projection)
                )
                
                datasets = []
                async for doc in self._async_cursor(cursor):
                    dataset = Dataset.from_mongo_dict(doc)
                    datasets.append(dataset)
                
                logger.info(f"Retrieved {len(datasets)} datasets from collection")
                return datasets
                
        except Exception as e:
            logger.error(f"Error getting all known datasets: {e}")
            raise

    async def batch_query_datasets(
        self, 
        queries: List[DatasetQuery]
    ) -> Dict[int, List[DatasetMatchResult]]:
        """
        Perform batch queries for multiple dataset search operations.
        
        Args:
            queries: List of dataset query objects
            
        Returns:
            Dictionary mapping query index to results
        """
        results = {}
        
        try:
            # Process queries concurrently
            tasks = []
            for i, query in enumerate(queries):
                task = self._process_single_query(i, query)
                tasks.append(task)
            
            # Wait for all queries to complete
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for query_idx, result in completed_results:
                if isinstance(result, Exception):
                    logger.error(f"Error in batch query {query_idx}: {result}")
                    results[query_idx] = []
                else:
                    results[query_idx] = result
                    
            logger.info(f"Completed batch query of {len(queries)} operations")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch query: {e}")
            raise

    async def _process_single_query(self, index: int, query: DatasetQuery) -> tuple[int, List[DatasetMatchResult]]:
        """Process a single query from a batch operation."""
        try:
            results = []
            
            # Process aliases if provided
            if query.aliases:
                alias_results = await self.get_datasets_by_aliases(
                    query.aliases,
                    fuzzy_threshold=query.fuzzy_threshold
                )
                results.extend(alias_results)
            
            # Process flag terms if provided
            if query.flag_terms:
                flag_results = await self.get_datasets_by_flag_terms(
                    query.flag_terms,
                    fuzzy_threshold=query.fuzzy_threshold
                )
                results.extend(flag_results)
            
            # Remove duplicates and sort by score
            seen_datasets = set()
            unique_results = []
            for result in sorted(results, key=lambda x: x.match_score, reverse=True):
                dataset_id = str(result.dataset.id) if result.dataset.id else result.dataset.name
                if dataset_id not in seen_datasets:
                    seen_datasets.add(dataset_id)
                    unique_results.append(result)
            
            # Apply limit if specified
            if query.limit:
                unique_results = unique_results[:query.limit]
            
            return index, unique_results
            
        except Exception as e:
            return index, e

    @cached_method(ttl_seconds=300)
    async def _get_all_datasets_cached(self) -> List[Dataset]:
        """Get all datasets with caching for fuzzy matching operations."""
        return await self.get_all_known_datasets(include_references=False)

    async def _async_cursor(self, cursor):
        """Convert blocking cursor iteration to async."""
        loop = asyncio.get_running_loop()
        
        def safe_cursor_next():
            """Safely get next document from cursor, returning None when exhausted."""
            try:
                return cursor.next()
            except StopIteration:
                return None
        
        while True:
            doc = await loop.run_in_executor(None, safe_cursor_next)
            if doc is None:
                break
            yield doc

    async def add_dataset(self, dataset: Dataset) -> Dataset:
        """
        Add a new dataset to the collection.
        
        Args:
            dataset: Dataset to add
            
        Returns:
            Added dataset with MongoDB ObjectId
            
        Raises:
            DuplicateKeyError: If dataset with same name already exists
        """
        try:
            await self.ensure_indexes()
            
            async with self.mongodb_client.ensure_connection():
                collection = self.mongodb_client.datasets_collection
                
                # Check for duplicate names
                existing = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: collection.find_one({"name": dataset.name})
                )
                
                if existing:
                    raise DuplicateKeyError(f"Dataset with name '{dataset.name}' already exists")
                
                # Insert dataset
                dataset_dict = dataset.to_mongo_dict()
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: collection.insert_one(dataset_dict)
                )
                
                # Return dataset with new ObjectId
                dataset.id = result.inserted_id
                logger.info(f"Successfully added dataset: {dataset.name}")
                return dataset
                
        except Exception as e:
            logger.error(f"Error adding dataset: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def get_dataset_by_publication(self, publication_id: str) -> List[Dataset]:
        """
        Get datasets mentioned in a specific publication.
        
        This method performs a two-step process:
        1. Find the publication in the dimensions.publications collection
        2. Extract mentioned_datasets and fetch full dataset details from general.datasets
        
        Args:
            publication_id: ID of the publication to search for
            
        Returns:
            List of datasets mentioned in the publication
            
        Raises:
            Exception: If there's an error in the database operations
        """
        try:
            await self.ensure_indexes()
            
            # Step 1: Find publication and extract mentioned_datasets
            mentioned_dataset_names = []
            
            async with self.mongodb_client.ensure_connection():
                # Access publications collection from dimensions database
                publications_collection = self.mongodb_client.publications_collection
                
                # Find publication by ID
                publication_doc = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: publications_collection.find_one({"id": publication_id})
                )
                
                if not publication_doc:
                    logger.warning(f"Publication with ID '{publication_id}' not found")
                    return []
                
                # Extract mentioned_datasets
                mentioned_datasets = publication_doc.get("mentioned_datasets", [])
                
                if not mentioned_datasets:
                    logger.info(f"No mentioned datasets found for publication '{publication_id}'")
                    return []
                
                # Extract dataset names/identifiers
                for dataset_ref in mentioned_datasets:
                    if isinstance(dataset_ref, dict):
                        # If mentioned_datasets contains objects with name/id fields
                        if "name" in dataset_ref:
                            mentioned_dataset_names.append(dataset_ref["name"])
                        elif "id" in dataset_ref:
                            mentioned_dataset_names.append(dataset_ref["id"])
                    elif isinstance(dataset_ref, str):
                        # If mentioned_datasets contains simple strings
                        mentioned_dataset_names.append(dataset_ref)
                
                logger.info(f"Found {len(mentioned_dataset_names)} mentioned datasets in publication '{publication_id}'")
            
            # Step 2: Fetch full dataset details from general.datasets collection
            datasets = []
            
            if mentioned_dataset_names:
                async with self.mongodb_client.ensure_connection():
                    datasets_collection = self.mongodb_client.datasets_collection
                    
                    # Build query to find datasets by name or aliases
                    query = {
                        "$or": [
                            {"name": {"$in": mentioned_dataset_names}},
                            {"aliases": {"$in": mentioned_dataset_names}}
                        ]
                    }
                    
                    cursor = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: datasets_collection.find(query)
                    )
                    
                    async for doc in self._async_cursor(cursor):
                        dataset = Dataset.from_mongo_dict(doc)
                        datasets.append(dataset)
                    
                    logger.info(f"Retrieved {len(datasets)} datasets from general collection")
            
            return datasets
            
        except Exception as e:
            logger.error(f"Error getting datasets by publication ID '{publication_id}': {e}")
            raise

    async def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get service statistics and connection information.
        
        Returns:
            Dictionary containing service stats
        """
        try:
            mongodb_info = self.mongodb_client.get_connection_info()
            
            # Get collection stats if connected
            collection_stats = {}
            if self.mongodb_client.is_connected:
                async with self.mongodb_client.ensure_connection():
                    collection = self.mongodb_client.datasets_collection
                    
                    # Get collection document count
                    count = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: collection.estimated_document_count()
                    )
                    collection_stats["document_count"] = count
            
            return {
                "service_info": {
                    "fuzzy_threshold": self.fuzzy_threshold,
                    "cache_ttl": self.cache_ttl,
                    "indexes_created": self._indexes_created
                },
                "mongodb_info": mongodb_info,
                "collection_stats": collection_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting connection stats: {e}")
            return {"error": str(e)} 