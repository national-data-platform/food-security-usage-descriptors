"""
Dimensions service for querying authors, institutions, and publications.

This module provides services for interacting with the dimensions database
collections, including fuzzy matching, filtering, and batch operations.
"""

import logging
import asyncio
from typing import List, Optional, Dict, Any, Set, Union, Tuple
from datetime import datetime, timedelta
from functools import wraps

from rapidfuzz import fuzz, process
from pymongo import IndexModel, ASCENDING, TEXT
from pymongo.errors import DuplicateKeyError, OperationFailure
from tenacity import retry, stop_after_attempt, wait_exponential

from .mongodb_client import MongoDBClient
from ..models.dimensions import (
    Author, Institution, Publication,
    AuthorQuery, InstitutionQuery, PublicationQuery
)
from ..config.settings import DatabaseSettings


logger = logging.getLogger(__name__)


class AuthorService:
    """
    Service for querying and managing authors.
    
    This service provides methods for searching authors with fuzzy matching
    and various filtering capabilities.
    """
    
    def __init__(self, mongodb_client: MongoDBClient, fuzzy_threshold: float = 0.8) -> None:
        """Initialize AuthorService."""
        self.mongodb_client = mongodb_client
        self.fuzzy_threshold = fuzzy_threshold
        self._indexes_created = False
        
        logger.info(f"AuthorService initialized with fuzzy_threshold={fuzzy_threshold}")

    async def ensure_indexes(self) -> None:
        """Ensure proper indexes exist on the authors collection."""
        if self._indexes_created:
            return
            
        try:
            async with self.mongodb_client.ensure_connection():
                collection = self.mongodb_client.authors_collection
                
                # Create indexes
                indexes = [
                    IndexModel([("name", TEXT)], name="name_text_index"),
                    IndexModel([("name", ASCENDING)], name="name_index"),
                    IndexModel([("orcid", ASCENDING)], name="orcid_index"),
                    IndexModel([("id", ASCENDING)], name="author_id_index"),
                ]
                
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: collection.create_indexes(indexes)
                )
                
                self._indexes_created = True
                logger.info("Successfully created author collection indexes")
                
        except DuplicateKeyError:
            self._indexes_created = True
            logger.debug("Author collection indexes already exist")
        except Exception as e:
            logger.error(f"Failed to create author collection indexes: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search_authors(self, query: AuthorQuery) -> List[Author]:
        """
        Search for authors based on query parameters.
        
        Args:
            query: Author query parameters
            
        Returns:
            List of matching authors
        """
        try:
            await self.ensure_indexes()
            
            async with self.mongodb_client.ensure_connection():
                collection = self.mongodb_client.authors_collection
                
                # Build MongoDB query
                mongo_query = {}
                
                if query.orcid:
                    mongo_query["orcid"] = query.orcid
                
                if query.name_pattern:
                    # Use text search if available, otherwise regex
                    mongo_query["$text"] = {"$search": query.name_pattern}
                
                cursor = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: collection.find(mongo_query).limit(query.limit or 100)
                )
                
                authors = []
                async for doc in self._async_cursor(cursor):
                    author = Author.from_mongo_dict(doc)
                    authors.append(author)
                
                # Apply fuzzy matching if needed
                if query.name_pattern and not query.orcid:
                    authors = self._apply_fuzzy_matching(authors, query.name_pattern, query.fuzzy_threshold)
                
                logger.info(f"Found {len(authors)} authors matching query")
                return authors
                
        except Exception as e:
            logger.error(f"Error searching authors: {e}")
            raise

    def _apply_fuzzy_matching(self, authors: List[Author], pattern: str, threshold: float) -> List[Author]:
        """Apply fuzzy matching to author names."""
        if not pattern:
            return authors
            
        scored_authors = []
        for author in authors:
            score = fuzz.ratio(pattern.lower(), author.name.lower()) / 100.0
            if score >= threshold:
                scored_authors.append((author, score))
        
        # Sort by score descending
        scored_authors.sort(key=lambda x: x[1], reverse=True)
        return [author for author, score in scored_authors]

    async def get_author_by_orcid(self, orcid: str) -> Optional[Author]:
        """Get author by ORCID identifier."""
        try:
            await self.ensure_indexes()
            
            async with self.mongodb_client.ensure_connection():
                collection = self.mongodb_client.authors_collection
                
                doc = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: collection.find_one({"orcid": orcid})
                )
                
                if doc:
                    return Author.from_mongo_dict(doc)
                return None
                
        except Exception as e:
            logger.error(f"Error getting author by ORCID: {e}")
            raise

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


class InstitutionService:
    """
    Service for querying and managing institutions.
    
    This service provides methods for searching institutions with geographic
    filtering and fuzzy matching capabilities.
    """
    
    def __init__(self, mongodb_client: MongoDBClient, fuzzy_threshold: float = 0.8) -> None:
        """Initialize InstitutionService."""
        self.mongodb_client = mongodb_client
        self.fuzzy_threshold = fuzzy_threshold
        self._indexes_created = False
        
        logger.info(f"InstitutionService initialized with fuzzy_threshold={fuzzy_threshold}")

    async def ensure_indexes(self) -> None:
        """Ensure proper indexes exist on the institutions collection."""
        if self._indexes_created:
            return
            
        try:
            async with self.mongodb_client.ensure_connection():
                collection = self.mongodb_client.institutions_collection
                
                # Create indexes
                indexes = [
                    IndexModel([("name", TEXT)], name="name_text_index"),
                    IndexModel([("name", ASCENDING)], name="name_index"),
                    IndexModel([("id", ASCENDING)], name="institution_id_index"),
                    IndexModel([("country", ASCENDING)], name="country_index"),
                    IndexModel([("state", ASCENDING)], name="state_index"),
                    IndexModel([("city", ASCENDING)], name="city_index"),
                    IndexModel([("country_code", ASCENDING)], name="country_code_index"),
                    IndexModel([
                        ("country", ASCENDING),
                        ("state", ASCENDING),
                        ("city", ASCENDING)
                    ], name="location_index"),
                ]
                
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: collection.create_indexes(indexes)
                )
                
                self._indexes_created = True
                logger.info("Successfully created institution collection indexes")
                
        except DuplicateKeyError:
            self._indexes_created = True
            logger.debug("Institution collection indexes already exist")
        except Exception as e:
            logger.error(f"Failed to create institution collection indexes: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search_institutions(self, query: InstitutionQuery) -> List[Institution]:
        """
        Search for institutions based on query parameters.
        
        Args:
            query: Institution query parameters
            
        Returns:
            List of matching institutions
        """
        try:
            await self.ensure_indexes()
            
            async with self.mongodb_client.ensure_connection():
                collection = self.mongodb_client.institutions_collection
                
                # Build MongoDB query
                mongo_query = {}
                
                if query.country:
                    mongo_query["country"] = {"$regex": query.country, "$options": "i"}
                
                if query.country_code:
                    mongo_query["country_code"] = query.country_code.upper()
                
                if query.state:
                    mongo_query["state"] = {"$regex": query.state, "$options": "i"}
                
                if query.city:
                    mongo_query["city"] = {"$regex": query.city, "$options": "i"}
                
                if query.name_pattern:
                    mongo_query["$text"] = {"$search": query.name_pattern}
                
                cursor = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: collection.find(mongo_query).limit(query.limit or 100)
                )
                
                institutions = []
                async for doc in self._async_cursor(cursor):
                    institution = Institution.from_mongo_dict(doc)
                    institutions.append(institution)
                
                # Apply fuzzy matching if needed
                if query.name_pattern:
                    institutions = self._apply_fuzzy_matching(institutions, query.name_pattern, query.fuzzy_threshold)
                
                logger.info(f"Found {len(institutions)} institutions matching query")
                return institutions
                
        except Exception as e:
            logger.error(f"Error searching institutions: {e}")
            raise

    def _apply_fuzzy_matching(self, institutions: List[Institution], pattern: str, threshold: float) -> List[Institution]:
        """Apply fuzzy matching to institution names."""
        if not pattern:
            return institutions
            
        scored_institutions = []
        for institution in institutions:
            score = fuzz.ratio(pattern.lower(), institution.name.lower()) / 100.0
            if score >= threshold:
                scored_institutions.append((institution, score))
        
        # Sort by score descending
        scored_institutions.sort(key=lambda x: x[1], reverse=True)
        return [institution for institution, score in scored_institutions]

    async def get_institutions_by_country(self, country_code: str) -> List[Institution]:
        """Get all institutions in a specific country."""
        try:
            await self.ensure_indexes()
            
            async with self.mongodb_client.ensure_connection():
                collection = self.mongodb_client.institutions_collection
                
                cursor = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: collection.find({"country_code": country_code.upper()})
                )
                
                institutions = []
                async for doc in self._async_cursor(cursor):
                    institution = Institution.from_mongo_dict(doc)
                    institutions.append(institution)
                
                logger.info(f"Found {len(institutions)} institutions in {country_code}")
                return institutions
                
        except Exception as e:
            logger.error(f"Error getting institutions by country: {e}")
            raise

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


class PublicationService:
    """
    Service for querying and managing publications.
    
    This service provides methods for searching publications with complex
    filtering based on authors, institutions, topics, and datasets.
    """
    
    def __init__(self, mongodb_client: MongoDBClient, fuzzy_threshold: float = 0.8) -> None:
        """Initialize PublicationService."""
        self.mongodb_client = mongodb_client
        self.fuzzy_threshold = fuzzy_threshold
        self._indexes_created = False
        
        logger.info(f"PublicationService initialized with fuzzy_threshold={fuzzy_threshold}")

    async def ensure_indexes(self) -> None:
        """Ensure proper indexes exist on the publications collection."""
        if self._indexes_created:
            return
            
        try:
            async with self.mongodb_client.ensure_connection():
                collection = self.mongodb_client.publications_collection
                
                # Create indexes
                indexes = [
                    IndexModel([("title", TEXT)], name="title_text_index"),
                    IndexModel([("id", ASCENDING)], name="publication_id_index"),
                    IndexModel([("doi", ASCENDING)], name="doi_index"),
                    IndexModel([("publication_year", ASCENDING)], name="year_index"),
                    IndexModel([("citation_count", ASCENDING)], name="citation_index"),
                    IndexModel([("authors.name", ASCENDING)], name="author_name_index"),
                    IndexModel([("authors.orcid", ASCENDING)], name="author_orcid_index"),
                    IndexModel([("authors.institutions.name", ASCENDING)], name="institution_name_index"),
                    IndexModel([("journals.name", ASCENDING)], name="journal_name_index"),
                    IndexModel([("topics.name", ASCENDING)], name="topic_name_index"),
                    IndexModel([("mentioned_datasets.name", ASCENDING)], name="dataset_name_index"),
                    IndexModel([
                        ("publication_year", ASCENDING),
                        ("citation_count", ASCENDING)
                    ], name="year_citation_index"),
                ]
                
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: collection.create_indexes(indexes)
                )
                
                self._indexes_created = True
                logger.info("Successfully created publication collection indexes")
                
        except DuplicateKeyError:
            self._indexes_created = True
            logger.debug("Publication collection indexes already exist")
        except Exception as e:
            logger.error(f"Failed to create publication collection indexes: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search_publications(self, query: PublicationQuery) -> List[Publication]:
        """
        Search for publications based on query parameters.
        
        Args:
            query: Publication query parameters
            
        Returns:
            List of matching publications
        """
        try:
            await self.ensure_indexes()
            
            async with self.mongodb_client.ensure_connection():
                collection = self.mongodb_client.publications_collection
                
                # Build MongoDB query
                mongo_query = {}
                
                if query.doi:
                    mongo_query["doi"] = query.doi
                
                if query.title_pattern:
                    mongo_query["$text"] = {"$search": query.title_pattern}
                
                if query.author_names:
                    mongo_query["authors.name"] = {"$in": query.author_names}
                
                if query.institution_names:
                    mongo_query["authors.institutions.name"] = {"$in": query.institution_names}
                
                if query.journal_names:
                    mongo_query["journals.name"] = {"$in": query.journal_names}
                
                if query.topics:
                    mongo_query["topics.name"] = {"$in": query.topics}
                
                if query.mentioned_datasets:
                    mongo_query["mentioned_datasets.name"] = {"$in": query.mentioned_datasets}
                
                if query.publication_years:
                    mongo_query["publication_year"] = {"$in": query.publication_years}
                
                # Citation count filters
                citation_filter = {}
                if query.min_citation_count is not None:
                    citation_filter["$gte"] = query.min_citation_count
                if query.max_citation_count is not None:
                    citation_filter["$lte"] = query.max_citation_count
                if citation_filter:
                    mongo_query["citation_count"] = citation_filter
                
                cursor = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: collection.find(mongo_query).limit(query.limit or 100)
                )
                
                publications = []
                async for doc in self._async_cursor(cursor):
                    publication = Publication.from_mongo_dict(doc)
                    publications.append(publication)
                
                logger.info(f"Found {len(publications)} publications matching query")
                return publications
                
        except Exception as e:
            logger.error(f"Error searching publications: {e}")
            raise

    async def get_publication_by_doi(self, doi: str) -> Optional[Publication]:
        """Get publication by DOI."""
        try:
            await self.ensure_indexes()
            
            async with self.mongodb_client.ensure_connection():
                collection = self.mongodb_client.publications_collection
                
                doc = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: collection.find_one({"doi": doi})
                )
                
                if doc:
                    return Publication.from_mongo_dict(doc)
                return None
                
        except Exception as e:
            logger.error(f"Error getting publication by DOI: {e}")
            raise

    async def get_publications_by_dataset(self, dataset_name: str) -> List[Publication]:
        """Get all publications that mention a specific dataset."""
        try:
            await self.ensure_indexes()
            
            async with self.mongodb_client.ensure_connection():
                collection = self.mongodb_client.publications_collection
                
                cursor = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: collection.find({"mentioned_datasets.name": dataset_name})
                )
                
                publications = []
                async for doc in self._async_cursor(cursor):
                    publication = Publication.from_mongo_dict(doc)
                    publications.append(publication)
                
                logger.info(f"Found {len(publications)} publications mentioning dataset: {dataset_name}")
                return publications
                
        except Exception as e:
            logger.error(f"Error getting publications by dataset: {e}")
            raise

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


class DimensionsService:
    """
    Unified service for querying all dimensions collections.
    
    This service combines AuthorService, InstitutionService, and PublicationService
    to provide a unified interface for complex queries across collections.
    """
    
    def __init__(self, mongodb_client: MongoDBClient, fuzzy_threshold: float = 0.8) -> None:
        """Initialize DimensionsService."""
        self.mongodb_client = mongodb_client
        self.fuzzy_threshold = fuzzy_threshold
        
        # Initialize individual services
        self.authors = AuthorService(mongodb_client, fuzzy_threshold)
        self.institutions = InstitutionService(mongodb_client, fuzzy_threshold)
        self.publications = PublicationService(mongodb_client, fuzzy_threshold)
        
        logger.info("DimensionsService initialized with all sub-services")

    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics for all dimensions collections."""
        try:
            mongodb_info = self.mongodb_client.get_connection_info()
            
            # Get collection stats if connected
            collection_stats = {}
            if self.mongodb_client.is_connected:
                async with self.mongodb_client.ensure_connection():
                    # Authors stats
                    authors_count = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.mongodb_client.authors_collection.estimated_document_count()
                    )
                    collection_stats["authors_count"] = authors_count
                    
                    # Institutions stats  
                    institutions_count = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.mongodb_client.institutions_collection.estimated_document_count()
                    )
                    collection_stats["institutions_count"] = institutions_count
                    
                    # Publications stats
                    publications_count = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.mongodb_client.publications_collection.estimated_document_count()
                    )
                    collection_stats["publications_count"] = publications_count
            
            return {
                "service_info": {
                    "fuzzy_threshold": self.fuzzy_threshold,
                },
                "mongodb_info": mongodb_info,
                "collection_stats": collection_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting dimensions connection stats: {e}")
            return {"error": str(e)} 