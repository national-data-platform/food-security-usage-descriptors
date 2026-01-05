"""
GROBID Parquet File Parser and Text Extraction Service.

This module provides utilities to parse GROBID-processed Parquet files and extract
structured text content from nested JSON structures. It handles the complex
GROBID output format with proper error handling and memory optimization.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Generator
import warnings

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator

from ..config.logging_config import get_logger

logger = get_logger(__name__)


class GROBIDSection(BaseModel):
    """Represents a GROBID text section with sentences and metadata."""
    
    title: Optional[Dict[str, Any]] = None
    sentences: List[Dict[str, Any]] = Field(default_factory=list)
    annotations: Optional[Dict[str, Any]] = None
    
    def extract_text(self) -> str:
        """Extract plain text from all sentences in the section."""
        if not self.sentences:
            return ""
        
        texts = []
        for sentence in self.sentences:
            if isinstance(sentence, dict) and "text" in sentence:
                texts.append(sentence["text"])
        
        return " ".join(texts)


class GROBIDTextContent(BaseModel):
    """Represents GROBID text content with sections."""
    
    sections: List[Union[GROBIDSection, Dict[str, Any]]] = Field(default_factory=list)
    annotations: Optional[Dict[str, Any]] = None
    
    def extract_text(self) -> str:
        """Extract plain text from all sections."""
        if not self.sections:
            return ""
        
        texts = []
        for section in self.sections:
            if isinstance(section, GROBIDSection):
                texts.append(section.extract_text())
            elif isinstance(section, dict):
                # Handle raw dict format
                if "sentences" in section:
                    sentence_texts = []
                    for sentence in section["sentences"]:
                        if isinstance(sentence, dict) and "text" in sentence:
                            sentence_texts.append(sentence["text"])
                    texts.append(" ".join(sentence_texts))
        
        return " ".join(texts)


class GROBIDFullText(BaseModel):
    """Represents the full GROBID text structure."""
    
    abstract: Optional[GROBIDTextContent] = None
    body: Optional[GROBIDTextContent] = None
    acknowledgement: Optional[GROBIDTextContent] = None
    availability: Optional[GROBIDTextContent] = None
    title: Optional[Dict[str, Any]] = None
    authorship: Optional[Dict[str, Any]] = None
    bibliography: Optional[Dict[str, Any]] = None
    identifiers: Optional[Dict[str, Any]] = None
    keywords: Optional[List[str]] = None
    
    def get_title_text(self) -> str:
        """Extract title text."""
        if self.title and isinstance(self.title, dict) and "text" in self.title:
            return self.title["text"]
        return ""
    
    def get_abstract_text(self) -> str:
        """Extract abstract text."""
        if self.abstract:
            return self.abstract.extract_text()
        return ""
    
    def get_body_text(self) -> str:
        """Extract body text."""
        if self.body:
            return self.body.extract_text()
        return ""
    
    def get_acknowledgement_text(self) -> str:
        """Extract acknowledgement text."""
        if self.acknowledgement:
            return self.acknowledgement.extract_text()
        return ""
    
    def get_availability_text(self) -> str:
        """Extract availability text."""
        if self.availability:
            return self.availability.extract_text()
        return ""
    
    def get_all_text(self) -> str:
        """Extract all text content combined."""
        texts = [
            self.get_title_text(),
            self.get_abstract_text(),
            self.get_body_text(),
            self.get_acknowledgement_text(),
            self.get_availability_text()
        ]
        return " ".join(text for text in texts if text)
    
    def get_keywords(self) -> List[str]:
        """Extract keywords."""
        if self.keywords and isinstance(self.keywords, list):
            return [str(kw) for kw in self.keywords if kw]
        return []
    
    def get_doi(self) -> Optional[str]:
        """Extract DOI from identifiers."""
        if self.identifiers and isinstance(self.identifiers, dict):
            return self.identifiers.get("doi")
        return None


class GROBIDPublication(BaseModel):
    """Represents a complete GROBID publication record."""
    
    publication_id: Optional[str] = None
    id: Optional[str] = None
    publication_ids: Optional[List[str]] = None
    fulltext: Optional[GROBIDFullText] = None
    processing: Optional[List[Dict[str, Any]]] = None
    file: Optional[List[Dict[str, Any]]] = None
    gbq_processing: Optional[List[Dict[str, Any]]] = None
    
    model_config = {
        "arbitrary_types_allowed": True,
        "populate_by_name": True,
        "validate_assignment": True
    }
    
    def get_publication_id(self) -> Optional[str]:
        """Get the primary publication ID."""
        if self.publication_id:
            return self.publication_id
        if self.publication_ids is not None and len(self.publication_ids) > 0:
            return str(self.publication_ids[0])
        return None


class GROBIDParser:
    """
    Parser for GROBID-processed Parquet files with text extraction capabilities.
    
    This class provides functionality to read GROBID Parquet files and extract
    structured text content from the nested JSON structure. It handles various
    sections including abstract, body, acknowledgements, and availability.
    """
    
    def __init__(self, chunk_size: int = 1000, memory_limit_mb: int = 512):
        """
        Initialize the GROBID parser.
        
        Args:
            chunk_size: Number of rows to process in each chunk for memory optimization
            memory_limit_mb: Memory limit in MB for batch processing
        """
        self.chunk_size = chunk_size
        self.memory_limit_mb = memory_limit_mb
        self.logger = logger
        
        # Configure pandas for memory efficiency
        pd.options.mode.chained_assignment = None
        warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
    
    def read_parquet_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Read a GROBID Parquet file into a pandas DataFrame.
        
        Args:
            file_path: Path to the Parquet file
            
        Returns:
            DataFrame containing the publication data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a valid Parquet file
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {file_path}")
        
        if not file_path.suffix.lower() == '.parquet':
            raise ValueError(f"File is not a Parquet file: {file_path}")
        
        try:
            self.logger.info(f"Reading Parquet file: {file_path}")
            df = pd.read_parquet(file_path)
            
            # Validate expected columns
            expected_columns = ['publication_id', 'fulltext']
            missing_columns = [col for col in expected_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.warning(f"Missing expected columns: {missing_columns}")
            
            self.logger.info(f"Successfully read {len(df)} records from {file_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading Parquet file {file_path}: {e}")
            raise ValueError(f"Failed to read Parquet file: {e}")
    
    def parse_publication_record(self, record: Dict[str, Any]) -> Optional[GROBIDPublication]:
        """
        Parse a single publication record from the DataFrame.
        
        Args:
            record: Dictionary containing publication data
            
        Returns:
            GROBIDPublication object or None if parsing fails
        """
        try:
            # Handle numpy arrays in publication_ids
            if 'publication_ids' in record and isinstance(record['publication_ids'], np.ndarray):
                record['publication_ids'] = record['publication_ids'].tolist()
            
            # Parse fulltext structure
            if 'fulltext' in record and isinstance(record['fulltext'], dict):
                fulltext_data = record['fulltext']
                
                # Create GROBIDFullText object
                fulltext = GROBIDFullText(
                    abstract=self._parse_text_content(fulltext_data.get('abstract')),
                    body=self._parse_text_content(fulltext_data.get('body')),
                    acknowledgement=self._parse_text_content(fulltext_data.get('acknowledgement')),
                    availability=self._parse_text_content(fulltext_data.get('availability')),
                    title=fulltext_data.get('title'),
                    authorship=fulltext_data.get('authorship'),
                    bibliography=fulltext_data.get('bibliography'),
                    identifiers=fulltext_data.get('identifiers'),
                    keywords=fulltext_data.get('keywords')
                )
                
                record['fulltext'] = fulltext
            
            return GROBIDPublication(**record)
            
        except Exception as e:
            self.logger.error(f"Error parsing publication record: {e}")
            return None
    
    def _parse_text_content(self, content: Optional[Dict[str, Any]]) -> Optional[GROBIDTextContent]:
        """
        Parse text content structure from GROBID format.
        
        Args:
            content: Dictionary containing text content structure
            
        Returns:
            GROBIDTextContent object or None if parsing fails
        """
        if not content or not isinstance(content, dict):
            return None
        
        try:
            sections = []
            if 'sections' in content and isinstance(content['sections'], list):
                for section_data in content['sections']:
                    if isinstance(section_data, dict):
                        section = GROBIDSection(
                            title=section_data.get('title'),
                            sentences=section_data.get('sentences', []),
                            annotations=section_data.get('annotations')
                        )
                        sections.append(section)
            
            return GROBIDTextContent(
                sections=sections,
                annotations=content.get('annotations')
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing text content: {e}")
            return None
    
    def extract_text_from_publication(self, publication: GROBIDPublication) -> Dict[str, str]:
        """
        Extract text content from all sections of a publication.
        
        Args:
            publication: GROBIDPublication object
            
        Returns:
            Dictionary containing extracted text from different sections
        """
        if not publication.fulltext:
            return {}
        
        return {
            'title': publication.fulltext.get_title_text(),
            'abstract': publication.fulltext.get_abstract_text(),
            'body': publication.fulltext.get_body_text(),
            'acknowledgement': publication.fulltext.get_acknowledgement_text(),
            'availability': publication.fulltext.get_availability_text(),
            'all_text': publication.fulltext.get_all_text()
        }
    
    def process_publication_file(self, file_path: Union[str, Path]) -> Generator[GROBIDPublication, None, None]:
        """
        Process a Parquet file and yield publication objects.
        
        Args:
            file_path: Path to the Parquet file
            
        Yields:
            GROBIDPublication objects
        """
        df = self.read_parquet_file(file_path)
        
        for _, row in df.iterrows():
            record = row.to_dict()
            publication = self.parse_publication_record(record)
            
            if publication:
                yield publication
    
    def process_publication_file_batch(
        self, 
        file_path: Union[str, Path],
        batch_size: Optional[int] = None
    ) -> Generator[List[GROBIDPublication], None, None]:
        """
        Process a Parquet file in batches for memory optimization.
        
        Args:
            file_path: Path to the Parquet file
            batch_size: Number of records per batch (uses self.chunk_size if None)
            
        Yields:
            Lists of GROBIDPublication objects
        """
        if batch_size is None:
            batch_size = self.chunk_size
        
        df = self.read_parquet_file(file_path)
        
        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            publications = []
            for _, row in batch_df.iterrows():
                record = row.to_dict()
                publication = self.parse_publication_record(record)
                
                if publication:
                    publications.append(publication)
            
            if publications:
                yield publications
    
    def extract_text_batch(
        self, 
        publications: List[GROBIDPublication]
    ) -> List[Dict[str, Any]]:
        """
        Extract text from a batch of publications.
        
        Args:
            publications: List of GROBIDPublication objects
            
        Returns:
            List of dictionaries containing publication ID and extracted text
        """
        results = []
        
        for publication in publications:
            pub_id = publication.get_publication_id()
            if not pub_id:
                continue
            
            text_content = self.extract_text_from_publication(publication)
            
            result = {
                'publication_id': pub_id,
                'text_content': text_content,
                'keywords': publication.fulltext.get_keywords() if publication.fulltext else [],
                'doi': publication.fulltext.get_doi() if publication.fulltext else None
            }
            
            results.append(result)
        
        return results
    
    def validate_publication_structure(self, publication: GROBIDPublication) -> bool:
        """
        Validate that a publication has the required structure.
        
        Args:
            publication: GROBIDPublication object to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not publication:
            return False
        
        if not publication.get_publication_id():
            self.logger.warning("Publication missing ID")
            return False
        
        if not publication.fulltext:
            self.logger.warning(f"Publication {publication.get_publication_id()} missing fulltext")
            return False
        
        return True
    
    def get_publication_metadata(self, publication: GROBIDPublication) -> Dict[str, Any]:
        """
        Extract metadata from a publication.
        
        Args:
            publication: GROBIDPublication object
            
        Returns:
            Dictionary containing publication metadata
        """
        metadata = {
            'publication_id': publication.get_publication_id(),
            'doi': None,
            'keywords': [],
            'has_abstract': False,
            'has_body': False,
            'has_acknowledgement': False,
            'has_availability': False
        }
        
        if publication.fulltext:
            metadata['doi'] = publication.fulltext.get_doi()
            metadata['keywords'] = publication.fulltext.get_keywords()
            metadata['has_abstract'] = bool(publication.fulltext.get_abstract_text())
            metadata['has_body'] = bool(publication.fulltext.get_body_text())
            metadata['has_acknowledgement'] = bool(publication.fulltext.get_acknowledgement_text())
            metadata['has_availability'] = bool(publication.fulltext.get_availability_text())
        
        return metadata 