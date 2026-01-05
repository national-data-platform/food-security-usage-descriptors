"""
Dataset models for the publication analysis agent.

This module contains Pydantic models for representing datasets and related entities.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Dataset(BaseModel):
    """Model representing a dataset."""
    
    id: Optional[str] = Field(None, alias="_id", description="Unique identifier for the dataset")
    name: str = Field(..., description="Name of the dataset")
    aliases: List[str] = Field(default_factory=list, description="Alternative names for the dataset")
    description: Optional[str] = Field(None, description="Description of the dataset")
    type: Optional[str] = Field(None, description="Type of dataset")
    domain: Optional[str] = Field(None, description="Domain of the dataset")
    flag_terms: Optional[List[str]] = Field(None, description="Flag terms associated with the dataset")
    publication_references: List['PublicationReference'] = Field(default_factory=list, description="Publication references")
    
    def get_all_identifiers(self) -> List[str]:
        """Get all identifiers for this dataset (name + aliases)."""
        identifiers = [self.name]
        identifiers.extend(self.aliases)
        return list(set(identifiers))  # Remove duplicates
    
    def add_publication_reference(self, ref: 'PublicationReference') -> None:
        """Add a publication reference to this dataset."""
        # Check if reference already exists
        for existing_ref in self.publication_references:
            if existing_ref.publication_id == ref.publication_id:
                return  # Already exists, don't add duplicate
        self.publication_references.append(ref)
    
    def to_mongo_dict(self) -> dict:
        """Convert dataset to MongoDB document format."""
        data = self.model_dump(by_alias=True)
        if self.id is None:
            data.pop('_id', None)
        return data
    
    @staticmethod
    def from_mongo_dict(data: dict) -> 'Dataset':
        """Create Dataset from MongoDB document."""
        # Convert MongoDB data to Dataset format
        data_copy = data.copy()
        
        # Handle ObjectId conversion
        if '_id' in data_copy:
            from bson import ObjectId
            if isinstance(data_copy['_id'], ObjectId):
                data_copy['_id'] = str(data_copy['_id'])
            # Rename _id to id if id doesn't exist
            if 'id' not in data_copy:
                data_copy['id'] = data_copy.pop('_id')
        
        return Dataset(**data_copy)
    
    model_config = {
        "populate_by_name": True,
        "validate_assignment": True
    }


class DatasetQuery(BaseModel):
    """Model for dataset queries."""
    
    query: Optional[str] = Field(None, description="Search query for datasets")
    aliases: Optional[List[str]] = Field(None, description="List of aliases to search for")
    flag_terms: Optional[List[str]] = Field(None, description="List of flag terms to search for")
    fuzzy_threshold: Optional[float] = Field(None, description="Fuzzy matching threshold")
    limit: Optional[int] = Field(10, description="Maximum number of results to return")
    include_aliases: bool = Field(True, description="Whether to search in aliases")
    include_descriptions: bool = Field(True, description="Whether to search in descriptions")


class DatasetMatchResult(BaseModel):
    """Model representing a dataset match result."""
    
    dataset: Dataset = Field(..., description="The matched dataset")
    score: Optional[float] = Field(None, description="Match score (0-1)")
    matched_fields: List[str] = Field(default_factory=list, description="Fields that matched")
    matched_text: Optional[str] = Field(None, description="The text that matched")
    match_score: Optional[float] = Field(None, description="Match score (alias for score)")
    matched_field: Optional[str] = Field(None, description="Matched field (alias for matched_fields)")
    matched_value: Optional[str] = Field(None, description="Matched value (alias for matched_text)")
    query: Optional[str] = Field(None, description="Query that produced this match")


class PublicationReference(BaseModel):
    """Model representing a publication reference."""
    
    publication_id: str = Field(None, description="Publication ID")
    title: Optional[str] = Field(None, description="Publication title")
    doi: Optional[str] = Field(None, description="DOI of the publication")
    year: Optional[int] = Field(None, description="Publication year")
    authors: Optional[List[str]] = Field(None, description="List of authors")
    journal: Optional[str] = Field(None, description="Journal name")
    dataset_mentions: Optional[List[str]] = Field(None, description="Datasets mentioned in the publication") 