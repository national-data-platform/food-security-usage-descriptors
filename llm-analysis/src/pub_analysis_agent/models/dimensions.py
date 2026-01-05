"""
Dimensions models for the publication analysis agent.

This module contains Pydantic models for representing Dimensions API entities.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Author(BaseModel):
    """Model representing an author."""
    
    id: str = Field(..., description="Author ID")
    name: str = Field(..., description="Author name")
    first_name: Optional[str] = Field(None, description="First name")
    last_name: Optional[str] = Field(None, description="Last name")
    orcid: Optional[str] = Field(None, description="ORCID identifier")
    affiliations: Optional[List[str]] = Field(None, description="Affiliations")


class Institution(BaseModel):
    """Model representing an institution."""
    
    id: str = Field(..., description="Institution ID")
    name: str = Field(..., description="Institution name")
    country: Optional[str] = Field(None, description="Country")
    city: Optional[str] = Field(None, description="City")
    type: Optional[str] = Field(None, description="Institution type")


class Publication(BaseModel):
    """Model representing a publication."""
    
    id: str = Field(..., description="Publication ID")
    title: str = Field(..., description="Publication title")
    doi: Optional[str] = Field(None, description="DOI")
    year: Optional[int] = Field(None, description="Publication year")
    authors: Optional[List[Author]] = Field(None, description="Authors")
    journal: Optional[str] = Field(None, description="Journal name")
    abstract: Optional[str] = Field(None, description="Abstract")
    keywords: Optional[List[str]] = Field(None, description="Keywords")


class AuthorQuery(BaseModel):
    """Model for author queries."""
    
    name_pattern: Optional[str] = Field(None, description="Name pattern to search for")
    orcid: Optional[str] = Field(None, description="ORCID identifier")
    limit: Optional[int] = Field(100, description="Maximum number of results")


class InstitutionQuery(BaseModel):
    """Model for institution queries."""
    
    name_pattern: Optional[str] = Field(None, description="Name pattern to search for")
    country: Optional[str] = Field(None, description="Country code")
    type: Optional[str] = Field(None, description="Institution type")
    limit: Optional[int] = Field(100, description="Maximum number of results")


class PublicationQuery(BaseModel):
    """Model for publication queries."""
    
    title_pattern: Optional[str] = Field(None, description="Title pattern to search for")
    doi: Optional[str] = Field(None, description="DOI")
    year_from: Optional[int] = Field(None, description="Start year")
    year_to: Optional[int] = Field(None, description="End year")
    author_id: Optional[str] = Field(None, description="Author ID")
    journal: Optional[str] = Field(None, description="Journal name")
    limit: Optional[int] = Field(100, description="Maximum number of results") 