"""
Repository package for MongoDB data access layer.

This package provides repository classes for managing different entity types
in MongoDB, including publications, authors, institutions, and datasets.
Each repository encapsulates database operations for its respective entity.
"""

from .publication_repository import PublicationRepository
from .author_repository import AuthorRepository
from .institution_repository import InstitutionRepository
from .dataset_repository import DatasetRepository