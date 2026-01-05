"""
Structured output creation for dataset relationships.

This module provides comprehensive structured output format for documenting
dataset relationships, integration patterns, and metadata.
"""

import json
import uuid
from datetime import datetime, UTC
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class RelationshipType(Enum):
    """Types of dataset relationships."""
    MERGE = "merge"
    FUSION = "fusion"
    JOIN = "join"
    LINKAGE = "linkage"
    INTEGRATION = "integration"
    CONCATENATION = "concatenation"
    AGGREGATION = "aggregation"
    TRANSFORMATION = "transformation"
    OTHER = "other"


class ComplexityLevel(Enum):
    """Complexity levels for dataset relationships."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RelationshipStrength(Enum):
    """Strength levels for dataset relationships."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class DatasetMetadata:
    """Metadata for a dataset in a relationship."""
    name: str
    source: Optional[str] = None
    version: Optional[str] = None
    size: Optional[str] = None
    record_count: Optional[int] = None
    fields: Optional[List[str]] = None
    data_types: Optional[Dict[str, str]] = None
    quality_score: Optional[float] = None
    last_updated: Optional[datetime] = None
    description: Optional[str] = None


@dataclass
class IntegrationMetadata:
    """Metadata about the integration process."""
    integration_date: Optional[datetime] = None
    processing_time: Optional[str] = None
    software_version: Optional[str] = None
    hardware_resources: Optional[Dict[str, Any]] = None
    data_volume_processed: Optional[str] = None
    success_rate: Optional[float] = None
    error_rate: Optional[float] = None
    validation_status: Optional[str] = None


@dataclass
class RelationshipMetrics:
    """Metrics for dataset relationships."""
    confidence_score: float
    relationship_strength: RelationshipStrength
    complexity_level: ComplexityLevel
    data_overlap_percentage: Optional[float] = None
    field_matching_score: Optional[float] = None
    temporal_alignment_score: Optional[float] = None
    quality_improvement_score: Optional[float] = None
    performance_impact_score: Optional[float] = None


@dataclass
class DatasetRelationship:
    """Structured representation of a dataset relationship."""
    # Core identification
    relationship_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    relationship_type: RelationshipType = RelationshipType.OTHER
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    # Dataset information
    source_datasets: List[DatasetMetadata] = field(default_factory=list)
    target_dataset: Optional[DatasetMetadata] = None
    
    # Relationship details
    relationship_metrics: Optional[RelationshipMetrics] = None
    integration_metadata: Optional[IntegrationMetadata] = None
    
    # Analysis results (from previous tasks)
    methodology: Optional[str] = None
    join_keys: Optional[List[str]] = None
    integration_challenges: Optional[List[Dict[str, Any]]] = None
    success_metrics: Optional[Dict[str, Any]] = None
    lessons_learned: Optional[List[Dict[str, Any]]] = None
    validation_methods: Optional[List[Dict[str, Any]]] = None
    risk_assessment: Optional[Dict[str, Any]] = None
    
    # Additional metadata
    publication_context: Optional[str] = None
    research_domain: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        
        # Convert enums to strings
        data['relationship_type'] = self.relationship_type.value
        if self.relationship_metrics:
            data['relationship_metrics']['relationship_strength'] = self.relationship_metrics.relationship_strength.value
            data['relationship_metrics']['complexity_level'] = self.relationship_metrics.complexity_level.value
        
        # Convert datetime objects to ISO format
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        
        if self.integration_metadata and self.integration_metadata.integration_date:
            data['integration_metadata']['integration_date'] = self.integration_metadata.integration_date.isoformat()
        
        for dataset in data['source_datasets']:
            if dataset.get('last_updated'):
                dataset['last_updated'] = dataset['last_updated'].isoformat()
        
        if data.get('target_dataset') and data['target_dataset'].get('last_updated'):
            data['target_dataset']['last_updated'] = data['target_dataset']['last_updated'].isoformat()
        
        return data
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def validate(self) -> List[str]:
        """Validate the relationship data and return list of errors."""
        errors = []
        
        # Required fields validation
        if not self.source_datasets:
            errors.append("At least one source dataset is required")
        
        if not self.relationship_metrics:
            errors.append("Relationship metrics are required")
        
        # Dataset validation
        for i, dataset in enumerate(self.source_datasets):
            if not dataset.name:
                errors.append(f"Source dataset {i} must have a name")
        
        # Metrics validation
        if self.relationship_metrics:
            if not (0 <= self.relationship_metrics.confidence_score <= 10):
                errors.append("Confidence score must be between 0 and 10")
            
            if self.relationship_metrics.data_overlap_percentage and not (0 <= self.relationship_metrics.data_overlap_percentage <= 100):
                errors.append("Data overlap percentage must be between 0 and 100")
        
        return errors


@dataclass
class RelationshipCollection:
    """Collection of dataset relationships."""
    collection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Dataset Relationships Collection"
    description: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    relationships: List[DatasetRelationship] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_relationship(self, relationship: DatasetRelationship) -> None:
        """Add a relationship to the collection."""
        self.relationships.append(relationship)
        self.updated_at = datetime.now(UTC)
    
    def remove_relationship(self, relationship_id: str) -> bool:
        """Remove a relationship by ID."""
        for i, rel in enumerate(self.relationships):
            if rel.relationship_id == relationship_id:
                del self.relationships[i]
                self.updated_at = datetime.now(UTC)
                return True
        return False
    
    def get_relationship(self, relationship_id: str) -> Optional[DatasetRelationship]:
        """Get a relationship by ID."""
        for rel in self.relationships:
            if rel.relationship_id == relationship_id:
                return rel
        return None
    
    def filter_by_type(self, relationship_type: RelationshipType) -> List[DatasetRelationship]:
        """Filter relationships by type."""
        return [rel for rel in self.relationships if rel.relationship_type == relationship_type]
    
    def filter_by_complexity(self, complexity_level: ComplexityLevel) -> List[DatasetRelationship]:
        """Filter relationships by complexity level."""
        return [
            rel for rel in self.relationships 
            if rel.relationship_metrics and rel.relationship_metrics.complexity_level == complexity_level
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self.relationships:
            return {
                "total_relationships": 0,
                "relationship_types": {},
                "complexity_distribution": {},
                "average_confidence": 0.0
            }
        
        # Count relationship types
        type_counts = {}
        complexity_counts = {}
        total_confidence = 0.0
        valid_confidence_count = 0
        
        for rel in self.relationships:
            # Type counts
            rel_type = rel.relationship_type.value
            type_counts[rel_type] = type_counts.get(rel_type, 0) + 1
            
            # Complexity counts
            if rel.relationship_metrics:
                complexity = rel.relationship_metrics.complexity_level.value
                complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
                
                # Confidence score
                if rel.relationship_metrics.confidence_score is not None:
                    total_confidence += rel.relationship_metrics.confidence_score
                    valid_confidence_count += 1
        
        return {
            "total_relationships": len(self.relationships),
            "relationship_types": type_counts,
            "complexity_distribution": complexity_counts,
            "average_confidence": total_confidence / valid_confidence_count if valid_confidence_count > 0 else 0.0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['relationships'] = [rel.to_dict() for rel in self.relationships]
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def validate_all(self) -> Dict[str, List[str]]:
        """Validate all relationships and return errors by relationship ID."""
        errors = {}
        for rel in self.relationships:
            rel_errors = rel.validate()
            if rel_errors:
                errors[rel.relationship_id] = rel_errors
        return errors


class RelationshipOutputManager:
    """Manager for creating and validating structured relationship outputs."""
    
    def __init__(self):
        self.collection = RelationshipCollection()
    
    def create_relationship_from_analysis(
        self,
        dataset1: str,
        dataset2: str,
        join_type: str,
        confidence_score: float,
        methodology: Optional[str] = None,
        join_keys: Optional[List[str]] = None,
        integration_challenges: Optional[List[Dict[str, Any]]] = None,
        success_metrics: Optional[Dict[str, Any]] = None,
        lessons_learned: Optional[List[Dict[str, Any]]] = None,
        validation_methods: Optional[List[Dict[str, Any]]] = None,
        risk_assessment: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> DatasetRelationship:
        """Create a relationship from analysis results."""
        
        # Map join type to relationship type
        relationship_type = self._map_join_type_to_relationship_type(join_type)
        
        # Create source datasets
        source_datasets = [
            DatasetMetadata(name=dataset1),
            DatasetMetadata(name=dataset2)
        ]
        
        # Determine relationship strength and complexity
        relationship_strength = self._calculate_relationship_strength(confidence_score)
        complexity_level = self._calculate_complexity_level(
            confidence_score, integration_challenges, success_metrics
        )
        
        # Create relationship metrics
        metrics = RelationshipMetrics(
            confidence_score=confidence_score,
            relationship_strength=relationship_strength,
            complexity_level=complexity_level
        )
        
        # Create relationship
        relationship = DatasetRelationship(
            relationship_type=relationship_type,
            source_datasets=source_datasets,
            relationship_metrics=metrics,
            methodology=methodology,
            join_keys=join_keys,
            integration_challenges=integration_challenges,
            success_metrics=success_metrics,
            lessons_learned=lessons_learned,
            validation_methods=validation_methods,
            risk_assessment=risk_assessment,
            **kwargs
        )
        
        return relationship
    
    def _map_join_type_to_relationship_type(self, join_type: str) -> RelationshipType:
        """Map join type string to RelationshipType enum."""
        join_type_lower = join_type.lower()
        
        if 'merge' in join_type_lower:
            return RelationshipType.MERGE
        elif 'fusion' in join_type_lower:
            return RelationshipType.FUSION
        elif 'join' in join_type_lower:
            return RelationshipType.JOIN
        elif 'linkage' in join_type_lower or 'link' in join_type_lower:
            return RelationshipType.LINKAGE
        elif 'integration' in join_type_lower:
            return RelationshipType.INTEGRATION
        elif 'concatenation' in join_type_lower or 'concat' in join_type_lower:
            return RelationshipType.CONCATENATION
        elif 'aggregation' in join_type_lower or 'aggregate' in join_type_lower:
            return RelationshipType.AGGREGATION
        elif 'transformation' in join_type_lower or 'transform' in join_type_lower:
            return RelationshipType.TRANSFORMATION
        else:
            return RelationshipType.OTHER
    
    def _calculate_relationship_strength(self, confidence_score: float) -> RelationshipStrength:
        """Calculate relationship strength based on confidence score."""
        if confidence_score >= 9.0:
            return RelationshipStrength.VERY_STRONG
        elif confidence_score >= 7.0:
            return RelationshipStrength.STRONG
        elif confidence_score >= 5.0:
            return RelationshipStrength.MODERATE
        else:
            return RelationshipStrength.WEAK
    
    def _calculate_complexity_level(
        self,
        confidence_score: float,
        integration_challenges: Optional[List[Dict[str, Any]]] = None,
        success_metrics: Optional[Dict[str, Any]] = None
    ) -> ComplexityLevel:
        """Calculate complexity level based on various factors."""
        complexity_score = 0
        
        # Base complexity from confidence (inverse relationship)
        complexity_score += (10 - confidence_score) * 0.3
        
        # Complexity from challenges
        if integration_challenges:
            for challenge in integration_challenges:
                if isinstance(challenge, dict):
                    severity = challenge.get('severity', 'medium')
                    if severity == 'critical':
                        complexity_score += 2.0
                    elif severity == 'high':
                        complexity_score += 1.5
                    elif severity == 'medium':
                        complexity_score += 1.0
                    else:
                        complexity_score += 0.5
        
        # Complexity from success metrics
        if success_metrics:
            data_loss = success_metrics.get('data_loss_percentage', '0%')
            if isinstance(data_loss, str):
                try:
                    loss_value = float(data_loss.replace('%', ''))
                    if loss_value > 10:
                        complexity_score += 1.0
                    elif loss_value > 5:
                        complexity_score += 0.5
                except ValueError:
                    pass
        
        # Map to complexity level
        if complexity_score >= 6.0:
            return ComplexityLevel.CRITICAL
        elif complexity_score >= 4.0:
            return ComplexityLevel.HIGH
        elif complexity_score >= 2.0:
            return ComplexityLevel.MEDIUM
        else:
            return ComplexityLevel.LOW
    
    def add_relationship(self, relationship: DatasetRelationship) -> None:
        """Add a relationship to the collection."""
        self.collection.add_relationship(relationship)
    
    def get_collection(self) -> RelationshipCollection:
        """Get the current collection."""
        return self.collection
    
    def export_to_json(self, filepath: str, indent: int = 2) -> None:
        """Export collection to JSON file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.collection.to_json(indent=indent))
            logger.info(f"Relationship collection exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export collection to {filepath}: {e}")
            raise
    
    def validate_collection(self) -> Dict[str, List[str]]:
        """Validate the entire collection."""
        return self.collection.validate_all()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return self.collection.get_statistics() 