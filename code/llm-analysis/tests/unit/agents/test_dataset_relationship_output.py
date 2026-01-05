"""
Tests for dataset relationship output functionality.

This module tests the structured output creation for dataset relationships,
including validation, serialization, and collection management.
"""

import json
import pytest
from datetime import datetime, UTC
from unittest.mock import Mock, patch

from pub_analysis_agent.agents.dataset_relationship_output import (
    RelationshipType,
    ComplexityLevel,
    RelationshipStrength,
    DatasetMetadata,
    IntegrationMetadata,
    RelationshipMetrics,
    DatasetRelationship,
    RelationshipCollection,
    RelationshipOutputManager
)


class TestRelationshipType:
    """Test RelationshipType enum."""
    
    def test_relationship_type_values(self):
        """Test relationship type enum values."""
        assert RelationshipType.MERGE.value == "merge"
        assert RelationshipType.FUSION.value == "fusion"
        assert RelationshipType.JOIN.value == "join"
        assert RelationshipType.LINKAGE.value == "linkage"
        assert RelationshipType.INTEGRATION.value == "integration"
        assert RelationshipType.CONCATENATION.value == "concatenation"
        assert RelationshipType.AGGREGATION.value == "aggregation"
        assert RelationshipType.TRANSFORMATION.value == "transformation"
        assert RelationshipType.OTHER.value == "other"


class TestComplexityLevel:
    """Test ComplexityLevel enum."""
    
    def test_complexity_level_values(self):
        """Test complexity level enum values."""
        assert ComplexityLevel.LOW.value == "low"
        assert ComplexityLevel.MEDIUM.value == "medium"
        assert ComplexityLevel.HIGH.value == "high"
        assert ComplexityLevel.CRITICAL.value == "critical"


class TestRelationshipStrength:
    """Test RelationshipStrength enum."""
    
    def test_relationship_strength_values(self):
        """Test relationship strength enum values."""
        assert RelationshipStrength.WEAK.value == "weak"
        assert RelationshipStrength.MODERATE.value == "moderate"
        assert RelationshipStrength.STRONG.value == "strong"
        assert RelationshipStrength.VERY_STRONG.value == "very_strong"


class TestDatasetMetadata:
    """Test DatasetMetadata dataclass."""
    
    def test_basic_creation(self):
        """Test basic dataset metadata creation."""
        metadata = DatasetMetadata(name="Test Dataset")
        
        assert metadata.name == "Test Dataset"
        assert metadata.source is None
        assert metadata.version is None
        assert metadata.size is None
        assert metadata.record_count is None
        assert metadata.fields is None
        assert metadata.data_types is None
        assert metadata.quality_score is None
        assert metadata.last_updated is None
        assert metadata.description is None
    
    def test_full_creation(self):
        """Test full dataset metadata creation."""
        last_updated = datetime.now(UTC)
        metadata = DatasetMetadata(
            name="Comprehensive Dataset",
            source="Database A",
            version="1.2.3",
            size="1GB",
            record_count=10000,
            fields=["id", "name", "value"],
            data_types={"id": "int", "name": "string", "value": "float"},
            quality_score=0.95,
            last_updated=last_updated,
            description="A comprehensive test dataset"
        )
        
        assert metadata.name == "Comprehensive Dataset"
        assert metadata.source == "Database A"
        assert metadata.version == "1.2.3"
        assert metadata.size == "1GB"
        assert metadata.record_count == 10000
        assert metadata.fields == ["id", "name", "value"]
        assert metadata.data_types == {"id": "int", "name": "string", "value": "float"}
        assert metadata.quality_score == 0.95
        assert metadata.last_updated == last_updated
        assert metadata.description == "A comprehensive test dataset"


class TestIntegrationMetadata:
    """Test IntegrationMetadata dataclass."""
    
    def test_basic_creation(self):
        """Test basic integration metadata creation."""
        metadata = IntegrationMetadata()
        
        assert metadata.integration_date is None
        assert metadata.processing_time is None
        assert metadata.software_version is None
        assert metadata.hardware_resources is None
        assert metadata.data_volume_processed is None
        assert metadata.success_rate is None
        assert metadata.error_rate is None
        assert metadata.validation_status is None
    
    def test_full_creation(self):
        """Test full integration metadata creation."""
        integration_date = datetime.now(UTC)
        metadata = IntegrationMetadata(
            integration_date=integration_date,
            processing_time="2.5 hours",
            software_version="v1.0.0",
            hardware_resources={"cpu": "8 cores", "memory": "16GB"},
            data_volume_processed="1TB",
            success_rate=0.98,
            error_rate=0.02,
            validation_status="passed"
        )
        
        assert metadata.integration_date == integration_date
        assert metadata.processing_time == "2.5 hours"
        assert metadata.software_version == "v1.0.0"
        assert metadata.hardware_resources == {"cpu": "8 cores", "memory": "16GB"}
        assert metadata.data_volume_processed == "1TB"
        assert metadata.success_rate == 0.98
        assert metadata.error_rate == 0.02
        assert metadata.validation_status == "passed"


class TestRelationshipMetrics:
    """Test RelationshipMetrics dataclass."""
    
    def test_basic_creation(self):
        """Test basic relationship metrics creation."""
        metrics = RelationshipMetrics(
            confidence_score=8.5,
            relationship_strength=RelationshipStrength.STRONG,
            complexity_level=ComplexityLevel.MEDIUM
        )
        
        assert metrics.confidence_score == 8.5
        assert metrics.relationship_strength == RelationshipStrength.STRONG
        assert metrics.complexity_level == ComplexityLevel.MEDIUM
        assert metrics.data_overlap_percentage is None
        assert metrics.field_matching_score is None
        assert metrics.temporal_alignment_score is None
        assert metrics.quality_improvement_score is None
        assert metrics.performance_impact_score is None
    
    def test_full_creation(self):
        """Test full relationship metrics creation."""
        metrics = RelationshipMetrics(
            confidence_score=9.0,
            relationship_strength=RelationshipStrength.VERY_STRONG,
            complexity_level=ComplexityLevel.LOW,
            data_overlap_percentage=85.5,
            field_matching_score=0.92,
            temporal_alignment_score=0.88,
            quality_improvement_score=0.15,
            performance_impact_score=0.95
        )
        
        assert metrics.confidence_score == 9.0
        assert metrics.relationship_strength == RelationshipStrength.VERY_STRONG
        assert metrics.complexity_level == ComplexityLevel.LOW
        assert metrics.data_overlap_percentage == 85.5
        assert metrics.field_matching_score == 0.92
        assert metrics.temporal_alignment_score == 0.88
        assert metrics.quality_improvement_score == 0.15
        assert metrics.performance_impact_score == 0.95


class TestDatasetRelationship:
    """Test DatasetRelationship dataclass."""
    
    def test_basic_creation(self):
        """Test basic dataset relationship creation."""
        source_datasets = [
            DatasetMetadata(name="Dataset A"),
            DatasetMetadata(name="Dataset B")
        ]
        metrics = RelationshipMetrics(
            confidence_score=8.0,
            relationship_strength=RelationshipStrength.STRONG,
            complexity_level=ComplexityLevel.MEDIUM
        )
        
        relationship = DatasetRelationship(
            relationship_type=RelationshipType.MERGE,
            source_datasets=source_datasets,
            relationship_metrics=metrics
        )
        
        assert relationship.relationship_type == RelationshipType.MERGE
        assert len(relationship.source_datasets) == 2
        assert relationship.source_datasets[0].name == "Dataset A"
        assert relationship.source_datasets[1].name == "Dataset B"
        assert relationship.relationship_metrics == metrics
        assert relationship.target_dataset is None
        assert relationship.integration_metadata is None
        assert relationship.methodology is None
        assert relationship.join_keys is None
        assert relationship.integration_challenges is None
        assert relationship.success_metrics is None
        assert relationship.lessons_learned is None
        assert relationship.validation_methods is None
        assert relationship.risk_assessment is None
        assert relationship.publication_context is None
        assert relationship.research_domain is None
        assert relationship.tags == []
        assert relationship.notes is None
    
    def test_full_creation(self):
        """Test full dataset relationship creation."""
        source_datasets = [
            DatasetMetadata(name="Source Dataset A"),
            DatasetMetadata(name="Source Dataset B")
        ]
        target_dataset = DatasetMetadata(name="Target Dataset")
        metrics = RelationshipMetrics(
            confidence_score=9.0,
            relationship_strength=RelationshipStrength.VERY_STRONG,
            complexity_level=ComplexityLevel.LOW
        )
        integration_metadata = IntegrationMetadata(
            processing_time="1.5 hours",
            success_rate=0.95
        )
        
        relationship = DatasetRelationship(
            relationship_type=RelationshipType.FUSION,
            source_datasets=source_datasets,
            target_dataset=target_dataset,
            relationship_metrics=metrics,
            integration_metadata=integration_metadata,
            methodology="Key-based join using common identifiers",
            join_keys=["id", "timestamp"],
            integration_challenges=[{"category": "data_quality", "description": "Missing values"}],
            success_metrics={"data_loss_percentage": "2%", "integration_success_rate": "95%"},
            lessons_learned=[{"category": "technical", "lesson": "Data validation is crucial"}],
            validation_methods=[{"method": "cross_validation", "description": "K-fold validation"}],
            risk_assessment={"identified_risks": ["Data loss"], "mitigation_strategies": ["Backup"]},
            publication_context="Research on data integration",
            research_domain="Computer Science",
            tags=["fusion", "high_confidence"],
            notes="Important relationship for analysis"
        )
        
        assert relationship.relationship_type == RelationshipType.FUSION
        assert len(relationship.source_datasets) == 2
        assert relationship.target_dataset == target_dataset
        assert relationship.relationship_metrics == metrics
        assert relationship.integration_metadata == integration_metadata
        assert relationship.methodology == "Key-based join using common identifiers"
        assert relationship.join_keys == ["id", "timestamp"]
        assert relationship.integration_challenges == [{"category": "data_quality", "description": "Missing values"}]
        assert relationship.success_metrics == {"data_loss_percentage": "2%", "integration_success_rate": "95%"}
        assert relationship.lessons_learned == [{"category": "technical", "lesson": "Data validation is crucial"}]
        assert relationship.validation_methods == [{"method": "cross_validation", "description": "K-fold validation"}]
        assert relationship.risk_assessment == {"identified_risks": ["Data loss"], "mitigation_strategies": ["Backup"]}
        assert relationship.publication_context == "Research on data integration"
        assert relationship.research_domain == "Computer Science"
        assert relationship.tags == ["fusion", "high_confidence"]
        assert relationship.notes == "Important relationship for analysis"
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        source_datasets = [
            DatasetMetadata(name="Dataset A"),
            DatasetMetadata(name="Dataset B")
        ]
        metrics = RelationshipMetrics(
            confidence_score=8.0,
            relationship_strength=RelationshipStrength.STRONG,
            complexity_level=ComplexityLevel.MEDIUM
        )
        
        relationship = DatasetRelationship(
            relationship_type=RelationshipType.MERGE,
            source_datasets=source_datasets,
            relationship_metrics=metrics
        )
        
        data = relationship.to_dict()
        
        assert data['relationship_type'] == "merge"
        assert len(data['source_datasets']) == 2
        assert data['source_datasets'][0]['name'] == "Dataset A"
        assert data['source_datasets'][1]['name'] == "Dataset B"
        assert data['relationship_metrics']['confidence_score'] == 8.0
        assert data['relationship_metrics']['relationship_strength'] == "strong"
        assert data['relationship_metrics']['complexity_level'] == "medium"
        assert 'created_at' in data
        assert 'updated_at' in data
        assert 'relationship_id' in data
    
    def test_to_json(self):
        """Test conversion to JSON string."""
        source_datasets = [
            DatasetMetadata(name="Dataset A"),
            DatasetMetadata(name="Dataset B")
        ]
        metrics = RelationshipMetrics(
            confidence_score=8.0,
            relationship_strength=RelationshipStrength.STRONG,
            complexity_level=ComplexityLevel.MEDIUM
        )
        
        relationship = DatasetRelationship(
            relationship_type=RelationshipType.MERGE,
            source_datasets=source_datasets,
            relationship_metrics=metrics
        )
        
        json_str = relationship.to_json()
        data = json.loads(json_str)
        
        assert data['relationship_type'] == "merge"
        assert len(data['source_datasets']) == 2
        assert data['relationship_metrics']['confidence_score'] == 8.0
    
    def test_validate_valid(self):
        """Test validation with valid data."""
        source_datasets = [
            DatasetMetadata(name="Dataset A"),
            DatasetMetadata(name="Dataset B")
        ]
        metrics = RelationshipMetrics(
            confidence_score=8.0,
            relationship_strength=RelationshipStrength.STRONG,
            complexity_level=ComplexityLevel.MEDIUM
        )
        
        relationship = DatasetRelationship(
            relationship_type=RelationshipType.MERGE,
            source_datasets=source_datasets,
            relationship_metrics=metrics
        )
        
        errors = relationship.validate()
        assert len(errors) == 0
    
    def test_validate_missing_source_datasets(self):
        """Test validation with missing source datasets."""
        metrics = RelationshipMetrics(
            confidence_score=8.0,
            relationship_strength=RelationshipStrength.STRONG,
            complexity_level=ComplexityLevel.MEDIUM
        )
        
        relationship = DatasetRelationship(
            relationship_type=RelationshipType.MERGE,
            source_datasets=[],
            relationship_metrics=metrics
        )
        
        errors = relationship.validate()
        assert len(errors) == 1
        assert "At least one source dataset is required" in errors[0]
    
    def test_validate_missing_metrics(self):
        """Test validation with missing metrics."""
        source_datasets = [
            DatasetMetadata(name="Dataset A"),
            DatasetMetadata(name="Dataset B")
        ]
        
        relationship = DatasetRelationship(
            relationship_type=RelationshipType.MERGE,
            source_datasets=source_datasets,
            relationship_metrics=None
        )
        
        errors = relationship.validate()
        assert len(errors) == 1
        assert "Relationship metrics are required" in errors[0]
    
    def test_validate_invalid_confidence_score(self):
        """Test validation with invalid confidence score."""
        source_datasets = [
            DatasetMetadata(name="Dataset A"),
            DatasetMetadata(name="Dataset B")
        ]
        metrics = RelationshipMetrics(
            confidence_score=15.0,  # Invalid: > 10
            relationship_strength=RelationshipStrength.STRONG,
            complexity_level=ComplexityLevel.MEDIUM
        )
        
        relationship = DatasetRelationship(
            relationship_type=RelationshipType.MERGE,
            source_datasets=source_datasets,
            relationship_metrics=metrics
        )
        
        errors = relationship.validate()
        assert len(errors) == 1
        assert "Confidence score must be between 0 and 10" in errors[0]
    
    def test_validate_invalid_data_overlap(self):
        """Test validation with invalid data overlap percentage."""
        source_datasets = [
            DatasetMetadata(name="Dataset A"),
            DatasetMetadata(name="Dataset B")
        ]
        metrics = RelationshipMetrics(
            confidence_score=8.0,
            relationship_strength=RelationshipStrength.STRONG,
            complexity_level=ComplexityLevel.MEDIUM,
            data_overlap_percentage=150.0  # Invalid: > 100
        )
        
        relationship = DatasetRelationship(
            relationship_type=RelationshipType.MERGE,
            source_datasets=source_datasets,
            relationship_metrics=metrics
        )
        
        errors = relationship.validate()
        assert len(errors) == 1
        assert "Data overlap percentage must be between 0 and 100" in errors[0]


class TestRelationshipCollection:
    """Test RelationshipCollection dataclass."""
    
    def test_basic_creation(self):
        """Test basic collection creation."""
        collection = RelationshipCollection()
        
        assert collection.name == "Dataset Relationships Collection"
        assert collection.description is None
        assert len(collection.relationships) == 0
        assert collection.metadata == {}
    
    def test_add_relationship(self):
        """Test adding relationships to collection."""
        collection = RelationshipCollection()
        
        source_datasets = [
            DatasetMetadata(name="Dataset A"),
            DatasetMetadata(name="Dataset B")
        ]
        metrics = RelationshipMetrics(
            confidence_score=8.0,
            relationship_strength=RelationshipStrength.STRONG,
            complexity_level=ComplexityLevel.MEDIUM
        )
        
        relationship = DatasetRelationship(
            relationship_type=RelationshipType.MERGE,
            source_datasets=source_datasets,
            relationship_metrics=metrics
        )
        
        collection.add_relationship(relationship)
        
        assert len(collection.relationships) == 1
        assert collection.relationships[0] == relationship
    
    def test_remove_relationship(self):
        """Test removing relationships from collection."""
        collection = RelationshipCollection()
        
        source_datasets = [
            DatasetMetadata(name="Dataset A"),
            DatasetMetadata(name="Dataset B")
        ]
        metrics = RelationshipMetrics(
            confidence_score=8.0,
            relationship_strength=RelationshipStrength.STRONG,
            complexity_level=ComplexityLevel.MEDIUM
        )
        
        relationship = DatasetRelationship(
            relationship_type=RelationshipType.MERGE,
            source_datasets=source_datasets,
            relationship_metrics=metrics
        )
        
        collection.add_relationship(relationship)
        assert len(collection.relationships) == 1
        
        # Remove by ID
        result = collection.remove_relationship(relationship.relationship_id)
        assert result is True
        assert len(collection.relationships) == 0
        
        # Try to remove non-existent relationship
        result = collection.remove_relationship("non-existent-id")
        assert result is False
    
    def test_get_relationship(self):
        """Test getting relationship by ID."""
        collection = RelationshipCollection()
        
        source_datasets = [
            DatasetMetadata(name="Dataset A"),
            DatasetMetadata(name="Dataset B")
        ]
        metrics = RelationshipMetrics(
            confidence_score=8.0,
            relationship_strength=RelationshipStrength.STRONG,
            complexity_level=ComplexityLevel.MEDIUM
        )
        
        relationship = DatasetRelationship(
            relationship_type=RelationshipType.MERGE,
            source_datasets=source_datasets,
            relationship_metrics=metrics
        )
        
        collection.add_relationship(relationship)
        
        # Get existing relationship
        found = collection.get_relationship(relationship.relationship_id)
        assert found == relationship
        
        # Get non-existent relationship
        found = collection.get_relationship("non-existent-id")
        assert found is None
    
    def test_filter_by_type(self):
        """Test filtering relationships by type."""
        collection = RelationshipCollection()
        
        # Add relationships of different types
        for rel_type in [RelationshipType.MERGE, RelationshipType.FUSION, RelationshipType.JOIN]:
            source_datasets = [
                DatasetMetadata(name=f"Dataset A {rel_type.value}"),
                DatasetMetadata(name=f"Dataset B {rel_type.value}")
            ]
            metrics = RelationshipMetrics(
                confidence_score=8.0,
                relationship_strength=RelationshipStrength.STRONG,
                complexity_level=ComplexityLevel.MEDIUM
            )
            
            relationship = DatasetRelationship(
                relationship_type=rel_type,
                source_datasets=source_datasets,
                relationship_metrics=metrics
            )
            
            collection.add_relationship(relationship)
        
        # Filter by merge type
        merge_relationships = collection.filter_by_type(RelationshipType.MERGE)
        assert len(merge_relationships) == 1
        assert merge_relationships[0].relationship_type == RelationshipType.MERGE
    
    def test_filter_by_complexity(self):
        """Test filtering relationships by complexity level."""
        collection = RelationshipCollection()
        
        # Add relationships of different complexity levels
        for complexity in [ComplexityLevel.LOW, ComplexityLevel.MEDIUM, ComplexityLevel.HIGH]:
            source_datasets = [
                DatasetMetadata(name=f"Dataset A {complexity.value}"),
                DatasetMetadata(name=f"Dataset B {complexity.value}")
            ]
            metrics = RelationshipMetrics(
                confidence_score=8.0,
                relationship_strength=RelationshipStrength.STRONG,
                complexity_level=complexity
            )
            
            relationship = DatasetRelationship(
                relationship_type=RelationshipType.MERGE,
                source_datasets=source_datasets,
                relationship_metrics=metrics
            )
            
            collection.add_relationship(relationship)
        
        # Filter by medium complexity
        medium_relationships = collection.filter_by_complexity(ComplexityLevel.MEDIUM)
        assert len(medium_relationships) == 1
        assert medium_relationships[0].relationship_metrics.complexity_level == ComplexityLevel.MEDIUM
    
    def test_get_statistics_empty(self):
        """Test statistics for empty collection."""
        collection = RelationshipCollection()
        
        stats = collection.get_statistics()
        
        assert stats['total_relationships'] == 0
        assert stats['relationship_types'] == {}
        assert stats['complexity_distribution'] == {}
        assert stats['average_confidence'] == 0.0
    
    def test_get_statistics_with_relationships(self):
        """Test statistics for collection with relationships."""
        collection = RelationshipCollection()
        
        # Add relationships
        for i, rel_type in enumerate([RelationshipType.MERGE, RelationshipType.FUSION, RelationshipType.MERGE]):
            source_datasets = [
                DatasetMetadata(name=f"Dataset A {i}"),
                DatasetMetadata(name=f"Dataset B {i}")
            ]
            metrics = RelationshipMetrics(
                confidence_score=8.0 + i,
                relationship_strength=RelationshipStrength.STRONG,
                complexity_level=ComplexityLevel.MEDIUM
            )
            
            relationship = DatasetRelationship(
                relationship_type=rel_type,
                source_datasets=source_datasets,
                relationship_metrics=metrics
            )
            
            collection.add_relationship(relationship)
        
        stats = collection.get_statistics()
        
        assert stats['total_relationships'] == 3
        assert stats['relationship_types']['merge'] == 2
        assert stats['relationship_types']['fusion'] == 1
        assert stats['complexity_distribution']['medium'] == 3
        assert stats['average_confidence'] == 9.0  # (8.0 + 9.0 + 10.0) / 3
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        collection = RelationshipCollection()
        
        source_datasets = [
            DatasetMetadata(name="Dataset A"),
            DatasetMetadata(name="Dataset B")
        ]
        metrics = RelationshipMetrics(
            confidence_score=8.0,
            relationship_strength=RelationshipStrength.STRONG,
            complexity_level=ComplexityLevel.MEDIUM
        )
        
        relationship = DatasetRelationship(
            relationship_type=RelationshipType.MERGE,
            source_datasets=source_datasets,
            relationship_metrics=metrics
        )
        
        collection.add_relationship(relationship)
        
        data = collection.to_dict()
        
        assert data['name'] == "Dataset Relationships Collection"
        assert len(data['relationships']) == 1
        assert data['relationships'][0]['relationship_type'] == "merge"
        assert 'created_at' in data
        assert 'updated_at' in data
        assert 'collection_id' in data
    
    def test_to_json(self):
        """Test conversion to JSON string."""
        collection = RelationshipCollection()
        
        source_datasets = [
            DatasetMetadata(name="Dataset A"),
            DatasetMetadata(name="Dataset B")
        ]
        metrics = RelationshipMetrics(
            confidence_score=8.0,
            relationship_strength=RelationshipStrength.STRONG,
            complexity_level=ComplexityLevel.MEDIUM
        )
        
        relationship = DatasetRelationship(
            relationship_type=RelationshipType.MERGE,
            source_datasets=source_datasets,
            relationship_metrics=metrics
        )
        
        collection.add_relationship(relationship)
        
        json_str = collection.to_json()
        data = json.loads(json_str)
        
        assert data['name'] == "Dataset Relationships Collection"
        assert len(data['relationships']) == 1
        assert data['relationships'][0]['relationship_type'] == "merge"
    
    def test_validate_all(self):
        """Test validation of all relationships."""
        collection = RelationshipCollection()
        
        # Add valid relationship
        source_datasets = [
            DatasetMetadata(name="Dataset A"),
            DatasetMetadata(name="Dataset B")
        ]
        metrics = RelationshipMetrics(
            confidence_score=8.0,
            relationship_strength=RelationshipStrength.STRONG,
            complexity_level=ComplexityLevel.MEDIUM
        )
        
        valid_relationship = DatasetRelationship(
            relationship_type=RelationshipType.MERGE,
            source_datasets=source_datasets,
            relationship_metrics=metrics
        )
        
        # Add invalid relationship (missing metrics)
        invalid_relationship = DatasetRelationship(
            relationship_type=RelationshipType.MERGE,
            source_datasets=source_datasets,
            relationship_metrics=None
        )
        
        collection.add_relationship(valid_relationship)
        collection.add_relationship(invalid_relationship)
        
        errors = collection.validate_all()
        
        assert len(errors) == 1
        assert invalid_relationship.relationship_id in errors
        assert "Relationship metrics are required" in errors[invalid_relationship.relationship_id][0]


class TestRelationshipOutputManager:
    """Test RelationshipOutputManager class."""
    
    def test_initialization(self):
        """Test manager initialization."""
        manager = RelationshipOutputManager()
        
        assert manager.collection is not None
        assert isinstance(manager.collection, RelationshipCollection)
        assert len(manager.collection.relationships) == 0
    
    def test_create_relationship_from_analysis(self):
        """Test creating relationship from analysis results."""
        manager = RelationshipOutputManager()
        
        relationship = manager.create_relationship_from_analysis(
            dataset1="Dataset A",
            dataset2="Dataset B",
            join_type="merge",
            confidence_score=8.5,
            methodology="Key-based join using common identifiers",
            join_keys=["id", "timestamp"],
            integration_challenges=[{"category": "data_quality", "description": "Missing values"}],
            success_metrics={"data_loss_percentage": "2%", "integration_success_rate": "95%"},
            lessons_learned=[{"category": "technical", "lesson": "Data validation is crucial"}],
            validation_methods=[{"method": "cross_validation", "description": "K-fold validation"}],
            risk_assessment={"identified_risks": ["Data loss"], "mitigation_strategies": ["Backup"]},
            publication_context="Research on data integration",
            tags=["merge", "high_confidence"]
        )
        
        assert relationship.relationship_type == RelationshipType.MERGE
        assert len(relationship.source_datasets) == 2
        assert relationship.source_datasets[0].name == "Dataset A"
        assert relationship.source_datasets[1].name == "Dataset B"
        assert relationship.relationship_metrics.confidence_score == 8.5
        assert relationship.relationship_metrics.relationship_strength == RelationshipStrength.STRONG
        assert relationship.methodology == "Key-based join using common identifiers"
        assert relationship.join_keys == ["id", "timestamp"]
        assert relationship.integration_challenges == [{"category": "data_quality", "description": "Missing values"}]
        assert relationship.success_metrics == {"data_loss_percentage": "2%", "integration_success_rate": "95%"}
        assert relationship.lessons_learned == [{"category": "technical", "lesson": "Data validation is crucial"}]
        assert relationship.validation_methods == [{"method": "cross_validation", "description": "K-fold validation"}]
        assert relationship.risk_assessment == {"identified_risks": ["Data loss"], "mitigation_strategies": ["Backup"]}
        assert relationship.publication_context == "Research on data integration"
        assert "merge" in relationship.tags
        assert "high_confidence" in relationship.tags
    
    def test_map_join_type_to_relationship_type(self):
        """Test mapping join types to relationship types."""
        manager = RelationshipOutputManager()
        
        # Test various join types
        assert manager._map_join_type_to_relationship_type("merge") == RelationshipType.MERGE
        assert manager._map_join_type_to_relationship_type("fusion") == RelationshipType.FUSION
        assert manager._map_join_type_to_relationship_type("join") == RelationshipType.JOIN
        assert manager._map_join_type_to_relationship_type("linkage") == RelationshipType.LINKAGE
        assert manager._map_join_type_to_relationship_type("integration") == RelationshipType.INTEGRATION
        assert manager._map_join_type_to_relationship_type("concatenation") == RelationshipType.CONCATENATION
        assert manager._map_join_type_to_relationship_type("aggregation") == RelationshipType.AGGREGATION
        assert manager._map_join_type_to_relationship_type("transformation") == RelationshipType.TRANSFORMATION
        assert manager._map_join_type_to_relationship_type("unknown") == RelationshipType.OTHER
        
        # Test case insensitive
        assert manager._map_join_type_to_relationship_type("MERGE") == RelationshipType.MERGE
        assert manager._map_join_type_to_relationship_type("Fusion") == RelationshipType.FUSION
    
    def test_calculate_relationship_strength(self):
        """Test calculating relationship strength from confidence score."""
        manager = RelationshipOutputManager()
        
        assert manager._calculate_relationship_strength(9.5) == RelationshipStrength.VERY_STRONG
        assert manager._calculate_relationship_strength(8.0) == RelationshipStrength.STRONG
        assert manager._calculate_relationship_strength(6.0) == RelationshipStrength.MODERATE
        assert manager._calculate_relationship_strength(3.0) == RelationshipStrength.WEAK
    
    def test_calculate_complexity_level(self):
        """Test calculating complexity level."""
        manager = RelationshipOutputManager()
        
        # Test base complexity from confidence score (inverse relationship)
        # Higher confidence = lower complexity
        assert manager._calculate_complexity_level(10.0) == ComplexityLevel.LOW
        assert manager._calculate_complexity_level(8.0) == ComplexityLevel.LOW
        assert manager._calculate_complexity_level(6.0) == ComplexityLevel.LOW  # (10-6)*0.3 = 1.2 < 2.0
        assert manager._calculate_complexity_level(4.0) == ComplexityLevel.LOW  # (10-4)*0.3 = 1.8 < 2.0
        assert manager._calculate_complexity_level(2.0) == ComplexityLevel.MEDIUM  # (10-2)*0.3 = 2.4 < 4.0
        
        # Test complexity from challenges
        challenges = [
            {"severity": "critical", "description": "Critical issue"},
            {"severity": "high", "description": "High severity issue"}
        ]
        complexity = manager._calculate_complexity_level(8.0, challenges)
        assert complexity == ComplexityLevel.HIGH  # 0.6 + 2.0 + 1.5 = 4.1 >= 4.0
        
        # Test complexity from success metrics
        success_metrics = {"data_loss_percentage": "15%"}
        complexity = manager._calculate_complexity_level(8.0, success_metrics=success_metrics)
        assert complexity == ComplexityLevel.LOW  # 0.6 + 1.0 = 1.6 < 2.0
    
    def test_add_relationship(self):
        """Test adding relationship to collection."""
        manager = RelationshipOutputManager()
        
        source_datasets = [
            DatasetMetadata(name="Dataset A"),
            DatasetMetadata(name="Dataset B")
        ]
        metrics = RelationshipMetrics(
            confidence_score=8.0,
            relationship_strength=RelationshipStrength.STRONG,
            complexity_level=ComplexityLevel.MEDIUM
        )
        
        relationship = DatasetRelationship(
            relationship_type=RelationshipType.MERGE,
            source_datasets=source_datasets,
            relationship_metrics=metrics
        )
        
        manager.add_relationship(relationship)
        
        assert len(manager.collection.relationships) == 1
        assert manager.collection.relationships[0] == relationship
    
    def test_get_collection(self):
        """Test getting the collection."""
        manager = RelationshipOutputManager()
        
        collection = manager.get_collection()
        
        assert collection is manager.collection
        assert isinstance(collection, RelationshipCollection)
    
    @patch('builtins.open', create=True)
    def test_export_to_json(self, mock_open):
        """Test exporting collection to JSON file."""
        manager = RelationshipOutputManager()
        
        # Add a relationship
        source_datasets = [
            DatasetMetadata(name="Dataset A"),
            DatasetMetadata(name="Dataset B")
        ]
        metrics = RelationshipMetrics(
            confidence_score=8.0,
            relationship_strength=RelationshipStrength.STRONG,
            complexity_level=ComplexityLevel.MEDIUM
        )
        
        relationship = DatasetRelationship(
            relationship_type=RelationshipType.MERGE,
            source_datasets=source_datasets,
            relationship_metrics=metrics
        )
        
        manager.add_relationship(relationship)
        
        # Mock file write
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Export
        manager.export_to_json("test_output.json")
        
        # Verify file was opened and written
        mock_open.assert_called_once_with("test_output.json", 'w', encoding='utf-8')
        mock_file.write.assert_called_once()
        
        # Verify JSON content
        written_content = mock_file.write.call_args[0][0]
        data = json.loads(written_content)
        assert data['name'] == "Dataset Relationships Collection"
        assert len(data['relationships']) == 1
        assert data['relationships'][0]['relationship_type'] == "merge"
    
    def test_validate_collection(self):
        """Test validating the collection."""
        manager = RelationshipOutputManager()
        
        # Add valid relationship
        source_datasets = [
            DatasetMetadata(name="Dataset A"),
            DatasetMetadata(name="Dataset B")
        ]
        metrics = RelationshipMetrics(
            confidence_score=8.0,
            relationship_strength=RelationshipStrength.STRONG,
            complexity_level=ComplexityLevel.MEDIUM
        )
        
        valid_relationship = DatasetRelationship(
            relationship_type=RelationshipType.MERGE,
            source_datasets=source_datasets,
            relationship_metrics=metrics
        )
        
        # Add invalid relationship
        invalid_relationship = DatasetRelationship(
            relationship_type=RelationshipType.MERGE,
            source_datasets=source_datasets,
            relationship_metrics=None
        )
        
        manager.add_relationship(valid_relationship)
        manager.add_relationship(invalid_relationship)
        
        errors = manager.validate_collection()
        
        assert len(errors) == 1
        assert invalid_relationship.relationship_id in errors
        assert "Relationship metrics are required" in errors[invalid_relationship.relationship_id][0]
    
    def test_get_statistics(self):
        """Test getting collection statistics."""
        manager = RelationshipOutputManager()
        
        # Add relationships
        for i, rel_type in enumerate([RelationshipType.MERGE, RelationshipType.FUSION, RelationshipType.MERGE]):
            source_datasets = [
                DatasetMetadata(name=f"Dataset A {i}"),
                DatasetMetadata(name=f"Dataset B {i}")
            ]
            metrics = RelationshipMetrics(
                confidence_score=8.0 + i,
                relationship_strength=RelationshipStrength.STRONG,
                complexity_level=ComplexityLevel.MEDIUM
            )
            
            relationship = DatasetRelationship(
                relationship_type=rel_type,
                source_datasets=source_datasets,
                relationship_metrics=metrics
            )
            
            manager.add_relationship(relationship)
        
        stats = manager.get_statistics()
        
        assert stats['total_relationships'] == 3
        assert stats['relationship_types']['merge'] == 2
        assert stats['relationship_types']['fusion'] == 1
        assert stats['complexity_distribution']['medium'] == 3
        assert stats['average_confidence'] == 9.0 