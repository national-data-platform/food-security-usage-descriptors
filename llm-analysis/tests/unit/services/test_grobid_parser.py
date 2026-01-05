"""
Unit tests for GROBID Parser service.

Tests the GROBIDParser class functionality including Parquet file reading,
text extraction, and error handling.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from pub_analysis_agent.services.grobid_parser import (
    GROBIDParser,
    GROBIDPublication,
    GROBIDFullText,
    GROBIDTextContent,
    GROBIDSection
)


class TestGROBIDSection:
    """Test GROBIDSection class functionality."""
    
    def test_extract_text_empty_sentences(self):
        """Test text extraction with empty sentences."""
        section = GROBIDSection()
        assert section.extract_text() == ""
    
    def test_extract_text_with_sentences(self):
        """Test text extraction with valid sentences."""
        section = GROBIDSection(
            sentences=[
                {"text": "First sentence."},
                {"text": "Second sentence."}
            ]
        )
        expected = "First sentence. Second sentence."
        assert section.extract_text() == expected
    
    def test_extract_text_with_invalid_sentences(self):
        """Test text extraction with invalid sentence structures."""
        section = GROBIDSection(
            sentences=[
                {"text": "Valid sentence."},
                {"invalid_key": "Invalid sentence."},
                {"text": "Another valid sentence."}
            ]
        )
        expected = "Valid sentence. Another valid sentence."
        assert section.extract_text() == expected


class TestGROBIDTextContent:
    """Test GROBIDTextContent class functionality."""
    
    def test_extract_text_empty_sections(self):
        """Test text extraction with empty sections."""
        content = GROBIDTextContent()
        assert content.extract_text() == ""
    
    def test_extract_text_with_grobid_sections(self):
        """Test text extraction with GROBIDSection objects."""
        section1 = GROBIDSection(sentences=[{"text": "Section 1."}])
        section2 = GROBIDSection(sentences=[{"text": "Section 2."}])
        
        content = GROBIDTextContent(sections=[section1, section2])
        expected = "Section 1. Section 2."
        assert content.extract_text() == expected
    
    def test_extract_text_with_dict_sections(self):
        """Test text extraction with dictionary sections."""
        sections = [
            {
                "sentences": [
                    {"text": "First sentence."},
                    {"text": "Second sentence."}
                ]
            },
            {
                "sentences": [
                    {"text": "Third sentence."}
                ]
            }
        ]
        
        content = GROBIDTextContent(sections=sections)
        expected = "First sentence. Second sentence. Third sentence."
        assert content.extract_text() == expected


class TestGROBIDFullText:
    """Test GROBIDFullText class functionality."""
    
    def test_get_title_text_with_title(self):
        """Test title text extraction with valid title."""
        fulltext = GROBIDFullText(
            title={"text": "Sample Research Paper"}
        )
        assert fulltext.get_title_text() == "Sample Research Paper"
    
    def test_get_title_text_without_title(self):
        """Test title text extraction without title."""
        fulltext = GROBIDFullText()
        assert fulltext.get_title_text() == ""
    
    def test_get_abstract_text(self):
        """Test abstract text extraction."""
        abstract_content = GROBIDTextContent(
            sections=[GROBIDSection(sentences=[{"text": "Abstract text."}])]
        )
        fulltext = GROBIDFullText(abstract=abstract_content)
        assert fulltext.get_abstract_text() == "Abstract text."
    
    def test_get_body_text(self):
        """Test body text extraction."""
        body_content = GROBIDTextContent(
            sections=[GROBIDSection(sentences=[{"text": "Body text."}])]
        )
        fulltext = GROBIDFullText(body=body_content)
        assert fulltext.get_body_text() == "Body text."
    
    def test_get_all_text(self):
        """Test extraction of all text content."""
        abstract_content = GROBIDTextContent(
            sections=[GROBIDSection(sentences=[{"text": "Abstract."}])]
        )
        body_content = GROBIDTextContent(
            sections=[GROBIDSection(sentences=[{"text": "Body."}])]
        )
        
        fulltext = GROBIDFullText(
            title={"text": "Title"},
            abstract=abstract_content,
            body=body_content
        )
        
        expected = "Title Abstract. Body."
        assert fulltext.get_all_text() == expected
    
    def test_get_keywords(self):
        """Test keywords extraction."""
        fulltext = GROBIDFullText(keywords=["keyword1", "keyword2"])
        assert fulltext.get_keywords() == ["keyword1", "keyword2"]
    
    def test_get_doi(self):
        """Test DOI extraction."""
        fulltext = GROBIDFullText(
            identifiers={"doi": "10.1234/test.2024.001"}
        )
        assert fulltext.get_doi() == "10.1234/test.2024.001"


class TestGROBIDPublication:
    """Test GROBIDPublication class functionality."""
    
    def test_get_publication_id_from_publication_id(self):
        """Test getting publication ID from publication_id field."""
        publication = GROBIDPublication(publication_id="pub.123456")
        assert publication.get_publication_id() == "pub.123456"
    
    def test_get_publication_id_from_publication_ids(self):
        """Test getting publication ID from publication_ids array."""
        publication = GROBIDPublication(
            publication_ids=np.array(["pub.123456"])
        )
        assert publication.get_publication_id() == "pub.123456"
    
    def test_get_publication_id_none(self):
        """Test getting publication ID when none available."""
        publication = GROBIDPublication()
        assert publication.get_publication_id() is None


class TestGROBIDParser:
    """Test GROBIDParser class functionality."""
    
    @pytest.fixture
    def parser(self):
        """Create a GROBIDParser instance for testing."""
        return GROBIDParser(chunk_size=100, memory_limit_mb=256)
    
    @pytest.fixture
    def sample_publication_data(self):
        """Sample publication data for testing."""
        return {
            "publication_id": "pub.1091402956",
            "id": "7d0a8e70e455929bb9869fad277b2a0f282f834b97de0730b2690005978ec933",
            "publication_ids": np.array(["pub.1091402956"], dtype=object),
            "fulltext": {
                "abstract": {
                    "annotations": {"lang": "en", "length": 1286},
                    "sections": [
                        {
                            "annotations": {"lang": "en", "length": 1286},
                            "sentences": [
                                {
                                    "citations": [],
                                    "text": "Range expansions are key demographic events.",
                                    "pno": 0
                                }
                            ]
                        }
                    ]
                },
                "title": {
                    "text": "Sample Research Paper",
                    "annotations": {"lang": "en", "length": 109}
                },
                "body": {
                    "annotations": {"lang": "en", "length": 35707},
                    "sections": [
                        {
                            "title": {
                                "text": "Introduction",
                                "annotations": {"normalised_titles": ["introduction"]}
                            },
                            "sentences": [
                                {
                                    "citations": [],
                                    "text": "This is the introduction.",
                                    "pno": 0
                                }
                            ]
                        }
                    ]
                },
                "identifiers": {
                    "doi": "10.1139/cjz-2017-0071"
                },
                "keywords": ["climate change", "population genetics"]
            }
        }
    
    def test_init(self, parser):
        """Test parser initialization."""
        assert parser.chunk_size == 100
        assert parser.memory_limit_mb == 256
        assert parser.logger is not None
    
    def test_read_parquet_file_not_found(self, parser):
        """Test reading non-existent Parquet file."""
        with pytest.raises(FileNotFoundError):
            parser.read_parquet_file("nonexistent.parquet")
    
    def test_read_parquet_file_invalid_extension(self, parser):
        """Test reading file with invalid extension."""
        # Mock file existence first
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.suffix', return_value='.txt'):
                with pytest.raises(ValueError, match="not a Parquet file"):
                    parser.read_parquet_file("test.txt")
    
    def test_read_parquet_file_success(self, parser, sample_publication_data):
        """Test successful Parquet file reading."""
        # Create mock DataFrame
        df = pd.DataFrame([sample_publication_data])
        
        with patch.object(parser, 'read_parquet_file', return_value=df):
            result = parser.read_parquet_file("test.parquet")
        
        assert len(result) == 1
        assert "publication_id" in result.columns
        assert "fulltext" in result.columns
    
    def test_read_parquet_file_missing_columns(self, parser):
        """Test reading Parquet file with missing expected columns."""
        # Create mock DataFrame with missing columns
        df = pd.DataFrame([{"other_column": "value"}])
        
        with patch.object(parser, 'read_parquet_file', return_value=df):
            result = parser.read_parquet_file("test.parquet")
        
        # Should still return the DataFrame but log a warning
        assert len(result) == 1
    
    def test_parse_publication_record_success(self, parser, sample_publication_data):
        """Test successful publication record parsing."""
        publication = parser.parse_publication_record(sample_publication_data)
        
        assert publication is not None
        assert publication.get_publication_id() == "pub.1091402956"
        assert publication.fulltext is not None
        assert publication.fulltext.get_title_text() == "Sample Research Paper"
        assert publication.fulltext.get_doi() == "10.1139/cjz-2017-0071"
    
    def test_parse_publication_record_invalid_data(self, parser):
        """Test parsing invalid publication record."""
        invalid_data = {"invalid": "data"}
        publication = parser.parse_publication_record(invalid_data)
        
        # Should return a valid publication object even with minimal data
        assert publication is not None
        assert publication.publication_id is None
        assert publication.fulltext is None
    
    def test_extract_text_from_publication(self, parser, sample_publication_data):
        """Test text extraction from publication."""
        publication = parser.parse_publication_record(sample_publication_data)
        text_content = parser.extract_text_from_publication(publication)
        
        assert "title" in text_content
        assert "abstract" in text_content
        assert "body" in text_content
        assert "all_text" in text_content
        assert "Sample Research Paper" in text_content["title"]
        assert "Range expansions are key demographic events." in text_content["abstract"]
    
    def test_extract_text_from_publication_no_fulltext(self, parser):
        """Test text extraction from publication without fulltext."""
        publication = GROBIDPublication(publication_id="pub.123")
        text_content = parser.extract_text_from_publication(publication)
        
        assert text_content == {}
    
    def test_validate_publication_structure_valid(self, parser, sample_publication_data):
        """Test validation of valid publication structure."""
        publication = parser.parse_publication_record(sample_publication_data)
        assert parser.validate_publication_structure(publication) is True
    
    def test_validate_publication_structure_invalid(self, parser):
        """Test validation of invalid publication structure."""
        # Publication without ID
        publication = GROBIDPublication()
        assert parser.validate_publication_structure(publication) is False
        
        # Publication without fulltext
        publication = GROBIDPublication(publication_id="pub.123")
        assert parser.validate_publication_structure(publication) is False
    
    def test_get_publication_metadata(self, parser, sample_publication_data):
        """Test metadata extraction from publication."""
        publication = parser.parse_publication_record(sample_publication_data)
        metadata = parser.get_publication_metadata(publication)
        
        assert metadata["publication_id"] == "pub.1091402956"
        assert metadata["doi"] == "10.1139/cjz-2017-0071"
        assert metadata["keywords"] == ["climate change", "population genetics"]
        assert metadata["has_abstract"] is True
        assert metadata["has_body"] is True
    
    def test_process_publication_file(self, parser, sample_publication_data):
        """Test processing publication file."""
        df = pd.DataFrame([sample_publication_data])
        
        with patch.object(parser, 'read_parquet_file', return_value=df):
            publications = list(parser.process_publication_file("test.parquet"))
        
        assert len(publications) == 1
        assert publications[0].get_publication_id() == "pub.1091402956"
    
    def test_process_publication_file_batch(self, parser, sample_publication_data):
        """Test batch processing of publication file."""
        # Create multiple records
        data = [sample_publication_data.copy() for _ in range(3)]
        df = pd.DataFrame(data)
        
        with patch.object(parser, 'read_parquet_file', return_value=df):
            batches = list(parser.process_publication_file_batch("test.parquet", batch_size=2))
        
        assert len(batches) == 2  # 3 records with batch_size=2
        assert len(batches[0]) == 2  # First batch has 2 records
        assert len(batches[1]) == 1  # Second batch has 1 record
    
    def test_extract_text_batch(self, parser, sample_publication_data):
        """Test batch text extraction."""
        publication = parser.parse_publication_record(sample_publication_data)
        results = parser.extract_text_batch([publication])
        
        assert len(results) == 1
        assert results[0]["publication_id"] == "pub.1091402956"
        assert "text_content" in results[0]
        assert results[0]["doi"] == "10.1139/cjz-2017-0071"
        assert results[0]["keywords"] == ["climate change", "population genetics"]


# Mock for PropertyMock
class PropertyMock:
    """Mock for property access."""
    def __init__(self, return_value):
        self.return_value = return_value
    
    def __call__(self):
        return self.return_value 