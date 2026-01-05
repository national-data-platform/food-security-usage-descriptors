"""
Pytest configuration and fixtures for pub-analysis-agent tests.
"""

import pytest
import asyncio
from typing import Generator, AsyncGenerator
from unittest.mock import Mock, AsyncMock
import pandas as pd
import numpy as np

# Configure asyncio for async tests
@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Real publication data fixtures based on actual parquet structure
@pytest.fixture
def sample_publication_data() -> dict:
    """Return realistic publication data based on actual parquet file structure."""
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
                                "text": "Range expansions are key demographic events driven by factors such as climate change and human intervention that ultimately influence the genetic composition of peripheral populations.",
                                "pno": 0
                            },
                            {
                                "citations": [],
                                "text": "The expansion of the Virginia opossum (Didelphis virginiana Kerr, 1792) into Michigan has been documented over the past 200 years, indicating relatively new colonizations in northern Michigan.",
                                "pno": 0
                            }
                        ]
                    }
                ]
            },
            "title": {
                "text": "Contemporary range expansion of the Virginia opossum (Didelphis virginiana) impacted by humans and snow cover",
                "annotations": {"lang": "en", "length": 109}
            },
            "authorship": {
                "authors": [
                    {
                        "forename": "Lisa",
                        "middlename": "L", 
                        "surname": "Walsh",
                        "email": "llwalsh@umich.edu",
                        "affiliations": [],
                        "author_id": None
                    },
                    {
                        "forename": "P",
                        "middlename": "K",
                        "surname": "Tucker", 
                        "email": None,
                        "affiliations": [],
                        "author_id": None
                    }
                ]
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
                                "citations": [
                                    {"pos": 161, "refid": "b17", "refno": 17, "text": "(Excoffier et al. 2009)", "type": "bib"}
                                ],
                                "text": "Dynamic range margins can strongly impact the natural history and genetics of a species, making range expansions important systems in both ecology and evolution (Excoffier et al. 2009).",
                                "pno": 0
                            }
                        ]
                    }
                ]
            },
            "bibliography": {
                "references": [
                    {
                        "id": "b0",
                        "title": "Michigan mammals",
                        "authors": [{"forename": "R", "middlename": "H", "surname": "Baker"}],
                        "imprint_date": {"type": "published", "when": "1983"},
                        "imprint_publisher": "Michigan State University Press"
                    }
                ]
            },
            "identifiers": {
                "doi": "10.1139/cjz-2017-0071",
                "md5": "4076C7F5F3A43925A215D9D548146D6F"
            },
            "keywords": [
                "agriculture", "climate change", "Didelphis virginiana",
                "land-use change", "Virginia opossum", "population genetics", "range expansion"
            ]
        },
        "processing": [
            {
                "error": None,
                "processor": "GROBID", 
                "processor_version": "0.8.1-SNAPSHOT",
                "timestamp": 1716841254.531689
            }
        ],
        "file": [
            {
                "source_hash": "7d0a8e70e455929bb9869fad277b2a0f282f834b97de0730b2690005978ec933",
                "source_size": 866985,
                "source_type": "PDF"
            }
        ],
        "gbq_processing": [
            {
                "date_imported": "2024-10-20T19:53:07+00:00",
                "import_series_id": "0000000001-0000001674-00004515-00000065"
            }
        ]
    }


@pytest.fixture
def sample_dataset_mention() -> dict:
    """Return sample publication that mentions datasets."""
    return {
        "publication_id": "pub.test_dataset",
        "fulltext": {
            "title": {"text": "Analysis of MNIST Dataset Performance in Deep Learning"},
            "abstract": {
                "sections": [{
                    "sentences": [{
                        "text": "We evaluated our model on the MNIST dataset and achieved 95% accuracy. The CIFAR-10 dataset was also used for comparison."
                    }]
                }]
            },
            "body": {
                "sections": [{
                    "sentences": [
                        {
                            "text": "The MNIST dataset contains 70,000 handwritten digits.",
                            "citations": []
                        },
                        {
                            "text": "We downloaded the data from the official repository.",
                            "citations": []
                        }
                    ]
                }]
            },
            "keywords": ["MNIST", "CIFAR-10", "deep learning", "computer vision"]
        }
    }


@pytest.fixture
def sample_data_analysis_paper() -> dict:
    """Return sample publication that performs actual data analysis."""
    return {
        "publication_id": "pub.test_analysis", 
        "fulltext": {
            "title": {"text": "Statistical Analysis of Climate Data Trends"},
            "abstract": {
                "sections": [{
                    "sentences": [{
                        "text": "We conducted a comprehensive statistical analysis of temperature data from 1980-2020. Our analysis reveals significant warming trends across multiple regions."
                    }]
                }]
            },
            "body": {
                "sections": [
                    {
                        "title": {"text": "Methods"},
                        "sentences": [
                            {
                                "text": "We applied linear regression analysis to the temperature time series.",
                                "citations": []
                            },
                            {
                                "text": "Statistical significance was tested using ANOVA with p < 0.05.",
                                "citations": []
                            }
                        ]
                    },
                    {
                        "title": {"text": "Results"},
                        "sentences": [
                            {
                                "text": "The analysis showed a statistically significant increasing trend (p < 0.001).",
                                "citations": []
                            }
                        ]
                    }
                ]
            },
            "keywords": ["statistical analysis", "climate data", "trend analysis", "regression"]
        }
    }


@pytest.fixture 
def sample_parquet_dataframe(sample_publication_data) -> pd.DataFrame:
    """Return a sample DataFrame matching the parquet file structure."""
    return pd.DataFrame([sample_publication_data])


@pytest.fixture
def mock_mongodb_datasets() -> list:
    """Return mock MongoDB dataset records with realistic dataset names and aliases."""
    return [
        {
            "_id": "mnist",
            "name": "MNIST",
            "aliases": ["mnist", "MNIST dataset", "Modified National Institute of Standards and Technology"],
            "description": "Database of handwritten digits",
            "type": "image_classification"
        },
        {
            "_id": "cifar10", 
            "name": "CIFAR-10",
            "aliases": ["cifar10", "CIFAR-10 dataset", "Canadian Institute for Advanced Research"],
            "description": "Object recognition dataset",
            "type": "image_classification"
        },
        {
            "_id": "imagenet",
            "name": "ImageNet", 
            "aliases": ["imagenet", "ImageNet dataset", "ILSVRC"],
            "description": "Large scale visual recognition challenge dataset",
            "type": "image_classification"
        }
    ]


# Test data fixtures
@pytest.fixture
def sample_publication_id() -> str:
    """Return a sample publication ID for testing."""
    return "pub.1091402956"


@pytest.fixture
def sample_grobid_content() -> dict:
    """Return sample GROBID parsed content for testing."""
    return {
        "title": "Sample Research Paper",
        "abstract": "This paper analyzes the MNIST dataset using machine learning techniques.",
        "body": "We trained a neural network on the MNIST dataset and achieved 95% accuracy.",
        "acknowledgement": "We thank the contributors of the MNIST dataset.",
        "availability": "Code is available at https://github.com/example/mnist-analysis",
    }


@pytest.fixture
def mock_llm_service() -> Mock:
    """Return a mock LLM service for testing."""
    mock = AsyncMock()
    mock.generate_response.return_value = {
        "response": "Sample LLM response",
        "confidence": 0.85,
        "tokens_used": 150,
        "datasets_mentioned": ["MNIST", "CIFAR-10"],
        "analysis_type": "classification"
    }
    return mock


@pytest.fixture
def mock_mongodb_service(mock_mongodb_datasets) -> Mock:
    """Return a mock MongoDB service for testing."""
    mock = Mock()
    mock.get_datasets.return_value = mock_mongodb_datasets
    mock.find_dataset_by_name.return_value = mock_mongodb_datasets[0]
    mock.insert_analysis_result.return_value = {"inserted_id": "test_analysis_123"}
    return mock


@pytest.fixture
def mock_elasticsearch_service() -> Mock:
    """Return a mock Elasticsearch service for testing."""
    mock = AsyncMock()
    mock.index_document.return_value = {"indexed": True, "id": "test_doc_123"}
    mock.search_publications.return_value = {
        "hits": {
            "total": {"value": 1},
            "hits": [{"_source": {"publication_id": "pub.test", "title": "Test Paper"}}]
        }
    }
    return mock


# Configuration fixtures
@pytest.fixture
def test_config() -> dict:
    """Return test configuration settings."""
    return {
        "mongodb": {
            "connection_string": "mongodb://localhost:27017/test_db",
            "database": "test_dimensions",
            "datasets_collection": "general.datasets",
            "results_collection": "dimensions.llm_analyses"
        },
        "llm": {
            "ollama_endpoint": "http://localhost:11434/api",
            "lmstudio_endpoint": "http://localhost:1234/v1/",
            "model_name": "gpt-oss:120b",
            "temperature": 0.1,
            "max_tokens": 2048,
        },
        "elasticsearch": {
            "endpoint": "http://localhost:9200",
            "index_name": "test_publications",
        },
        "processing": {
            "batch_size": 10,
            "max_retries": 3,
            "timeout_seconds": 30,
        },
    }


# LangGraph workflow fixtures
@pytest.fixture
def mock_workflow_state():
    """Return a mock LangGraph workflow state for testing."""
    return {
        "publication_id": "pub.test",
        "processing_stage": "dataset_extraction",
        "datasets_found": ["MNIST", "CIFAR-10"],
        "analysis_results": {
            "is_data_analysis": True,
            "confidence": 0.9,
            "evidence": ["statistical analysis", "regression analysis"]
        },
        "errors": [],
        "metadata": {
            "start_time": "2024-01-01T00:00:00Z",
            "agent_version": "1.0.0"
        }
    }


# Integration test marks
def pytest_configure(config):
    """Configure pytest marks."""
    config.addinivalue_line(
        "markers", 
        "integration: mark test as integration test requiring external services"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow-running test"
    )
    config.addinivalue_line(
        "markers",
        "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers",
        "requires_parquet: mark test as requiring parquet file access"
    )
    config.addinivalue_line(
        "markers", 
        "requires_llm: mark test as requiring LLM service"
    ) 