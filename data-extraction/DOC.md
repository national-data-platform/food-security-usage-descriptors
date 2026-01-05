

# Technical Documentation - Academic Publications Processing System

## Overview
System for collecting, processing, and indexing academic publications from different sources (OpenAlex and OpenAIRE), focusing on data normalization and citation analysis.

## Architecture

### Directory Structure
```
src/
├── infra/               # Infrastructure and configurations
├── use_cases/          # Application use cases
└── __init__.py
```

## Core Components

### 1. Configuration (`infra/configuration.py`)
Manages system configurations via environment variables:
- MongoDB connections
- OCI Queues
- API credentials (OpenAlex, OpenAIRE)
- Elasticsearch settings

### 2. Queues (`infra/queue/`)
Implements interface with Oracle Cloud Infrastructure (OCI) Queues for asynchronous processing:
- Message publishing
- Message consumption
- Receipt management

### 3. Use Cases

#### 3.1 FlattenPublication
```python
class FlattenPublicationUseCase:
    """
    Normalizes publications into a flat format suitable for indexing.
    
    Flow:
    1. Receives publication from MongoDB
    2. Processes authors, institutions, journals, and topics
    3. Calculates citation metrics
    4. Indexes in Elasticsearch
    """
```

#### 3.2 ProcessDatasets
```python
class ProcessDatasetsUseCase:
    """
    Processes datasets from different origins.
    
    Features:
    - Collects data from OpenAlex/OpenAIRE
    - Normalizes format
    - Saves to MongoDB
    - Queues for further processing
    """
```

#### 3.3 SearchDatasets
```python
class SearchDatasetsUseCase:
    """
    Manages search and registration of new datasets.
    
    Attributes:
    - origin: Enum[openalex, openaire]
    - dataset: str
    - flag_terms: List[str]
    - aliases: List[str]
    """
```

## Data Models

### Publication
```python
class Publication:
    """
    Core model representing an academic publication.
    
    Main attributes:
    - id: str
    - name: str
    - doi: str
    - citation_count: int
    - year: int
    - authors: List[Author]
    - institutions: List[Institution]
    - journals: List[Journal]
    - topics: List[Topic]
    """
```

### FlatPublication
```python
class FlatPublication:
    """
    Denormalized version of publication for indexing.
    
    Characteristics:
    - Denormalized data
    - Aggregated metrics
    - Geolocation
    - Citation counts
    """
```

## Data Flow

1. **Dataset Input**
   - Through SearchDatasetsUseCase
   - Validation and MongoDB registration
   - Queuing for processing

2. **Processing**
   - ProcessDatasetsUseCase consumes queue
   - Collects API data
   - Normalizes and saves to MongoDB
   - Queues for flattening

3. **Flatten and Indexing**
   - FlattenPublicationUseCase processes
   - Aggregates metrics
   - Denormalizes data
   - Indexes in Elasticsearch

## Integrations

- **MongoDB**: Main storage
- **Elasticsearch**: Indexing and search
- **OpenAlex API**: Academic data source
- **OpenAIRE API**: Alternative data source
- **OCI Queues**: Asynchronous processing
- **GeoNames API**: Geographic data

## Technical Considerations

1. **Performance**
   - Asynchronous processing via queues
   - Elasticsearch bulk operations
   - Optimized MongoDB aggregations

2. **Resilience**
   - Retry policies for external APIs
   - Layered error handling
   - Structured logging

3. **Scalability**
   - Event-based architecture
   - Distributed processing via queues
   - Clear separation of concerns

4. **Maintainability**
   - Modular code
   - Consistent patterns
   - Strong typing with Pydantic
   - Centralized configuration

## Main Dependencies

- `pymongo`: MongoDB interface
- `elasticsearch`: Elasticsearch client
- `pyalex`: OpenAlex client
- `requests`: HTTP calls
- `pydantic`: Data validation
- `python-dotenv`: Configuration management
- `oci`: Oracle Cloud SDK

## Environment Setup

```env
MONGODB_CONN=mongodb://...
MONGODB_CONN_DATABASE=your_database
ELASTIC_URL=https://...
ELASTIC_API_KEY=your_key
PYALEX_EMAIL=your_email
PUBLICATION_QUEUE_ENDPOINT=your_endpoint
DATASET_QUEUE_ENDPOINT=your_endpoint
```

## Usage Examples

### Adding a New Dataset
```python
dataset = DatasetDTO(
    origin=OpenAlexEnum.openalex,
    dataset="research_dataset",
    aliases=["research", "study"],
    flag_terms=["term1", "term2"]
)
use_case = SearchDatasetsUseCase()
use_case.execute(dataset)
```

### Processing Publications
```python
use_case = ProcessDatasetsUseCase()
use_case.execute(json.dumps({
    "origin": "openalex",
    "dataset": "research_dataset",
    "alias": "research"
}))
```

## Error Handling

The system implements various error handling strategies:
- API request retries
- Queue message visibility timeout
- Database connection pooling
- Exception logging and monitoring

## Monitoring and Logging

- Structured logging for all operations
- Queue processing metrics
- API call monitoring
- Database performance tracking

## Security

- API key management
- Secure configuration handling
- MongoDB authentication
- Elasticsearch security features

## Future Improvements

1. **Performance Optimizations**
   - Caching layer implementation
   - Batch processing optimization
   - Index optimization

2. **Feature Additions**
   - Additional data sources
   - Enhanced metrics
   - Advanced search capabilities

3. **Infrastructure**
   - Container orchestration
   - Automated scaling
   - Enhanced monitoring

4. **Code Quality**
   - Increased test coverage
   - Documentation improvements
   - Code style standardization
