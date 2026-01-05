# Academic Publications Metadata Extraction

A service for extracting and processing metadata from OpenAlex API, focusing on dataset usage and related information in scientific publications.

## Overview

This system collects, processes, and indexes academic publications from OpenAlex, with a focus on data normalization and citation analysis. It uses an event-driven architecture with RabbitMQ for asynchronous processing and stores data in MongoDB with Elasticsearch for search capabilities.

## Architecture

```text
┌─────────────┐     ┌─────────────┐     ┌──────────────────────┐
│   FastAPI   │────►│  RabbitMQ   │────►│  Worker Services     │
│     API     │     │   Queues    │     │  - process_dataset   │
└─────────────┘     └─────────────┘     │  - process_publication│
                                        │  - process_institution│
                                        │  - flatten_publication│
                                        └──────────┬───────────┘
                                                   │
                          ┌────────────────────────┼────────────────────────┐
                          ▼                        ▼                        ▼
                    ┌──────────┐            ┌─────────────┐          ┌──────────┐
                    │ MongoDB  │            │ Elasticsearch│          │   ROR    │
                    │ Storage  │            │   Indexing   │          │   API    │
                    └──────────┘            └─────────────┘          └──────────┘
```

## Prerequisites

- Python 3.8 or higher
- Docker and Docker Compose
- Git

## Project Structure

```text
data-extraction/
├── api.py                    # FastAPI REST API
├── process_publication.py    # Publication processing worker
├── process_institution.py    # Institution enrichment worker (ROR API)
├── flatten_publication.py    # Data flattening for Elasticsearch
├── finish_process_notification.py  # Pipeline completion handler
├── docker-compose.yml        # Docker services configuration
├── Dockerfile               # Container build configuration
├── requirements.txt         # Python dependencies
├── .env.sample              # Environment variables template
├── rabbitmq/                # RabbitMQ configuration
│   ├── definitions.json     # Queue and exchange definitions
│   ├── rabbitmq.conf        # RabbitMQ settings
│   └── enabled_plugins      # Enabled plugins list
└── src/                     # Source code modules
    ├── infra/               # Infrastructure (config, queue, db)
    └── use_cases/           # Business logic use cases
```

## Services

| Service | Description |
|---------|-------------|
| **api** | FastAPI REST API for triggering pipelines and checking status |
| **process_dataset** | Searches OpenAlex for publications matching dataset aliases |
| **process_publication** | Processes and stores publication metadata |
| **process_institution** | Enriches institution data with state information from ROR API |
| **flatten_publication** | Transforms hierarchical data into flat format for Elasticsearch |

## Environment Setup

### 1. Clone the Repository

```bash
git clone [repository-url]
cd data-extraction
```

### 2. Environment Configuration

Create the `.env` file using `.env.sample` as a template:

```env
# RabbitMQ Configuration
RABBITMQ_HOST=rabbitmq
RABBITMQ_USER=guest
RABBITMQ_PASS=guest
PUBLICATION_QUEUE=publication_queue
PUBLICATION_EXCHANGE=publication_exchange
PROCESS_PUBLICATION_QUEUE=process_publication_queue
PROCESS_PUBLICATION_EXCHANGE=process_publication_exchange
PUBLICATION_ERROR_QUEUE=publication_error_queue
PUBLICATION_ERROR_EXCHANGE=publication_error_exchange
DATASET_QUEUE=dataset_queue
DATASET_EXCHANGE=dataset_exchange
PIPELINE_START_QUEUE=pipeline_start_queue
PIPELINE_START_EXCHANGE=pipeline_start_exchange
INSTITUTION_QUEUE=institution_queue
INSTITUTION_EXCHANGE=institution_exchange

# OpenAlex Configuration
PYALEX_EMAIL=your_email@example.com

# MongoDB Configuration
MONGODB_CONN=mongodb://mongo:27017/
MONGODB_PUBLICATIONS_TABLE=publications
MONGODB_AUTHORS_TABLE=authors
MONGODB_DATASETS_TABLE=datasets
MONGODB_INSTITUTIONS_TABLE=institutions
MONGODB_TOPICS_TABLE=topics
MONGODB_JOURNALS_TABLE=journals

# Elasticsearch Configuration
ELASTIC_URL=http://elastic:9200
ELASTIC_API_KEY=your_api_key
ELASTIC_CA_CERTIFICATE=

# API Authentication
BASIC_USERNAME=admin
BASIC_PASSWORD=your_secure_password

# Server Configuration
PORT=80

# Dimensions API (optional)
DIMENSIONS_API_KEY=
DIMENSIONS_ENDPOINT=https://app.dimensions.ai
```

## Running the Application

### Using Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Services Endpoints

| Service | URL |
|---------|-----|
| **API Documentation** | http://localhost/docs |
| **RabbitMQ Management** | http://localhost:15672 (guest/guest) |
| **Elasticsearch** | http://localhost:9200 |
| **Kibana** | http://localhost:5601 |
| **MongoDB** | localhost:27017 |

## API Endpoints

### Start Pipeline

```bash
POST /pipelines/start
```

Starts a pipeline to search for publications that use the specified datasets.

**Request Body:**

```json
{
  "group": {
    "name": "USDA Census of Agriculture",
    "datasets": [
      {
        "name": "Census of Agriculture",
        "aliases": ["NASS Census of Agriculture", "Agricultural Census", "USDA Census", "AG Census"]
      }
    ]
  }
}
```

**Response:**

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Check Pipeline Status

```bash
GET /pipelines/{task_id}/status
```

Returns the current status of a pipeline task.

### Download Results

```bash
GET /pipelines/{task_id}/result/download
```

Downloads the results of a completed pipeline.

## Data Schema

The system stores the following entities in MongoDB:

| Collection | Fields | Description |
|------------|--------|-------------|
| **publication** | `_id`, `publication_id`, `publication_external_id`, `title`, `year`, `doi`, `citation_count`, `journal_id`, `open_access_url` | Publication details |
| **journal** | `_id`, `journal_id`, `journal_external_id`, `name`, `issn` | Journal information |
| **author** | `_id`, `author_id`, `author_external_id`, `name`, `orcid` | Author information |
| **institution** | `_id`, `institution_id`, `institution_external_id`, `name`, `state`, `country`, `ror` | Institution details |
| **topic** | `_id`, `topic_id`, `topic_external_id`, `name`, `type` | Topic/subject classification |
| **dataset** | `_id`, `dataset_id`, `name` | Dataset information |
| **dataset_alias** | `_id`, `alias_id`, `dataset_id`, `name` | Dataset aliases for search |

## Data Flow

1. **Pipeline Start** - API receives dataset group with aliases
2. **Dataset Search** - OpenAlex API is queried using full-text search with aliases
3. **Publication Processing** - Publications are extracted and stored in MongoDB
4. **Institution Enrichment** - ROR API adds state/region information
5. **Data Flattening** - Hierarchical data is denormalized for Elasticsearch
6. **Indexing** - Flattened data is indexed in Elasticsearch for search

## Technical Details

### Message Queues

The system uses RabbitMQ with the following queues:

- `pipeline_start_queue` - Initial pipeline requests
- `dataset_queue` - Dataset search tasks
- `process_publication_queue` - Publication processing tasks
- `institution_queue` - Institution enrichment tasks
- `publication_queue` - Flattening tasks
- `publication_error_queue` - Failed message handling

### External APIs

- **OpenAlex API** - Primary source for publication metadata
- **ROR API** - Research Organization Registry for institution enrichment
- **Dimensions API** - Alternative data source (optional)

## Development

### Local Development Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Unix/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run API locally
python api.py
```

### Running Individual Workers

```bash
# Process publications
python process_publication.py

# Process institutions
python process_institution.py

# Flatten publications
python flatten_publication.py
```

## License

This project is licensed under the MIT License.
