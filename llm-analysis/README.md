# Publication Analysis Agent

An AI-powered pipeline for analyzing scientific publications to extract structured information about dataset usage, discover new datasets, and analyze how datasets are combined in research.

## Overview

This system uses LangGraph to orchestrate a multi-agent workflow that processes full-text scientific publications. Each agent performs a specific analysis task, and the results are aggregated into a structured output stored in MongoDB and indexed in Elasticsearch.

## Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          WorkflowOrchestrator                               │
│                            (LangGraph)                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐ │
│  │   Triage     │──►│  Validation  │──►│  Discovery   │──►│  Join        │ │
│  │   Agent      │   │  Agent       │   │  Agent       │   │  Analysis    │ │
│  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘ │
│         │                                                        │          │
│         ▼                                                        ▼          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                    │
│  │   Code       │──►│   GitHub     │──►│   JSON       │                    │
│  │   Extraction │   │   Verify     │   │   Assembly   │                    │
│  └──────────────┘   └──────────────┘   └──────────────┘                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              ┌──────────┐   ┌─────────────┐  ┌──────────┐
              │ MongoDB  │   │Elasticsearch│  │   LLM    │
              │ Storage  │   │   Index     │  │ (Ollama) │
              └──────────┘   └─────────────┘  └──────────┘
```

## Workflow Steps

| Step | Agent | Description |
|------|-------|-------------|
| 1 | **TriageAgent** | Classifies if the publication is a data analysis paper |
| 2 | **DatasetValidationAgent** | Validates known datasets mentioned in the text |
| 3 | **DatasetDiscoveryAgent** | Discovers new/unknown datasets in the publication |
| 4 | **DatasetJoinAnalysisAgent** | Analyzes how datasets are combined/joined |
| 5 | **CodeExtractionAgent** | Extracts code snippets and external links |
| 6 | **GitHubRepositoryVerificationAgent** | Verifies GitHub repository links |
| 7 | **JSONAssemblyAgent** | Assembles final structured output |

## Core Technologies

| Component | Technology |
|-----------|------------|
| **Orchestration** | LangGraph |
| **LLM Serving** | Ollama, LM Studio, OpenAI, Anthropic |
| **Language** | Python 3.12+ |
| **Database** | MongoDB |
| **Search** | Elasticsearch |
| **Validation** | Pydantic |
| **Testing** | pytest |

## Project Structure

```text
llm-analysis/
├── src/pub_analysis_agent/
│   ├── agents/                    # LLM-powered analysis agents
│   │   ├── triage_agent.py        # Data analysis classification
│   │   ├── dataset_validation_agent.py    # Known dataset validation
│   │   ├── dataset_discovery_agent.py     # New dataset discovery
│   │   ├── dataset_join_agent.py          # Dataset join analysis
│   │   ├── code_extraction_agent.py       # Code/link extraction
│   │   ├── github_repo_verification_agent.py  # GitHub verification
│   │   └── json_assembly_agent.py         # Output assembly
│   ├── config/                    # Configuration management
│   │   ├── settings.py            # Pydantic settings
│   │   ├── logging_config.py      # Logging setup
│   │   └── environment.py         # Environment management
│   ├── models/                    # Data models
│   │   ├── analysis_result.py     # Analysis result schema
│   │   ├── dataset.py             # Dataset models
│   │   └── schema_validator.py    # JSON schema validation
│   ├── services/                  # Business logic services
│   │   ├── mongodb_client.py      # MongoDB operations
│   │   ├── dataset_service.py     # Dataset management
│   │   ├── results_service.py     # Analysis results storage
│   │   ├── llm_service.py         # LLM interactions
│   │   ├── grobid_parser.py       # GROBID XML parsing
│   │   ├── elasticsearch_sync_service.py  # ES synchronization
│   │   └── data_quality_validator.py      # Data quality checks
│   ├── workflows/                 # LangGraph workflows
│   │   ├── workflow_orchestrator.py   # Main orchestrator
│   │   ├── state_models.py            # Workflow state definitions
│   │   └── state_converters.py        # State conversion utilities
│   └── utils/                     # Utility functions
├── tests/
│   ├── unit/                      # Unit tests
│   │   ├── agents/                # Agent tests
│   │   ├── services/              # Service tests
│   │   └── workflows/             # Workflow tests
│   └── integration/               # Integration tests
├── config/                        # Configuration files
├── requirements.txt               # Production dependencies
├── requirements-dev.txt           # Development dependencies
└── setup.py                       # Package setup
```

## Prerequisites

- Python 3.12+
- MongoDB
- Elasticsearch (optional)
- Ollama or LM Studio (for local LLMs)

## Installation

### Quick Setup

```bash
# Clone and enter directory
git clone <repository-url>
cd llm-analysis

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Environment Configuration

Copy `.env.example` to `.env` and configure:

```env
# MongoDB
MONGODB_CONNECTION_STRING=mongodb://localhost:27017/
MONGODB_DATABASE=publication_analysis

# Elasticsearch (optional)
ELASTICSEARCH_URL=https://localhost:9200
ELASTICSEARCH_USERNAME=elastic
ELASTICSEARCH_PASSWORD=

# LLM Configuration
OLLAMA_ENDPOINT=http://localhost:11434/api
LMSTUDIO_ENDPOINT=http://localhost:1234/v1/

# Optional: Cloud LLM APIs
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# Application
ENVIRONMENT=development
LOG_LEVEL=INFO
```

## Usage

### Programmatic Usage

```python
from pub_analysis_agent import setup_development_environment
from pub_analysis_agent.workflows import WorkflowOrchestrator
from pub_analysis_agent.agents import (
    triage_agent_step,
    dataset_validation_agent_step,
    dataset_discovery_agent_step,
    dataset_join_analysis_agent_step,
    code_extraction_agent_step
)

# Initialize environment
settings = setup_development_environment()

# Create orchestrator with agents
orchestrator = WorkflowOrchestrator()
orchestrator.register_agent("classify_data_analysis", triage_agent_step)
orchestrator.register_agent("validate_datasets", dataset_validation_agent_step)
orchestrator.register_agent("discover_new_datasets", dataset_discovery_agent_step)
orchestrator.register_agent("analyze_dataset_joins", dataset_join_analysis_agent_step)
orchestrator.register_agent("extract_code_snippets", code_extraction_agent_step)

# Execute workflow
result = await orchestrator.execute_workflow(
    publication_id="pub_123",
    initial_data={"fulltext": publication_content}
)

# Access results
print(f"Is data analysis: {result.is_data_analysis}")
print(f"Validated datasets: {len(result.validated_datasets)}")
print(f"Discovered datasets: {len(result.newly_discovered_datasets)}")
print(f"Dataset joins: {len(result.dataset_joins)}")
```

### CLI Usage

```bash
# Validate configuration
python -m pub_analysis_agent.config.cli validate

# Test connections
python -m pub_analysis_agent.config.cli test-connections

# Analyze publication
python -m pub_analysis_agent.config.cli analyze-publication -p /path/to/file.parquet
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pub_analysis_agent --cov-report=html

# Run specific test categories
pytest tests/unit/agents/ -v
pytest tests/unit/services/ -v
pytest tests/unit/workflows/ -v

# Run with markers
pytest -m "not slow"
```

## Services

### MongoDBClient

Handles all MongoDB operations with connection pooling and retry logic.

### DatasetService

Manages known datasets, including CRUD operations and similarity matching.

### ResultsService

Stores and retrieves analysis results with transaction support.

### LLMService

Provides unified interface for different LLM providers (Ollama, LM Studio, OpenAI, Anthropic).

### ElasticsearchSyncService

Synchronizes analysis results to Elasticsearch for search and aggregation.

### GROBIDParser

Parses GROBID XML output to extract structured publication content.

## Data Models

### AnalysisState

Main workflow state containing:

- `publication_id` - Unique publication identifier
- `is_data_analysis` - Classification result
- `validated_datasets` - List of validated known datasets
- `newly_discovered_datasets` - List of discovered new datasets
- `dataset_joins` - List of dataset join analyses
- `extracted_code_snippets` - Code snippets found
- `extracted_github_repos` - GitHub repositories found
- `final_json` - Assembled output

### DatasetMention

Represents a dataset mention with:

- `name` - Dataset name
- `confidence_score_mention` - Confidence of mention detection
- `confidence_score_use` - Confidence of actual usage
- `text_quote` - Original text quote
- `context` - Surrounding context
- `source` - Data source information

## Code Quality

```bash
# Format code
ruff format .

# Check linting
ruff check .

# Type checking
mypy src/

# Pre-commit hooks
pre-commit run --all-files
```

## Troubleshooting

### MongoDB Connection Issues

```bash
# Start MongoDB with Docker
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

### Elasticsearch Issues

```bash
# Start Elasticsearch with Docker
docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.12.0
```

### LLM Connection Issues

Ensure Ollama is running:

```bash
ollama serve
ollama pull llama3.2  # or your preferred model
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest`)
5. Run code quality checks (`ruff check . && mypy src/`)
6. Commit changes (`git commit -am 'Add new feature'`)
7. Push to branch (`git push origin feature/new-feature`)
8. Open a Pull Request

## License

MIT License