# Food Security Usage Data Descriptors

This repository contains data files and code supporting the article:

**"Usage data descriptors as metadata: the case of food security and the National Data Platform (2015-2025)"**

Published in: Nature Scientific Data

---

## Quick Start

### View the Data
The `data/` folder contains all output CSV files from the study. See [Data Files](#data-files) below for descriptions.

### Run the Code
Each code component has its own Quick Start guide:
- **Data Extraction**: See [`code/data-extraction/README.md`](code/data-extraction/README.md)
- **LLM Analysis**: See [`code/llm-analysis/README.md`](code/llm-analysis/README.md)

Both components include sample data files in their `samples/` folders for testing without a full database setup.

---

## Repository Structure

```
├── data/                              # Output data files from the study
│   ├── fileA_usda_publication_dataset_pairs.csv
│   ├── fileB_additional_datasets_publication_pairs.csv
│   ├── fileC_dataset_joins.csv
│   ├── summary_by_dataset_year.csv
│   ├── data_dyads.csv
│   ├── usda_seed_datasets.csv
│   ├── additional_discovered_datasets.csv
│   └── data_dictionary.csv
│
├── code/                              # Source code for data collection & analysis
│   ├── data-extraction/               # Publication metadata extraction service
│   │   ├── api.py                     # FastAPI REST API (main entrypoint)
│   │   ├── samples/                   # Sample input data for testing
│   │   ├── docker-compose.yml         # Docker configuration
│   │   └── README.md                  # Setup and usage instructions
│   │
│   └── llm-analysis/                  # LLM-powered publication analysis
│       ├── src/pub_analysis_agent/    # Main analysis package
│       ├── samples/                   # Sample input data for testing
│       ├── tests/                     # Unit and integration tests
│       └── README.md                  # Setup and usage instructions
│
└── README.md                          # This file
```

---

## Data Files

### Primary Files

| File | Description | Records |
|------|-------------|---------|
| `fileA_usda_publication_dataset_pairs.csv` | Verified instances where USDA datasets are mentioned/used in publications | 1,249 pairs (859 publications) |
| `fileB_additional_datasets_publication_pairs.csv` | Newly discovered datasets from LLM extraction in Stage 2 | 5,159 pairs (1,399 publications) |
| `fileC_dataset_joins.csv` | Instances where multiple datasets are integrated within publications | 1,608 joins (535 publications) |

### Supporting Files

| File | Description | Records |
|------|-------------|---------|
| `summary_by_dataset_year.csv` | Publication counts by dataset and year | 68 rows |
| `data_dyads.csv` | Dataset names and aliases used for string search | 46 rows |
| `usda_seed_datasets.csv` | List of 11 USDA seed datasets (Stage 1) | 12 rows |
| `additional_discovered_datasets.csv` | List of 18 additional datasets (Stage 2) | 18 rows |
| `data_dictionary.csv` | Field definitions and data types for all files | - |

### JSON-LD Output (`data/json-ld/`)

Dataset-centric JSON-LD files using [schema.org](https://schema.org/) vocabulary. Each file represents one curated dataset and aggregates all publications that reference it as `Review` entries.

| File | Description |
|------|-------------|
| `context.jsonld` | Shared `@context` definition (pure schema.org) |
| `catalog.jsonld` | `DataCatalog` listing all 31 datasets with reviews |
| `datasets/*.jsonld` | One file per dataset (34 total, 31 with reviews) |

**Schema mapping — uses only native schema.org vocabulary (no custom namespace):**

| schema.org Type/Property | Usage |
|--------------------------|-------|
| `Dataset` | Root entity — one per file |
| `Review` | Each publication's assessment of the dataset |
| `Rating` | Confidence scores (`mention` and `use`, 0–10 scale) |
| `ScholarlyArticle` | Publication cited in the review |
| `creator` | Organization responsible for the dataset (on root `Dataset`) |
| `about` | Research domain classification (on root `Dataset`) |
| `url` | Dimensions page for the cited publication (on `ScholarlyArticle`) |

**Review aspects** distinguish how the dataset-publication link was identified:

| `reviewAspect` | Source | Description |
|----------------|--------|-------------|
| `"validation"` | File A | Dataset confirmed by LLM against known seed/additional list |
| `"discovery"` | File B | Dataset discovered by LLM from full-text extraction |

**Regenerate:**
```bash
python code/csv-to-jsonld/convert.py
```

---

## Code Components

### CSV to JSON-LD (`code/csv-to-jsonld/`)

Converts the CSV data files into dataset-centric JSON-LD using schema.org vocabulary.

```bash
python code/csv-to-jsonld/convert.py [--data-dir data/]
```

### Data Extraction (`code/data-extraction/`)

A service for extracting and processing metadata from OpenAlex API, focusing on dataset usage in scientific publications.

**Key Features:**
- Extracts publication metadata using dataset alias full-text searches
- Processes and stores information about publications, authors, institutions, journals, and topics
- Uses RabbitMQ for message queue processing
- Stores data in MongoDB with Elasticsearch for search capabilities

**Tech Stack:** Python, FastAPI, RabbitMQ, MongoDB, Elasticsearch, Docker

**Getting Started:**
```bash
cd code/data-extraction
cp .env.sample .env
docker-compose up -d
open http://localhost/docs
```

See [`code/data-extraction/README.md`](code/data-extraction/README.md) for detailed documentation.

### LLM Analysis (`code/llm-analysis/`)

An AI-powered pipeline for analyzing full-text scientific publications to extract structured information about dataset usage.

- **[Prompt Versioning](code/llm-analysis/PROMPT_VERSIONING.md)** — Documents all 11 LLM prompt templates and 2 query templates with SHA-256 hashes, model configuration, threshold parameters, and end-to-end workflow mapping. See also [`prompt_versions.json`](code/llm-analysis/prompt_versions.json) for the machine-readable prompt inventory.

**Workflow Steps:**
1. **TriageAgent** - Classifies if the publication is a data analysis paper
2. **DatasetValidationAgent** - Validates known datasets mentioned in the text
3. **DatasetDiscoveryAgent** - Discovers new/unknown datasets
4. **DatasetJoinAnalysisAgent** - Analyzes how datasets are combined/joined
5. **CodeExtractionAgent** - Extracts code snippets and external links
6. **GitHubRepositoryVerificationAgent** - Verifies GitHub repository links
7. **JSONAssemblyAgent** - Assembles final structured output

**Tech Stack:** Python 3.12+, LangGraph, Ollama/LM Studio, MongoDB, Elasticsearch

**Getting Started:**
```bash
cd code/llm-analysis
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env
python -m pub_analysis_agent.config.cli validate
```

See [`code/llm-analysis/README.md`](code/llm-analysis/README.md) for detailed documentation.

---

## Data Description

### File A: USDA Publication-Dataset Pairs

Each record represents a verified instance where a USDA dataset is mentioned within a food-security-related publication. Publications were identified through:
1. String search for 11 USDA seed datasets in Dimensions corpus
2. Filtering for "food security" or "food insecurity" topics with relevance score ≥ 0.6
3. LLM-based validation with confidence score ≥ 6

**Key fields:**
- `publication_id`: Dimensions publication identifier
- `publication_doi`: Digital Object Identifier
- `validated_dataset_name`: Canonical name of the USDA dataset
- `confidence_score_mention`: Confidence the dataset was correctly identified (0-10)
- `confidence_score_use`: Confidence the dataset was used in analysis (0-10)

### File B: Additional Datasets Publication Pairs

Each record documents a newly discovered dataset (not in the original USDA seed list) that was detected through LLM extraction from the Stage 2 food security publications.

**Key fields:**
- `publication_id`: Dimensions publication identifier
- `new_name`: Name of the newly discovered dataset
- `new_confidence_score_mention`: Confidence score for dataset identification (0-10)
- `new_confidence_score_use`: Confidence score for substantive use (0-10)
- `new_description`: Brief description of the dataset
- `new_source`: Organization responsible for the dataset
- `new_domain`: Thematic domain (e.g., public health, labor statistics)

### File C: Dataset Joins

Each record documents an instance where multiple datasets are integrated within a single publication. Fields include:
- Dataset pair names
- Join type (e.g., merge, linkage, integration)
- Methodology description
- Join keys used

---

## Data Extraction Parameters

- **Dimensions Corpus Filters**: English-language, peer-reviewed, 2015-2025, US-affiliated author
- **Food Security Threshold**: Topic relevance score ≥ 0.6
- **Validation Threshold**: Confidence score ≥ 6
- **Data Cutoff Date**: 2025-11-06T10:14:00 UTC

---

## Citation

If using this data or code, please cite:

> Chenarides, L., Ladislau, R., Parashar, M., Hook, D., Porter, S., & Lane, J. (2026). Usage data descriptors as metadata: the case of food security and the National Data Platform (2015-2025). *Nature Scientific Data*.

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). See [LICENSE](LICENSE) for details.

## Contact

For questions about this dataset or code, contact the corresponding author.
