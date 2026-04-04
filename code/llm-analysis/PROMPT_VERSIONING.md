# LLM Prompt & Query Template Versioning

**Extraction date:** 2026-03-19
**Data file:** [`prompt_versions.json`](prompt_versions.json)

## Scope

This report documents the stochastic and configurable components of the workflow described in the paper. **All analyses reported in the paper used the Dimensions database exclusively** — the OpenAlex connector exists in the codebase as an alternative integration but was **not used** for any results in the manuscript. The query template for OpenAlex is included in `prompt_versions.json` for completeness of the codebase audit but is marked as unused below.

## Prompt Inventory

| # | Agent | Prompt Name | SHA-256 (prefix) | Temp | Max Tokens |
|---|-------|-------------|-------------------|------|------------|
| 1 | DatasetValidationAgent | `dataset_validation` | `534cbc3c668a` | 0.7 | 4,000 |
| 2 | DatasetDiscoveryAgent | `dataset_discovery` | `06d3d6123149` | 0.7 | 40,000 |
| 3 | DatasetJoinAnalysisAgent | `join_detection` | `a059562062f4` | 0.7 | 5,000 |
| 4 | DatasetJoinAnalysisAgent | `methodology_extraction` | `8bb02184f6eb` | 0.7 | 4,000 |
| 5 | DatasetJoinAnalysisAgent | `challenge_documentation` | `37e81cd2629c` | 0.7 | 4,000 |
| 6 | TriageAgent | `triage_classification` | `54104d2d7eb0` | 0.7 | 4,000 |
| 7 | CodeExtractionAgent | `content_discovery` | `44bb26250fe5` | 0.7 | 3,000 |
| 8 | CodeExtractionAgent | `link_analysis` | `cf93646fab67` | 0.7 | 4,000 |
| 9 | CodeAnalysisLLM | `comprehensive_code_analysis` | `cecd5f07cb25` | 0.7 | 4,000 |
| 10 | CodeAnalysisLLM | `language_identification` | `90716ec5ee12` | 0.7 | 4,000 |
| 11 | GitHubRepositoryVerificationAgent | `repo_alignment_analysis` | `6cfe80d4f0e1` | 0.7 | 4,000 |

**Total: 11 prompt templates** across 6 agent modules.

### Prompt Purposes

#### Stage 1 — Validate seed-dataset mentions (Steps 3 & 4 of the paper workflow)

- **triage_classification** — Binary classification of whether a publication qualifies as a data-analysis paper (boolean + confidence 0-10). This is the gate that determines whether a publication proceeds to deeper analysis.
- **dataset_validation** — Validates whether known datasets (the 11 seed datasets from the "Agricultural and Food Security Data" group) are genuinely mentioned and/or used in a publication. Outputs two independent 0-10 scores: `confidence_score_mention` and `confidence_score_use`. The paper's threshold of >6 on these scores produces the 859 mention-validated and 602 use-validated publication counts.
- **dataset_discovery** — Discovers additional datasets (known and unknown) mentioned in publication text with structured scoring. This is the prompt behind Step 4 (LLM extraction), which identified 774 dataset names from the 859 mention-validated publications, subsequently clustered into 77 candidates and 16 unique additional datasets (Table A.1).

#### Stage 2 — Deep analysis of validated publications

- **join_detection** — Identifies instances where multiple datasets are integrated or joined in research.
- **methodology_extraction** — Extracts detailed join methodology (keys, tools, algorithms) for detected dataset joins.
- **challenge_documentation** — Documents integration challenges, success metrics, lessons learned, and risk assessments for dataset joins.
- **content_discovery** — Discovers code snippets, GitHub repositories, and external links via LLM (validates regex findings and finds additional content).
- **link_analysis** — Categorizes and scores relevance of external links found in publications.
- **comprehensive_code_analysis** — Full code snippet analysis: language, purpose, implementation type, relevance, complexity.
- **language_identification** — Precise programming language identification for uncertain code snippets.
- **repo_alignment_analysis** — Assesses alignment between a GitHub repository's content and the publication's methods/data/objectives.

## Query Templates

| # | Name | Source | SHA-256 (prefix) | Used in Paper |
|---|------|--------|-------------------|---------------|
| 1 | `dimensions_dsl_search` | `dd-metadata-extraction/.../dimensions.py` | `353489fc5965` | **Yes** |
| 2 | `openalex_search` | `dd-metadata-extraction/.../openalex.py` | `64853b0e0b8d` | No |

### Dimensions DSL Query Structure (the query used in the paper)

This is the query that drives Steps 1-2 of the paper workflow — the string search that produces the initial publication candidate set (11,915 for seed datasets; 233,446 aggregated across Stage 2 datasets). The Dimensions DSL (Domain Specific Language) operates against the full-text Dimensions corpus.

```
search publications
  in full_data for "(<aliases OR-joined> [AND (<flag_terms OR-joined>)] [NOT (<exclude_terms NOT-joined>)])"
  where year >= {start_year} and
        year <= {end_year} and
        type in ["article", "chapter", "proceeding", "monograph", "preprint"] and
        research_org_countries = "US"
  return publications[basics + journal_lists + concepts_scores + doi + issn + isbn + linkout + dimensions_url + times_cited + abstract + category_for]
```

**Query construction logic:**

1. **Alias OR-joining**: Each dataset alias is double-quoted and joined with `OR` — e.g., `"Food Access Research Atlas" OR "FARA" OR "Food Desert Atlas"`.
2. **Flag-term AND logic**: If flag terms are configured for a dataset, they are OR-joined within their group, then AND-joined with the aliases — e.g., `("FARA" OR "Food Access Research Atlas") AND ("USDA" OR "ERS")`.
3. **Exclusion NOT logic**: If exclusion terms are configured, they are NOT-joined and subtracted — e.g., `(...) NOT ("unrelated term")`.
4. **Fixed filters**: Year range (2015-2025 for the paper), five publication types, US research-organization affiliation.
5. **Iterative pagination**: Results are retrieved via `dsl.query_iterative()` which handles Dimensions' pagination automatically.


## Model Configuration

| Parameter | Value |
|-----------|-------|
| Default model | `gpt-oss:120b` |
| Provider | Ollama (via LM Studio endpoint `http://localhost:1234/v1/`) |
| Global temperature | 0.1 (settings.py default) |
| Agent temperature | 0.7 (all agents override global) |
| Top-p | 0.9 |
| Request timeout | 60 seconds |
| Max retries | 3 |
| Retry delay | 1.0 seconds |

### Per-Agent Max Tokens

| Agent | Max Tokens |
|-------|------------|
| DatasetValidationAgent | 4,000 |
| DatasetDiscoveryAgent | 40,000 |
| DatasetJoinAnalysisAgent (detection) | 5,000 |
| DatasetJoinAnalysisAgent (methodology) | 4,000 |
| DatasetJoinAnalysisAgent (challenges) | 4,000 |
| TriageAgent | 4,000 |
| CodeExtractionAgent (discovery) | 3,000 |
| CodeExtractionAgent (link analysis) | 4,000 |
| CodeAnalysisLLM | 4,000 |
| GitHubRepositoryVerificationAgent | 4,000 |

## End-to-End Workflow Mapping

The table below maps each pipeline step described in the paper to the specific prompts and queries that drive it.

| Paper Step | Description | Query/Prompt Used | Key Thresholds |
|------------|-------------|-------------------|----------------|
| Step 1 | String-search Dimensions (2015-2025, US) | `dimensions_dsl_search` | Year: 2015-2025; Types: article/chapter/proceeding/monograph/preprint; Affiliation: US |
| Step 2 | Concept filter ("food security"/"food insecurity") | Dimensions concepts_scores field | Relevance > 0.6 |
| Step 3 | LLM validation (mention + use confidence) | `triage_classification` → `dataset_validation` | Triage confidence ≥ 6.0; Mention > 6 → 859 pubs; Use > 6 → 602 pubs |
| Step 4 | LLM dataset discovery | `dataset_discovery` | Discovery confidence ≥ 6.0; 774 names → 77 clusters → 16 unique |
| Step 5 | String-search + concept filter for new datasets | `dimensions_dsl_search` + concept filter | Same as Steps 1-2 per dataset |
| Step 6 | LLM validation of Stage 2 publications | `dataset_validation` | Same thresholds as Step 3 |
| Deep analysis | Join detection, code/link extraction, repo verification | `join_detection`, `methodology_extraction`, `challenge_documentation`, `content_discovery`, `link_analysis`, `comprehensive_code_analysis`, `language_identification`, `repo_alignment_analysis` | Join confidence ≥ 6.0; Link relevance ≥ 6.0; Alignment ≥ 6.0 |

## Reproducibility Notes

### What Is Pinned

- **Prompt templates**: Full text + SHA-256 hash for each of 11 prompts and 1 query template used in the paper (see [`prompt_versions.json`](prompt_versions.json)).
- **Scoring thresholds**: Confidence thresholds are hardcoded in agent config dataclasses (e.g., `confidence_threshold: 7.0` for validation, `6.0` for discovery/joins/triage).
- **Temperature and token limits**: Per-agent values documented above.
- **Dimensions query filters**: Year range (2015-2025), five publication types, US affiliation, concept relevance threshold (0.6).
- **Fuzzy match threshold**: 80 (dataset validation agent).
- **Dimensions API authentication**: Via API key + endpoint configured in `dd-metadata-extraction`.

### What Varies Between Runs

- **Model weights**: `gpt-oss:120b` is served via Ollama/LM Studio. The exact model binary is not pinned by digest — the same model name may point to different checkpoints after updates.
- **Input data**: The Dimensions corpus is a living database. New publications are continuously indexed, so the same DSL query may return different result counts over time. The paper's numbers (11,915 → 1,014 → 859/602) represent a snapshot.
- **LLM non-determinism**: Even at temperature 0.7, LLM outputs are stochastic. Identical inputs may produce different confidence scores and reasoning across runs.
- **Dimensions API versioning**: The Dimensions API may update its full-text indexing, concept scoring algorithms, or metadata fields between runs.
- **Publication full-text availability**: GROBID-parsed full text depends on PDF availability at the time of processing. Some publications may become available or unavailable over time.

### Hashing Methodology

SHA-256 hashes are computed on the **invariant template** — the prompt text after:
1. Collapsing escaped braces (`{{` → `{`, `}}` → `}`)
2. Replacing variable placeholders (`{variable_name}`) with `<VAR>`
3. Stripping leading/trailing whitespace

This ensures that the hash captures the prompt's structure and instructions, not the variable names used in the Python code.

## Recommendations for Replicators

1. **Pin the model digest**: Record the SHA-256 digest of the model binary (e.g., `ollama show gpt-oss:120b --digest`) alongside the prompt hashes. This closes the gap between "same model name" and "same model weights."

2. **Add prompt version tracking to the pipeline**: Store `prompt_versions.json` hashes in each `llm_analyses` document so that results can be traced back to the exact prompts that produced them.

3. **Seed the random number generator**: Where the LLM API supports a `seed` parameter, pass a fixed seed for exact reproducibility of individual analyses.

4. **Snapshot the Dimensions corpus**: Record the Dimensions API query timestamp, result count, and API version at each pipeline run to document the input data version. This is especially important because the paper's numbers are point-in-time snapshots of a continuously updated database.

5. **Version the extraction script**: Track prompt extraction tooling in version control alongside the pipeline code so that future prompt changes automatically produce updated hashes.

6. **Document the Dimensions-only scope**: Clarify in the Methods section that all publication retrieval and concept filtering was performed exclusively through the Dimensions API (not OpenAlex or other sources), so that replicators know exactly which data source to use.
