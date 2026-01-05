# Sample Data for LLM Analysis Pipeline

This folder contains sample input files for testing the LLM analysis pipeline without requiring a full database setup.

## Files

### `sample_publication.json`

A sample publication document with fulltext content that mentions datasets. Use this to test the analysis workflow.

### `sample_known_datasets.json`

A list of known datasets that the validation agent uses to identify dataset mentions. This represents the reference datasets to search for in publications.

## Quick Test

```python
import json
from pub_analysis_agent.workflows import WorkflowOrchestrator
from pub_analysis_agent.agents import (
    triage_agent_step,
    dataset_validation_agent_step,
    dataset_discovery_agent_step
)

# Load sample data
with open('samples/sample_publication.json') as f:
    publication = json.load(f)

with open('samples/sample_known_datasets.json') as f:
    known_datasets = json.load(f)

# Create and configure orchestrator
orchestrator = WorkflowOrchestrator()
orchestrator.register_agent("classify_data_analysis", triage_agent_step)
orchestrator.register_agent("validate_datasets", dataset_validation_agent_step)
orchestrator.register_agent("discover_new_datasets", dataset_discovery_agent_step)

# Execute workflow
result = await orchestrator.execute_workflow(
    publication_id=publication['publication_id'],
    initial_data={
        "fulltext": publication['fulltext'],
        "known_datasets": known_datasets
    }
)

print(f"Is data analysis paper: {result.is_data_analysis}")
print(f"Validated datasets: {result.validated_datasets}")
print(f"Discovered datasets: {result.newly_discovered_datasets}")
```

## Expected Output

The sample publication should:
1. Be classified as a data analysis paper (is_data_analysis: True)
2. Validate mentions of "Census of Agriculture" and "National Resources Inventory"
3. Detect a dataset join between the two datasets
4. Extract the GitHub repository link
