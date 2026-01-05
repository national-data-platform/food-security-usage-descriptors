# Sample Data for Data Extraction Pipeline

This folder contains sample input files for testing the data extraction pipeline.

## Files

### `sample_dataset_input.json`

Example input for the `/pipelines/start` API endpoint. This file demonstrates the expected format for specifying datasets and their aliases to search for in OpenAlex.

**Usage:**

```bash
curl -X POST http://localhost/pipelines/start \
  -H "Content-Type: application/json" \
  -d @samples/sample_dataset_input.json
```

## Expected Output

When the pipeline completes successfully, you will receive publications matching the specified dataset aliases. Results are stored in MongoDB and can be retrieved via the `/pipelines/{task_id}/result/download` endpoint.
