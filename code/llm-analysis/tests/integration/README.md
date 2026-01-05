# Integration Tests - Pub Analysis Agent

This directory contains complete integration tests for the publication analysis system, which execute all integrated flows without altering any existing implementations.

## Overview

The integration tests were designed to:

- ✅ **Execute all system flows** without modifying existing implementations
- ✅ **Use appropriate mocks** to simulate external services (LLM, database, etc.)
- ✅ **Test data integrity** throughout the entire workflow
- ✅ **Verify state transitions** between different steps
- ✅ **Monitor performance** and error recovery
- ✅ **Validate final output** of the system

## Test Structure

### 1. Complete Workflow Test (`test_full_workflow_integration.py`)

Comprehensive pytest test that covers:

- **Complete workflow execution** with all 7 steps
- **State transitions** between steps
- **Error recovery** and retry logic
- **Data integrity** at each stage
- **Performance monitoring**
- **Concurrent execution** of multiple workflows
- **Edge cases** of error handling
- **Configuration validation** of the workflow
- **Serialization** of state models
- **Status reporting** of the workflow

### 2. Simple Execution Script (`scripts/run_integration_test.py`)

Standalone script that can be executed directly:

```bash
# Run the integration test
python scripts/run_integration_test.py

# Or execute directly
./scripts/run_integration_test.py
```

## Tested Flows

The system tests the following main flows:

### 1. **Data Analysis Classification**
- Determines if the document contains data analysis
- Sets processing flags (`is_data_analysis`)

### 2. **Dataset Identification**
- Identifies datasets mentioned in the document
- Extracts context and confidence from mentions

### 3. **Dataset Validation**
- Validates identified datasets against knowledge base
- Confirms existence and relevance

### 4. **New Dataset Discovery**
- Discovers uncataloged datasets
- Identifies additional data sources

### 5. **Dataset Join Analysis**
- Analyzes possible integrations between datasets
- Identifies join types and confidence

### 6. **Code and Link Extraction**
- Extracts relevant code snippets
- Identifies external links and GitHub repositories
- Validates link accessibility

### 7. **Final Output Generation**
- Assembles final JSON with all results
- Includes processing metadata

## How to Run

### Option 1: Using pytest (Recommended)

```bash
# Run all integration tests
pytest tests/integration/ -v -s

# Run specific test
pytest tests/integration/test_full_workflow_integration.py::TestFullWorkflowIntegration::test_complete_workflow_execution -v -s

# Run with detailed report
pytest tests/integration/ -v -s --tb=long
```

### Option 2: Using the standalone script

```bash
# Run basic test + performance
python scripts/run_integration_test.py

# Run only basic test (modify the script)
```

### Option 3: Manual execution in Python

```python
import asyncio
from scripts.run_integration_test import run_integration_test

# Run test
success = asyncio.run(run_integration_test())
print(f"Test {'passed' if success else 'failed'}")
```

## Test Data

The tests use realistic simulated data:

### Simulated Grobid Content
- Title and abstract of data analysis
- Document body with Python code
- Authors and references
- Links to GitHub repositories

### Mocked LLM Responses
- Data analysis classification
- Dataset identification (Census 2020, Continuous PNAD)
- Dataset validation
- New dataset discovery (RAIS 2021)
- Dataset join analysis
- Code and link extraction
- Final output generation

### Mocked Datasets
- Census 2020
- Continuous PNAD
- RAIS 2021 (discovered)

## Performed Verifications

### Data Integrity
- ✅ Data type validation
- ✅ Required field verification
- ✅ Confidence score control (0-1)
- ✅ URL and link validation
- ✅ Metadata verification

### Workflow Flow
- ✅ Sequential execution of 7 steps
- ✅ Correct state transitions
- ✅ Step completion marking
- ✅ Timestamp updates
- ✅ Unique workflow_id generation

### Performance
- ✅ Execution time < 30 seconds
- ✅ Concurrent execution of multiple workflows
- ✅ Memory usage monitoring
- ✅ Performance statistics

### Error Recovery
- ✅ Automatic retry on failures
- ✅ Detailed error logging
- ✅ Workflow continuation after recovery
- ✅ Edge case handling

## Expected Output

### Execution Logs
```
2024-01-15 10:30:00 - root - INFO - ============================================================
2024-01-15 10:30:00 - root - INFO - STARTING COMPLETE INTEGRATION TEST
2024-01-15 10:30:00 - root - INFO - ============================================================
2024-01-15 10:30:00 - root - INFO - 1. Creating test content...
2024-01-15 10:30:00 - root - INFO - 2. Configuring mocked agents...
2024-01-15 10:30:00 - root - INFO - 3. Initializing workflow orchestrator...
2024-01-15 10:30:00 - root - INFO - 4. Registering agents in workflow...
2024-01-15 10:30:00 - root - INFO - 5. Executing complete workflow...
2024-01-15 10:30:00 - root - INFO - Executing: Data analysis classification
2024-01-15 10:30:00 - root - INFO - Executing: Dataset identification
2024-01-15 10:30:00 - root - INFO - Executing: Dataset validation
2024-01-15 10:30:00 - root - INFO - Executing: New dataset discovery
2024-01-15 10:30:00 - root - INFO - Executing: Dataset join analysis
2024-01-15 10:30:00 - root - INFO - Executing: Code and link extraction
2024-01-15 10:30:00 - root - INFO - Executing: Final output generation
2024-01-15 10:30:00 - root - INFO - 6. Verifying results...
2024-01-15 10:30:00 - root - INFO - ============================================================
2024-01-15 10:30:00 - root - INFO - INTEGRATION TEST COMPLETED SUCCESSFULLY!
2024-01-15 10:30:00 - root - INFO - ============================================================
2024-01-15 10:30:00 - root - INFO - Execution time: 2.45 seconds
2024-01-15 10:30:00 - root - INFO - Steps executed: 7
2024-01-15 10:30:00 - root - INFO - Validated datasets: 2
2024-01-15 10:30:00 - root - INFO - New datasets: 1
2024-01-15 10:30:00 - root - INFO - Joins analyzed: 1
2024-01-15 10:30:00 - root - INFO - Code snippets: 1
2024-01-15 10:30:00 - root - INFO - External links: 1
2024-01-15 10:30:00 - root - INFO - GitHub repositories: 1
```

### Final JSON
```json
{
  "publication_id": "test_pub_001",
  "workflow_id": "test_workflow_001",
  "analysis_summary": "Complete analysis performed",
  "datasets_found": 3,
  "code_snippets_extracted": 1,
  "execution_time": 2.45,
  "total_datasets": 3,
  "total_joins": 1,
  "total_code_snippets": 1,
  "total_links": 1,
  "total_github_repos": 1
}
```

## Troubleshooting

### Error: "Module not found"
```bash
# Make sure you're in the project root directory
cd /path/to/pub-analysis-agent

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Error: "Import error"
```bash
# Add src directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or execute with correct path
PYTHONPATH=src python scripts/run_integration_test.py
```

### Error: "Async/await issues"
```bash
# Make sure to use Python 3.7+
python --version

# For older versions, use the script with compatibility
python scripts/run_integration_test.py
```

## Contributing

To add new integration tests:

1. **Create new test methods** in the `TestFullWorkflowIntegration` class
2. **Use the established mock pattern**
3. **Add data integrity verifications**
4. **Document** the test purpose
5. **Execute** all tests to ensure compatibility

## Next Steps

- [ ] Add integration tests with real services (optional)
- [ ] Implement load tests with large data volumes
- [ ] Add integration tests with Elasticsearch
- [ ] Create integration tests with real MongoDB
- [ ] Implement integration tests with real LLMs 