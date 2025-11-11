# Tests

This directory contains unit tests and integration tests for the project.

## Running Tests

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src tests/
```

Run specific test file:
```bash
pytest tests/test_data_loader.py
```

Run with verbose output:
```bash
pytest -v
```

## Test Organization

```
tests/
├── unit/             # Unit tests for individual modules
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_llm.py
│   └── ...
├── integration/      # Integration tests
│   └── test_pipeline.py
└── README.md         # This file
```

## Writing Tests

Example test structure:

```python
import pytest
from src.data import DataLoader

def test_data_loader_initialization():
    loader = DataLoader()
    assert loader is not None

def test_data_loader_with_config():
    loader = DataLoader(config_path="config.yaml")
    assert loader.image_size == (512, 512)
```

## Best Practices

1. Write tests before or alongside code (TDD)
2. Aim for high code coverage (>80%)
3. Test edge cases and error conditions
4. Keep tests independent and isolated
5. Use fixtures for common setup
6. Mock external dependencies (APIs, databases)

## Continuous Integration

Tests are automatically run on:
- Every pull request
- Every push to main branch
- Scheduled daily builds
