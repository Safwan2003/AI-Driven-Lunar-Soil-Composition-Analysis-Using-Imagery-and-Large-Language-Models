# Documentation

This directory contains project documentation.

## Contents

- **API Documentation**: Detailed API references
- **Tutorials**: Step-by-step guides
- **Architecture**: System design and architecture docs
- **Deployment**: Deployment and production guides

## Building Documentation

This project uses Sphinx for documentation. To build:

```bash
cd docs
pip install -r requirements.txt  # If separate doc requirements exist
make html
```

View the built documentation:
```bash
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
```

## Contributing to Documentation

1. Write docstrings in Google or NumPy style
2. Add new pages as needed
3. Keep documentation up-to-date with code changes
4. Include examples and use cases

## Documentation Standards

- Use clear, concise language
- Include code examples
- Add diagrams where helpful
- Cross-reference related sections
