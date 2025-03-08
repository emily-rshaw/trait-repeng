# CLAUDE.md - Guidelines for AI Assistants

## Running Commands

```bash
# Run main DCT personality modeling script
python scripts/main_demo.py

# Run IPIP NEO personality trait modeling
python scripts/main_demo_ipip_mod.py --trait a1_trust --direction max

# List available datasets
python scripts/main_demo_ipip_mod.py --list-datasets
```

## Code Style Guidelines

- **Imports**: Standard library first, third-party next, local imports last
- **Naming**: snake_case for functions/variables, CamelCase for classes
- **Docstrings**: Triple-quoted docstrings for modules, classes, and functions
- **Type annotations**: Used selectively, not consistently throughout
- **Indentation**: 4 spaces, max line length ~100 characters
- **Error handling**: Explicit validation with descriptive error messages
- **Classes**: Clear responsibility boundaries, helper methods prefixed with underscore

The codebase is a Python machine learning project for personality trait modeling using DCT techniques with PyTorch and Hugging Face transformers.