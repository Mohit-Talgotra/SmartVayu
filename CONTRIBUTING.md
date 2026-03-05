# Contributing to SmartVayu

Thank you for your interest in contributing to SmartVayu! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, inclusive, and professional in all interactions.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in Issues
2. Create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and OS
   - Error messages and stack traces

### Suggesting Features

1. Check if the feature has been suggested
2. Create a new issue describing:
   - The problem you're solving
   - Your proposed solution
   - Alternative approaches
   - Examples or mockups

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Add tests for new functionality
5. Update documentation
6. Commit with clear messages: `git commit -m 'Add feature: description'`
7. Push to your fork: `git push origin feature/your-feature`
8. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/smartvayu.git
cd smartvayu

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install pytest black flake8 mypy
```

## Code Standards

- Follow PEP 8 style guidelines
- Use type hints for function parameters
- Write docstrings for all functions and classes
- Keep functions focused and modular
- Add comments for complex logic

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions/classes
- Include usage examples
- Update HANDOVER.md for architectural changes

## Questions?

Open an issue or start a discussion on GitHub.

Thank you for contributing! 🙏
