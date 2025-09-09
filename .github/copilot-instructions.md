# PyRenew: Multi-signal Bayesian Renewal Modeling

PyRenew is a Python package for simulation and statistical inference of epidemiological models using JAX and NumPyro, emphasizing renewal models for infectious disease forecasting and analysis.

Always reference these instructions first and fallback to additional search or context gathering only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Prerequisites and Installation
- **CRITICAL**: Install `uv` package manager first: `pip install uv`
- **Python version requirement**: Python 3.13+ (as specified in pyproject.toml)
- Install development environment: `uv sync --extra dev`
- Install test dependencies: `uv sync --extra test` 
- Install documentation dependencies: `uv sync --extra docs`
- Install ALL dependencies at once: `uv sync --all-extras`

### Build and Test Process
- **NEVER CANCEL**: Test suite takes ~2.5 minutes. ALWAYS set timeout to 5+ minutes for pytest commands.
- **NEVER CANCEL**: Documentation build takes ~11 seconds, but first-time dependency installation takes 1-2 minutes.
- Test installation: `uv sync --extra dev` -- takes ~1 minute on first run, ~0.3 seconds on subsequent runs
- Run tests: `uv run pytest --mpl --mpl-default-tolerance=10` -- takes 2.5 minutes. NEVER CANCEL.
- Run tests with coverage: `uv run pytest --mpl --mpl-default-tolerance=10 --cov=pyrenew --cov-report term --cov-report xml`

### Documentation
- **Requires Quarto CLI**: Install from https://github.com/quarto-dev/quarto-cli/releases
- Build documentation: `cd docs && uv run make html` -- takes ~11 seconds
- Documentation source: `docs/source/`
- Tutorials: Quarto (.qmd) files in `docs/source/tutorials/`
- API reference: Auto-generated from docstrings

### Pre-commit and Code Quality
- Install pre-commit manually: `uv pip install pre-commit` (not included in project dependencies)
- Setup hooks: `uv run pre-commit install`
- **NOTE**: Pre-commit is NOT included in the dev dependencies by design
- Linting tools used: ruff (formatting and linting), numpydoc-validation, secret detection, typos checking

### Key Commands Summary
```bash
# Installation (choose one based on needs)
uv sync --extra dev        # Development dependencies only
uv sync --extra test       # Test dependencies only  
uv sync --extra docs       # Documentation dependencies only
uv sync --all-extras       # All dependencies (recommended for full development)

# Testing - NEVER CANCEL, takes ~2.5 minutes
uv run pytest --mpl --mpl-default-tolerance=10

# Documentation build
cd docs && uv run make html

# Basic Python module test
uv run python -c "import pyrenew; print('PyRenew imported successfully')"
```

## Validation Scenarios

### After Making Code Changes
1. **ALWAYS** test import: `uv run python -c "import pyrenew; print('Import successful')"`
2. **ALWAYS** run affected tests: `uv run pytest test/test_[relevant_module].py -v`
3. **ALWAYS** run full test suite before committing: `uv run pytest --mpl --mpl-default-tolerance=10` -- NEVER CANCEL, takes 2.5 minutes
4. **Check code style**: Run ruff formatting (if pre-commit installed)

### For Documentation Changes
1. **ALWAYS** rebuild docs: `cd docs && uv run make html`
2. **Check tutorials**: Ensure .qmd files in tutorials/ render correctly
3. **Verify API docs**: Check that module docstrings appear in generated docs

### For New Features or Models
1. **Test basic functionality**: Create minimal example using the new feature
2. **Run relevant test modules**: Focus on affected areas (e.g., test_model.py for model changes)
3. **Validate with real data**: Use example datasets in pyrenew.datasets if available

## Repository Structure

### Key Directories
- `pyrenew/`: Main package source code
  - `deterministic/`: Deterministic variables and components
  - `distributions/`: Custom probability distributions
  - `latent/`: Latent variable models
  - `model/`: Complete model implementations
  - `observation/`: Observation process models
  - `process/`: Time series and stochastic processes
  - `randomvariable/`: Random variable abstractions
- `test/`: Comprehensive test suite (174 tests)
- `docs/`: Sphinx documentation with Quarto tutorials
- `hook_scripts/`: Pre-commit utility scripts

### Important Files
- `pyproject.toml`: Package configuration, dependencies, and tool settings
- `Makefile`: Simplified commands (install, test targets)
- `.pre-commit-config.yaml`: Code quality automation
- `docs/source/conf.py`: Sphinx documentation configuration

## Common Issues and Solutions

### Build Issues
- **Missing uv**: Install with `pip install uv`
- **Wrong Python version**: Requires Python 3.13+, check with `python --version`
- **Missing Quarto**: Required for documentation, install from GitHub releases
- **Test failures**: Some tests require specific plot tolerances (--mpl flags)

### Import Errors
- **Missing dependencies**: Run `uv sync` with appropriate extras
- **Path issues**: Ensure working directory is repository root
- **JAX/NumPyro issues**: These are heavy dependencies; consider environment compatibility

### Documentation Warnings
- **Import warnings during doc build**: Normal for mocked heavy dependencies (JAX, NumPyro)
- **Duplicate labels**: Expected for placeholder tutorial files

## Development Workflow

### Typical Development Session
1. `uv sync --all-extras` (first time or after dependency changes)
2. Make code changes
3. `uv run python -c "import pyrenew"` (quick import test)
4. `uv run pytest test/test_[relevant].py -v` (focused testing)
5. `uv run pytest --mpl --mpl-default-tolerance=10` (full test suite - NEVER CANCEL)
6. `cd docs && uv run make html` (if documentation changed)

### Adding New Code
- **Follow existing patterns**: Check similar modules for structure
- **Add tests**: Every new feature should have corresponding tests
- **Document thoroughly**: Use NumPy-style docstrings
- **Consider tutorials**: Complex features may benefit from Quarto tutorial examples

## Timing Expectations
- **uv sync operations**: 0.3-2 minutes depending on cache state
- **Test suite**: 2.5 minutes (NEVER CANCEL)
- **Documentation build**: 11 seconds
- **Simple imports**: <1 second
- **Full environment setup**: 3-5 minutes for completely fresh install

## CI/CD Integration
- **GitHub Actions**: Uses `uv` for package management
- **Test workflow**: Runs on Ubuntu with Python 3.13
- **Documentation**: Auto-deployed to GitHub Pages
- **Coverage**: Integrated with Codecov

Remember: This is a scientific computing package dealing with epidemiological modeling. Precision and reproducibility are critical - always validate mathematical components thoroughly and never skip the full test suite.