# Contributing to PurkinjeUV

Thank you for your interest in contributing to purkinje-uv!

## Code of Conduct

Please follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). Report issues to the maintainers.

## Getting Started

### Clone & Environment

```bash
git clone https://github.com/ricardogr07/purkinje-uv.git
cd PurkinjeUV
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Branching & Pull Requests

- **Protected** `main`: no direct pushes.
- Create a feature branch: `git checkout -b feature/short-description`
- Commit, push, open a PR against `main`.
- All checks (CI, lint, tests) must pass.
- Use “Create a merge commit” to preserve history.

### Enforced Rules

| Rule            | Policy                                            |
|-----------------|---------------------------------------------------|
| PRs only        | All changes via PR, no direct main pushes         |
| Signed commits  | GPG/SSH signature required                        |
| Status checks   | CI, lint, tests, coverage must pass               |
| No force pushes | main cannot be force-pushed or deleted            |

## Coding Standards

- **Formatter**: Black (`black src/`)
- **Imports**: ruff (`ruff --select I --fix`)
- **Linting**: ruff (`ruff check src/`)
- **Typing**: mypy (`mypy --strict src/`)
- **Docstrings**: Google style (pydocstyle/ruff `D` rules)
- **Tests**: pytest + pytest-cov (≥ 90% coverage)
- **Commit messages**: Conventional Commits

## Testing

```bash
pytest --cov=purkinje_uv --cov-report=term-missing
```

## Documentation

Built with Sphinx and furo:

```bash
pip install -e ".[docs]"
sphinx-build -W -b html docs docs/_build/html
```

## Docstring Guidelines

We follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for all docstrings—**public** and **internal** alike.

#### Basic template

```python
def my_function(arg1: int, arg2: str) -> bool:
    """One-line summary of what the function does.

    More elaborate description, if needed. Describe side effects or behaviors.

    Args:
        arg1 (int): Description of the first argument.
        arg2 (str): Description of the second argument.

    Returns:
        bool: Description of the return value.

    Raises:
        ValueError: If an invalid value is passed.

    Examples:
        >>> my_function(1, "hello")
        True
    """
    ...


## Issues & Feature Requests

- Search existing issues
- Provide minimal reproduction steps
- Describe use case and desired behavior

## Contact

Raise questions via GitHub Issues or contact maintainers in `pyproject.toml`.
