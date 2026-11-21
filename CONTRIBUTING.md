# Contributing to `hawkeye-lite`

Thank you for your interest in contributing!

## Development Setup

```bash
uv sync
cd web && npm install
```

## Quality Gates

Every pull request must pass all Python and Web checks:

```bash
# 1. Lint & formatting
uv run ruff check .
uv run ruff format --check .

# 2. Strict static type checking
uv run mypy

# 3. Unit, property & end-to-end video pipeline tests (>= 90% branch coverage)
uv run pytest --cov

# 4. Verify exported benchmark report & web assets are up to date
uv run hawkeye-lite export --check

# 5. Web studio checks
cd web && npm run check
```
