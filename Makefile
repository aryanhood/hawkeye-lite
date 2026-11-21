.PHONY: setup test lint export web check

setup:
	uv sync
	cd web && npm install

test:
	uv run pytest --cov
	cd web && npm test

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	cd web && npm run typecheck && npm run lint

export:
	uv run hawkeye-lite export

web:
	cd web && npm run dev

check: lint test
	uv run hawkeye-lite export --check
	cd web && npm run build
