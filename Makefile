.PHONY: install test lint process api viewer

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

lint:
	ruff check packages/ tests/ apps/

process:
	@echo "Usage: dt-pipeline process <path-to-ply-file> [-o output.json]"

api:
	uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000

viewer:
	cd viewer && npm install && npm run dev
