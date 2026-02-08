.PHONY: install test lint process

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

lint:
	ruff check packages/ tests/

process:
	@echo "Usage: dt-pipeline process <path-to-ply-file> [-o output.json]"
