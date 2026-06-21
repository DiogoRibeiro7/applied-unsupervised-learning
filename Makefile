.PHONY: install test lint typecheck notebooks

install:
	poetry install

test:
	poetry run pytest

lint:
	poetry run ruff check .

typecheck:
	poetry run mypy src

notebooks:
	poetry run python scripts/run_all_notebooks.py
