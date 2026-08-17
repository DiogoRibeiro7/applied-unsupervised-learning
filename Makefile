.PHONY: install install-api install-deep format lint typecheck test check notebooks notebook-smoke build clean docs-prepare docs-serve docs-build

install:
	poetry install

install-api:
	poetry install --with api

install-deep:
	poetry install --with deep

format:
	poetry run ruff format .
	poetry run ruff check . --fix

test:
	poetry run pytest -q -rs

lint:
	poetry run ruff check .

typecheck:
	poetry run mypy src

check:
	poetry check
	poetry run ruff format . --check
	poetry run ruff check .
	poetry run mypy src
	poetry run pytest -q -rs
	poetry run python scripts/run_all_notebooks.py --only 00_project_overview.ipynb

docs-prepare:
	# Notebooks live in notebooks/ because they are the product, not a docs asset.
	# Copy them in for the build; the copy is gitignored.
	python -c "import pathlib,shutil; d=pathlib.Path('docs/notebooks'); shutil.rmtree(d, ignore_errors=True); shutil.copytree('notebooks', d, ignore=shutil.ignore_patterns('.ipynb_checkpoints'))"

docs-serve: docs-prepare
	poetry run mkdocs serve

docs-build: docs-prepare
	poetry run mkdocs build --strict

notebooks:
	poetry run python scripts/run_all_notebooks.py

notebook-smoke:
	poetry run python scripts/run_all_notebooks.py --only 00_project_overview.ipynb

build:
	poetry build

clean:
	poetry run python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
