.PHONY: install install-api install-deep format lint typecheck test check notebooks notebook-smoke build clean

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

notebooks:
	poetry run python scripts/run_all_notebooks.py

notebook-smoke:
	poetry run python scripts/run_all_notebooks.py --only 00_project_overview.ipynb

build:
	poetry build

clean:
	poetry run python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
