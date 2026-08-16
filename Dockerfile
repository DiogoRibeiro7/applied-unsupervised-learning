# Minimal image for the applied-unsupervised-learning API and CLI.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=2.1.3 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

RUN pip install "poetry==${POETRY_VERSION}"

# Install dependencies first to leverage Docker layer caching.
COPY pyproject.toml poetry.lock README.md ./
COPY src ./src
# `--with` is ignored when `--only` is given, so the groups are listed together:
# `--only main --with api` would silently ship an image without uvicorn.
RUN poetry install --only main,api --no-interaction

# Train default models so the API has something to serve out of the box.
RUN applied-unsupervised-learning train-clustering && applied-unsupervised-learning detect-anomalies

EXPOSE 8000

CMD ["uvicorn", "unsup_lab.api:app", "--host", "0.0.0.0", "--port", "8000"]
