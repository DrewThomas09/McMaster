FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

# Install the lightweight runtime by default. For GPU inference build with
#   --build-arg EXTRAS=ml,faiss,llm
ARG EXTRAS=faiss,llm
RUN pip install ".[${EXTRAS}]"

EXPOSE 8000
CMD ["mcv", "serve", "--host", "0.0.0.0", "--port", "8000"]
