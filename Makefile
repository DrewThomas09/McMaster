.PHONY: install install-all lint test demo serve clean

install:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff format src tests
	ruff check --fix src tests

test:
	pytest

# Build a synthetic catalog, embed it, index it, and identify a sample photo.
demo:
	mcv demo --parts 300

serve:
	mcv serve

clean:
	rm -rf build dist .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
