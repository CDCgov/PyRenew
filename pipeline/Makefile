PYTHON := poetry run python

lock:
	poetry lock

install:
	poetry install

run: install
	$(PYTHON) pipeline/main.py -h

test:
	poetry run pytest

fmt:
	poetry run ruff format

check:
	poetry run ruff check
