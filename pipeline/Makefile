PYTHON := poetry run python

lock:
	poetry lock

install:
	poetry install

run: install
	$(PYTHON) pipeline/main.py -h

fmt:
	poetry run ruff format

check:
	poetry run ruff check
