install:
	poetry install

test:
	poetry run pytest

docs: docs/pyrenew_demo.md docs/getting-started.md

docs/pyrenew_demo.md: docs/pyrenew_demo.qmd
	quarto render docs/pyrenew_demo.qmd

docs/getting-started.md: docs/getting-started.qmd
	quarto render docs/getting-started.qmd

.PHONY: install test
