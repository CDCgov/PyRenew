install:
	poetry install

test:
	poetry run pytest

docs: docs/pyrenew_demo.md docs/getting-started.md

docs/pyrenew_demo.md: docs/pyrenew_demo.qmd
	poetry run quarto render docs/pyrenew_demo.qmd

docs/getting-started.md: docs/getting-started.qmd
	poetry run quarto render docs/getting-started.qmd

clean:
	rm -rf docs/*_files/
	rm -f docs/getting-started.ipynb
	rm -f docs/pyrenew_demo.ipynb

.PHONY: install test docs clean
