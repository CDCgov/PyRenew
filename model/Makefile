install:
	poetry install

test:
	poetry run pytest

docs: docs/pyrenew_demo.md docs/getting-started.md \
	docs/example-with-datasets.md

docs/pyrenew_demo.md: docs/pyrenew_demo.qmd
	poetry run quarto render docs/pyrenew_demo.qmd

docs/getting-started.md: docs/getting-started.qmd
	poetry run quarto render docs/getting-started.qmd

docs/example-with-datasets.md: docs/example-with-datasets.qmd
	poetry run quarto render docs/example-with-datasets.qmd

notebooks:
	quarto convert docs/pyrenew_demo.qmd --output docs/pyrenew_demo.ipynb
	quarto convert docs/getting-started.qmd --output docs/getting-started.ipynb
	jupyter nbconvert --to python docs/pyrenew_demo.ipynb
	jupyter nbconvert --to python docs/getting-started.ipynb
	rm -f docs/pyrenew_demo.ipynb docs/getting-started.ipynb

clean:
	rm -rf docs/*_files/
	rm -f docs/getting-started.ipynb
	rm -f docs/pyrenew_demo.ipynb

.PHONY: install test docs clean notebooks
