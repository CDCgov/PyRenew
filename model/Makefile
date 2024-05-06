install:
	poetry install

test:
	poetry run pytest --mpl --mpl-default-tolerance=10

docs: docs/pyrenew_demo.md docs/getting-started.md \
	docs/example-with-datasets.md

docs/pyrenew_demo.md: docs/pyrenew_demo.qmd
	poetry run quarto render docs/pyrenew_demo.qmd

docs/getting-started.md: docs/getting-started.qmd
	poetry run quarto render docs/getting-started.qmd

docs/example-with-datasets.md: docs/example-with-datasets.qmd
	poetry run quarto render docs/example-with-datasets.qmd

docs/py: docs/notebooks
	jupyter nbconvert --to python docs/pyrenew_demo.ipynb
	jupyter nbconvert --to python docs/getting-started.ipynb
	jupyter nbconvert --to python docs/example-with-datasets.ipynb

docs/notebooks:
	quarto convert docs/pyrenew_demo.qmd --output docs/pyrenew_demo.ipynb
	quarto convert docs/getting-started.qmd --output docs/getting-started.ipynb
	quarto convert docs/example-with-datasets.qmd --output \
		docs/example-with-datasets.ipynb

test_images:
	echo "Generating reference images for tests"
	poetry run pytest --mpl-generate-path=src/test/baseline

clean:
	rm -rf docs/*_files/
	for i in .py, .md, .ipynb; do \
		rm -f docs/getting-started.$$i; \
		rm -f docs/pyrenew_demo.$$i; \
		rm -f docs/example-with-datasets.$$i; \
	done

image-build: Dockerfile
	docker build -t pyrenew:latest .

.PHONY: install test docs clean docs/notebooks docs/py test_images image-build
