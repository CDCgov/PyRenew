CONTAINER=podman
# Pass as environment variables
# CONTAINER_REGISTRY=
# CONTAINER_URL=


# Help target listing the available targets
help:
	@echo "Available targets"
	@echo "================="
	@echo "  help           : Show this help message"
	@echo "  install        : Install the project dependencies"
	@echo "  test           : Run the tests"
	@echo "  docs           : Render the documentation"
	@echo "  clean          : Clean the generated files"
	@echo "  docs/notebooks : Convert the Quarto files to Jupyter notebooks"
	@echo "  test_images    : Generate reference images for tests"
	@echo "  image-build    : Build the podman/Docker image"
	@echo "  image-run      : Run the podman/Docker image"
	@echo "  image-push     : Push the podman/Docker image to the registry"

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
	for i in docs/*.qmd; do \
		jupyter nbconvert --to python $$(basename $$i .qmd).ipynb ; \
	done

docs/notebooks:
	for i in docs/*.qmd; do \
		if [ $$i -nt $$(basename $$i .qmd).ipynb ]; then \
			quarto convert $$i --output $$(basename $$i .qmd).ipynb ; \
		fi \
	done

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
	$(CONTAINER) build -t pyrenew:latest .

image-run:
	$(CONTAINER) run -it --rm -v $(PWD):/mnt pyrenew:latest

image-az-login:
	if [ -z $(CONTAINER_REGISTRY) ]; then \
		echo "Please set the CONTAINER_REGISTRY environment variable"; \
		exit 1; \
	fi
	az login
	az acr login --name $(CONTAINER_REGISTRY)

image-push:
	if [ -z $(CONTAINER_URL) ]; then \
		echo "Please set the CONTAINER_URL environment variable"; \
		exit 1; \
	fi
	$(CONTAINER) push pyrenew:latest $(CONTAINER_URL)/pyrenew:latest

.PHONY: install test docs clean docs/notebooks docs/py test_images image-build \
	image-run
