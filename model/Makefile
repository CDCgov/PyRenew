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

# Automatically get all .qmd files in the docs directory
QMD_FILES := $(wildcard docs/*.qmd)

# Derive the corresponding markdown files
MD_FILES := $(QMD_FILES:.qmd=.md)

# Target to build all markdown files
docs: $(MD_FILES)

# Pattern rule to build a markdown file from a qmd file
docs/%.md: docs/%.qmd
	poetry run quarto render $<

# Phony target to ensure docs are built
.PHONY: docs

docs/py: docs/notebooks
	for i in docs/*.ipynb; do \
		jupyter nbconvert --to python $$i ; \
	done

docs/notebooks:
	for i in docs/*.qmd; do \
		if [ $$i -nt $$(basename $$i .qmd).ipynb ]; then \
			quarto convert $$i --output docs/$$(basename $$i .qmd).ipynb ; \
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
