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
	@echo "  docs/py        : Convert the Jupyter notebooks to Python scripts"
	@echo "  test_images    : Generate reference images for tests"
	@echo "  image-build    : Build the podman/Docker image"
	@echo "  image-run      : Run the podman/Docker image"
	@echo "  image-push     : Push the podman/Docker image to the registry"

install:
	poetry install

test:
	poetry run pytest --mpl --mpl-default-tolerance=10

# Pattern rule to build a markdown file from a qmd file
docs/%.md: docs/%.qmd
	poetry run quarto render $<

# Pattern rule to build a notebook file from a qmd file
docs/%.ipynb: docs/%.qmd
	poetry run quarto convert $< --output $@

# Pattern rule to build a Python file from a notebook file
docs/%.py: docs/%.ipynb
	poetry run jupyter nbconvert --to python $<

# Automatically get all .qmd files in the docs directory
QMD_FILES := $(wildcard docs/*.qmd)

# Derive the corresponding markdown files
MD_FILES := $(QMD_FILES:.qmd=.md)

# Derive the corresponding notebook files
NOTEBOOK_FILES := $(QMD_FILES:.qmd=.ipynb)

# Derive the corresponding Python files
PY_FILES := $(NOTEBOOK_FILES:.ipynb=.py)

# Target to build all markdown files
docs: $(MD_FILES)

# Target to build all notebook files
docs/notebooks: $(NOTEBOOK_FILES)

# Target to build all Python files
docs/py: $(PY_FILES)

# Phony target to ensure docs are built
.PHONY: docs docs/notebooks docs/py

test_images:
	echo "Generating reference images for tests"
	poetry run pytest --mpl-generate-path=src/test/baseline

clean:
	rm -rf docs/*_files/
	rm -f $(MD_FILES) $(NOTEBOOK_FILES) $(PY_FILES)

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
