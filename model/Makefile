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
	@echo "  docs_clean     : Remove the generated documentation files"
	@echo "  docs_md        : Convert the Quarto files to rendered Markdown files"
	@echo "  docs_ipynb     : Convert the Quarto files to Jupyter notebooks"
	@echo "  docs_py        : Convert the Jupyter notebooks to Python scripts"
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

# Pattern rule to build a reStructuredText file from a qmd file
docs/%.rst: docs/%.qmd
	poetry run quarto render $< --to rst
	poetry run python ../docs/post_process_rst.py $@

# Pattern rule to build a notebook file from a qmd file
docs/%.ipynb: docs/%.qmd
	poetry run quarto convert $< --output $@

# Pattern rule to build a Python file from a notebook file
docs/%.py: docs/%.ipynb
	sed -i.bak 's/"cell_type": "raw"/"cell_type": "markdown"/g' $<
	poetry run jupyter nbconvert --to python $<
	rm $<.bak

# Automatically get all .qmd files in the docs directory
QMD_FILES := $(wildcard docs/*.qmd)

# Derive the corresponding markdown files
MD_FILES := $(QMD_FILES:.qmd=.md)

# Derive the corresponding markdown files
RST_FILES := $(QMD_FILES:.qmd=.rst)

# Derive the corresponding notebook files
IPYNB_FILES := $(QMD_FILES:.qmd=.ipynb)

# Derive the corresponding Python files
PY_FILES := $(IPYNB_FILES:.ipynb=.py)

# Target to build all docs
docs: docs_md docs_ipynb docs_py

# Target to build all markdown files
docs_md: $(MD_FILES)

# Target to build all reStructuredText files
docs_rst: $(RST_FILES)

# Target to build all notebook files
docs_ipynb: $(IPYNB_FILES)

# Target to build all Python files
docs_py: $(PY_FILES)

docs_clean:
	rm -rf docs/*_files/
	rm -f $(MD_FILES) $(RST_FILES) $(IPYNB_FILES) $(PY_FILES)

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

.PHONY: install test docs docs_md docs_ipynb docs_py docs_clean test_images \
		image-build image-run
