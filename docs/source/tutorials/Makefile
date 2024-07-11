help:
	@echo "Available targets"
	@echo "================="
	@echo "  help           : Show this help message"
	@echo "  docs           : Render the documentation"
	@echo "  docs_clean     : Remove the generated documentation files"
	@echo "  docs_md        : Convert the Quarto files to rendered Markdown files"
	@echo "  docs_ipynb     : Convert the Quarto files to Jupyter notebooks"
	@echo "  docs_py        : Convert the Jupyter notebooks to Python scripts"
	@echo ""

.PHONY: help docs docs_md docs_rst docs_ipynb docs_py docs_clean


# Pattern rule to build a markdown file from a qmd file
%.md: %.qmd
	poetry run quarto render $<

# Pattern rule to build a reStructuredText file from a qmd file
%.rst: %.qmd
	if [ ! -d $(dir $@) ]; then mkdir rendered_rst; fi
	poetry run quarto render $< --to rst --output-dir rendered_rst
	poetry run python ../docs/post_process_rst.py $@

# Pattern rule to build a notebook file from a qmd file
%.ipynb: %.qmd
	poetry run quarto convert $< --output $@

# Pattern rule to build a Python file from a notebook file
%.py: %.ipynb
	sed -i.bak 's/"cell_type": "raw"/"cell_type": "markdown"/g' $<
	poetry run jupyter nbconvert --to python $<
	rm $<.bak

# Automatically get all .qmd files in the docs directory
QMD_FILES := $(wildcard *.qmd)

# Derive the corresponding markdown files
MD_FILES := $(QMD_FILES:.qmd=.md)

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
	rm -f $(MD_FILES) $(IPYNB_FILES) $(PY_FILES)
