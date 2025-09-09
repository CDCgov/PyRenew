# Makefile for MkDocs documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
MKDOCSBUILD   ?= mkdocs
SOURCEDIR     = .
BUILDDIR      = site

# Put it first so that "make" without argument is like "make help".
help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  serve      to serve the documentation locally"
	@echo "  tutorials  to build all tutorial formats"
	@echo "  tutorials_md     to build tutorial markdown files"
	@echo "  tutorials_ipynb  to build tutorial notebook files"
	@echo "  tutorials_py     to build tutorial Python files"
	@echo "  tutorials_clean  to clean tutorial build artifacts"

.PHONY: help html serve Makefile

TUTORIALSDIR = tutorials

.PHONY: tutorials tutorials_md tutorials_ipynb tutorials_py tutorials_clean echo_tutorial_source

# Pattern rule to build a Github flavored markdown file from a qmd file
$(TUTORIALSDIR)/%.md: $(TUTORIALSDIR)/%.qmd
	quarto render $< --to gfm

# Pattern rule to build a notebook file from a qmd file
$(TUTORIALSDIR)/%.ipynb: $(TUTORIALSDIR)/%.qmd
	quarto convert $< --output $@

# Pattern rule to build a Python file from a notebook file
$(TUTORIALSDIR)/%.py: $(TUTORIALSDIR)/%.ipynb
	sed -i.bak 's/"cell_type": "raw"/"cell_type": "markdown"/g' $<
	jupyter nbconvert --to python $<
	rm $<.bak

# Automatically get all .qmd files in the tutorials directory
TUTORIALS_QMD := $(wildcard $(TUTORIALSDIR)/*.qmd)

# Derive the corresponding markdown files
TUTORIALS_MD := $(TUTORIALS_QMD:.qmd=.md)

# Derive the corresponding notebook files
TUTORIALS_IPYNB := $(TUTORIALS_QMD:.qmd=.ipynb)

# Derive the corresponding Python files
TUTORIALS_PY := $(TUTORIALS_QMD:.qmd=.py)

# Target to render tutorials to all formats
tutorials: tutorials_md tutorials_ipynb tutorials_py

# Target to render tutorials as Github-flavored markdown
tutorials_md: $(TUTORIALS_MD)

# Target to render tutorials as Jupyter notebooks
tutorials_ipynb: $(TUTORIALS_IPYNB)

# Target to render tutorials as .py python files
tutorials_py: $(TUTORIALS_PY)

tutorials_clean:
	@rm -rf ./$(TUTORIALSDIR)/*_files/
	@rm -f $(TUTORIALS_MD) $(TUTORIALS_IPYNB) $(TUTORIALS_PY)

.PHONY: html
html: $(TUTORIALS_MD)
	@cd .. && $(MKDOCSBUILD) build

.PHONY: serve
serve: $(TUTORIALS_MD)
	@cd .. && $(MKDOCSBUILD) serve
