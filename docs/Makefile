# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

TUTORIALSDIR = $(SOURCEDIR)/tutorials

.PHONY: tutorials tutorials_md tutorials_ipynb tutorials_py tutorials_clean echo_tutorial_source

# Pattern rule to build a Github flavored markdown file from a qmd file
$(TUTORIALSDIR)/%.md: $(TUTORIALSDIR)/%.qmd
	quarto render $< --to gfm

# Pattern rule to build a notebook file from a qmd file
$(TUTORIALS_DIR)/%.ipynb: $(TUTORIALS_DIR)/%.qmd
	quarto convert $< --output $@

# Pattern rule to build a Python file from a notebook file
$(TUTORIALS_DIR)/%.py: $(TUTORIALS_DIR)/%.ipynb
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
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
.DEFAULT: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
