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
	@echo "Available targets"
	@echo "================="
	@echo "  help           : Show this help message"
	@echo "  docs           : Render the documentation"
	@echo "  docs_clean     : Remove the generated documentation files"
	@echo "  docs_md        : Convert the Quarto files to rendered Markdown files"
	@echo "  docs_ipynb     : Convert the Quarto files to Jupyter notebooks"
	@echo "  docs_py        : Convert the Jupyter notebooks to Python scripts"
	@echo ""
	@echo "Sphix-related targets"
	@echo "====================="
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	
.PHONY: help Makefile docs docs_md docs_rst docs_ipynb docs_py docs_clean

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


