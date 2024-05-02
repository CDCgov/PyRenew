# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.insert(0, os.path.abspath("../../model/src"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "CFA Multisignal Renewal"
copyright = "2024, CFA"
author = "CFA"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",  # numpydoc
    "sphinx.ext.duration",
    "sphinx.ext.githubpages",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinxcontrib.mermaid",
    "sphinx_autodoc_typehints",
]

# Simplifies printing of type hints
set_type_checking_flag = True
typehints_fully_qualified = False

# Avoid appending the full module name to the class name
add_module_names = False

templates_path = ["_templates"]
exclude_patterns = []

# Default deph for documentation
toc_deph = 2

# We don't want that explicitly
todo_include_todos = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "classic"
html_static_path = ["_static"]
html_css_files = ["msei.css"]
