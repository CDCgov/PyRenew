# numpydoc ignore=GL08

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Multisignal Renewal Models for Epi Inference"
copyright = "2024, CDC CFA"
author = "CDC's Center for Forecasting Analytics"
release = "0.1.0"

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
    "myst_parser",
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

# Numpydocs checks
# numpydoc_validation_checks = {"SA01", "EX01"}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/CDCgov/PyRenew",
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "repository_branch": "main",
    "path_to_docs": "docs/source",
    "use_download_button": True,
}

html_static_path = ["_static"]
html_css_files = ["msei.css"]

html_sidebars = {
    "**": [
        "navbar-logo.html",
        "search-field.html",
        "sbt-sidebar-nav.html",
    ]
}
master_doc = "general/ctoc"
