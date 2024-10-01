# numpydoc ignore=GL08

import os
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Multisignal Renewal Models for Epi Inference"
copyright = "2024, CDC CFA"
author = "CDC Center for Forecasting Analytics"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

sys.path.insert(0, os.path.abspath("../../pyrenew"))

# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",  # numpydoc
    "sphinx.ext.duration",
    "sphinx.ext.githubpages",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinxcontrib.katex",
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
html_css_files = ["pyrenew.css"]

html_sidebars = {
    "**": [
        "navbar-logo.html",
        "search-field.html",
        "sbt-sidebar-nav.html",
    ]
}

master_doc = "general/ctoc"

myst_fence_as_directive = ["mermaid"]
myst_enable_extensions = ["amsmath", "dollarmath"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "numpyro": ("https://num.pyro.ai/en/latest/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "polars": ("https://docs.pola.rs/api/python/stable/", None),
}

napoleon_preprocess_types = True
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_type_aliases = {
    "ArrayLike": ":obj:`ArrayLike <jax.typing.ArrayLike>`",
    "RandomVariable": ":class:`RandomVariable <pyrenew.metaclass.52RandomVariable>`",
    "Any": ":obj:`Any <typing.Any>`",
}
napoleon_type_aliases = autodoc_type_aliases
