# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
# import pathlib
# import sys
# sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

import os
import sys
sys.path.insert(0, os.path.abspath('../../model/src'))


project = 'CFA Multisignal Renewal'
copyright = '2024, CFA'
author = 'CFA'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.duration',
    'sphinx.ext.githubpages',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'classic'
html_static_path = ['_static']
