# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'Pyomo'
copyright = '2017, Sandia National Laboratory'
author = 'Sandia National Laboratory'

release = '0.1'
version = '0.1.0'

import os
import sys
sys.path.insert(0, os.path.abspath('/Users/wangjialu/pyomo/pyomo/contrib/doe'))

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
