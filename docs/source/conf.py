# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import pathlib
import sys
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().joinpath('source').as_posix())
#sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().joinpath('source').joinpath('hjb').as_posix())


# -- Project information -----------------------------------------------------

project = 'Institute of Numerical Mathematics Code Documentation'
copyright = '2024, Institute of Numerical Mathematics, Ulm University'
author = 'Institute of Numerical Mathematics'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.viewcode',
]

show_authors = True # show code and section authors

# autosummary_generate = True  # Turn on sphinx.ext.autosummary

autoclass_content = 'both'  # integrate __init__ doc in class description 

todo_include_todos = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


latex_elements = {'preamble': r'\usepackage{amsmath}\n\usepackage{amssymb}'}


autodoc_default_options = {
    'members': True,
    'inherited-members': True,
    'member-order': 'groupwise',
    'undoc-members': True,
}

# -----------------------------------------------------------------------------
# Intersphinx configuration
# -----------------------------------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/dev', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('http://docs.scipy.org/doc/scipy/reference', None),
}
