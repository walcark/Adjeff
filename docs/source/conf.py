# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------

project = "Adjeff"
copyright = "2026, Walcarius Kévin"
author = "Walcarius Kévin"
release = "v0.6.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "nbsphinx",
]

nbsphinx_kernel_name = "python3"
nbsphinx_execute = "never"

# Napoleon (NumPy / Google docstrings)
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True

# Autodoc
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
add_module_names = False

# Intersphinx — cross-references to external libraries
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

templates_path = ["_templates"]
exclude_patterns = []

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "pygments_light_style": "default",
    "pygments_dark_style": "monokai",
    "navbar_center": ["navbar-nav"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/walcark/adjeff",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ],
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]
