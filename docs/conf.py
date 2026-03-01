# pyhqiv Sphinx configuration
# Build with: sphinx-build -b html docs docs/_build
# Read the Docs will use this automatically.

project = "pyhqiv"
copyright = "2026 Steven Ettinger Jr, HQIV Team"
author = "Steven Ettinger Jr, HQIV Team"
release = "0.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = ["_static"]
html_show_sourcelink = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
