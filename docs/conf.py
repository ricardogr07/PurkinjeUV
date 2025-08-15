# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os

project = 'purkinje-uv'
copyright = '2025, Ricardo García Ramírez'
author = 'Ricardo García Ramírez'
release = '0.2.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinxcontrib.mermaid",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinxcontrib.bibtex",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
bibtex_bibfiles = ["references.bib"]
autodoc_mock_imports = ["cupy"]
autosummary_generate = True
numfig = True
bibtex_default_style = "unsrt"
autodoc_typehints = "description"
# Don’t warn when the same object is documented multiple times (we control indexing via :noindex:)
suppress_warnings = ["autosectionlabel.*", "ref.doc", "ref.python", "duplicate.object"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = []

# --- sphinx-multiversion (SMV) minimal config ---
extensions.append("sphinx_multiversion")


# Build: main + semver tags (vX.Y.Z). Can be overridden via env in CI.
smv_branch_whitelist = os.getenv("SMV_BRANCH_WHITELIST", r"^(main)$")
smv_tag_whitelist = os.getenv("SMV_TAG_WHITELIST", r"^v\d+\.\d+\.\d+$")

# Put each version under /<version>/ (e.g., /main/, /v0.2.2/)
smv_outputdir_format = "{ref.name}"

# --- Furo version dropdown (sidebar template) ---
# Ensure templates path includes "_templates"
try:
    templates_path
except NameError:
    templates_path = []
if "_templates" not in templates_path:
    templates_path.append("_templates")

# Insert our version switcher into the default Furo sidebar
default_sidebar = [
    "sidebar/brand.html",
    "sidebar/search.html",
    "version-switcher.html",  # custom file
    "sidebar/navigation.html",
    "sidebar/ethical-ads.html",
]
html_sidebars = globals().get("html_sidebars", {}) or {}
html_sidebars.setdefault("**", default_sidebar)

# Force a simple title without the version
html_title = f"{project} documentation"

# Shorten the browser tab title
html_short_title = f"{project}"
