# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os

# -- Project information -----------------------------------------------------
project = "purkinje-uv"
author = "Ricardo García Ramírez"
copyright = "2025, Ricardo García Ramírez"
release = "0.3.0"

# -- General configuration ---------------------------------------------------
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
    "nbsphinx",
    "nbsphinx_link",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_multiversion",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
autodoc_mock_imports = ["cupy"]
autosummary_generate = True
autodoc_typehints = "description"
numfig = True

# Don’t warn when the same object is documented multiple times
suppress_warnings = ["autosectionlabel.*", "ref.doc", "ref.python", "duplicate.object"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = []

# --- Notebooks (nbsphinx) ---
nbsphinx_execute = "never"
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# --- sphinx-multiversion (SMV) config: build main + semver tags, prefer remote refs ---
smv_branch_whitelist = os.getenv("SMV_BRANCH_WHITELIST", r"^(main)$")
smv_tag_whitelist = os.getenv("SMV_TAG_WHITELIST", r"^v\d+\.\d+\.\d+$")
smv_remote_whitelist = r"^origin$"
smv_prefer_remote_refs = True  # avoid output dir conflict for main vs origin/main
smv_released_pattern = r"^refs/tags/.*$"
smv_outputdir_format = "{ref.name}"  # /main/, /v0.3.0/, etc.
smv_rename_latest_version = ("main", "latest")

# --- Sidebar (Furo) with version switcher ---
default_sidebar = [
    "sidebar/brand.html",
    "sidebar/search.html",
    "version-switcher.html",  # show SMV version dropdown
    "sidebar/navigation.html",
    "sidebar/ethical-ads.html",
]
html_sidebars = {"**": default_sidebar}

# Titles
html_title = f"{project} documentation"
html_short_title = f"{project}"

# --- Colab badge for every nbsphinx-rendered notebook (works with .nblink) ---
html_context = {
    "github_user": "ricardogr07",
    "github_repo": "purkinje-uv",
    "github_version": "main",
}

nbsphinx_prolog = r"""
.. raw:: html

   <div style="margin: 0 0 1rem 0;">
     <a href="https://colab.research.google.com/github/{{ env.config.html_context.github_user }}/{{ env.config.html_context.github_repo }}/blob/{{ env.config.html_context.github_version | default('main') }}/{{ env.doc2path(env.docname, base=None) | replace('.nblink', '.ipynb') }}" target="_blank" rel="noopener noreferrer">
       <img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
     </a>
   </div>
"""
