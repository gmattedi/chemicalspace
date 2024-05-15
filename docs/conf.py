# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

import chemicalspace

sys.path.insert(0, os.path.abspath("../chemicalspace"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "chemicalspace"
copyright = "2023, Giulio Mattedi"
author = "Giulio Mattedi"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
]
autodoc_member_order = "bysource"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_context = {
    "display_github": True,
    "github_user": "gmattedi",
    "github_repo": "chemicalspace",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

version = chemicalspace.__version__


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None
    filename = info["module"].replace(".", "/")

    user = html_context["github_user"]
    repo = html_context["github_repo"]
    version = html_context["github_version"]

    return f"https://github.com/{user}/{repo}/blob/{version}/{filename}.py"
