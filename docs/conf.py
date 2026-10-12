"""Sphinx configuration for the pyrunoff documentation."""

import tomllib
from pathlib import Path

# Read the version from the source tree, not installed metadata, which may be
# stale (e.g. a leftover ``pyrunoff.egg-info`` in the working directory).
with open(Path(__file__).parents[1] / "pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)

project = "pyrunoff"
author = "Darshan Baral"
copyright = "Darshan Baral"
release = pyproject["project"]["version"]
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]

# Show members in source order, with type hints in the descriptions rather
# than the signatures, which stay readable for the long dataclass ones.
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {"members": True, "show-inheritance": True}
autoclass_content = "class"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

# Give Markdown headings anchors so pages can link to sections.
myst_heading_anchors = 3

html_theme = "furo"
# The sidebar shows the title, so it carries the version.
html_title = f"pyrunoff {release}"

# No "view source" button for the page's own reST/Markdown; the API's
# [source] links (viewcode) cover the code.
html_show_sourcelink = False

repository_url = "https://github.com/darshanbaral/pyrunoff"
html_theme_options = {
    # Repository URL as plain text in the footer.
    "footer_icons": [
        {"name": "GitHub", "url": repository_url, "html": "GitHub", "class": ""},
    ],
}
