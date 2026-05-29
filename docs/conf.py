"""Sphinx configuration for lattice-dsp."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "lattice-dsp"
author = "Shohruh Miryusupov"
copyright = f"{datetime.now().year}, {author}"
release = "0.1.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# Most documentation is reStructuredText. MyST is enabled so existing Markdown
# pages in docs/ can still be included or linked from the Sphinx tree.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "lattice-dsp documentation"
html_theme_options = {
    "description": "Efficient stable IIR lattice filters and matrix/MIMO lattice DSP",
    "github_user": "smiryusupov",
    "github_repo": "lattice-dsp",
    "fixed_sidebar": True,
    "show_powered_by": False,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Build docs even if optional examples dependencies are missing. The package
# itself should be installed for full API autodoc; on Read the Docs this happens
# through pip install -e .[docs].
autodoc_mock_imports = []

# Keep the build quiet while legacy Markdown notes remain outside the main Sphinx toctree.
suppress_warnings = ["toc.not_included", "ref.citation", "autodoc.duplicate_object"]


def setup(app):  # type: ignore[no-untyped-def]
    """Generate tutorial pages before Sphinx reads the toctree.

    By default this creates narrative pages without executing the examples.  Use
    ``scripts/build_docs_with_results.sh`` to execute examples/benchmarks and
    embed captured output, figures, and data.
    """

    def generate_tutorial_pages(app):  # type: ignore[no-untyped-def]
        if os.environ.get("LATTICE_DSP_SKIP_GALLERY_GENERATION") == "1":
            return
        sys.path.insert(0, str(ROOT))
        from tools.generate_sphinx_gallery import generate_gallery

        run_results = os.environ.get("LATTICE_DSP_GENERATE_TUTORIAL_RESULTS") == "1"
        timeout = float(os.environ.get("LATTICE_DSP_TUTORIAL_TIMEOUT", "120"))
        generate_gallery(
            repo_root=ROOT,
            docs_dir=Path(__file__).resolve().parent,
            run_examples=run_results,
            run_benchmarks=run_results,
            timeout=timeout,
        )

    app.connect("builder-inited", generate_tutorial_pages)
