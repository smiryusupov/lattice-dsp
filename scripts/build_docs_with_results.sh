#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Do not prepend the source checkout to PYTHONPATH.
# The examples should import the installed package so the compiled
# lattice_dsp._core extension is available on Read the Docs.
export MPLBACKEND="${MPLBACKEND:-Agg}"

# Sanity-check the installed package from outside the source tree.
(
  cd /tmp
  python -c "import lattice_dsp, lattice_dsp._core as core; print(lattice_dsp.__file__); print(core.__file__)"
)

python tools/generate_sphinx_gallery.py \
  --repo-root "$PWD" \
  --docs-dir docs \
  --run-examples \
  --run-benchmarks \
  --timeout "${LATTICE_DSP_TUTORIAL_TIMEOUT:-120}"

# The pages have already been generated above; avoid running all examples twice
# from the Sphinx configuration hook.
LATTICE_DSP_SKIP_GALLERY_GENERATION=1 sphinx-build -b html docs docs/_build/html

echo
echo "Open the rendered tutorials:"
echo "  docs/_build/html/examples/index.html"
echo "  docs/_build/html/benchmarks/index.html"
