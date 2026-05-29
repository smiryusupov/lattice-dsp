#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

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
