#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

pytest -q
pytest -q tests/test_release_trust_claims.py
./scripts/build_docs_with_results.sh
python tools/audit_public_api.py
python tools/audit_public_api.py --hide-deprecated
python -m build
python -m twine check dist/*
