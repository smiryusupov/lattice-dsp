# Examples tutorials

The public-facing examples are now rendered as Sphinx tutorial pages.  Build the
local HTML gallery with:

```bash
python -m pip install -e '.[dev,docs,examples,benchmark]'
./scripts/build_docs_with_results.sh
```

Then open:

```bash
xdg-open docs/_build/html/examples/index.html
```

Each generated page contains:

- context and motivation;
- the key equation or algorithmic idea;
- how to interpret the output;
- the exact command;
- captured stdout/stderr, only when non-empty;
- generated figures and data downloads;
- the source code.

The spectral diagnostics tutorials include periodograms, Levinson/Burg AR spectra,
and Capon/MVDR spectra.  Generated figures and CSV files are written under the
Sphinx generated artifact directories or `reports/`, not the repository root.
