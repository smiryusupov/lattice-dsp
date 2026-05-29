Installation
============

Editable development install
----------------------------

The package builds a small pybind11/C++ extension.  A typical development setup
is:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -U pip
   python -m pip install -e .[dev,examples,benchmark,docs]

Run tests:

.. code-block:: bash

   pytest -q

Build a wheel:

.. code-block:: bash

   python -m build

Build documentation:

.. code-block:: bash

   cd docs
   make html

Open ``docs/_build/html/index.html`` in a browser.

OpenMP
------

If OpenMP is available, the C++ extension enables parallel batch processing.
Check from Python:

.. code-block:: python

   import lattice_dsp as ld
   print(ld.HAS_OPENMP)

OpenMP is most useful for independent filter jobs, for example processing many
channels, many filters, or many frequency samples in matrix-lattice response
evaluation.

Optional dependencies
---------------------

``numpy`` is required.  ``scipy`` and ``matplotlib`` are optional and are used by
examples and benchmarks.  ``sphinx`` and ``myst-parser`` are optional and are
used only for documentation builds.
