Streaming and block processing
==============================

Why block APIs matter
---------------------

DSP filters are often run in blocks in real systems, while examples and tests
are often written as one-shot array operations.  ``lattice-dsp`` provides helper
classes to make the block path explicit and testable.

``BlockProcessor``
------------------

``BlockProcessor`` wraps a stateful processor and feeds an input signal in fixed
chunks.  It returns a ``BlockResult`` containing the concatenated output and
per-block metadata.

``AdaptiveBlockProcessor``
--------------------------

``AdaptiveBlockProcessor`` is similar but accepts desired/reference signals for
adaptive filters.  It is used to test that streaming adaptation remains stable
and close to one-shot adaptation.

Relevant APIs
-------------

* ``BlockResult``
* ``BlockProcessor``
* ``AdaptiveBlockProcessor``

Examples
--------

* ``examples/streaming_block_processing.py``
