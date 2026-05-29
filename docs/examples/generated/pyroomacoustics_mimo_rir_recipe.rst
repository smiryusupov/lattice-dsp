Pyroomacoustics MIMO RIR interoperability recipe
================================================

.. admonition:: Tutorial goal

   Convert Pyroomacoustics-style room impulse responses to the MIMO Markov tensor shape used by lattice-dsp.

.. note::

   New to the terminology? See the :doc:`lattice DSP concept map <../../algorithms/concept_map>` and the :doc:`causality/data-use guide <../../theory/causality_and_data_use>` for how online, offline, block, and MIMO examples should be read.

Context
-------

Room-acoustics simulators can produce one impulse response for each
microphone/source pair.  This recipe keeps Pyroomacoustics outside the
package dependency tree while documenting the data convention used to
bring those paths into MIMO block-Hankel reduction.

Key idea and equations
----------------------

The mapping is

.. code-block:: text

   room.rir[microphone][source][tap]
        -> markov[tap, microphone, source]

How to read the result
----------------------

Use the printed shape and relative Markov-response error to verify the conversion and reduced MIMO model.

Run command
-----------

.. code-block:: bash

   python examples/pyroomacoustics_mimo_rir_recipe.py

Run status
----------

Return code: ``1``

Captured stdout
---------------

.. code-block:: text

   converted shape: (96, 2, 2)
   mapping: markov[tap, microphone, source]

Captured stderr
---------------

.. code-block:: text

   Traceback (most recent call last):
     File "examples/pyroomacoustics_mimo_rir_recipe.py", line 124, in <module>
       main()
     File "examples/pyroomacoustics_mimo_rir_recipe.py", line 111, in main
       print("reduced order:", result["order"])
                               ~~~~~~^^^^^^^^^
   KeyError: 'order'

Source code
-----------

.. literalinclude:: ../../../examples/pyroomacoustics_mimo_rir_recipe.py
   :language: python
   :linenos:
