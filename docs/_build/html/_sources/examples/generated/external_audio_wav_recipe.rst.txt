External WAV and eSpeak/eSpeak NG interoperability recipe
=========================================================

.. admonition:: Tutorial goal

   Bring an external WAV signal into lattice-dsp without adding audio I/O dependencies to the package.

.. note::

   New to the terminology? See the :doc:`lattice DSP concept map <../../algorithms/concept_map>` and the :doc:`causality/data-use guide <../../theory/causality_and_data_use>` for how online, offline, block, and MIMO examples should be read.

Context
-------

Speech synthesizers, recording tools, DAWs, and simulators can all write
WAV files.  This recipe uses the Python standard library for a minimal
PCM WAV boundary and leaves richer loaders such as librosa, soundfile,
or scipy.io.wavfile as user-side choices.

Key idea and equations
----------------------

The boundary is intentionally simple:

.. code-block:: text

   external tool -> WAV file -> NumPy array -> lattice-dsp filter/model

How to read the result
----------------------

The printed RMS values confirm that the signal crossed the WAV-to-array boundary and was processed by a lattice filter.

Run command
-----------

.. code-block:: bash

   python examples/external_audio_wav_recipe.py

Run status
----------

Return code: ``0``

Captured stdout
---------------

.. code-block:: text

   loaded sample rate: 16000
   loaded samples: 4000
   input RMS: 0.166148
   filtered RMS: 0.106986
   example WAV path: external_audio_recipe_input.wav

Source code
-----------

.. literalinclude:: ../../../examples/external_audio_wav_recipe.py
   :language: python
   :linenos:
