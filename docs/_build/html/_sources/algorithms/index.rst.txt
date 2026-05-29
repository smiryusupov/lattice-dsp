Algorithms
==========

This section explains the algorithms behind ``lattice-dsp``.  Start with
:doc:`../theory/choosing_algorithms` if you are choosing an API by task.  Start with the
concept map if the terminology is unfamiliar: it shows how Schur recursions,
Szegő polynomials, Christoffel-Darboux identities, all-pass completions,
Hankel/Padé approximation, and matrix/MIMO lattice filters fit together.  The
emphasis is on stable recursive parameterizations and on multichannel
extensions that are uncommon in Python-first DSP packages.

The scalar path starts with reflection/PARCOR coefficients and lattice-ladder
IIR realizations.  The adaptive path uses those coordinates to demonstrate
recursive identification without treating denominator coefficients as free,
unconstrained parameters.  The H∞/minimax LMS tutorial adds a complementary
robust-filtering viewpoint: adaptive filters can be judged by worst-case
disturbance energy gain, not only average squared error.  The spectral path connects AR/Burg/Levinson tools to periodogram and Capon
diagnostics.  The model-reduction path explains why Hankel operators, Nehari
approximation, and AAK theory motivate the finite-section SISO reduction APIs.  The matrix/MIMO path explores matrix all-pass, paraunitary, and multichannel AR examples as tutorial and diagnostic material.

.. toctree::
   :maxdepth: 2

   concept_map
   lattice_filters
   adaptive_filters
   robust_lms_hinf
   ar_estimators
   spectral_diagnostics
   model_reduction_hankel_aak
   multichannel_ar
   matrix_lattice
   tangential_schur
   tangential_schur_verification
   mimo_verification_map
   streaming_blocks
