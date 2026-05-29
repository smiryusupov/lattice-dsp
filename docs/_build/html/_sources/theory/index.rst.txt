Theory guide
============

This section gives the conceptual map needed to choose and interpret the
algorithms in ``lattice-dsp``.  The goal is not to replace a textbook treatment;
it is to identify the mathematical objects that appear in the examples and to
state the package scope clearly.

Start with the package-positioning page to understand the niche: stable IIR
lattice coordinates, model-reduction diagnostics, and SISO/MIMO examples in one
workflow.  Then use the algorithm-selection page when deciding which API or
tutorial to use.  Read the causality/data-use page before interpreting online versus offline
claims.  Read the filtering-relationships page if you want the basic DSP
vocabulary behind equalization, echo cancellation, prediction, adaptivity,
recursivity, minimum phase, and maximum phase.  Read the Hardy/Hankel/state-space
page before the finite-Hankel, Nehari/AAK, and MIMO reduction examples.  The interpolation/Schur page explains why Pick matrices,
inner/outer factors, Kronecker finite-rank structure, and lattice coordinates
appear together in the package motivation.  The tangential-Schur page gives the
finite right-tangential Pick and J-inner baseline used by the matrix/MIMO docs.

.. toctree::
   :maxdepth: 2

   package_positioning
   choosing_algorithms
   causality_and_data_use
   filtering_relationships
   hardy_hankel_state_space
   interpolation_schur_motivation
   tangential_schur
