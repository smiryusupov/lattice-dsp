References and further reading
==============================

This page lists books and papers that are useful for understanding the algorithms
and examples in ``lattice-dsp``.  The package documentation does not attempt to
replace these sources; it explains how the package uses the ideas.

Scalar lattice filters, PARCOR, and AR modeling
-----------------------------------------------

.. [Makhoul1975] J. Makhoul, "Linear Prediction: A Tutorial Review,"
   *Proceedings of the IEEE*, 1975.  A classic tutorial on linear prediction,
   AR models, prediction-error filters, and reflection/PARCOR ideas.

.. [MarkelGray1976] J. D. Markel and A. H. Gray, Jr., *Linear Prediction of
   Speech*. Springer, 1976.  A classic source for speech linear prediction and
   PARCOR/lattice structures.

.. [Durbin1960] J. Durbin, "The Fitting of Time-Series Models," *Revue de
   l'Institut International de Statistique*, 1960.  Classic Toeplitz/Yule-Walker
   recursion background.

.. [Burg1967] J. P. Burg, "Maximum Entropy Spectral Analysis," presented at the
   37th Annual International SEG Meeting, 1967.  Original maximum-entropy/Burg
   spectral estimation work.

.. [UlrychBishop1975] T. J. Ulrych and T. N. Bishop, "Maximum entropy spectral
   analysis and autoregressive decomposition," *Reviews of Geophysics*, 1975.
   A widely cited review connecting maximum entropy spectral analysis and AR
   modeling.


Schur, Szegő, OPUC, and rational approximation background
-----------------------------------------------------------


.. [Pick1916] G. Pick, "Über die Beschränkungen analytischer Funktionen,
   welche durch vorgegebene Funktionswerte bewirkt werden," *Mathematische
   Annalen*, 1916.  Classical source for the interpolation condition now known
   as the Pick or Nevanlinna--Pick problem.

.. [Dym1989] H. Dym, *J Contractive Matrix Functions, Reproducing Kernel
   Hilbert Spaces and Interpolation*, American Mathematical Society, 1989.
   Reference for Schur-class interpolation, reproducing kernels, and contractive
   matrix-function viewpoints.

.. [Schur1917] I. Schur, "Über Potenzreihen, die im Innern des
   Einheitskreises beschränkt sind," *Journal für die reine und angewandte
   Mathematik*, 1917.  Classical source for the Schur algorithm and Schur
   functions.

.. [Szego1975] G. Szegő, *Orthogonal Polynomials*, 4th ed., American
   Mathematical Society, 1975.  Classical reference for orthogonal polynomial
   recurrences and Christoffel-Darboux identities.

.. [Simon2005] B. Simon, *Orthogonal Polynomials on the Unit Circle*, Parts 1
   and 2, American Mathematical Society, 2005.  Modern reference for OPUC,
   Verblunsky coefficients, Szegő recurrences, and unit-circle spectral theory.

.. [Bultheel2000] A. Bultheel and M. Van Barel, "Rational approximation in
   linear systems and control," *Journal of Computational and Applied
   Mathematics*, 2000.  Useful bridge between Schur algorithms, rational
   approximation, and control-oriented system theory.


.. [Potapov1955] V. P. Potapov, "The multiplicative structure of
   J-contractive matrix functions," *Trudy Moskov. Mat. Obshch.*, 4, 1955.
   Classical source for Potapov factorization and J-contractive matrix
   functions.

.. [BultheelMuller1998] A. Bultheel and K. Müller, "On several aspects of
   J-inner functions in Schur analysis," *Bulletin of the Belgian Mathematical
   Society - Simon Stevin*, 1998.  Survey-style discussion of J-inner functions,
   Potapov factorization, and generalized Schur interpolation context.

.. [HanzonOliviPeeters2010] B. Hanzon, M. Olivi, and R. L. M. Peeters,
   "Balanced realizations of discrete-time stable all-pass systems and the
   tangential Schur algorithm," arXiv:1012.3272, 2010.  Explains the connection
   between tangential Schur algorithms, RKHS ideas, Schur parameters, and
   balanced realizations of multivariable lossless systems.

.. [OliviMarmoratHanzonPeeters2003] M. Olivi, J.-P. Marmorat, B. Hanzon, and
   R. L. M. Peeters, "Schur parametrizations and balanced realizations of real
   discrete-time stable all-pass systems," CDC, 2003.  Real tangential Schur
   parametrization and balanced-realization context for stable all-pass systems.

.. [MarmoratHanzonOliviPeeters2002] J.-P. Marmorat, B. Hanzon, M. Olivi, and
   R. L. M. Peeters, "Matrix rational H2 approximation: a state-space approach
   using Schur parameters," CDC, 2002.  Uses Schur-parameter coordinates on
   the manifold of stable all-pass/lossless systems for matrix rational
   approximation.

.. [GonnetGuettelTrefethen2013] P. Gonnet, S. Güttel, and L. N. Trefethen,
   "Robust Padé Approximation via SVD," *SIAM Review*, 2013.  Practical
   numerical-linear-algebra view of Padé approximation and near-degenerate
   rational approximants.

.. [Trefethen2023] L. N. Trefethen, "Square blocks and equioscillation in the
   Padé, Walsh, and Carathéodory-Fejér tables," 2023.  Modern discussion of
   Padé/Walsh rational-approximation tables and degeneracy structure.

Adaptive filtering
------------------

This section includes the Widrow--Hoff LMS origin, Odile Macchi's
transmission-oriented LMS treatment, standard adaptive-filter texts, and the
H∞/minimax reinterpretation of LMS.

.. [WidrowHoff1960] B. Widrow and M. E. Hoff, Jr., "Adaptive Switching
   Circuits," 1960 IRE WESCON Convention Record, 1960.  Classical origin point
   for the LMS/delta-rule adaptive-update idea.

.. [Macchi1995] Odile Macchi, *Adaptive Processing: The Least Mean Squares
   Approach with Applications in Transmission*, Wiley, 1995.  Focused treatment
   of LMS-based adaptive processing, transmission/equalization applications,
   tracking, sign algorithms, and adaptive IIR context.

.. [Haykin2002] S. Haykin, *Adaptive Filter Theory*, 4th ed., Prentice Hall,
   2002.  Standard reference for LMS, NLMS, and RLS families.

.. [Haykin2014] S. Haykin, *Adaptive Filter Theory*, 5th ed., Pearson, 2014.
   Later edition with extensive examples and computer experiments for LMS and
   RLS adaptive filters.

.. [WidrowStearns1985] B. Widrow and S. D. Stearns, *Adaptive Signal Processing*,
   Prentice Hall, 1985.  Foundational adaptive signal processing text.

.. [Sayed2008] A. H. Sayed, *Adaptive Filters*, Wiley/IEEE Press, 2008.  Modern
   treatment of adaptive-filter theory and analysis.

.. [FarhangBoroujeny2013] B. Farhang-Boroujeny, *Adaptive Filters: Theory and
   Applications*, 2nd ed., Wiley, 2013.

.. [HassibiSayedKailath1996] B. Hassibi, A. H. Sayed, and T. Kailath,
   "H∞ Optimality of the LMS Algorithm," *IEEE Transactions on Signal
   Processing*, 44(2), 1996.  Shows that LMS and normalized LMS have exact
   H∞/minimax interpretations, explaining a robustness property that is not
   visible from the usual least-squares story.


Multichannel AR, block Toeplitz systems, and block Levinson recursions
----------------------------------------------------------------------

.. [Whittle1963] P. Whittle, "On the fitting of multivariate autoregressions,
   and the approximate canonical factorization of a spectral density matrix,"
   *Biometrika*, 1963.  Classical source for multivariate autoregression and
   spectral factorization ideas.

.. [WigginsRobinson1965] R. A. Wiggins and E. A. Robinson, "Recursive solution
   to the multichannel filtering problem," *Journal of Geophysical Research*,
   1965.  Early multichannel recursive filtering work often associated with the
   LWR family of algorithms.

.. [KailathSayedHassibi2000] T. Kailath, A. H. Sayed, and B. Hassibi,
   *Linear Estimation*, Prentice Hall, 2000.  Modern reference for structured
   covariance, Toeplitz systems, innovations, and recursive estimation.

Multirate DSP, paraunitary systems, and matrix all-pass filters
---------------------------------------------------------------

.. [Vaidyanathan1993] P. P. Vaidyanathan, *Multirate Systems and Filter Banks*,
   Prentice Hall, 1993.  Core reference for perfect reconstruction,
   paraunitary/orthogonal filter banks, polyphase systems, and lattice
   realizations.

.. [StrangNguyen1996] G. Strang and T. Nguyen, *Wavelets and Filter Banks*,
   Wellesley-Cambridge Press, 1996.  Accessible treatment of filter banks and
   wavelet connections.

.. [Mehta2023] P. Mehta, A. S. Bharath, K. Appaiah, R. Velmurugan, and
   D. Pal, "Lattice All-Pass Filter based Precoder Adaptation for MIMO Wireless
   Channels," arXiv:2302.11204, 2023.  Useful modern example of matrix-lattice
   all-pass filters as compact unitary MIMO representations.  The package uses
   this as motivation for general matrix-lattice DSP, not as a claim to be a 5G
   simulator.



Model reduction, Hankel operators, Nehari, and AAK
------------------------------------------------------------

.. [Nehari1957] Z. Nehari, "On bounded bilinear forms," *Annals of
   Mathematics*, 65(1), 1957.  Classical source for the Nehari approximation
   problem and its Hankel-operator interpretation.

.. [AdamjanArovKrein1971] V. M. Adamjan, D. Z. Arov, and M. G. Krein,
   "Analytic properties of Schmidt pairs for a Hankel operator and the
   generalized Schur-Takagi problem," *Mathematics of the USSR-Sbornik*, 1971.
   Foundational AAK work connecting Hankel singular structure and rational
   approximation.

.. [Glover1984] K. Glover, "All optimal Hankel-norm approximations of linear
   multivariable systems and their L∞-error bounds," *International Journal of
   Control*, 1984.  Classic control-theory source for Hankel-norm model
   reduction.

.. [Peller2003] V. V. Peller, *Hankel Operators and Their Applications*,
   Springer, 2003.  Comprehensive operator-theoretic reference for Hankel
   operators, Nehari-type problems, and AAK theory.

.. [Antoulas2005] A. C. Antoulas, *Approximation of Large-Scale Dynamical
   Systems*, SIAM, 2005.  Standard model-reduction reference, useful for
   balanced truncation, Hankel singular values, and system approximation.

Orthogonal convolutions and ML-adjacent unitary systems
----------------------------------------------------------

.. [Wang2020] J. Wang, Y. Chen, R. Chakraborty, and S. X. Yu, "Orthogonal
   Convolutional Neural Networks," CVPR, 2020.  Orthogonality constraints for
   convolutional layers and stability/regularization motivation.

.. [Achour2022] E. M. Achour, F. Malgouyres, and F. Mamalet, "Existence,
   Stability and Scalability of Orthogonal Convolutional Neural Networks,"
   *Journal of Machine Learning Research*, 2022.  Theoretical properties of
   orthogonal convolutional transforms.

.. [Su2022] J. Su, W. Byeon, and F. Huang, "Scaling-up Diverse Orthogonal
   Convolutional Networks by a Paraunitary Framework," ICML, 2022.  Connects
   orthogonal convolution layers in the spatial domain with paraunitary systems
   in the spectral domain.

How references map to package features
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Package feature
     - Main references
   * - Reflection/PARCOR coefficients and Schur/Pick motivation
     - [Makhoul1975]_, [MarkelGray1976]_, [Pick1916]_, [Schur1917]_, [Dym1989]_
   * - Tangential Schur and J-inner/Potapov diagnostics
     - [Dym1989]_, [Potapov1955]_, [BultheelMuller1998]_, [HanzonOliviPeeters2010]_, [OliviMarmoratHanzonPeeters2003]_, [MarmoratHanzonOliviPeeters2002]_
   * - Levinson-Durbin and Burg AR tools
     - [Durbin1960]_, [Burg1967]_, [UlrychBishop1975]_, [Szego1975]_, [Simon2005]_
   * - Multichannel/block Levinson AR tools
     - [Whittle1963]_, [WigginsRobinson1965]_, [KailathSayedHassibi2000]_
   * - NLMS/RLS adaptive filtering
     - [WidrowHoff1960]_, [Macchi1995]_, [Haykin2002]_, [WidrowStearns1985]_, [Sayed2008]_
   * - H∞/minimax LMS interpretation
     - [WidrowHoff1960]_, [Macchi1995]_, [HassibiSayedKailath1996]_
   * - Model reduction and Hankel/AAK theory
     - [Nehari1957]_, [AdamjanArovKrein1971]_, [Glover1984]_, [Peller2003]_, [Antoulas2005]_, [Bultheel2000]_, [GonnetGuettelTrefethen2013]_
   * - Paraunitary and matrix lattice systems
     - [Vaidyanathan1993]_, [StrangNguyen1996]_, [Mehta2023]_
   * - ML unitary convolution motivation
     - [Wang2020]_, [Achour2022]_, [Su2022]_
