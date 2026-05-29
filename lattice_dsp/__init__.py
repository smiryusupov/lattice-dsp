"""Stable lattice/lattice-ladder DSP filters backed by C++.

`lattice-dsp` is a stability-first IIR toolkit: reflection/PARCOR
parameterization, lattice-ladder realizations, adaptive recursive filtering,
OpenMP batch processing, and small dependency-light examples.

It is not a production acoustic echo canceller. Echo helpers are retained as
synthetic ERLE/system-identification examples, not product claims.
"""

__version__ = "0.1.0"

from .applications import (
    EchoCancellationResult,
    HybridEchoCanceller,
    ResidualAttenuator,
    SpectralResidualSuppressor,
    make_residual_processor_from_model,
    residual_attenuator,
    spectral_residual_suppressor,
)
from .metrics import (
    EchoMetrics,
    echo_metrics,
    erle_db,
    improvement_db,
    mse,
    power,
    power_db,
    segmental_erle_db,
)
from .synthetic import EchoProblem, generate_echo_problem, iir_filter
from .tuning import tune_reflection_update_period
from .streaming import AdaptiveBlockProcessor, BlockProcessor, BlockResult

from .multichannel_ar import (
    MIMOLatticePredictor,
    MultichannelARResult,
    block_levinson_durbin,
    causal_mimo_lattice_predict,
    block_toeplitz_from_autocorrelation,
    companion_spectral_radius,
    matrix_ar_frequency_response,
    multichannel_autocorrelation,
    multichannel_prediction_error,
    solve_block_yule_walker_direct,
)

from .nehari import (
    FiniteNehariCandidateCriteria,
    finite_aak_siso_certificate,
    finite_aak_reduce_iir,
    finite_aak_reduce_tail,
    finite_hankel_matrix_from_tail,
    finite_nehari_rational_candidates,
    fit_rational_tail,
    rational_tail_response,
    relative_error,
    select_finite_nehari_candidate,
)

from .matrix_lattice import (
    MatrixLatticeAllPass,
    OnlineMatrixLatticeAllPass,
    contractive_matrix_from_raw,
    experimental_mimo_state_space_to_matrix_lattice,
    fit_static_matrix_gains,
    is_matrix_reflection_stable,
    matrix_lattice_scaffold_from_markov,
    matrix_lattice_stage_blocks,
    matrix_lattice_finite_adjoint,
    matrix_lattice_impulse_response_convolution,
    online_matrix_lattice_allpass_process,
    matrix_spectral_norm,
    mimo_state_space_frequency_response,
    polar_factor_response,
    project_matrix_reflection,
    psd_matrix_sqrt,
    unitary_polar_factor,
)

from .tangential_schur import (
    PotapovProduct,
    RightTangentialSchurData,
    TangentialPotapovFactor,
    constant_schur_solution,
    disk_blaschke,
    elementary_potapov_factor,
    is_pick_positive_semidefinite,
    is_tangential_schur_solvable,
    j_signature,
    j_unitarity_residual,
    max_tangential_residual,
    pick_matrix_eigenvalues,
    potapov_product_from_rank_one_data,
    right_tangential_pick_matrix,
    tangential_interpolation_residual,
)

from ._core import (  # type: ignore[attr-defined]
    HAS_OPENMP,
    AdaptiveLatticeLadderNLMS,
    AdaptiveNotch,
    LatticeIIR,
    LatticeLadderIIR,
    LatticeLadderNLMS,
    LatticeLadderRLS,
    adaptive_process_batch,
    autocorrelation,
    burg_denominator,
    burg_reflection,
    bounded_reflection_from_raw,
    denominator_raw_jacobian,
    denominator_raw_jacobian_finite_difference,
    denominator_reflection_jacobian,
    denominator_to_reflection,
    finite_hankel_reduce_impulse,
    finite_hankel_reduce_iir,
    finite_hankel_reduce_mimo,
    finite_nehari_approximate_tail,
    hankel_singular_values,
    iir_impulse_response,
    mimo_state_space_markov_response,
    mimo_state_space_process_batch,
    ladder_to_numerator,
    levinson_durbin_denominator,
    levinson_durbin_error,
    levinson_durbin_reflection,
    matrix_lattice_frequency_response,
    numerator_to_ladder,
    process_batch,
    reflection_to_denominator,
    rls_process_batch,
)


def finite_hankel_aak_reduce_impulse(*args, **kwargs):
    """Deprecated alias for :func:`finite_hankel_reduce_impulse`.

    The implementation is a finite-Hankel/Ho--Kalman baseline, not an
    exact infinite-dimensional AAK solver.  New code should call
    ``finite_hankel_reduce_impulse``.
    """

    import warnings

    warnings.warn(
        "finite_hankel_aak_reduce_impulse is deprecated; use "
        "finite_hankel_reduce_impulse for the finite-Hankel/Ho--Kalman baseline",
        DeprecationWarning,
        stacklevel=2,
    )
    return finite_hankel_reduce_impulse(*args, **kwargs)


def finite_hankel_aak_reduce_iir(*args, **kwargs):
    """Deprecated alias for :func:`finite_hankel_reduce_iir`.

    The implementation is a finite-Hankel/Ho--Kalman baseline, not an
    exact infinite-dimensional AAK solver.  New code should call
    ``finite_hankel_reduce_iir``.
    """

    import warnings

    warnings.warn(
        "finite_hankel_aak_reduce_iir is deprecated; use "
        "finite_hankel_reduce_iir for the finite-Hankel/Ho--Kalman baseline",
        DeprecationWarning,
        stacklevel=2,
    )
    return finite_hankel_reduce_iir(*args, **kwargs)


__all__ = [
    "__version__",
    "AdaptiveLatticeLadderNLMS",
    "BlockResult",
    "BlockProcessor",
    "AdaptiveBlockProcessor",
    "AdaptiveNotch",
    "EchoCancellationResult",
    "EchoMetrics",
    "EchoProblem",
    "HAS_OPENMP",
    "HybridEchoCanceller",
    "LatticeIIR",
    "LatticeLadderIIR",
    "LatticeLadderNLMS",
    "LatticeLadderRLS",
    "MatrixLatticeAllPass",
    "OnlineMatrixLatticeAllPass",
    "MIMOLatticePredictor",
    "MultichannelARResult",
    "block_levinson_durbin",
    "causal_mimo_lattice_predict",
    "block_toeplitz_from_autocorrelation",
    "companion_spectral_radius",
    "matrix_ar_frequency_response",
    "multichannel_autocorrelation",
    "multichannel_prediction_error",
    "solve_block_yule_walker_direct",
    "PotapovProduct",
    "RightTangentialSchurData",
    "TangentialPotapovFactor",
    "constant_schur_solution",
    "disk_blaschke",
    "elementary_potapov_factor",
    "is_pick_positive_semidefinite",
    "is_tangential_schur_solvable",
    "j_signature",
    "j_unitarity_residual",
    "max_tangential_residual",
    "pick_matrix_eigenvalues",
    "potapov_product_from_rank_one_data",
    "right_tangential_pick_matrix",
    "tangential_interpolation_residual",
    "contractive_matrix_from_raw",
    "experimental_mimo_state_space_to_matrix_lattice",
    "fit_static_matrix_gains",
    "is_matrix_reflection_stable",
    "matrix_lattice_frequency_response",
    "matrix_lattice_scaffold_from_markov",
    "matrix_lattice_stage_blocks",
    "matrix_lattice_finite_adjoint",
    "matrix_lattice_impulse_response_convolution",
    "online_matrix_lattice_allpass_process",
    "matrix_spectral_norm",
    "mimo_state_space_frequency_response",
    "polar_factor_response",
    "ResidualAttenuator",
    "SpectralResidualSuppressor",
    "adaptive_process_batch",
    "autocorrelation",
    "burg_denominator",
    "burg_reflection",
    "bounded_reflection_from_raw",
    "denominator_raw_jacobian",
    "denominator_raw_jacobian_finite_difference",
    "denominator_reflection_jacobian",
    "denominator_to_reflection",
    "finite_hankel_reduce_impulse",
    "finite_hankel_reduce_iir",
    "finite_hankel_reduce_mimo",
    "finite_nehari_approximate_tail",
    "finite_aak_siso_certificate",
    "finite_aak_reduce_iir",
    "finite_aak_reduce_tail",
    "finite_hankel_matrix_from_tail",
    "FiniteNehariCandidateCriteria",
    "finite_nehari_rational_candidates",
    "fit_rational_tail",
    "rational_tail_response",
    "relative_error",
    "select_finite_nehari_candidate",
    "finite_hankel_aak_reduce_impulse",
    "finite_hankel_aak_reduce_iir",
    "hankel_singular_values",
    "iir_impulse_response",
    "mimo_state_space_markov_response",
    "mimo_state_space_process_batch",
    "echo_metrics",
    "erle_db",
    "generate_echo_problem",
    "iir_filter",
    "improvement_db",
    "ladder_to_numerator",
    "levinson_durbin_denominator",
    "levinson_durbin_error",
    "levinson_durbin_reflection",
    "make_residual_processor_from_model",
    "mse",
    "numerator_to_ladder",
    "power",
    "power_db",
    "process_batch",
    "project_matrix_reflection",
    "psd_matrix_sqrt",
    "reflection_to_denominator",
    "rls_process_batch",
    "residual_attenuator",
    "segmental_erle_db",
    "unitary_polar_factor",
    "spectral_residual_suppressor",
    "tune_reflection_update_period",
]
