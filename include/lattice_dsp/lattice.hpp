// Copyright 2026 Shohruh Miryusupov
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace lattice_dsp {


// Finite-section Hankel SISO model-reduction helpers.
//
// These routines are intentionally finite-dimensional numerical tools: they
// build truncated Hankel matrices from an impulse response, compute Hankel
// singular-value diagnostics, and construct a reduced Ho-Kalman realization
// from the leading finite Hankel factors. They are useful approximations and
// diagnostics for Hankel/Nehari/AAK-style model reduction, but they do not claim to
// solve the infinite-dimensional optimal Nehari or AAK problem exactly.
struct FiniteHankelReduction {
  std::vector<double> numerator;
  std::vector<double> denominator;
  std::vector<double> reflection;
  std::vector<double> hankel_singular_values;
  double retained_hankel_energy{};
  double relative_impulse_error{};
  bool stable{};
  std::string method;
};

std::vector<double> iir_impulse_response(const std::vector<double>& denominator,
                                         const std::vector<double>& numerator,
                                         std::size_t n_samples);

std::vector<double> hankel_singular_values(const std::vector<double>& impulse_response,
                                           std::size_t rows,
                                           std::size_t cols,
                                           std::size_t offset = 1);



// Finite-section Nehari/AAK teaching helper for SISO anticausal tails.
//
// Given a finite tail gamma_1, gamma_2, ... this routine builds the Hankel
// matrix H[i,j] = gamma_{i+j+1}, computes the best unconstrained rank-r
// approximation, then projects that approximation back onto the finite Hankel
// subspace by anti-diagonal averaging. It is a finite-dimensional baseline and
// diagnostic, not an exact infinite-dimensional Nehari or AAK solver.
struct FiniteNehariApproximation {
  std::vector<double> approximated_tail;
  std::vector<double> hankel_singular_values;
  double sigma_next{};
  double unconstrained_hankel_error{};
  double hankelized_hankel_error{};
  double relative_tail_error{};
  std::string method;
  std::size_t rank{};
  std::size_t rows{};
  std::size_t cols{};
};

FiniteNehariApproximation finite_nehari_approximate_tail(
    const std::vector<double>& anticausal_tail,
    std::size_t rank,
    std::size_t rows,
    std::size_t cols,
    double regularization = 1e-12);

FiniteHankelReduction finite_hankel_reduce_impulse(
    const std::vector<double>& impulse_response,
    std::size_t reduced_order,
    std::size_t rows,
    std::size_t cols,
    double regularization = 1e-12);

FiniteHankelReduction finite_hankel_reduce_iir(
    const std::vector<double>& reflection,
    const std::vector<double>& numerator,
    std::size_t reduced_order,
    std::size_t n_impulse = 512,
    std::size_t rows = 64,
    std::size_t cols = 64,
    double regularization = 1e-12);

// Backward-compatible aliases retained for one release cycle. Prefer the
// shorter finite_hankel_reduce_* names: these routines are finite-section
// Hankel/Ho-Kalman reducers, not exact infinite-dimensional AAK solvers.
FiniteHankelReduction finite_hankel_aak_reduce_impulse(
    const std::vector<double>& impulse_response,
    std::size_t reduced_order,
    std::size_t rows,
    std::size_t cols,
    double regularization = 1e-12);

FiniteHankelReduction finite_hankel_aak_reduce_iir(
    const std::vector<double>& reflection,
    const std::vector<double>& numerator,
    std::size_t reduced_order,
    std::size_t n_impulse = 512,
    std::size_t rows = 64,
    std::size_t cols = 64,
    double regularization = 1e-12);


// Finite-section block-Hankel MIMO model-reduction helpers.
//
// MIMO systems are represented by Markov parameters M_k with shape
// [samples, outputs, inputs]. The reduction returns a state-space realization
// A, B, C, D because MIMO transfer functions do not have a single scalar
// numerator/denominator representation. This is a finite-Hankel/Ho-Kalman
// baseline, not an exact matrix Nehari/AAK solver.
struct FiniteHankelMimoReduction {
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> c;
  std::vector<double> d;
  std::vector<double> hankel_singular_values;
  double retained_hankel_energy{};
  double relative_markov_error{};
  bool stable{};
  std::string method;
  std::size_t state_order{};
  std::size_t n_outputs{};
  std::size_t n_inputs{};
};

std::vector<double> mimo_state_space_markov_response(
    const std::vector<double>& a,
    const std::vector<double>& b,
    const std::vector<double>& c,
    const std::vector<double>& d,
    std::size_t state_order,
    std::size_t n_outputs,
    std::size_t n_inputs,
    std::size_t n_samples);

FiniteHankelMimoReduction finite_hankel_reduce_mimo(
    const std::vector<double>& markov_parameters,
    std::size_t n_samples,
    std::size_t n_outputs,
    std::size_t n_inputs,
    std::size_t reduced_order,
    std::size_t block_rows,
    std::size_t block_cols,
    double regularization = 1e-12);

// Convert stable reflection/PARCOR coefficients to an all-pole denominator
// polynomial A(z) = 1 + a1 z^-1 + ... + aN z^-N.
std::vector<double> reflection_to_denominator(const std::vector<double>& reflection);

// Inverse Schur/step-down recursion. The denominator can be monic or non-monic;
// the returned reflection coefficients satisfy |k_i| < 1 when the polynomial is stable.
std::vector<double> denominator_to_reflection(const std::vector<double>& denominator,
                                              double stability_tol = 1e-12);

// Map unconstrained parameters to stable reflection coefficients using tanh.
std::vector<double> bounded_reflection_from_raw(const std::vector<double>& raw,
                                                double margin = 1e-6);

// Analytic Jacobian utilities for the Schur/step-up recursion. The returned
// matrix has shape [order][order], where row j contains derivatives of
// denominator coefficients a_1...a_N with respect to coefficient j.
std::vector<std::vector<double>> denominator_reflection_jacobian(
    const std::vector<double>& reflection);
std::vector<std::vector<double>> denominator_raw_jacobian(
    const std::vector<double>& raw, double margin = 1e-6);

// Finite-difference Jacobian retained as a slow debug/reference helper.
std::vector<std::vector<double>> denominator_raw_jacobian_finite_difference(
    const std::vector<double>& raw, double margin = 1e-6, double step_scale = 1e-6);

// Ladder/direct numerator conversion for the synthesis lattice-ladder structure.
// The denominator is implied by the reflection coefficients.
std::vector<double> ladder_to_numerator(const std::vector<double>& reflection,
                                        const std::vector<double>& ladder);
std::vector<double> numerator_to_ladder(const std::vector<double>& reflection,
                                        const std::vector<double>& numerator);

// True synthesis lattice-ladder realization.
//
// Given reflection coefficients k_1...k_M, the recursive lattice implements a
// stable denominator A_M(z). Ladder coefficients c_0...c_M combine the backward
// lattice variables to realize B(z) / A_M(z), where B is obtained by
// ladder_to_numerator(reflection, ladder). This is the first non-direct-form
// realization in the package.
class LatticeLadderIIR {
public:
  LatticeLadderIIR(std::vector<double> reflection, std::vector<double> ladder);

  std::size_t order() const noexcept { return order_; }
  void reset(double value = 0.0);

  double process_sample(double x);
  std::vector<double> process_block(const std::vector<double>& x);

  const std::vector<double>& reflection() const noexcept { return reflection_; }
  const std::vector<double>& ladder() const noexcept { return ladder_; }
  const std::vector<double>& taps() const noexcept { return ladder_; }
  const std::vector<double>& denominator() const noexcept { return denominator_; }
  const std::vector<double>& numerator() const noexcept { return numerator_; }
  const std::vector<double>& state() const noexcept { return state_; }
  const std::vector<double>& last_forward() const noexcept { return last_forward_; }
  const std::vector<double>& last_backward() const noexcept { return last_backward_; }

  void set_reflection(const std::vector<double>& reflection);
  void set_ladder(const std::vector<double>& ladder);
  void set_taps(const std::vector<double>& ladder) { set_ladder(ladder); }

private:
  std::size_t order_{};
  std::vector<double> reflection_;
  std::vector<double> ladder_;
  std::vector<double> denominator_;
  std::vector<double> numerator_;
  std::vector<double> state_;         // delayed backward variables b_0...b_{M-1}
  std::vector<double> last_forward_;  // current f_0...f_M
  std::vector<double> last_backward_; // current b_0...b_M

  void validate() const;
  void update_polynomials_and_state();
};

// Direct-form reference IIR driven by a denominator generated from reflection
// coefficients. This keeps the public API stability-first while giving a
// simple, testable processing path for the research-oriented alpha toolkit.
class LatticeIIR {
public:
  LatticeIIR(std::vector<double> reflection, std::vector<double> numerator);

  std::size_t order() const noexcept { return order_; }
  void reset(double value = 0.0);

  double process_sample(double x);
  std::vector<double> process_block(const std::vector<double>& x);

  const std::vector<double>& reflection() const noexcept { return reflection_; }
  const std::vector<double>& denominator() const noexcept { return denominator_; }
  const std::vector<double>& numerator() const noexcept { return numerator_; }
  const std::vector<double>& state() const noexcept { return state_; }
  const std::vector<double>& last_basis() const noexcept { return last_basis_; }

  void set_reflection(const std::vector<double>& reflection);
  void set_reflection_preserve_state(const std::vector<double>& reflection);
  void set_numerator(const std::vector<double>& numerator);
  void add_scaled_to_numerator(const std::vector<double>& direction, double scale);

private:
  std::size_t order_{};
  std::vector<double> reflection_;
  std::vector<double> denominator_;
  std::vector<double> numerator_;
  std::vector<double> state_;       // DF-II transposed filter states.
  std::vector<double> x_history_;   // x[n-1], x[n-2], ... for numerator basis generation.
  std::vector<double> basis_state_; // (order+1) all-pole DF-II states, flattened.
  std::vector<double> last_basis_;  // exact dy/db_i for last sample, used by adaptation.

  void validate() const;
  void update_denominator(bool reset_state = true);
};

// Experimental filtered-gradient adaptation of both numerator taps and stable
// reflection coefficients. Reflection coefficients are never updated directly;
// unconstrained raw parameters are mapped through bounded_reflection_from_raw(),
// so |k_i| remains below 1 - margin throughout adaptation.
class AdaptiveLatticeLadderNLMS {
public:
  AdaptiveLatticeLadderNLMS(std::vector<double> initial_reflection,
                            std::vector<double> initial_numerator,
                            double mu_taps = 0.05,
                            double mu_reflection = 0.001,
                            double epsilon = 1e-8,
                            double margin = 1e-4,
                            bool freeze_reflection = false,
                            std::string gradient_mode = "analytic",
                            std::size_t reflection_update_period = 1,
                            bool scale_reflection_mu_by_period = false);

  // Returns {output, error} for desired sample d.
  std::pair<double, double> adapt_sample(double x, double d);
  std::vector<double> adapt_block(const std::vector<double>& x,
                                  const std::vector<double>& desired);

  void reset(double value = 0.0);

  const LatticeIIR& filter() const noexcept { return filter_; }
  const std::vector<double>& numerator() const noexcept { return filter_.numerator(); }
  const std::vector<double>& taps() const noexcept { return filter_.numerator(); }
  std::vector<double> ladder() const;
  const std::vector<double>& reflection() const noexcept { return filter_.reflection(); }
  const std::vector<double>& raw_reflection() const noexcept { return raw_reflection_; }
  const std::vector<double>& denominator() const noexcept { return filter_.denominator(); }
  const std::vector<double>& last_tap_gradient() const noexcept { return last_tap_gradient_; }
  const std::vector<double>& last_reflection_gradient() const noexcept { return last_reflection_gradient_; }
  const std::vector<double>& last_raw_gradient() const noexcept { return last_raw_gradient_; }

  double mu_taps() const noexcept { return mu_taps_; }
  double mu_reflection() const noexcept { return mu_reflection_; }
  double margin() const noexcept { return margin_; }
  bool freeze_reflection() const noexcept { return freeze_reflection_; }
  std::size_t reflection_update_period() const noexcept { return reflection_update_period_; }
  bool scale_reflection_mu_by_period() const noexcept { return scale_reflection_mu_by_period_; }
  std::string gradient_mode() const;

  void set_mu_taps(double mu);
  void set_mu_reflection(double mu);
  void set_freeze_reflection(bool freeze) noexcept { freeze_reflection_ = freeze; }
  void set_reflection_update_period(std::size_t period);
  void set_scale_reflection_mu_by_period(bool scale) noexcept { scale_reflection_mu_by_period_ = scale; }
  void set_gradient_mode(const std::string& mode);

private:
  LatticeIIR filter_;
  std::vector<double> raw_reflection_;
  double mu_taps_;
  double mu_reflection_;
  double epsilon_;
  double margin_;
  bool freeze_reflection_;
  bool use_finite_difference_gradient_;
  std::size_t reflection_update_period_;
  bool scale_reflection_mu_by_period_;
  std::size_t adaptation_step_;

  std::vector<double> y_history_;
  std::vector<double> denominator_basis_state_;
  std::vector<double> denominator_gradient_;
  std::vector<double> last_tap_gradient_;
  std::vector<double> last_reflection_gradient_;
  std::vector<double> last_raw_gradient_;

  void validate_common() const;
  std::vector<double> reflection_from_raw() const;
  void update_denominator_gradients(bool compute_parameter_gradient);
  void update_output_history(double y);
};

// NLMS adaptation of numerator/ladder coefficients for a fixed stable
// reflection-parameterized denominator. This is useful for system
// identification and equalization prototypes where denominator stability must
// remain guaranteed during adaptation.
class LatticeLadderNLMS {
public:
  LatticeLadderNLMS(std::vector<double> reflection,
                    std::vector<double> initial_numerator,
                    double mu = 0.1,
                    double epsilon = 1e-8);

  // Returns {output, error} for desired sample d.
  std::pair<double, double> adapt_sample(double x, double d);

  // Returns error signal.
  std::vector<double> adapt_block(const std::vector<double>& x,
                                  const std::vector<double>& desired);

  void reset(double value = 0.0) { filter_.reset(value); }

  const LatticeIIR& filter() const noexcept { return filter_; }
  const std::vector<double>& numerator() const noexcept { return filter_.numerator(); }
  const std::vector<double>& taps() const noexcept { return filter_.numerator(); }
  const std::vector<double>& reflection() const noexcept { return filter_.reflection(); }
  const std::vector<double>& denominator() const noexcept { return filter_.denominator(); }

  double mu() const noexcept { return mu_; }
  void set_mu(double mu);

private:
  LatticeIIR filter_;
  double mu_;
  double epsilon_;
};



// Autocorrelation sequence r[0]...r[max_lag]. If biased=true the result is
// divided by N; otherwise lag k is divided by N-k.
std::vector<double> autocorrelation(const std::vector<double>& x,
                                    std::size_t max_lag,
                                    bool biased = true);

// Levinson-Durbin recursion for all-pole AR modeling from an autocorrelation
// sequence. The returned reflection coefficients are the PARCOR coefficients
// k_1...k_order. `regularization` is added to r[0] for numerical safety.
std::vector<double> levinson_durbin_reflection(const std::vector<double>& autocorr,
                                               std::size_t order,
                                               double regularization = 1e-12);
std::vector<double> levinson_durbin_denominator(const std::vector<double>& autocorr,
                                                std::size_t order,
                                                double regularization = 1e-12);
double levinson_durbin_error(const std::vector<double>& autocorr,
                             std::size_t order,
                             double regularization = 1e-12);

// Burg AR estimator directly from a signal. Returns stable reflection/PARCOR
// coefficients and can be converted to a denominator with
// reflection_to_denominator().
std::vector<double> burg_reflection(const std::vector<double>& x,
                                    std::size_t order,
                                    double regularization = 1e-12);
std::vector<double> burg_denominator(const std::vector<double>& x,
                                     std::size_t order,
                                     double regularization = 1e-12);

// Fixed-denominator RLS adaptation of lattice-ladder/direct numerator taps.
// The denominator is stable because it is parameterized by reflection
// coefficients. This is a serious adaptive-filter baseline without pretending
// to be a complete AEC system.
class LatticeLadderRLS {
public:
  LatticeLadderRLS(std::vector<double> reflection,
                   std::vector<double> initial_numerator,
                   double forgetting_factor = 0.995,
                   double initial_inverse_covariance = 1000.0,
                   double epsilon = 1e-12);

  std::pair<double, double> adapt_sample(double x, double desired);
  std::vector<double> adapt_block(const std::vector<double>& x,
                                  const std::vector<double>& desired);

  void reset(double value = 0.0);
  void reset_inverse_covariance(double initial_inverse_covariance);

  const LatticeIIR& filter() const noexcept { return filter_; }
  const std::vector<double>& numerator() const noexcept { return filter_.numerator(); }
  const std::vector<double>& taps() const noexcept { return filter_.numerator(); }
  const std::vector<double>& reflection() const noexcept { return filter_.reflection(); }
  const std::vector<double>& denominator() const noexcept { return filter_.denominator(); }
  const std::vector<double>& inverse_covariance() const noexcept { return inverse_covariance_; }
  const std::vector<double>& last_gain() const noexcept { return last_gain_; }

  double forgetting_factor() const noexcept { return forgetting_factor_; }
  double epsilon() const noexcept { return epsilon_; }
  void set_forgetting_factor(double forgetting_factor);

private:
  LatticeIIR filter_;
  double forgetting_factor_;
  double epsilon_;
  std::vector<double> inverse_covariance_; // flattened row-major P matrix.
  std::vector<double> last_gain_;

  void validate() const;
};

// Adaptive second-order notch filter using a normalized gradient update for
// the notch angle parameter a = -2 cos(theta). Useful as a compact demo of
// adaptive IIR behavior.
class AdaptiveNotch {
public:
  AdaptiveNotch(double theta = 0.25, double pole_radius = 0.98,
                double mu = 0.005, double epsilon = 1e-8);

  void reset();
  double process_sample(double x);
  std::vector<double> process_block(const std::vector<double>& x);

  double theta() const noexcept;
  double coefficient() const noexcept { return a_; }
  double pole_radius() const noexcept { return r_; }
  double mu() const noexcept { return mu_; }

  void set_mu(double mu);
  void set_theta(double theta);
  void set_coefficient(double a);

private:
  double a_;      // a = -2 cos(theta), constrained to (-2, 2)
  double r_;
  double mu_;
  double epsilon_;

  // x[n-1], x[n-2], y[n-1], y[n-2]
  double x1_{};
  double x2_{};
  double y1_{};
  double y2_{};

  // derivative memories g[n-1], g[n-2]
  double g1_{};
  double g2_{};

  void validate() const;
  void clamp_a();
};

} // namespace lattice_dsp
