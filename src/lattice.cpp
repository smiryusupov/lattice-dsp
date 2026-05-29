#include "lattice_dsp/lattice.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace lattice_dsp {
namespace {

void require_finite(double v, const char* name) {
  if (!std::isfinite(v)) {
    throw std::invalid_argument(std::string(name) + " contains a non-finite value");
  }
}

void require_vector_finite(const std::vector<double>& values, const char* name) {
  for (double v : values) {
    require_finite(v, name);
  }
}

constexpr double kPi = 3.141592653589793238462643383279502884;

} // namespace

std::vector<double> reflection_to_denominator(const std::vector<double>& reflection) {
  require_vector_finite(reflection, "reflection");
  std::vector<double> a{1.0};
  for (double k : reflection) {
    if (std::abs(k) >= 1.0) {
      throw std::invalid_argument("reflection coefficients must satisfy |k| < 1");
    }
    const std::size_t m = a.size();
    std::vector<double> next(m + 1, 0.0);
    next[0] = 1.0;
    for (std::size_t i = 1; i < m; ++i) {
      next[i] = a[i] + k * a[m - i];
    }
    next[m] = k;
    a = std::move(next);
  }
  return a;
}

std::vector<double> denominator_to_reflection(const std::vector<double>& denominator,
                                              double stability_tol) {
  if (denominator.empty()) {
    throw std::invalid_argument("denominator must not be empty");
  }
  require_vector_finite(denominator, "denominator");
  if (!std::isfinite(stability_tol) || stability_tol < 0.0) {
    throw std::invalid_argument("stability_tol must be finite and non-negative");
  }
  if (std::abs(denominator[0]) <= 0.0) {
    throw std::invalid_argument("denominator[0] must be non-zero");
  }

  std::vector<double> current = denominator;
  const double a0 = current[0];
  for (double& v : current) {
    v /= a0;
  }

  const std::size_t order = current.size() - 1;
  std::vector<double> reflection(order, 0.0);

  for (std::size_t m = order; m > 0; --m) {
    const double k = current[m];
    if (std::abs(k) >= 1.0 - stability_tol) {
      throw std::invalid_argument("denominator is not strictly stable under this lattice convention");
    }
    reflection[m - 1] = k;
    const double denom = 1.0 - k * k;
    std::vector<double> previous(m, 0.0);
    previous[0] = 1.0;
    for (std::size_t i = 1; i < m; ++i) {
      previous[i] = (current[i] - k * current[m - i]) / denom;
    }
    current = std::move(previous);
  }
  return reflection;
}

std::vector<double> bounded_reflection_from_raw(const std::vector<double>& raw, double margin) {
  if (!(margin >= 0.0 && margin < 1.0) || !std::isfinite(margin)) {
    throw std::invalid_argument("margin must be finite and in [0, 1)");
  }
  require_vector_finite(raw, "raw");
  std::vector<double> out;
  out.reserve(raw.size());
  const double scale = 1.0 - margin;
  const double strict_limit = std::nextafter(scale, 0.0);
  for (double v : raw) {
    double k = scale * std::tanh(v);
    if (std::abs(k) >= scale) {
      k = std::copysign(strict_limit, k);
    }
    out.push_back(k);
  }
  return out;
}

std::vector<std::vector<double>> denominator_reflection_jacobian(
    const std::vector<double>& reflection) {
  require_vector_finite(reflection, "reflection");
  const std::size_t order = reflection.size();
  std::vector<std::vector<double>> jac(order, std::vector<double>(order, 0.0));
  if (order == 0) {
    return jac;
  }

  std::vector<double> a{1.0};
  // da[j] stores d A_m / d k_j, including coefficient a_0, for the
  // current recursion order m. Future parameters have all-zero derivatives
  // until their own reflection coefficient enters the step-up recursion.
  std::vector<std::vector<double>> da(order, std::vector<double>(1, 0.0));

  for (std::size_t m = 1; m <= order; ++m) {
    const double k = reflection[m - 1];
    if (std::abs(k) >= 1.0) {
      throw std::invalid_argument("reflection coefficients must satisfy |k| < 1");
    }

    const std::vector<double> old_a = a;
    const std::size_t old_size = old_a.size(); // equal to m
    std::vector<double> next_a(old_size + 1, 0.0);
    next_a[0] = 1.0;
    for (std::size_t i = 1; i < old_size; ++i) {
      next_a[i] = old_a[i] + k * old_a[old_size - i];
    }
    next_a[old_size] = k;

    std::vector<std::vector<double>> next_da(order, std::vector<double>(old_size + 1, 0.0));
    for (std::size_t j = 0; j < order; ++j) {
      // a_0 is always one, so its derivative is zero.
      for (std::size_t i = 1; i < old_size; ++i) {
        const double direct = (j == m - 1) ? old_a[old_size - i] : 0.0;
        next_da[j][i] = da[j][i] + direct + k * da[j][old_size - i];
      }
      next_da[j][old_size] = (j == m - 1) ? 1.0 : 0.0;
    }

    a = std::move(next_a);
    da = std::move(next_da);
  }

  for (std::size_t j = 0; j < order; ++j) {
    for (std::size_t p = 0; p < order; ++p) {
      jac[j][p] = da[j][p + 1];
    }
  }
  return jac;
}

std::vector<std::vector<double>> denominator_raw_jacobian(
    const std::vector<double>& raw, double margin) {
  const std::size_t order = raw.size();
  std::vector<std::vector<double>> jac(order, std::vector<double>(order, 0.0));
  if (order == 0) {
    return jac;
  }
  if (!(margin >= 0.0 && margin < 1.0) || !std::isfinite(margin)) {
    throw std::invalid_argument("margin must be finite and in [0, 1)");
  }

  const double scale = 1.0 - margin;
  const auto reflection = bounded_reflection_from_raw(raw, margin);
  const auto jac_k = denominator_reflection_jacobian(reflection);
  for (std::size_t j = 0; j < order; ++j) {
    // k_j = scale * tanh(raw_j), so dk_j/draw_j = scale * sech^2(raw_j).
    // Writing this as scale * (1 - tanh^2(raw_j)) avoids cosh overflow for
    // large raw values and correctly tends to zero near the stability bound.
    const double t = std::tanh(raw[j]);
    const double dk_draw = scale * (1.0 - t * t);
    for (std::size_t p = 0; p < order; ++p) {
      jac[j][p] = jac_k[j][p] * dk_draw;
    }
  }
  return jac;
}

std::vector<std::vector<double>> denominator_raw_jacobian_finite_difference(
    const std::vector<double>& raw, double margin, double step_scale) {
  const std::size_t order = raw.size();
  std::vector<std::vector<double>> jac(order, std::vector<double>(order, 0.0));
  if (order == 0) {
    return jac;
  }
  if (!(step_scale > 0.0) || !std::isfinite(step_scale)) {
    throw std::invalid_argument("step_scale must be positive and finite");
  }

  for (std::size_t j = 0; j < order; ++j) {
    const double h = step_scale * std::max(1.0, std::abs(raw[j]));
    std::vector<double> plus = raw;
    std::vector<double> minus = raw;
    plus[j] += h;
    minus[j] -= h;
    const auto a_plus = reflection_to_denominator(bounded_reflection_from_raw(plus, margin));
    const auto a_minus = reflection_to_denominator(bounded_reflection_from_raw(minus, margin));
    for (std::size_t pidx = 0; pidx < order; ++pidx) {
      jac[j][pidx] = (a_plus[pidx + 1] - a_minus[pidx + 1]) / (2.0 * h);
    }
  }
  return jac;
}

namespace {

std::vector<std::vector<double>> ladder_basis_polynomials(const std::vector<double>& reflection) {
  // Basis polynomial P_m(z) is the reversed order-m denominator generated by
  // the first m reflection coefficients. The synthesis lattice's backward
  // variable b_m has transfer P_m(z) / A_M(z), where A_M is the full
  // denominator. P_m has degree m and coefficient 1 at z^-m, so the basis is
  // triangular and easy to invert by back-substitution.
  std::vector<std::vector<double>> basis;
  basis.reserve(reflection.size() + 1);
  for (std::size_t m = 0; m <= reflection.size(); ++m) {
    std::vector<double> prefix(reflection.begin(), reflection.begin() + static_cast<std::ptrdiff_t>(m));
    std::vector<double> poly = reflection_to_denominator(prefix);
    std::reverse(poly.begin(), poly.end());
    basis.push_back(std::move(poly));
  }
  return basis;
}

} // namespace

std::vector<double> ladder_to_numerator(const std::vector<double>& reflection,
                                        const std::vector<double>& ladder) {
  require_vector_finite(reflection, "reflection");
  require_vector_finite(ladder, "ladder");
  if (ladder.size() != reflection.size() + 1) {
    throw std::invalid_argument("ladder/taps must have length order + 1");
  }

  const auto basis = ladder_basis_polynomials(reflection);
  std::vector<double> numerator(reflection.size() + 1, 0.0);
  for (std::size_t m = 0; m < basis.size(); ++m) {
    for (std::size_t i = 0; i < basis[m].size(); ++i) {
      numerator[i] += ladder[m] * basis[m][i];
    }
  }
  return numerator;
}

std::vector<double> numerator_to_ladder(const std::vector<double>& reflection,
                                        const std::vector<double>& numerator) {
  require_vector_finite(reflection, "reflection");
  require_vector_finite(numerator, "numerator");
  if (numerator.size() != reflection.size() + 1) {
    throw std::invalid_argument("numerator must have length order + 1");
  }

  const auto basis = ladder_basis_polynomials(reflection);
  std::vector<double> residual = numerator;
  std::vector<double> ladder(reflection.size() + 1, 0.0);

  for (std::size_t m_plus_1 = basis.size(); m_plus_1 > 0; --m_plus_1) {
    const std::size_t m = m_plus_1 - 1;
    const double coeff = residual[m]; // basis[m][m] is exactly 1.
    ladder[m] = coeff;
    for (std::size_t i = 0; i <= m; ++i) {
      residual[i] -= coeff * basis[m][i];
    }
  }
  return ladder;
}

LatticeLadderIIR::LatticeLadderIIR(std::vector<double> reflection, std::vector<double> ladder)
    : order_(reflection.size()), reflection_(std::move(reflection)), ladder_(std::move(ladder)) {
  update_polynomials_and_state();
  validate();
}

void LatticeLadderIIR::validate() const {
  if (ladder_.size() != order_ + 1) {
    throw std::invalid_argument("ladder/taps must have length order + 1");
  }
  require_vector_finite(reflection_, "reflection");
  require_vector_finite(ladder_, "ladder");
  for (double k : reflection_) {
    if (std::abs(k) >= 1.0) {
      throw std::invalid_argument("reflection coefficients must satisfy |k| < 1 for stability");
    }
  }
}

void LatticeLadderIIR::update_polynomials_and_state() {
  order_ = reflection_.size();
  denominator_ = reflection_to_denominator(reflection_);
  numerator_ = ladder_to_numerator(reflection_, ladder_);
  state_.assign(order_, 0.0);
  last_forward_.assign(order_ + 1, 0.0);
  last_backward_.assign(order_ + 1, 0.0);
}

void LatticeLadderIIR::set_reflection(const std::vector<double>& reflection) {
  reflection_ = reflection;
  update_polynomials_and_state();
  validate();
}

void LatticeLadderIIR::set_ladder(const std::vector<double>& ladder) {
  ladder_ = ladder;
  numerator_ = ladder_to_numerator(reflection_, ladder_);
  validate();
}

void LatticeLadderIIR::reset(double value) {
  require_finite(value, "reset value");
  std::fill(state_.begin(), state_.end(), value);
  std::fill(last_forward_.begin(), last_forward_.end(), value);
  std::fill(last_backward_.begin(), last_backward_.end(), value);
}

double LatticeLadderIIR::process_sample(double x) {
  require_finite(x, "input");

  // Synthesis lattice recursion. state_[m] stores b_m[n-1]. First compute
  // forward variables down from f_M[n] = x[n], then current backward variables.
  last_forward_[order_] = x;
  for (std::size_t m_plus_1 = order_; m_plus_1 > 0; --m_plus_1) {
    const std::size_t m = m_plus_1; // stage number in 1..order_
    last_forward_[m - 1] = last_forward_[m] - reflection_[m - 1] * state_[m - 1];
  }

  last_backward_[0] = last_forward_[0];
  for (std::size_t m = 1; m <= order_; ++m) {
    last_backward_[m] = reflection_[m - 1] * last_forward_[m - 1] + state_[m - 1];
  }

  double y = 0.0;
  for (std::size_t m = 0; m <= order_; ++m) {
    y += ladder_[m] * last_backward_[m];
  }

  for (std::size_t m = 0; m < order_; ++m) {
    state_[m] = last_backward_[m];
  }
  return y;
}

std::vector<double> LatticeLadderIIR::process_block(const std::vector<double>& x) {
  std::vector<double> y;
  y.reserve(x.size());
  for (double v : x) {
    y.push_back(process_sample(v));
  }
  return y;
}

LatticeIIR::LatticeIIR(std::vector<double> reflection, std::vector<double> numerator)
    : order_(reflection.size()),
      reflection_(std::move(reflection)),
      numerator_(std::move(numerator)),
      state_(order_, 0.0),
      x_history_(order_, 0.0),
      basis_state_((order_ + 1) * order_, 0.0),
      last_basis_(order_ + 1, 0.0) {
  update_denominator();
  validate();
}

void LatticeIIR::validate() const {
  if (numerator_.size() != order_ + 1) {
    throw std::invalid_argument("numerator/taps must have length order + 1");
  }
  require_vector_finite(reflection_, "reflection");
  require_vector_finite(numerator_, "numerator");
  for (double k : reflection_) {
    if (std::abs(k) >= 1.0) {
      throw std::invalid_argument("reflection coefficients must satisfy |k| < 1 for stability");
    }
  }
}

void LatticeIIR::update_denominator(bool reset_state) {
  const std::size_t previous_order = order_;
  order_ = reflection_.size();
  denominator_ = reflection_to_denominator(reflection_);
  const bool order_changed = order_ != previous_order;
  if (reset_state || order_changed || state_.size() != order_) {
    state_.assign(order_, 0.0);
    x_history_.assign(order_, 0.0);
    basis_state_.assign((order_ + 1) * order_, 0.0);
    last_basis_.assign(order_ + 1, 0.0);
  } else {
    last_basis_.assign(order_ + 1, 0.0);
  }
}

void LatticeIIR::set_reflection(const std::vector<double>& reflection) {
  reflection_ = reflection;
  update_denominator(true);
  validate();
}

void LatticeIIR::set_reflection_preserve_state(const std::vector<double>& reflection) {
  reflection_ = reflection;
  update_denominator(false);
  validate();
}

void LatticeIIR::set_numerator(const std::vector<double>& numerator) {
  numerator_ = numerator;
  validate();
}

void LatticeIIR::add_scaled_to_numerator(const std::vector<double>& direction, double scale) {
  require_finite(scale, "numerator update scale");
  if (direction.size() != numerator_.size()) {
    throw std::invalid_argument("numerator update direction must match numerator length");
  }
  for (std::size_t i = 0; i < numerator_.size(); ++i) {
    require_finite(direction[i], "numerator update direction");
    numerator_[i] += scale * direction[i];
    require_finite(numerator_[i], "updated numerator");
  }
}

void LatticeIIR::reset(double value) {
  require_finite(value, "reset value");
  std::fill(state_.begin(), state_.end(), value);
  std::fill(x_history_.begin(), x_history_.end(), value);
  std::fill(basis_state_.begin(), basis_state_.end(), value);
  std::fill(last_basis_.begin(), last_basis_.end(), value);
}

double LatticeIIR::process_sample(double x) {
  require_finite(x, "input");

  // Transposed direct-form II reference realization:
  // y[n] = sum_i b_i x[n-i] - sum_i a_i y[n-i], a[0] = 1.
  // last_basis_ tracks dy/db_i for this sample using the same denominator.
  const double y = numerator_[0] * x + (order_ > 0 ? state_[0] : 0.0);

  if (order_ > 0) {
    for (std::size_t i = 0; i + 1 < order_; ++i) {
      state_[i] = state_[i + 1] + numerator_[i + 1] * x - denominator_[i + 1] * y;
    }
    state_[order_ - 1] = numerator_[order_] * x - denominator_[order_] * y;
  }

  // Exact numerator-gradient basis: phi_j[n] = z^-j x[n] / A(z).
  // Each basis channel is an all-pole IIR with the same stable denominator.
  for (std::size_t j = 0; j < order_ + 1; ++j) {
    const double basis_input = (j == 0) ? x : x_history_[j - 1];
    const std::size_t offset = j * order_;
    const double phi = basis_input + (order_ > 0 ? basis_state_[offset] : 0.0);
    last_basis_[j] = phi;

    if (order_ > 0) {
      for (std::size_t i = 0; i + 1 < order_; ++i) {
        basis_state_[offset + i] = basis_state_[offset + i + 1] - denominator_[i + 1] * phi;
      }
      basis_state_[offset + order_ - 1] = -denominator_[order_] * phi;
    }
  }

  if (order_ > 0) {
    for (std::size_t i = order_ - 1; i > 0; --i) {
      x_history_[i] = x_history_[i - 1];
    }
    x_history_[0] = x;
  }

  return y;
}

std::vector<double> LatticeIIR::process_block(const std::vector<double>& x) {
  std::vector<double> y;
  y.reserve(x.size());
  for (double v : x) {
    y.push_back(process_sample(v));
  }
  return y;
}


namespace {

std::vector<double> raw_from_reflection(const std::vector<double>& reflection, double margin) {
  if (!(margin >= 0.0 && margin < 1.0) || !std::isfinite(margin)) {
    throw std::invalid_argument("margin must be finite and in [0, 1)");
  }
  const double bound = 1.0 - margin;
  if (!(bound > 0.0)) {
    throw std::invalid_argument("margin leaves no stable reflection range");
  }
  std::vector<double> raw;
  raw.reserve(reflection.size());
  for (double k : reflection) {
    require_finite(k, "reflection");
    if (std::abs(k) >= bound) {
      throw std::invalid_argument("initial reflection must satisfy |k| < 1 - margin");
    }
    const double q = k / bound;
    raw.push_back(0.5 * std::log((1.0 + q) / (1.0 - q)));
  }
  return raw;
}


} // namespace

AdaptiveLatticeLadderNLMS::AdaptiveLatticeLadderNLMS(std::vector<double> initial_reflection,
                                                     std::vector<double> initial_numerator,
                                                     double mu_taps,
                                                     double mu_reflection,
                                                     double epsilon,
                                                     double margin,
                                                     bool freeze_reflection,
                                                     std::string gradient_mode,
                                                     std::size_t reflection_update_period,
                                                     bool scale_reflection_mu_by_period)
    : filter_(bounded_reflection_from_raw(raw_from_reflection(initial_reflection, margin), margin),
              std::move(initial_numerator)),
      raw_reflection_(raw_from_reflection(initial_reflection, margin)),
      mu_taps_(mu_taps),
      mu_reflection_(mu_reflection),
      epsilon_(epsilon),
      margin_(margin),
      freeze_reflection_(freeze_reflection),
      use_finite_difference_gradient_(false),
      reflection_update_period_(reflection_update_period),
      scale_reflection_mu_by_period_(scale_reflection_mu_by_period),
      adaptation_step_(0),
      y_history_(initial_reflection.size(), 0.0),
      denominator_basis_state_(initial_reflection.size() * initial_reflection.size(), 0.0),
      denominator_gradient_(initial_reflection.size(), 0.0),
      last_tap_gradient_(initial_reflection.size() + 1, 0.0),
      last_reflection_gradient_(initial_reflection.size(), 0.0),
      last_raw_gradient_(initial_reflection.size(), 0.0) {
  set_gradient_mode(gradient_mode);
  validate_common();
}

std::string AdaptiveLatticeLadderNLMS::gradient_mode() const {
  return use_finite_difference_gradient_ ? "finite_difference" : "analytic";
}

void AdaptiveLatticeLadderNLMS::set_gradient_mode(const std::string& mode) {
  if (mode == "analytic") {
    use_finite_difference_gradient_ = false;
    return;
  }
  if (mode == "finite_difference" || mode == "finite-difference" || mode == "fd") {
    use_finite_difference_gradient_ = true;
    return;
  }
  throw std::invalid_argument("gradient_mode must be 'analytic' or 'finite_difference'");
}

void AdaptiveLatticeLadderNLMS::validate_common() const {
  if (!(epsilon_ > 0.0) || !std::isfinite(epsilon_)) {
    throw std::invalid_argument("epsilon must be positive and finite");
  }
  if (!(margin_ >= 0.0 && margin_ < 1.0) || !std::isfinite(margin_)) {
    throw std::invalid_argument("margin must be finite and in [0, 1)");
  }
  if (!std::isfinite(mu_taps_) || mu_taps_ < 0.0 || mu_taps_ > 2.0) {
    throw std::invalid_argument("mu_taps should be finite and in [0, 2]");
  }
  if (!std::isfinite(mu_reflection_) || mu_reflection_ < 0.0 || mu_reflection_ > 2.0) {
    throw std::invalid_argument("mu_reflection should be finite and in [0, 2]");
  }
  if (reflection_update_period_ == 0) {
    throw std::invalid_argument("reflection_update_period must be at least 1");
  }
}

std::vector<double> AdaptiveLatticeLadderNLMS::reflection_from_raw() const {
  return bounded_reflection_from_raw(raw_reflection_, margin_);
}

std::vector<double> AdaptiveLatticeLadderNLMS::ladder() const {
  return numerator_to_ladder(filter_.reflection(), filter_.numerator());
}

void AdaptiveLatticeLadderNLMS::set_mu_taps(double mu) {
  mu_taps_ = mu;
  validate_common();
}

void AdaptiveLatticeLadderNLMS::set_mu_reflection(double mu) {
  mu_reflection_ = mu;
  validate_common();
}

void AdaptiveLatticeLadderNLMS::set_reflection_update_period(std::size_t period) {
  reflection_update_period_ = period;
  validate_common();
}

void AdaptiveLatticeLadderNLMS::reset(double value) {
  filter_.reset(value);
  std::fill(y_history_.begin(), y_history_.end(), value);
  std::fill(denominator_basis_state_.begin(), denominator_basis_state_.end(), 0.0);
  std::fill(denominator_gradient_.begin(), denominator_gradient_.end(), 0.0);
  std::fill(last_tap_gradient_.begin(), last_tap_gradient_.end(), 0.0);
  std::fill(last_reflection_gradient_.begin(), last_reflection_gradient_.end(), 0.0);
  std::fill(last_raw_gradient_.begin(), last_raw_gradient_.end(), 0.0);
  adaptation_step_ = 0;
}

void AdaptiveLatticeLadderNLMS::update_denominator_gradients(bool compute_parameter_gradient) {
  const std::size_t order = raw_reflection_.size();
  std::fill(last_reflection_gradient_.begin(), last_reflection_gradient_.end(), 0.0);
  std::fill(last_raw_gradient_.begin(), last_raw_gradient_.end(), 0.0);
  if (order == 0) {
    return;
  }

  const auto& a = filter_.denominator();
  if (compute_parameter_gradient) {
    std::fill(denominator_gradient_.begin(), denominator_gradient_.end(), 0.0);
  }

  for (std::size_t p = 0; p < order; ++p) {
    const double basis_input = -y_history_[p]; // -y[n-(p+1)] / A(z)
    const std::size_t offset = p * order;
    const double psi = basis_input + denominator_basis_state_[offset];
    if (compute_parameter_gradient) {
      denominator_gradient_[p] = psi;
    }

    for (std::size_t i = 0; i + 1 < order; ++i) {
      denominator_basis_state_[offset + i] =
          denominator_basis_state_[offset + i + 1] - a[i + 1] * psi;
    }
    denominator_basis_state_[offset + order - 1] = -a[order] * psi;
  }

  if (!compute_parameter_gradient) {
    return;
  }

  const auto reflection_jac = denominator_reflection_jacobian(filter_.reflection());
  const auto raw_jac = use_finite_difference_gradient_
                           ? denominator_raw_jacobian_finite_difference(raw_reflection_, margin_)
                           : denominator_raw_jacobian(raw_reflection_, margin_);

  for (std::size_t j = 0; j < order; ++j) {
    for (std::size_t p = 0; p < order; ++p) {
      last_reflection_gradient_[j] += denominator_gradient_[p] * reflection_jac[j][p];
      last_raw_gradient_[j] += denominator_gradient_[p] * raw_jac[j][p];
    }
  }
}

void AdaptiveLatticeLadderNLMS::update_output_history(double y) {
  if (y_history_.empty()) {
    return;
  }
  for (std::size_t i = y_history_.size() - 1; i > 0; --i) {
    y_history_[i] = y_history_[i - 1];
  }
  y_history_[0] = y;
}

std::pair<double, double> AdaptiveLatticeLadderNLMS::adapt_sample(double x, double d) {
  require_finite(d, "desired");
  const double y = filter_.process_sample(x);
  const double e = d - y;

  last_tap_gradient_ = filter_.last_basis();
  const bool track_reflection_sensitivity =
      !freeze_reflection_ && mu_reflection_ > 0.0 && !raw_reflection_.empty();
  const bool update_reflection_this_sample =
      track_reflection_sensitivity && (adaptation_step_ % reflection_update_period_ == 0);
  if (track_reflection_sensitivity) {
    update_denominator_gradients(update_reflection_this_sample);
  } else {
    std::fill(last_reflection_gradient_.begin(), last_reflection_gradient_.end(), 0.0);
    std::fill(last_raw_gradient_.begin(), last_raw_gradient_.end(), 0.0);
  }

  if (mu_taps_ > 0.0) {
    double norm = epsilon_;
    for (double v : last_tap_gradient_) {
      norm += v * v;
    }
    const double gain = mu_taps_ * e / norm;
    filter_.add_scaled_to_numerator(last_tap_gradient_, gain);
  }

  if (update_reflection_this_sample && !last_raw_gradient_.empty()) {
    double norm = epsilon_;
    for (double v : last_raw_gradient_) {
      norm += v * v;
    }
    const double effective_mu_reflection =
        mu_reflection_ * (scale_reflection_mu_by_period_ ? static_cast<double>(reflection_update_period_) : 1.0);
    const double gain = effective_mu_reflection * e / norm;
    for (std::size_t i = 0; i < raw_reflection_.size(); ++i) {
      raw_reflection_[i] += gain * last_raw_gradient_[i];
      raw_reflection_[i] = std::clamp(raw_reflection_[i], -20.0, 20.0);
    }
    filter_.set_reflection_preserve_state(reflection_from_raw());
  }

  update_output_history(y);
  ++adaptation_step_;
  return {y, e};
}

std::vector<double> AdaptiveLatticeLadderNLMS::adapt_block(const std::vector<double>& x,
                                                          const std::vector<double>& desired) {
  if (x.size() != desired.size()) {
    throw std::invalid_argument("x and desired must have the same length");
  }
  std::vector<double> err;
  err.reserve(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    err.push_back(adapt_sample(x[i], desired[i]).second);
  }
  return err;
}

LatticeLadderNLMS::LatticeLadderNLMS(std::vector<double> reflection,
                                     std::vector<double> initial_numerator,
                                     double mu,
                                     double epsilon)
    : filter_(std::move(reflection), std::move(initial_numerator)), mu_(mu), epsilon_(epsilon) {
  set_mu(mu);
  if (!(epsilon_ > 0.0) || !std::isfinite(epsilon_)) {
    throw std::invalid_argument("epsilon must be positive and finite");
  }
}

void LatticeLadderNLMS::set_mu(double mu) {
  if (!std::isfinite(mu) || mu < 0.0 || mu > 2.0) {
    throw std::invalid_argument("mu should be finite and in [0, 2] for NLMS-style adaptation");
  }
  mu_ = mu;
}

std::pair<double, double> LatticeLadderNLMS::adapt_sample(double x, double d) {
  require_finite(d, "desired");
  const double y = filter_.process_sample(x);
  const double e = d - y;

  const auto& phi = filter_.last_basis();
  double norm = epsilon_;
  for (double v : phi) {
    norm += v * v;
  }

  const double gain = mu_ * e / norm;
  filter_.add_scaled_to_numerator(phi, gain);
  return {y, e};
}

std::vector<double> LatticeLadderNLMS::adapt_block(const std::vector<double>& x,
                                                   const std::vector<double>& desired) {
  if (x.size() != desired.size()) {
    throw std::invalid_argument("x and desired must have the same length");
  }
  std::vector<double> err;
  err.reserve(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    err.push_back(adapt_sample(x[i], desired[i]).second);
  }
  return err;
}


std::vector<double> autocorrelation(const std::vector<double>& x,
                                    std::size_t max_lag,
                                    bool biased) {
  require_vector_finite(x, "x");
  if (x.empty()) {
    throw std::invalid_argument("x must not be empty");
  }
  if (max_lag >= x.size()) {
    max_lag = x.size() - 1;
  }
  std::vector<double> r(max_lag + 1, 0.0);
  const double n = static_cast<double>(x.size());
  for (std::size_t lag = 0; lag <= max_lag; ++lag) {
    double acc = 0.0;
    for (std::size_t i = lag; i < x.size(); ++i) {
      acc += x[i] * x[i - lag];
    }
    const double denom = biased ? n : static_cast<double>(x.size() - lag);
    r[lag] = acc / denom;
  }
  return r;
}

namespace {

std::pair<std::vector<double>, double> levinson_impl(const std::vector<double>& autocorr,
                                                     std::size_t order,
                                                     double regularization) {
  require_vector_finite(autocorr, "autocorr");
  if (autocorr.empty()) {
    throw std::invalid_argument("autocorr must contain at least r[0]");
  }
  if (order + 1 > autocorr.size()) {
    throw std::invalid_argument("autocorr must have at least order + 1 samples");
  }
  if (!std::isfinite(regularization) || regularization < 0.0) {
    throw std::invalid_argument("regularization must be finite and non-negative");
  }
  double err = autocorr[0] + regularization;
  if (!(err > 0.0) || !std::isfinite(err)) {
    throw std::invalid_argument("regularized zero-lag autocorrelation must be positive");
  }

  std::vector<double> a(order + 1, 0.0);
  std::vector<double> reflection(order, 0.0);
  a[0] = 1.0;
  for (std::size_t m = 1; m <= order; ++m) {
    double num = autocorr[m];
    for (std::size_t i = 1; i < m; ++i) {
      num += a[i] * autocorr[m - i];
    }
    double k = -num / err;
    // Numerical guard: a stable AR model has |k| < 1. For nearly singular
    // autocorrelation estimates, clamp just inside the stable region rather
    // than letting a tiny roundoff produce an unusable model.
    constexpr double limit = 1.0 - 1e-12;
    if (k > limit) {
      k = limit;
    } else if (k < -limit) {
      k = -limit;
    }
    reflection[m - 1] = k;

    std::vector<double> next = a;
    next[m] = k;
    for (std::size_t i = 1; i < m; ++i) {
      next[i] = a[i] + k * a[m - i];
    }
    a = std::move(next);
    err *= (1.0 - k * k);
    if (err < regularization) {
      err = regularization;
    }
  }
  return {reflection, err};
}

} // namespace

std::vector<double> levinson_durbin_reflection(const std::vector<double>& autocorr,
                                               std::size_t order,
                                               double regularization) {
  return levinson_impl(autocorr, order, regularization).first;
}

std::vector<double> levinson_durbin_denominator(const std::vector<double>& autocorr,
                                                std::size_t order,
                                                double regularization) {
  return reflection_to_denominator(levinson_durbin_reflection(autocorr, order, regularization));
}

double levinson_durbin_error(const std::vector<double>& autocorr,
                             std::size_t order,
                             double regularization) {
  return levinson_impl(autocorr, order, regularization).second;
}

std::vector<double> burg_reflection(const std::vector<double>& x,
                                    std::size_t order,
                                    double regularization) {
  require_vector_finite(x, "x");
  if (x.empty()) {
    throw std::invalid_argument("x must not be empty");
  }
  if (order >= x.size()) {
    throw std::invalid_argument("order must be smaller than the number of samples");
  }
  if (!std::isfinite(regularization) || regularization < 0.0) {
    throw std::invalid_argument("regularization must be finite and non-negative");
  }

  // Burg recursion is easiest to implement with compact forward/backward
  // prediction-error vectors. At order m, ef/eb have length N - m - 1 and
  // represent the valid overlapping samples for estimating the next PARCOR
  // coefficient. The previous implementation kept full-length arrays and
  // updated eb[n - 1] in-place over a sliding region; this accidentally reused
  // stale boundary samples and produced an alternating first-order coefficient
  // for higher orders.
  std::vector<double> ef(x.begin() + 1, x.end());
  std::vector<double> eb(x.begin(), x.end() - 1);
  std::vector<double> reflection(order, 0.0);
  for (std::size_t m = 0; m < order; ++m) {
    double num = 0.0;
    double den = regularization;
    for (std::size_t n = 0; n < ef.size(); ++n) {
      num += eb[n] * ef[n];
      den += ef[n] * ef[n] + eb[n] * eb[n];
    }
    if (!(den > 0.0)) {
      reflection[m] = 0.0;
    } else {
      double k = -2.0 * num / den;
      constexpr double limit = 1.0 - 1e-12;
      if (k > limit) {
        k = limit;
      } else if (k < -limit) {
        k = -limit;
      }
      reflection[m] = k;
    }

    if (m + 1 == order) {
      break;
    }

    const double k = reflection[m];
    std::vector<double> next_ef;
    std::vector<double> next_eb;
    next_ef.reserve(ef.size() - 1);
    next_eb.reserve(eb.size() - 1);
    for (std::size_t n = 1; n < ef.size(); ++n) {
      next_ef.push_back(ef[n] + k * eb[n]);
      next_eb.push_back(eb[n - 1] + k * ef[n - 1]);
    }
    ef = std::move(next_ef);
    eb = std::move(next_eb);
  }
  return reflection;
}

std::vector<double> burg_denominator(const std::vector<double>& x,
                                     std::size_t order,
                                     double regularization) {
  return reflection_to_denominator(burg_reflection(x, order, regularization));
}

LatticeLadderRLS::LatticeLadderRLS(std::vector<double> reflection,
                                   std::vector<double> initial_numerator,
                                   double forgetting_factor,
                                   double initial_inverse_covariance,
                                   double epsilon)
    : filter_(std::move(reflection), std::move(initial_numerator)),
      forgetting_factor_(forgetting_factor),
      epsilon_(epsilon),
      inverse_covariance_(),
      last_gain_(filter_.numerator().size(), 0.0) {
  validate();
  reset_inverse_covariance(initial_inverse_covariance);
}

void LatticeLadderRLS::validate() const {
  if (!std::isfinite(forgetting_factor_) || forgetting_factor_ <= 0.0 || forgetting_factor_ > 1.0) {
    throw std::invalid_argument("forgetting_factor must be finite and in (0, 1]");
  }
  if (!std::isfinite(epsilon_) || epsilon_ < 0.0) {
    throw std::invalid_argument("epsilon must be finite and non-negative");
  }
}

void LatticeLadderRLS::set_forgetting_factor(double forgetting_factor) {
  forgetting_factor_ = forgetting_factor;
  validate();
}

void LatticeLadderRLS::reset(double value) {
  filter_.reset(value);
  std::fill(last_gain_.begin(), last_gain_.end(), 0.0);
}

void LatticeLadderRLS::reset_inverse_covariance(double initial_inverse_covariance) {
  if (!std::isfinite(initial_inverse_covariance) || initial_inverse_covariance <= 0.0) {
    throw std::invalid_argument("initial_inverse_covariance must be positive and finite");
  }
  const std::size_t n = filter_.numerator().size();
  inverse_covariance_.assign(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    inverse_covariance_[i * n + i] = initial_inverse_covariance;
  }
  last_gain_.assign(n, 0.0);
}

std::pair<double, double> LatticeLadderRLS::adapt_sample(double x, double desired) {
  require_finite(desired, "desired");
  const double y = filter_.process_sample(x);
  const double e = desired - y;
  const auto& phi = filter_.last_basis();
  const std::size_t n = phi.size();
  if (inverse_covariance_.size() != n * n) {
    reset_inverse_covariance(1000.0);
  }

  std::vector<double> p_phi(n, 0.0);
  std::vector<double> phi_p(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const double pij = inverse_covariance_[i * n + j];
      p_phi[i] += pij * phi[j];
      phi_p[j] += phi[i] * pij;
    }
  }

  double denom = forgetting_factor_ + epsilon_;
  for (std::size_t i = 0; i < n; ++i) {
    denom += phi[i] * p_phi[i];
  }
  if (!(denom > 0.0) || !std::isfinite(denom)) {
    return {y, e};
  }

  for (std::size_t i = 0; i < n; ++i) {
    last_gain_[i] = p_phi[i] / denom;
  }
  filter_.add_scaled_to_numerator(last_gain_, e);

  const double inv_lambda = 1.0 / forgetting_factor_;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      inverse_covariance_[i * n + j] =
          inv_lambda * (inverse_covariance_[i * n + j] - last_gain_[i] * phi_p[j]);
    }
  }
  return {y, e};
}

std::vector<double> LatticeLadderRLS::adapt_block(const std::vector<double>& x,
                                                  const std::vector<double>& desired) {
  if (x.size() != desired.size()) {
    throw std::invalid_argument("x and desired must have the same length");
  }
  std::vector<double> err;
  err.reserve(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    err.push_back(adapt_sample(x[i], desired[i]).second);
  }
  return err;
}

AdaptiveNotch::AdaptiveNotch(double theta, double pole_radius, double mu, double epsilon)
    : a_(-2.0 * std::cos(theta)), r_(pole_radius), mu_(mu), epsilon_(epsilon) {
  validate();
  clamp_a();
}

void AdaptiveNotch::validate() const {
  require_finite(a_, "notch coefficient");
  if (!(r_ > 0.0 && r_ < 1.0) || !std::isfinite(r_)) {
    throw std::invalid_argument("pole_radius must be finite and in (0, 1)");
  }
  if (!std::isfinite(mu_) || mu_ < 0.0 || mu_ > 2.0) {
    throw std::invalid_argument("mu must be finite and in [0, 2]");
  }
  if (!(epsilon_ > 0.0) || !std::isfinite(epsilon_)) {
    throw std::invalid_argument("epsilon must be positive and finite");
  }
}

void AdaptiveNotch::clamp_a() {
  constexpr double limit = 1.999999;
  if (a_ > limit) {
    a_ = limit;
  } else if (a_ < -limit) {
    a_ = -limit;
  }
}

void AdaptiveNotch::reset() {
  x1_ = x2_ = y1_ = y2_ = g1_ = g2_ = 0.0;
}

void AdaptiveNotch::set_mu(double mu) {
  mu_ = mu;
  validate();
}

void AdaptiveNotch::set_theta(double theta) {
  require_finite(theta, "theta");
  if (!(theta > 0.0 && theta < kPi)) {
    throw std::invalid_argument("theta must be in (0, pi)");
  }
  a_ = -2.0 * std::cos(theta);
  clamp_a();
}

void AdaptiveNotch::set_coefficient(double a) {
  require_finite(a, "notch coefficient");
  a_ = a;
  clamp_a();
}

double AdaptiveNotch::theta() const noexcept {
  double v = -a_ / 2.0;
  if (v > 1.0) v = 1.0;
  if (v < -1.0) v = -1.0;
  return std::acos(v);
}

double AdaptiveNotch::process_sample(double x) {
  require_finite(x, "input");

  const double y = x + a_ * x1_ + x2_ - r_ * a_ * y1_ - r_ * r_ * y2_;

  // Sensitivity of y with respect to a.
  const double g = x1_ - r_ * y1_ - r_ * a_ * g1_ - r_ * r_ * g2_;
  a_ -= mu_ * y * g / (epsilon_ + g * g);
  clamp_a();

  x2_ = x1_;
  x1_ = x;
  y2_ = y1_;
  y1_ = y;
  g2_ = g1_;
  g1_ = g;
  return y;
}

std::vector<double> AdaptiveNotch::process_block(const std::vector<double>& x) {
  std::vector<double> y;
  y.reserve(x.size());
  for (double v : x) {
    y.push_back(process_sample(v));
  }
  return y;
}

} // namespace lattice_dsp
