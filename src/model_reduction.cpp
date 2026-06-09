// Copyright 2026 Shohruh Miryusupov
// SPDX-License-Identifier: Apache-2.0

#include "lattice_dsp/lattice.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

using Matrix = std::vector<double>;

std::size_t idx(std::size_t row, std::size_t col, std::size_t cols) {
  return row * cols + col;
}

Matrix identity(std::size_t n) {
  Matrix eye(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    eye[idx(i, i, n)] = 1.0;
  }
  return eye;
}

Matrix matmul(const Matrix& a, std::size_t a_rows, std::size_t a_cols,
              const Matrix& b, std::size_t b_rows, std::size_t b_cols) {
  if (a_cols != b_rows) {
    throw std::invalid_argument("matrix dimensions do not agree");
  }
  Matrix c(a_rows * b_cols, 0.0);
  for (std::size_t i = 0; i < a_rows; ++i) {
    for (std::size_t k = 0; k < a_cols; ++k) {
      const double aik = a[idx(i, k, a_cols)];
      if (aik == 0.0) {
        continue;
      }
      for (std::size_t j = 0; j < b_cols; ++j) {
        c[idx(i, j, b_cols)] += aik * b[idx(k, j, b_cols)];
      }
    }
  }
  return c;
}

Matrix transpose(const Matrix& a, std::size_t rows, std::size_t cols) {
  Matrix t(cols * rows, 0.0);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      t[idx(j, i, rows)] = a[idx(i, j, cols)];
    }
  }
  return t;
}

Matrix hankel_matrix(const std::vector<double>& h, std::size_t rows, std::size_t cols,
                     std::size_t offset) {
  if (rows == 0 || cols == 0) {
    throw std::invalid_argument("Hankel matrix rows and cols must be positive");
  }
  const std::size_t need = offset + rows + cols - 1;
  if (h.size() < need) {
    throw std::invalid_argument("impulse_response is too short for requested Hankel dimensions");
  }
  Matrix out(rows * cols, 0.0);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      out[idx(i, j, cols)] = h[offset + i + j];
    }
  }
  return out;
}

struct SymmetricEigenResult {
  std::vector<double> values;
  Matrix vectors; // columns are eigenvectors, row-major matrix of shape n x n.
};

SymmetricEigenResult jacobi_symmetric_eigen(Matrix a, std::size_t n) {
  if (a.size() != n * n) {
    throw std::invalid_argument("Jacobi eigen input has wrong shape");
  }
  Matrix v = identity(n);
  if (n == 0) {
    return {{}, {}};
  }
  if (n == 1) {
    return {{a[0]}, {1.0}};
  }

  double diag_scale = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    diag_scale = std::max(diag_scale, std::abs(a[idx(i, i, n)]));
  }
  const double tol = std::max(1e-14, 1e-14 * diag_scale);
  const std::size_t max_iter = 80 * n * n;

  for (std::size_t iter = 0; iter < max_iter; ++iter) {
    std::size_t p = 0;
    std::size_t q = 1;
    double max_off = std::abs(a[idx(p, q, n)]);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) {
        const double off = std::abs(a[idx(i, j, n)]);
        if (off > max_off) {
          max_off = off;
          p = i;
          q = j;
        }
      }
    }
    if (max_off <= tol) {
      break;
    }

    const double app = a[idx(p, p, n)];
    const double aqq = a[idx(q, q, n)];
    const double apq = a[idx(p, q, n)];
    if (apq == 0.0) {
      continue;
    }
    const double tau = (aqq - app) / (2.0 * apq);
    const double t = (tau >= 0.0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
    const double c = 1.0 / std::sqrt(1.0 + t * t);
    const double s = t * c;

    for (std::size_t k = 0; k < n; ++k) {
      if (k == p || k == q) {
        continue;
      }
      const double aik = a[idx(k, p, n)];
      const double akq = a[idx(k, q, n)];
      const double new_kp = c * aik - s * akq;
      const double new_kq = s * aik + c * akq;
      a[idx(k, p, n)] = new_kp;
      a[idx(p, k, n)] = new_kp;
      a[idx(k, q, n)] = new_kq;
      a[idx(q, k, n)] = new_kq;
    }

    a[idx(p, p, n)] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
    a[idx(q, q, n)] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
    a[idx(p, q, n)] = 0.0;
    a[idx(q, p, n)] = 0.0;

    for (std::size_t k = 0; k < n; ++k) {
      const double vip = v[idx(k, p, n)];
      const double viq = v[idx(k, q, n)];
      v[idx(k, p, n)] = c * vip - s * viq;
      v[idx(k, q, n)] = s * vip + c * viq;
    }
  }

  std::vector<double> values(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    values[i] = a[idx(i, i, n)];
  }
  return {std::move(values), std::move(v)};
}

std::vector<std::size_t> argsort_descending(const std::vector<double>& values) {
  std::vector<std::size_t> order(values.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return values[a] > values[b];
  });
  return order;
}

std::vector<double> singular_values_from_hankel(const Matrix& h, std::size_t rows, std::size_t cols,
                                                Matrix* right_vectors = nullptr,
                                                std::vector<double>* eigenvalues_sorted = nullptr) {
  const Matrix ht = transpose(h, rows, cols);
  const Matrix gram = matmul(ht, cols, rows, h, rows, cols);
  auto eig = jacobi_symmetric_eigen(gram, cols);
  const auto order = argsort_descending(eig.values);

  std::vector<double> singular;
  singular.reserve(cols);
  Matrix sorted_v;
  if (right_vectors != nullptr) {
    sorted_v.assign(cols * cols, 0.0);
  }
  if (eigenvalues_sorted != nullptr) {
    eigenvalues_sorted->clear();
    eigenvalues_sorted->reserve(cols);
  }

  for (std::size_t new_col = 0; new_col < order.size(); ++new_col) {
    const std::size_t old_col = order[new_col];
    const double lambda = std::max(0.0, eig.values[old_col]);
    singular.push_back(std::sqrt(lambda));
    if (eigenvalues_sorted != nullptr) {
      eigenvalues_sorted->push_back(lambda);
    }
    if (right_vectors != nullptr) {
      for (std::size_t row = 0; row < cols; ++row) {
        sorted_v[idx(row, new_col, cols)] = eig.vectors[idx(row, old_col, cols)];
      }
    }
  }

  if (right_vectors != nullptr) {
    *right_vectors = std::move(sorted_v);
  }
  return singular;
}


Matrix hankelize_by_antidiagonal_average(const Matrix& matrix, std::size_t rows, std::size_t cols,
                                         std::vector<double>* tail_out = nullptr) {
  std::vector<double> values(rows + cols - 1, 0.0);
  std::vector<double> counts(rows + cols - 1, 0.0);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      values[i + j] += matrix[idx(i, j, cols)];
      counts[i + j] += 1.0;
    }
  }
  for (std::size_t k = 0; k < values.size(); ++k) {
    values[k] /= counts[k];
  }
  if (tail_out != nullptr) {
    *tail_out = values;
  }
  Matrix projected(rows * cols, 0.0);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      projected[idx(i, j, cols)] = values[i + j];
    }
  }
  return projected;
}

Matrix matrix_difference(const Matrix& a, const Matrix& b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("matrix sizes do not agree");
  }
  Matrix out(a.size(), 0.0);
  for (std::size_t i = 0; i < a.size(); ++i) {
    out[i] = a[i] - b[i];
  }
  return out;
}

double spectral_norm_matrix(const Matrix& matrix, std::size_t rows, std::size_t cols) {
  const auto singular = singular_values_from_hankel(matrix, rows, cols);
  return singular.empty() ? 0.0 : singular.front();
}

std::vector<double> state_impulse(const Matrix& a, const std::vector<double>& b,
                                  const std::vector<double>& c, double d,
                                  std::size_t n_samples) {
  const std::size_t n = b.size();
  std::vector<double> h(n_samples, 0.0);
  if (n_samples == 0) {
    return h;
  }
  h[0] = d;
  if (n == 0) {
    return h;
  }
  std::vector<double> x = b;
  for (std::size_t sample = 1; sample < n_samples; ++sample) {
    double y = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      y += c[i] * x[i];
    }
    h[sample] = y;
    std::vector<double> next(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        next[i] += a[idx(i, j, n)] * x[j];
      }
    }
    x = std::move(next);
  }
  return h;
}

std::vector<double> characteristic_denominator(const Matrix& a, std::size_t n) {
  if (n == 0) {
    return {1.0};
  }
  Matrix b = identity(n);
  std::vector<double> coeff(n + 1, 0.0);
  coeff[0] = 1.0;
  for (std::size_t k = 1; k <= n; ++k) {
    const Matrix ab = matmul(a, n, n, b, n, n);
    double tr = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      tr += ab[idx(i, i, n)];
    }
    const double ck = -tr / static_cast<double>(k);
    coeff[k] = ck;
    b = ab;
    for (std::size_t i = 0; i < n; ++i) {
      b[idx(i, i, n)] += ck;
    }
  }
  return coeff;
}


Matrix block_hankel_matrix(const std::vector<double>& markov,
                           std::size_t n_samples,
                           std::size_t n_outputs,
                           std::size_t n_inputs,
                           std::size_t block_rows,
                           std::size_t block_cols,
                           std::size_t offset) {
  if (block_rows == 0 || block_cols == 0) {
    throw std::invalid_argument("block_rows and block_cols must be positive");
  }
  if (n_outputs == 0 || n_inputs == 0) {
    throw std::invalid_argument("n_outputs and n_inputs must be positive");
  }
  const std::size_t block_size = n_outputs * n_inputs;
  if (markov.size() != n_samples * block_size) {
    throw std::invalid_argument("markov_parameters size does not match n_samples * n_outputs * n_inputs");
  }
  const std::size_t required = offset + block_rows + block_cols - 1;
  if (n_samples < required) {
    throw std::invalid_argument("markov_parameters is too short for requested block-Hankel dimensions");
  }

  const std::size_t rows = block_rows * n_outputs;
  const std::size_t cols = block_cols * n_inputs;
  Matrix out(rows * cols, 0.0);
  for (std::size_t br = 0; br < block_rows; ++br) {
    for (std::size_t bc = 0; bc < block_cols; ++bc) {
      const std::size_t sample = offset + br + bc;
      for (std::size_t y = 0; y < n_outputs; ++y) {
        for (std::size_t u = 0; u < n_inputs; ++u) {
          const std::size_t row = br * n_outputs + y;
          const std::size_t col = bc * n_inputs + u;
          const std::size_t markov_index = (sample * n_outputs + y) * n_inputs + u;
          out[idx(row, col, cols)] = markov[markov_index];
        }
      }
    }
  }
  return out;
}

double relative_markov_error(const std::vector<double>& reference,
                             const std::vector<double>& estimate) {
  if (reference.size() != estimate.size()) {
    throw std::invalid_argument("Markov response sizes do not match");
  }
  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double e = reference[i] - estimate[i];
    num += e * e;
    den += reference[i] * reference[i];
  }
  return den > 0.0 ? num / den : 0.0;
}

std::vector<double> numerator_from_impulse_and_denominator(const std::vector<double>& impulse,
                                                           const std::vector<double>& denominator) {
  const std::size_t order = denominator.empty() ? 0 : denominator.size() - 1;
  if (impulse.size() < order + 1) {
    throw std::invalid_argument("impulse response is too short to form numerator");
  }
  std::vector<double> numerator(order + 1, 0.0);
  for (std::size_t i = 0; i <= order; ++i) {
    double v = 0.0;
    for (std::size_t j = 0; j <= i; ++j) {
      v += denominator[j] * impulse[i - j];
    }
    numerator[i] = v;
  }
  return numerator;
}

} // namespace

std::vector<double> iir_impulse_response(const std::vector<double>& denominator,
                                         const std::vector<double>& numerator,
                                         std::size_t n_samples) {
  if (denominator.empty()) {
    throw std::invalid_argument("denominator must not be empty");
  }
  if (std::abs(denominator[0]) < 1e-15) {
    throw std::invalid_argument("denominator[0] must be non-zero");
  }
  require_vector_finite(denominator, "denominator");
  require_vector_finite(numerator, "numerator");
  if (n_samples == 0) {
    return {};
  }

  std::vector<double> a = denominator;
  std::vector<double> b = numerator;
  const double a0 = a[0];
  for (double& v : a) {
    v /= a0;
  }
  for (double& v : b) {
    v /= a0;
  }

  std::vector<double> h(n_samples, 0.0);
  for (std::size_t n = 0; n < n_samples; ++n) {
    double y = (n < b.size()) ? b[n] : 0.0;
    for (std::size_t k = 1; k < a.size(); ++k) {
      if (n >= k) {
        y -= a[k] * h[n - k];
      }
    }
    h[n] = y;
  }
  return h;
}

std::vector<double> hankel_singular_values(const std::vector<double>& impulse_response,
                                           std::size_t rows,
                                           std::size_t cols,
                                           std::size_t offset) {
  require_vector_finite(impulse_response, "impulse_response");
  const Matrix h = hankel_matrix(impulse_response, rows, cols, offset);
  return singular_values_from_hankel(h, rows, cols);
}

FiniteHankelReduction finite_hankel_reduce_impulse(
    const std::vector<double>& impulse_response,
    std::size_t reduced_order,
    std::size_t rows,
    std::size_t cols,
    double regularization) {
  require_vector_finite(impulse_response, "impulse_response");
  if (!std::isfinite(regularization) || regularization < 0.0) {
    throw std::invalid_argument("regularization must be finite and non-negative");
  }
  if (reduced_order > std::min(rows, cols)) {
    throw std::invalid_argument("reduced_order cannot exceed min(rows, cols)");
  }
  if (impulse_response.empty()) {
    throw std::invalid_argument("impulse_response must not be empty");
  }

  const std::size_t required = rows + cols + 1;  // H1 uses offset=2.
  if (impulse_response.size() < required) {
    throw std::invalid_argument("impulse_response must have at least rows + cols + 1 samples");
  }

  FiniteHankelReduction result;
  result.method = "finite_hankel_ho_kalman";

  const Matrix h0 = hankel_matrix(impulse_response, rows, cols, 1);
  const Matrix h1 = hankel_matrix(impulse_response, rows, cols, 2);

  Matrix v_full;
  result.hankel_singular_values = singular_values_from_hankel(h0, rows, cols, &v_full);

  const double total_energy = std::inner_product(result.hankel_singular_values.begin(),
                                                result.hankel_singular_values.end(),
                                                result.hankel_singular_values.begin(), 0.0);
  double kept_energy = 0.0;
  for (std::size_t i = 0; i < std::min(reduced_order, result.hankel_singular_values.size()); ++i) {
    kept_energy += result.hankel_singular_values[i] * result.hankel_singular_values[i];
  }
  result.retained_hankel_energy = total_energy > 0.0 ? kept_energy / total_energy : 1.0;

  if (reduced_order == 0) {
    result.denominator = {1.0};
    result.numerator = {impulse_response[0]};
    result.reflection = {};
    result.stable = true;
    const auto reduced = iir_impulse_response(result.denominator, result.numerator, impulse_response.size());
    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < impulse_response.size(); ++i) {
      const double e = impulse_response[i] - reduced[i];
      num += e * e;
      den += impulse_response[i] * impulse_response[i];
    }
    result.relative_impulse_error = den > 0.0 ? num / den : 0.0;
    return result;
  }

  for (std::size_t i = 0; i < reduced_order; ++i) {
    if (result.hankel_singular_values[i] <= regularization) {
      throw std::invalid_argument("requested reduced_order exceeds numerical Hankel rank");
    }
  }

  Matrix v(cols * reduced_order, 0.0);
  for (std::size_t j = 0; j < reduced_order; ++j) {
    for (std::size_t i = 0; i < cols; ++i) {
      v[idx(i, j, reduced_order)] = v_full[idx(i, j, cols)];
    }
  }

  Matrix u(rows * reduced_order, 0.0);
  for (std::size_t j = 0; j < reduced_order; ++j) {
    const double inv_s = 1.0 / result.hankel_singular_values[j];
    for (std::size_t i = 0; i < rows; ++i) {
      double acc = 0.0;
      for (std::size_t k = 0; k < cols; ++k) {
        acc += h0[idx(i, k, cols)] * v[idx(k, j, reduced_order)];
      }
      u[idx(i, j, reduced_order)] = acc * inv_s;
    }
  }

  Matrix temp = matmul(transpose(u, rows, reduced_order), reduced_order, rows,
                       h1, rows, cols);
  Matrix uh1v = matmul(temp, reduced_order, cols, v, cols, reduced_order);

  Matrix a(reduced_order * reduced_order, 0.0);
  std::vector<double> b(reduced_order, 0.0);
  std::vector<double> c(reduced_order, 0.0);
  for (std::size_t i = 0; i < reduced_order; ++i) {
    const double si_sqrt = std::sqrt(result.hankel_singular_values[i]);
    c[i] = u[idx(0, i, reduced_order)] * si_sqrt;
    b[i] = si_sqrt * v[idx(0, i, reduced_order)];
    for (std::size_t j = 0; j < reduced_order; ++j) {
      const double scale = 1.0 / (std::sqrt(result.hankel_singular_values[i]) *
                                  std::sqrt(result.hankel_singular_values[j]));
      a[idx(i, j, reduced_order)] = uh1v[idx(i, j, reduced_order)] * scale;
    }
  }

  const double d = impulse_response[0];
  const auto reduced_impulse = state_impulse(a, b, c, d, impulse_response.size());
  result.denominator = characteristic_denominator(a, reduced_order);
  result.numerator = numerator_from_impulse_and_denominator(reduced_impulse, result.denominator);

  try {
    result.reflection = denominator_to_reflection(result.denominator, 1e-10);
    result.stable = true;
  } catch (const std::invalid_argument&) {
    result.reflection.clear();
    result.stable = false;
  }

  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = 0; i < impulse_response.size(); ++i) {
    const double e = impulse_response[i] - reduced_impulse[i];
    num += e * e;
    den += impulse_response[i] * impulse_response[i];
  }
  result.relative_impulse_error = den > 0.0 ? num / den : 0.0;
  return result;
}

FiniteHankelReduction finite_hankel_reduce_iir(
    const std::vector<double>& reflection,
    const std::vector<double>& numerator,
    std::size_t reduced_order,
    std::size_t n_impulse,
    std::size_t rows,
    std::size_t cols,
    double regularization) {
  const auto denominator = reflection_to_denominator(reflection);
  const auto impulse = iir_impulse_response(denominator, numerator, n_impulse);
  return finite_hankel_reduce_impulse(impulse, reduced_order, rows, cols, regularization);
}



std::vector<double> mimo_state_space_markov_response(
    const std::vector<double>& a,
    const std::vector<double>& b,
    const std::vector<double>& c,
    const std::vector<double>& d,
    std::size_t state_order,
    std::size_t n_outputs,
    std::size_t n_inputs,
    std::size_t n_samples) {
  require_vector_finite(a, "a");
  require_vector_finite(b, "b");
  require_vector_finite(c, "c");
  require_vector_finite(d, "d");
  if (n_outputs == 0 || n_inputs == 0) {
    throw std::invalid_argument("n_outputs and n_inputs must be positive");
  }
  if (a.size() != state_order * state_order) {
    throw std::invalid_argument("A must have shape (state_order, state_order)");
  }
  if (b.size() != state_order * n_inputs) {
    throw std::invalid_argument("B must have shape (state_order, n_inputs)");
  }
  if (c.size() != n_outputs * state_order) {
    throw std::invalid_argument("C must have shape (n_outputs, state_order)");
  }
  if (d.size() != n_outputs * n_inputs) {
    throw std::invalid_argument("D must have shape (n_outputs, n_inputs)");
  }

  std::vector<double> markov(n_samples * n_outputs * n_inputs, 0.0);
  if (n_samples == 0) {
    return markov;
  }
  for (std::size_t y = 0; y < n_outputs; ++y) {
    for (std::size_t u = 0; u < n_inputs; ++u) {
      markov[(0 * n_outputs + y) * n_inputs + u] = d[idx(y, u, n_inputs)];
    }
  }
  if (state_order == 0) {
    return markov;
  }

  Matrix power_b = b;  // A^0 B, shape state_order x n_inputs.
  for (std::size_t sample = 1; sample < n_samples; ++sample) {
    const Matrix c_power_b = matmul(c, n_outputs, state_order, power_b, state_order, n_inputs);
    for (std::size_t y = 0; y < n_outputs; ++y) {
      for (std::size_t u = 0; u < n_inputs; ++u) {
        markov[(sample * n_outputs + y) * n_inputs + u] = c_power_b[idx(y, u, n_inputs)];
      }
    }
    power_b = matmul(a, state_order, state_order, power_b, state_order, n_inputs);
  }
  return markov;
}

FiniteHankelMimoReduction finite_hankel_reduce_mimo(
    const std::vector<double>& markov_parameters,
    std::size_t n_samples,
    std::size_t n_outputs,
    std::size_t n_inputs,
    std::size_t reduced_order,
    std::size_t block_rows,
    std::size_t block_cols,
    double regularization) {
  require_vector_finite(markov_parameters, "markov_parameters");
  if (!std::isfinite(regularization) || regularization < 0.0) {
    throw std::invalid_argument("regularization must be finite and non-negative");
  }
  if (n_outputs == 0 || n_inputs == 0) {
    throw std::invalid_argument("n_outputs and n_inputs must be positive");
  }
  if (block_rows == 0 || block_cols == 0) {
    throw std::invalid_argument("block_rows and block_cols must be positive");
  }
  const std::size_t rows = block_rows * n_outputs;
  const std::size_t cols = block_cols * n_inputs;
  if (reduced_order > std::min(rows, cols)) {
    throw std::invalid_argument("reduced_order cannot exceed min(block_rows*n_outputs, block_cols*n_inputs)");
  }
  const std::size_t required = block_rows + block_cols + 1;  // H1 uses offset=2.
  if (n_samples < required) {
    throw std::invalid_argument("markov_parameters must contain at least block_rows + block_cols + 1 samples");
  }
  if (markov_parameters.size() != n_samples * n_outputs * n_inputs) {
    throw std::invalid_argument("markov_parameters size does not match n_samples * n_outputs * n_inputs");
  }

  FiniteHankelMimoReduction result;
  result.method = "finite_block_hankel_ho_kalman";
  result.state_order = reduced_order;
  result.n_outputs = n_outputs;
  result.n_inputs = n_inputs;

  const Matrix h0 = block_hankel_matrix(markov_parameters, n_samples, n_outputs, n_inputs,
                                        block_rows, block_cols, 1);
  const Matrix h1 = block_hankel_matrix(markov_parameters, n_samples, n_outputs, n_inputs,
                                        block_rows, block_cols, 2);

  Matrix v_full;
  result.hankel_singular_values = singular_values_from_hankel(h0, rows, cols, &v_full);
  const double total_energy = std::inner_product(result.hankel_singular_values.begin(),
                                                result.hankel_singular_values.end(),
                                                result.hankel_singular_values.begin(), 0.0);
  double kept_energy = 0.0;
  for (std::size_t i = 0; i < std::min(reduced_order, result.hankel_singular_values.size()); ++i) {
    kept_energy += result.hankel_singular_values[i] * result.hankel_singular_values[i];
  }
  result.retained_hankel_energy = total_energy > 0.0 ? kept_energy / total_energy : 1.0;

  result.d.assign(n_outputs * n_inputs, 0.0);
  for (std::size_t y = 0; y < n_outputs; ++y) {
    for (std::size_t u = 0; u < n_inputs; ++u) {
      result.d[idx(y, u, n_inputs)] = markov_parameters[(0 * n_outputs + y) * n_inputs + u];
    }
  }

  if (reduced_order == 0) {
    result.a = {};
    result.b = {};
    result.c = {};
    result.stable = true;
    const auto reduced = mimo_state_space_markov_response(result.a, result.b, result.c, result.d,
                                                          0, n_outputs, n_inputs, n_samples);
    result.relative_markov_error = relative_markov_error(markov_parameters, reduced);
    return result;
  }

  for (std::size_t i = 0; i < reduced_order; ++i) {
    if (result.hankel_singular_values[i] <= regularization) {
      throw std::invalid_argument("requested reduced_order exceeds numerical block-Hankel rank");
    }
  }

  Matrix v(cols * reduced_order, 0.0);
  for (std::size_t j = 0; j < reduced_order; ++j) {
    for (std::size_t i = 0; i < cols; ++i) {
      v[idx(i, j, reduced_order)] = v_full[idx(i, j, cols)];
    }
  }

  Matrix u(rows * reduced_order, 0.0);
  for (std::size_t j = 0; j < reduced_order; ++j) {
    const double inv_s = 1.0 / result.hankel_singular_values[j];
    for (std::size_t i = 0; i < rows; ++i) {
      double acc = 0.0;
      for (std::size_t k = 0; k < cols; ++k) {
        acc += h0[idx(i, k, cols)] * v[idx(k, j, reduced_order)];
      }
      u[idx(i, j, reduced_order)] = acc * inv_s;
    }
  }

  Matrix temp = matmul(transpose(u, rows, reduced_order), reduced_order, rows,
                       h1, rows, cols);
  Matrix uh1v = matmul(temp, reduced_order, cols, v, cols, reduced_order);

  result.a.assign(reduced_order * reduced_order, 0.0);
  result.b.assign(reduced_order * n_inputs, 0.0);
  result.c.assign(n_outputs * reduced_order, 0.0);

  for (std::size_t i = 0; i < reduced_order; ++i) {
    const double si_sqrt = std::sqrt(result.hankel_singular_values[i]);
    for (std::size_t input = 0; input < n_inputs; ++input) {
      result.b[idx(i, input, n_inputs)] = si_sqrt * v[idx(input, i, reduced_order)];
    }
    for (std::size_t output = 0; output < n_outputs; ++output) {
      result.c[idx(output, i, reduced_order)] = u[idx(output, i, reduced_order)] * si_sqrt;
    }
    for (std::size_t j = 0; j < reduced_order; ++j) {
      const double scale = 1.0 / (std::sqrt(result.hankel_singular_values[i]) *
                                  std::sqrt(result.hankel_singular_values[j]));
      result.a[idx(i, j, reduced_order)] = uh1v[idx(i, j, reduced_order)] * scale;
    }
  }

  try {
    const auto denom = characteristic_denominator(result.a, reduced_order);
    (void)denominator_to_reflection(denom, 1e-8);
    result.stable = true;
  } catch (const std::invalid_argument&) {
    result.stable = false;
  }

  const auto reduced = mimo_state_space_markov_response(result.a, result.b, result.c, result.d,
                                                        reduced_order, n_outputs, n_inputs, n_samples);
  result.relative_markov_error = relative_markov_error(markov_parameters, reduced);
  return result;
}



FiniteNehariApproximation finite_nehari_approximate_tail(
    const std::vector<double>& anticausal_tail,
    std::size_t rank,
    std::size_t rows,
    std::size_t cols,
    double regularization) {
  require_vector_finite(anticausal_tail, "anticausal_tail");
  if (!std::isfinite(regularization) || regularization < 0.0) {
    throw std::invalid_argument("regularization must be finite and non-negative");
  }
  if (rows == 0 || cols == 0) {
    throw std::invalid_argument("rows and cols must be positive");
  }
  if (rank > std::min(rows, cols)) {
    throw std::invalid_argument("rank cannot exceed min(rows, cols)");
  }
  const std::size_t need = rows + cols - 1;
  if (anticausal_tail.size() < need) {
    throw std::invalid_argument("anticausal_tail must contain at least rows + cols - 1 values");
  }

  FiniteNehariApproximation result;
  result.method = "finite_nehari_hankelized_svd";
  result.rank = rank;
  result.rows = rows;
  result.cols = cols;

  const Matrix h = hankel_matrix(anticausal_tail, rows, cols, 0);
  Matrix v_full;
  result.hankel_singular_values = singular_values_from_hankel(h, rows, cols, &v_full, nullptr);
  result.sigma_next = rank < result.hankel_singular_values.size()
                          ? result.hankel_singular_values[rank]
                          : 0.0;

  if (rank == 0) {
    Matrix zero(h.size(), 0.0);
    std::vector<double> tail;
    const Matrix hankelized = hankelize_by_antidiagonal_average(zero, rows, cols, &tail);
    result.approximated_tail = std::move(tail);
    result.unconstrained_hankel_error = spectral_norm_matrix(h, rows, cols);
    result.hankelized_hankel_error = spectral_norm_matrix(h, rows, cols);
    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < need; ++i) {
      num += anticausal_tail[i] * anticausal_tail[i];
      den += anticausal_tail[i] * anticausal_tail[i];
    }
    result.relative_tail_error = den > 0.0 ? std::sqrt(num / den) : 0.0;
    return result;
  }

  for (std::size_t i = 0; i < rank; ++i) {
    if (result.hankel_singular_values[i] <= regularization) {
      throw std::invalid_argument("requested rank exceeds numerical Hankel rank");
    }
  }

  Matrix v(cols * rank, 0.0);
  for (std::size_t j = 0; j < rank; ++j) {
    for (std::size_t i = 0; i < cols; ++i) {
      v[idx(i, j, rank)] = v_full[idx(i, j, cols)];
    }
  }

  Matrix u(rows * rank, 0.0);
  for (std::size_t j = 0; j < rank; ++j) {
    const double inv_s = 1.0 / result.hankel_singular_values[j];
    for (std::size_t i = 0; i < rows; ++i) {
      double acc = 0.0;
      for (std::size_t k = 0; k < cols; ++k) {
        acc += h[idx(i, k, cols)] * v[idx(k, j, rank)];
      }
      u[idx(i, j, rank)] = acc * inv_s;
    }
  }

  Matrix truncated(rows * cols, 0.0);
  for (std::size_t mode = 0; mode < rank; ++mode) {
    const double sigma = result.hankel_singular_values[mode];
    for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
        truncated[idx(i, j, cols)] += sigma * u[idx(i, mode, rank)] * v[idx(j, mode, rank)];
      }
    }
  }

  result.unconstrained_hankel_error = spectral_norm_matrix(matrix_difference(h, truncated), rows, cols);
  std::vector<double> tail;
  const Matrix hankelized = hankelize_by_antidiagonal_average(truncated, rows, cols, &tail);
  result.approximated_tail = std::move(tail);
  result.hankelized_hankel_error = spectral_norm_matrix(matrix_difference(h, hankelized), rows, cols);

  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = 0; i < need; ++i) {
    const double e = anticausal_tail[i] - result.approximated_tail[i];
    num += e * e;
    den += anticausal_tail[i] * anticausal_tail[i];
  }
  result.relative_tail_error = den > 0.0 ? std::sqrt(num / den) : 0.0;
  return result;
}

FiniteHankelReduction finite_hankel_aak_reduce_impulse(
    const std::vector<double>& impulse_response,
    std::size_t reduced_order,
    std::size_t rows,
    std::size_t cols,
    double regularization) {
  return finite_hankel_reduce_impulse(impulse_response, reduced_order, rows, cols, regularization);
}

FiniteHankelReduction finite_hankel_aak_reduce_iir(
    const std::vector<double>& reflection,
    const std::vector<double>& numerator,
    std::size_t reduced_order,
    std::size_t n_impulse,
    std::size_t rows,
    std::size_t cols,
    double regularization) {
  return finite_hankel_reduce_iir(reflection, numerator, reduced_order, n_impulse, rows, cols, regularization);
}

} // namespace lattice_dsp
