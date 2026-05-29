#include "lattice_dsp/lattice.hpp"

#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef LATTICE_DSP_HAS_OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using lattice_dsp::LatticeIIR;
using lattice_dsp::LatticeLadderIIR;
using lattice_dsp::LatticeLadderNLMS;
using lattice_dsp::LatticeLadderRLS;
using lattice_dsp::AdaptiveLatticeLadderNLMS;
using lattice_dsp::AdaptiveNotch;

namespace {

template <typename Filter>
py::array_t<double> process_array(Filter& filter,
                                  py::array_t<double, py::array::c_style | py::array::forcecast> x) {
  if (x.ndim() != 1) {
    throw std::invalid_argument("process expects a 1-D NumPy array");
  }
  auto xin = x.unchecked<1>();
  py::array_t<double> y(x.shape(0));
  auto yout = y.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < x.shape(0); ++i) {
    yout(i) = filter.process_sample(xin(i));
  }
  return y;
}

py::array_t<double> process_batch(const std::vector<double>& reflection,
                                  const std::vector<double>& taps,
                                  py::array_t<double, py::array::c_style | py::array::forcecast> x,
                                  int n_threads,
                                  const std::string& realization) {
  const bool use_lattice = realization == "lattice";
  const bool use_direct = realization == "direct";
  if (!use_lattice && !use_direct) {
    throw std::invalid_argument("realization must be 'direct' or 'lattice'");
  }

  if (x.ndim() == 1) {
    if (use_lattice) {
      LatticeLadderIIR f(reflection, lattice_dsp::numerator_to_ladder(reflection, taps));
      return process_array(f, x);
    }
    LatticeIIR f(reflection, taps);
    return process_array(f, x);
  }
  if (x.ndim() != 2) {
    throw std::invalid_argument("process_batch expects a 1-D or 2-D NumPy array");
  }

  const py::ssize_t channels = x.shape(0);
  const py::ssize_t samples = x.shape(1);
  py::array_t<double> y({channels, samples});

  auto xin = x.unchecked<2>();
  auto yout = y.mutable_unchecked<2>();
  const std::vector<double> ladder = use_lattice ? lattice_dsp::numerator_to_ladder(reflection, taps)
                                                 : std::vector<double>{};

#ifdef LATTICE_DSP_HAS_OPENMP
  if (n_threads > 0) {
#pragma omp parallel for schedule(static) num_threads(n_threads)
    for (py::ssize_t ch = 0; ch < channels; ++ch) {
      if (use_lattice) {
        LatticeLadderIIR f(reflection, ladder);
        for (py::ssize_t n = 0; n < samples; ++n) {
          yout(ch, n) = f.process_sample(xin(ch, n));
        }
      } else {
        LatticeIIR f(reflection, taps);
        for (py::ssize_t n = 0; n < samples; ++n) {
          yout(ch, n) = f.process_sample(xin(ch, n));
        }
      }
    }
  } else {
#pragma omp parallel for schedule(static)
    for (py::ssize_t ch = 0; ch < channels; ++ch) {
      if (use_lattice) {
        LatticeLadderIIR f(reflection, ladder);
        for (py::ssize_t n = 0; n < samples; ++n) {
          yout(ch, n) = f.process_sample(xin(ch, n));
        }
      } else {
        LatticeIIR f(reflection, taps);
        for (py::ssize_t n = 0; n < samples; ++n) {
          yout(ch, n) = f.process_sample(xin(ch, n));
        }
      }
    }
  }
#else
  (void)n_threads;
  for (py::ssize_t ch = 0; ch < channels; ++ch) {
    if (use_lattice) {
      LatticeLadderIIR f(reflection, ladder);
      for (py::ssize_t n = 0; n < samples; ++n) {
        yout(ch, n) = f.process_sample(xin(ch, n));
      }
    } else {
      LatticeIIR f(reflection, taps);
      for (py::ssize_t n = 0; n < samples; ++n) {
        yout(ch, n) = f.process_sample(xin(ch, n));
      }
    }
  }
#endif

  return y;
}

py::tuple adapt_sample(LatticeLadderNLMS& adaptive, double x, double d) {
  auto [y, e] = adaptive.adapt_sample(x, d);
  return py::make_tuple(y, e);
}

py::tuple adapt_full_sample(AdaptiveLatticeLadderNLMS& adaptive, double x, double d) {
  auto [y, e] = adaptive.adapt_sample(x, d);
  return py::make_tuple(y, e);
}


template <typename AdaptiveFilter>
py::tuple adapt_array(AdaptiveFilter& adaptive,
                      py::array_t<double, py::array::c_style | py::array::forcecast> x,
                      py::array_t<double, py::array::c_style | py::array::forcecast> desired) {
  if (x.ndim() != 1 || desired.ndim() != 1) {
    throw std::invalid_argument("process_adapt expects 1-D NumPy arrays");
  }
  if (x.shape(0) != desired.shape(0)) {
    throw std::invalid_argument("x and desired must have the same length");
  }

  const py::ssize_t samples = x.shape(0);
  py::array_t<double> y(samples);
  py::array_t<double> e(samples);

  auto xin = x.unchecked<1>();
  auto din = desired.unchecked<1>();
  auto yout = y.mutable_unchecked<1>();
  auto eout = e.mutable_unchecked<1>();

  {
    py::gil_scoped_release release;
    for (py::ssize_t n = 0; n < samples; ++n) {
      auto [yn, en] = adaptive.adapt_sample(xin(n), din(n));
      yout(n) = yn;
      eout(n) = en;
    }
  }

  return py::make_tuple(y, e);
}

void validate_finite_2d(py::array_t<double, py::array::c_style | py::array::forcecast> arr,
                        const char* name) {
  auto view = arr.unchecked<2>();
  for (py::ssize_t i = 0; i < arr.shape(0); ++i) {
    for (py::ssize_t j = 0; j < arr.shape(1); ++j) {
      if (!std::isfinite(view(i, j))) {
        throw std::invalid_argument(std::string(name) + " must contain only finite values");
      }
    }
  }
}

void validate_finite_3d(py::array_t<double, py::array::c_style | py::array::forcecast> arr,
                        const char* name) {
  auto view = arr.unchecked<3>();
  for (py::ssize_t i = 0; i < arr.shape(0); ++i) {
    for (py::ssize_t j = 0; j < arr.shape(1); ++j) {
      for (py::ssize_t k = 0; k < arr.shape(2); ++k) {
        if (!std::isfinite(view(i, j, k))) {
          throw std::invalid_argument(std::string(name) + " must contain only finite values");
        }
      }
    }
  }
}


py::array_t<double> mimo_state_space_process_batch(
    py::array_t<double, py::array::c_style | py::array::forcecast> A,
    py::array_t<double, py::array::c_style | py::array::forcecast> B,
    py::array_t<double, py::array::c_style | py::array::forcecast> C,
    py::array_t<double, py::array::c_style | py::array::forcecast> D,
    py::array_t<double, py::array::c_style | py::array::forcecast> x,
    int n_threads) {
  if (A.ndim() != 2 || A.shape(0) != A.shape(1)) {
    throw std::invalid_argument("A must have shape (state_order, state_order)");
  }
  if (B.ndim() != 2) {
    throw std::invalid_argument("B must have shape (state_order, inputs)");
  }
  if (C.ndim() != 2) {
    throw std::invalid_argument("C must have shape (outputs, state_order)");
  }
  if (D.ndim() != 2) {
    throw std::invalid_argument("D must have shape (outputs, inputs)");
  }
  if (x.ndim() != 3) {
    throw std::invalid_argument("x must have shape (batch, samples, inputs)");
  }

  const py::ssize_t state_order = A.shape(0);
  const py::ssize_t n_inputs = D.shape(1);
  const py::ssize_t n_outputs = D.shape(0);
  if (B.shape(0) != state_order || B.shape(1) != n_inputs) {
    throw std::invalid_argument("B must have shape (state_order, inputs)");
  }
  if (C.shape(0) != n_outputs || C.shape(1) != state_order) {
    throw std::invalid_argument("C must have shape (outputs, state_order)");
  }
  if (x.shape(2) != n_inputs) {
    throw std::invalid_argument("x must have shape (batch, samples, inputs)");
  }
  if (n_inputs <= 0 || n_outputs <= 0) {
    throw std::invalid_argument("inputs and outputs must be positive");
  }

  validate_finite_2d(A, "A");
  validate_finite_2d(B, "B");
  validate_finite_2d(C, "C");
  validate_finite_2d(D, "D");
  validate_finite_3d(x, "x");

  const py::ssize_t batch = x.shape(0);
  const py::ssize_t samples = x.shape(1);
  py::array_t<double> y({batch, samples, n_outputs});

  auto av = A.unchecked<2>();
  auto bv = B.unchecked<2>();
  auto cv = C.unchecked<2>();
  auto dv = D.unchecked<2>();
  auto xv = x.unchecked<3>();
  auto yv = y.mutable_unchecked<3>();

  auto process_one_batch = [&](py::ssize_t bidx) {
    std::vector<double> state(static_cast<std::size_t>(state_order), 0.0);
    std::vector<double> next_state(static_cast<std::size_t>(state_order), 0.0);

    for (py::ssize_t n = 0; n < samples; ++n) {
      for (py::ssize_t out = 0; out < n_outputs; ++out) {
        double value = 0.0;
        for (py::ssize_t st = 0; st < state_order; ++st) {
          value += cv(out, st) * state[static_cast<std::size_t>(st)];
        }
        for (py::ssize_t in = 0; in < n_inputs; ++in) {
          value += dv(out, in) * xv(bidx, n, in);
        }
        yv(bidx, n, out) = value;
      }

      if (state_order > 0) {
        for (py::ssize_t st = 0; st < state_order; ++st) {
          double value = 0.0;
          for (py::ssize_t prev = 0; prev < state_order; ++prev) {
            value += av(st, prev) * state[static_cast<std::size_t>(prev)];
          }
          for (py::ssize_t in = 0; in < n_inputs; ++in) {
            value += bv(st, in) * xv(bidx, n, in);
          }
          next_state[static_cast<std::size_t>(st)] = value;
        }
        state.swap(next_state);
      }
    }
  };

  {
    py::gil_scoped_release release;
#ifdef LATTICE_DSP_HAS_OPENMP
    if (n_threads > 0) {
#pragma omp parallel for schedule(static) num_threads(n_threads)
      for (py::ssize_t bidx = 0; bidx < batch; ++bidx) {
        process_one_batch(bidx);
      }
    } else {
#pragma omp parallel for schedule(static)
      for (py::ssize_t bidx = 0; bidx < batch; ++bidx) {
        process_one_batch(bidx);
      }
    }
#else
    (void)n_threads;
    for (py::ssize_t bidx = 0; bidx < batch; ++bidx) {
      process_one_batch(bidx);
    }
#endif
  }

  return y;
}

py::tuple adaptive_process_batch(const std::vector<double>& initial_reflection,
                                 const std::vector<double>& initial_taps,
                                 py::array_t<double, py::array::c_style | py::array::forcecast> x,
                                 py::array_t<double, py::array::c_style | py::array::forcecast> desired,
                                 double mu_taps,
                                 double mu_reflection,
                                 double epsilon,
                                 double margin,
                                 bool freeze_reflection,
                                 const std::string& gradient_mode,
                                 std::size_t reflection_update_period,
                                 bool scale_reflection_mu_by_period,
                                 int n_threads) {
  if (x.ndim() != 2 || desired.ndim() != 2) {
    throw std::invalid_argument("adaptive_process_batch expects 2-D channel-by-sample NumPy arrays");
  }
  if (x.shape(0) != desired.shape(0) || x.shape(1) != desired.shape(1)) {
    throw std::invalid_argument("x and desired must have the same shape");
  }

  const py::ssize_t channels = x.shape(0);
  const py::ssize_t samples = x.shape(1);
  const py::ssize_t order = static_cast<py::ssize_t>(initial_reflection.size());
  const py::ssize_t n_taps = static_cast<py::ssize_t>(initial_taps.size());

  // Validate once before the OpenMP region so numerical input errors are
  // reported as normal Python exceptions rather than escaping a worker thread.
  validate_finite_2d(x, "x");
  validate_finite_2d(desired, "desired");
  AdaptiveLatticeLadderNLMS validator(initial_reflection, initial_taps, mu_taps, mu_reflection,
                                      epsilon, margin, freeze_reflection, gradient_mode,
                                      reflection_update_period, scale_reflection_mu_by_period);
  (void)validator;

  py::array_t<double> y({channels, samples});
  py::array_t<double> e({channels, samples});
  py::array_t<double> final_reflection({channels, order});
  py::array_t<double> final_taps({channels, n_taps});

  auto xin = x.unchecked<2>();
  auto din = desired.unchecked<2>();
  auto yout = y.mutable_unchecked<2>();
  auto eout = e.mutable_unchecked<2>();
  auto rout = final_reflection.mutable_unchecked<2>();
  auto tout = final_taps.mutable_unchecked<2>();

  {
    py::gil_scoped_release release;
#ifdef LATTICE_DSP_HAS_OPENMP
    if (n_threads > 0) {
#pragma omp parallel for schedule(static) num_threads(n_threads)
      for (py::ssize_t ch = 0; ch < channels; ++ch) {
        AdaptiveLatticeLadderNLMS adaptive(initial_reflection, initial_taps, mu_taps, mu_reflection,
                                           epsilon, margin, freeze_reflection, gradient_mode,
                                           reflection_update_period, scale_reflection_mu_by_period);
        for (py::ssize_t n = 0; n < samples; ++n) {
          auto [yn, en] = adaptive.adapt_sample(xin(ch, n), din(ch, n));
          yout(ch, n) = yn;
          eout(ch, n) = en;
        }
        const auto& ref = adaptive.reflection();
        const auto& taps = adaptive.taps();
        for (py::ssize_t i = 0; i < order; ++i) {
          rout(ch, i) = ref[static_cast<std::size_t>(i)];
        }
        for (py::ssize_t i = 0; i < n_taps; ++i) {
          tout(ch, i) = taps[static_cast<std::size_t>(i)];
        }
      }
    } else {
#pragma omp parallel for schedule(static)
      for (py::ssize_t ch = 0; ch < channels; ++ch) {
        AdaptiveLatticeLadderNLMS adaptive(initial_reflection, initial_taps, mu_taps, mu_reflection,
                                           epsilon, margin, freeze_reflection, gradient_mode,
                                           reflection_update_period, scale_reflection_mu_by_period);
        for (py::ssize_t n = 0; n < samples; ++n) {
          auto [yn, en] = adaptive.adapt_sample(xin(ch, n), din(ch, n));
          yout(ch, n) = yn;
          eout(ch, n) = en;
        }
        const auto& ref = adaptive.reflection();
        const auto& taps = adaptive.taps();
        for (py::ssize_t i = 0; i < order; ++i) {
          rout(ch, i) = ref[static_cast<std::size_t>(i)];
        }
        for (py::ssize_t i = 0; i < n_taps; ++i) {
          tout(ch, i) = taps[static_cast<std::size_t>(i)];
        }
      }
    }
#else
    (void)n_threads;
    for (py::ssize_t ch = 0; ch < channels; ++ch) {
      AdaptiveLatticeLadderNLMS adaptive(initial_reflection, initial_taps, mu_taps, mu_reflection,
                                         epsilon, margin, freeze_reflection, gradient_mode,
                                         reflection_update_period, scale_reflection_mu_by_period);
      for (py::ssize_t n = 0; n < samples; ++n) {
        auto [yn, en] = adaptive.adapt_sample(xin(ch, n), din(ch, n));
        yout(ch, n) = yn;
        eout(ch, n) = en;
      }
      const auto& ref = adaptive.reflection();
      const auto& taps = adaptive.taps();
      for (py::ssize_t i = 0; i < order; ++i) {
        rout(ch, i) = ref[static_cast<std::size_t>(i)];
      }
      for (py::ssize_t i = 0; i < n_taps; ++i) {
        tout(ch, i) = taps[static_cast<std::size_t>(i)];
      }
    }
#endif
  }

  return py::make_tuple(y, e, final_reflection, final_taps);
}

} // namespace


py::tuple rls_process_batch(const std::vector<double>& reflection,
                            const std::vector<double>& initial_taps,
                            py::array_t<double, py::array::c_style | py::array::forcecast> x,
                            py::array_t<double, py::array::c_style | py::array::forcecast> desired,
                            double forgetting_factor,
                            double initial_inverse_covariance,
                            double epsilon,
                            int n_threads) {
  if (x.ndim() != 2 || desired.ndim() != 2) {
    throw std::invalid_argument("rls_process_batch expects 2-D channel-by-sample NumPy arrays");
  }
  if (x.shape(0) != desired.shape(0) || x.shape(1) != desired.shape(1)) {
    throw std::invalid_argument("x and desired must have the same shape");
  }
  const py::ssize_t channels = x.shape(0);
  const py::ssize_t samples = x.shape(1);
  const py::ssize_t n_taps = static_cast<py::ssize_t>(initial_taps.size());

  validate_finite_2d(x, "x");
  validate_finite_2d(desired, "desired");
  LatticeLadderRLS validator(reflection, initial_taps, forgetting_factor,
                             initial_inverse_covariance, epsilon);
  (void)validator;

  py::array_t<double> y({channels, samples});
  py::array_t<double> e({channels, samples});
  py::array_t<double> final_taps({channels, n_taps});

  auto xin = x.unchecked<2>();
  auto din = desired.unchecked<2>();
  auto yout = y.mutable_unchecked<2>();
  auto eout = e.mutable_unchecked<2>();
  auto tout = final_taps.mutable_unchecked<2>();

  {
    py::gil_scoped_release release;
#ifdef LATTICE_DSP_HAS_OPENMP
    if (n_threads > 0) {
#pragma omp parallel for schedule(static) num_threads(n_threads)
      for (py::ssize_t ch = 0; ch < channels; ++ch) {
        LatticeLadderRLS adaptive(reflection, initial_taps, forgetting_factor,
                                  initial_inverse_covariance, epsilon);
        for (py::ssize_t n = 0; n < samples; ++n) {
          auto [yn, en] = adaptive.adapt_sample(xin(ch, n), din(ch, n));
          yout(ch, n) = yn;
          eout(ch, n) = en;
        }
        const auto& taps = adaptive.taps();
        for (py::ssize_t i = 0; i < n_taps; ++i) {
          tout(ch, i) = taps[static_cast<std::size_t>(i)];
        }
      }
    } else {
#pragma omp parallel for schedule(static)
      for (py::ssize_t ch = 0; ch < channels; ++ch) {
        LatticeLadderRLS adaptive(reflection, initial_taps, forgetting_factor,
                                  initial_inverse_covariance, epsilon);
        for (py::ssize_t n = 0; n < samples; ++n) {
          auto [yn, en] = adaptive.adapt_sample(xin(ch, n), din(ch, n));
          yout(ch, n) = yn;
          eout(ch, n) = en;
        }
        const auto& taps = adaptive.taps();
        for (py::ssize_t i = 0; i < n_taps; ++i) {
          tout(ch, i) = taps[static_cast<std::size_t>(i)];
        }
      }
    }
#else
    (void)n_threads;
    for (py::ssize_t ch = 0; ch < channels; ++ch) {
      LatticeLadderRLS adaptive(reflection, initial_taps, forgetting_factor,
                                initial_inverse_covariance, epsilon);
      for (py::ssize_t n = 0; n < samples; ++n) {
        auto [yn, en] = adaptive.adapt_sample(xin(ch, n), din(ch, n));
        yout(ch, n) = yn;
        eout(ch, n) = en;
      }
      const auto& taps = adaptive.taps();
      for (py::ssize_t i = 0; i < n_taps; ++i) {
        tout(ch, i) = taps[static_cast<std::size_t>(i)];
      }
    }
#endif
  }
  return py::make_tuple(y, e, final_taps);
}


namespace {

using Complex = std::complex<double>;

std::vector<Complex> matmul_complex(const std::vector<Complex>& a,
                                    const std::vector<Complex>& b,
                                    std::size_t m) {
  std::vector<Complex> out(m * m, Complex{0.0, 0.0});
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t k = 0; k < m; ++k) {
      const Complex aik = a[i * m + k];
      for (std::size_t j = 0; j < m; ++j) {
        out[i * m + j] += aik * b[k * m + j];
      }
    }
  }
  return out;
}

std::vector<Complex> solve_complex_matrix(std::vector<Complex> a,
                                          std::vector<Complex> b,
                                          std::size_t m) {
  // Gaussian elimination with partial pivoting for A X = B.  Matrices are
  // row-major m x m.  This is intentionally small-matrix code for MIMO stages.
  for (std::size_t col = 0; col < m; ++col) {
    std::size_t pivot = col;
    double best = std::abs(a[col * m + col]);
    for (std::size_t row = col + 1; row < m; ++row) {
      const double candidate = std::abs(a[row * m + col]);
      if (candidate > best) {
        best = candidate;
        pivot = row;
      }
    }
    if (!(best > 0.0) || !std::isfinite(best)) {
      throw std::runtime_error("singular matrix while evaluating matrix lattice response");
    }
    if (pivot != col) {
      for (std::size_t j = 0; j < m; ++j) {
        std::swap(a[col * m + j], a[pivot * m + j]);
        std::swap(b[col * m + j], b[pivot * m + j]);
      }
    }

    const Complex diag = a[col * m + col];
    for (std::size_t j = 0; j < m; ++j) {
      a[col * m + j] /= diag;
      b[col * m + j] /= diag;
    }
    for (std::size_t row = 0; row < m; ++row) {
      if (row == col) {
        continue;
      }
      const Complex factor = a[row * m + col];
      if (factor == Complex{0.0, 0.0}) {
        continue;
      }
      for (std::size_t j = 0; j < m; ++j) {
        a[row * m + j] -= factor * a[col * m + j];
        b[row * m + j] -= factor * b[col * m + j];
      }
    }
  }
  return b;
}

py::array_t<Complex> matrix_lattice_frequency_response_cpp(
    py::array_t<Complex, py::array::c_style | py::array::forcecast> stage_blocks,
    py::array_t<Complex, py::array::c_style | py::array::forcecast> residue,
    py::array_t<double, py::array::c_style | py::array::forcecast> omega,
    int n_threads) {
  if (stage_blocks.ndim() != 4) {
    throw std::invalid_argument("stage_blocks must have shape (stages, 4, dim, dim)");
  }
  if (stage_blocks.shape(1) != 4 || stage_blocks.shape(2) != stage_blocks.shape(3)) {
    throw std::invalid_argument("stage_blocks must have shape (stages, 4, dim, dim)");
  }
  if (residue.ndim() != 2 || residue.shape(0) != residue.shape(1)) {
    throw std::invalid_argument("residue must be a square 2-D complex array");
  }
  if (omega.ndim() != 1) {
    throw std::invalid_argument("omega must be a 1-D array");
  }
  const py::ssize_t stages = stage_blocks.shape(0);
  const py::ssize_t m_py = stage_blocks.shape(2);
  if (residue.shape(0) != m_py) {
    throw std::invalid_argument("residue dimension must match stage_blocks");
  }
  const std::size_t m = static_cast<std::size_t>(m_py);
  const py::ssize_t n_freq = omega.shape(0);
  py::array_t<Complex> response({n_freq, m_py, m_py});

  auto blocks = stage_blocks.unchecked<4>();
  auto res = residue.unchecked<2>();
  auto w = omega.unchecked<1>();
  auto out = response.mutable_unchecked<3>();

  {
    py::gil_scoped_release release;
#ifdef LATTICE_DSP_HAS_OPENMP
    if (n_threads > 0) {
#pragma omp parallel for schedule(static) num_threads(n_threads)
      for (py::ssize_t wi = 0; wi < n_freq; ++wi) {
        std::vector<Complex> g(m * m);
        for (std::size_t i = 0; i < m; ++i) {
          for (std::size_t j = 0; j < m; ++j) {
            g[i * m + j] = res(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(j));
          }
        }
        const Complex z = std::exp(Complex{0.0, -w(wi)});
        for (py::ssize_t stage = stages - 1; stage >= 0; --stage) {
          std::vector<Complex> t11(m * m), t12(m * m), t21(m * m), t22(m * m), zg(m * m);
          for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < m; ++j) {
              const auto ii = static_cast<py::ssize_t>(i);
              const auto jj = static_cast<py::ssize_t>(j);
              t11[i * m + j] = blocks(stage, 0, ii, jj);
              t12[i * m + j] = blocks(stage, 1, ii, jj);
              t21[i * m + j] = blocks(stage, 2, ii, jj);
              t22[i * m + j] = blocks(stage, 3, ii, jj);
              zg[i * m + j] = z * g[i * m + j];
            }
          }
          auto t22_zg = matmul_complex(t22, zg, m);
          std::vector<Complex> a(m * m, Complex{0.0, 0.0});
          for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < m; ++j) {
              a[i * m + j] = (i == j ? Complex{1.0, 0.0} : Complex{0.0, 0.0}) - t22_zg[i * m + j];
            }
          }
          auto x = solve_complex_matrix(std::move(a), t21, m);
          auto zg_x = matmul_complex(zg, x, m);
          auto term = matmul_complex(t12, zg_x, m);
          for (std::size_t i = 0; i < m * m; ++i) {
            g[i] = t11[i] + term[i];
          }
        }
        for (std::size_t i = 0; i < m; ++i) {
          for (std::size_t j = 0; j < m; ++j) {
            out(wi, static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(j)) = g[i * m + j];
          }
        }
      }
    } else {
#pragma omp parallel for schedule(static)
      for (py::ssize_t wi = 0; wi < n_freq; ++wi) {
        std::vector<Complex> g(m * m);
        for (std::size_t i = 0; i < m; ++i) {
          for (std::size_t j = 0; j < m; ++j) {
            g[i * m + j] = res(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(j));
          }
        }
        const Complex z = std::exp(Complex{0.0, -w(wi)});
        for (py::ssize_t stage = stages - 1; stage >= 0; --stage) {
          std::vector<Complex> t11(m * m), t12(m * m), t21(m * m), t22(m * m), zg(m * m);
          for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < m; ++j) {
              const auto ii = static_cast<py::ssize_t>(i);
              const auto jj = static_cast<py::ssize_t>(j);
              t11[i * m + j] = blocks(stage, 0, ii, jj);
              t12[i * m + j] = blocks(stage, 1, ii, jj);
              t21[i * m + j] = blocks(stage, 2, ii, jj);
              t22[i * m + j] = blocks(stage, 3, ii, jj);
              zg[i * m + j] = z * g[i * m + j];
            }
          }
          auto t22_zg = matmul_complex(t22, zg, m);
          std::vector<Complex> a(m * m, Complex{0.0, 0.0});
          for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < m; ++j) {
              a[i * m + j] = (i == j ? Complex{1.0, 0.0} : Complex{0.0, 0.0}) - t22_zg[i * m + j];
            }
          }
          auto x = solve_complex_matrix(std::move(a), t21, m);
          auto zg_x = matmul_complex(zg, x, m);
          auto term = matmul_complex(t12, zg_x, m);
          for (std::size_t i = 0; i < m * m; ++i) {
            g[i] = t11[i] + term[i];
          }
        }
        for (std::size_t i = 0; i < m; ++i) {
          for (std::size_t j = 0; j < m; ++j) {
            out(wi, static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(j)) = g[i * m + j];
          }
        }
      }
    }
#else
    (void)n_threads;
    for (py::ssize_t wi = 0; wi < n_freq; ++wi) {
      std::vector<Complex> g(m * m);
      for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
          g[i * m + j] = res(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(j));
        }
      }
      const Complex z = std::exp(Complex{0.0, -w(wi)});
      for (py::ssize_t stage = stages - 1; stage >= 0; --stage) {
        std::vector<Complex> t11(m * m), t12(m * m), t21(m * m), t22(m * m), zg(m * m);
        for (std::size_t i = 0; i < m; ++i) {
          for (std::size_t j = 0; j < m; ++j) {
            const auto ii = static_cast<py::ssize_t>(i);
            const auto jj = static_cast<py::ssize_t>(j);
            t11[i * m + j] = blocks(stage, 0, ii, jj);
            t12[i * m + j] = blocks(stage, 1, ii, jj);
            t21[i * m + j] = blocks(stage, 2, ii, jj);
            t22[i * m + j] = blocks(stage, 3, ii, jj);
            zg[i * m + j] = z * g[i * m + j];
          }
        }
        auto t22_zg = matmul_complex(t22, zg, m);
        std::vector<Complex> a(m * m, Complex{0.0, 0.0});
        for (std::size_t i = 0; i < m; ++i) {
          for (std::size_t j = 0; j < m; ++j) {
            a[i * m + j] = (i == j ? Complex{1.0, 0.0} : Complex{0.0, 0.0}) - t22_zg[i * m + j];
          }
        }
        auto x = solve_complex_matrix(std::move(a), t21, m);
        auto zg_x = matmul_complex(zg, x, m);
        auto term = matmul_complex(t12, zg_x, m);
        for (std::size_t i = 0; i < m * m; ++i) {
          g[i] = t11[i] + term[i];
        }
      }
      for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
          out(wi, static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(j)) = g[i * m + j];
        }
      }
    }
#endif
  }
  return response;
}



py::array_t<double> flat_to_array2(const std::vector<double>& values,
                                   std::size_t rows,
                                   std::size_t cols) {
  py::array_t<double> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
  auto view = out.mutable_unchecked<2>();
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      view(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(j)) = values[i * cols + j];
    }
  }
  return out;
}

py::array_t<double> flat_to_array3(const std::vector<double>& values,
                                   std::size_t n0,
                                   std::size_t n1,
                                   std::size_t n2) {
  py::array_t<double> out({static_cast<py::ssize_t>(n0), static_cast<py::ssize_t>(n1), static_cast<py::ssize_t>(n2)});
  auto view = out.mutable_unchecked<3>();
  for (std::size_t i = 0; i < n0; ++i) {
    for (std::size_t j = 0; j < n1; ++j) {
      for (std::size_t k = 0; k < n2; ++k) {
        view(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(j), static_cast<py::ssize_t>(k)) =
            values[(i * n1 + j) * n2 + k];
      }
    }
  }
  return out;
}

py::dict finite_hankel_mimo_result_to_dict(const lattice_dsp::FiniteHankelMimoReduction& result) {
  py::dict out;
  out["A"] = flat_to_array2(result.a, result.state_order, result.state_order);
  out["B"] = flat_to_array2(result.b, result.state_order, result.n_inputs);
  out["C"] = flat_to_array2(result.c, result.n_outputs, result.state_order);
  out["D"] = flat_to_array2(result.d, result.n_outputs, result.n_inputs);
  out["hankel_singular_values"] = result.hankel_singular_values;
  out["retained_hankel_energy"] = result.retained_hankel_energy;
  out["relative_markov_error"] = result.relative_markov_error;
  out["stable"] = result.stable;
  out["method"] = result.method;
  out["state_order"] = result.state_order;
  out["n_outputs"] = result.n_outputs;
  out["n_inputs"] = result.n_inputs;
  return out;
}

std::vector<double> flatten_3d_markov(py::array_t<double, py::array::c_style | py::array::forcecast> markov) {
  if (markov.ndim() != 3) {
    throw std::invalid_argument("markov_parameters must have shape (samples, outputs, inputs)");
  }
  if (markov.shape(0) <= 0 || markov.shape(1) <= 0 || markov.shape(2) <= 0) {
    throw std::invalid_argument("markov_parameters dimensions must be positive");
  }
  auto view = markov.unchecked<3>();
  std::vector<double> flat(static_cast<std::size_t>(markov.shape(0) * markov.shape(1) * markov.shape(2)));
  for (py::ssize_t n = 0; n < markov.shape(0); ++n) {
    for (py::ssize_t y = 0; y < markov.shape(1); ++y) {
      for (py::ssize_t u = 0; u < markov.shape(2); ++u) {
        const double v = view(n, y, u);
        if (!std::isfinite(v)) {
          throw std::invalid_argument("markov_parameters must contain only finite values");
        }
        flat[static_cast<std::size_t>((n * markov.shape(1) + y) * markov.shape(2) + u)] = v;
      }
    }
  }
  return flat;
}

std::vector<double> flatten_2d(py::array_t<double, py::array::c_style | py::array::forcecast> arr,
                               py::ssize_t rows,
                               py::ssize_t cols,
                               const char* name) {
  if (arr.ndim() != 2 || arr.shape(0) != rows || arr.shape(1) != cols) {
    throw std::invalid_argument(std::string(name) + " has wrong shape");
  }
  auto view = arr.unchecked<2>();
  std::vector<double> flat(static_cast<std::size_t>(rows * cols));
  for (py::ssize_t i = 0; i < rows; ++i) {
    for (py::ssize_t j = 0; j < cols; ++j) {
      const double v = view(i, j);
      if (!std::isfinite(v)) {
        throw std::invalid_argument(std::string(name) + " must contain only finite values");
      }
      flat[static_cast<std::size_t>(i * cols + j)] = v;
    }
  }
  return flat;
}


py::dict finite_nehari_result_to_dict(const lattice_dsp::FiniteNehariApproximation& result) {
  py::dict out;
  out["approximated_tail"] = result.approximated_tail;
  out["hankel_singular_values"] = result.hankel_singular_values;
  out["sigma_next"] = result.sigma_next;
  out["unconstrained_hankel_error"] = result.unconstrained_hankel_error;
  out["hankelized_hankel_error"] = result.hankelized_hankel_error;
  out["relative_tail_error"] = result.relative_tail_error;
  out["method"] = result.method;
  out["rank"] = result.rank;
  out["rows"] = result.rows;
  out["cols"] = result.cols;
  return out;
}

py::dict finite_hankel_result_to_dict(const lattice_dsp::FiniteHankelReduction& result) {
  py::dict out;
  out["numerator"] = result.numerator;
  out["denominator"] = result.denominator;
  out["reflection"] = result.reflection;
  out["hankel_singular_values"] = result.hankel_singular_values;
  out["retained_hankel_energy"] = result.retained_hankel_energy;
  out["relative_impulse_error"] = result.relative_impulse_error;
  out["stable"] = result.stable;
  out["method"] = result.method;
  return out;
}

} // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "C++ core for stable lattice/lattice-ladder DSP filters";

#ifdef LATTICE_DSP_HAS_OPENMP
  m.attr("HAS_OPENMP") = true;
#else
  m.attr("HAS_OPENMP") = false;
#endif

  py::class_<LatticeIIR>(m, "LatticeIIR")
      .def(py::init<std::vector<double>, std::vector<double>>(),
           py::arg("reflection"), py::arg("taps"),
           "Create a stable reflection-parameterized IIR filter.")
      .def_property_readonly("order", &LatticeIIR::order)
      .def("reset", &LatticeIIR::reset, py::arg("value") = 0.0)
      .def("process_sample", &LatticeIIR::process_sample, py::arg("x"))
      .def("process", &process_array<LatticeIIR>, py::arg("x"))
      .def("set_reflection", &LatticeIIR::set_reflection, py::arg("reflection"))
      .def("set_reflection_preserve_state", &LatticeIIR::set_reflection_preserve_state, py::arg("reflection"))
      .def("set_numerator", &LatticeIIR::set_numerator, py::arg("numerator"))
      .def("set_taps", &LatticeIIR::set_numerator, py::arg("taps"))
      .def_property_readonly("reflection", &LatticeIIR::reflection)
      .def_property_readonly("denominator", &LatticeIIR::denominator)
      .def_property_readonly("numerator", &LatticeIIR::numerator)
      .def_property_readonly("taps", &LatticeIIR::numerator)
      .def_property_readonly("state", &LatticeIIR::state)
      .def_property_readonly("last_basis", &LatticeIIR::last_basis);

  py::class_<LatticeLadderIIR>(m, "LatticeLadderIIR")
      .def(py::init<std::vector<double>, std::vector<double>>(),
           py::arg("reflection"), py::arg("ladder_taps"),
           "Create a true synthesis lattice-ladder IIR filter.")
      .def_property_readonly("order", &LatticeLadderIIR::order)
      .def("reset", &LatticeLadderIIR::reset, py::arg("value") = 0.0)
      .def("process_sample", &LatticeLadderIIR::process_sample, py::arg("x"))
      .def("process", &process_array<LatticeLadderIIR>, py::arg("x"))
      .def("set_reflection", &LatticeLadderIIR::set_reflection, py::arg("reflection"))
      .def("set_ladder", &LatticeLadderIIR::set_ladder, py::arg("ladder_taps"))
      .def("set_taps", &LatticeLadderIIR::set_ladder, py::arg("ladder_taps"))
      .def_property_readonly("reflection", &LatticeLadderIIR::reflection)
      .def_property_readonly("ladder", &LatticeLadderIIR::ladder)
      .def_property_readonly("taps", &LatticeLadderIIR::taps)
      .def_property_readonly("denominator", &LatticeLadderIIR::denominator)
      .def_property_readonly("numerator", &LatticeLadderIIR::numerator)
      .def_property_readonly("state", &LatticeLadderIIR::state)
      .def_property_readonly("last_forward", &LatticeLadderIIR::last_forward)
      .def_property_readonly("last_backward", &LatticeLadderIIR::last_backward);

  py::class_<AdaptiveLatticeLadderNLMS>(m, "AdaptiveLatticeLadderNLMS")
      .def(py::init<std::vector<double>, std::vector<double>, double, double, double, double, bool, std::string, std::size_t, bool>(),
           py::arg("initial_reflection"), py::arg("initial_taps"),
           py::arg("mu_taps") = 0.05, py::arg("mu_reflection") = 0.001,
           py::arg("epsilon") = 1e-8, py::arg("margin") = 1e-4,
           py::arg("freeze_reflection") = false,
           py::arg("gradient_mode") = "analytic",
           py::arg("reflection_update_period") = 1,
           py::arg("scale_reflection_mu_by_period") = false,
           "Experimental stable adaptive IIR: NLMS numerator updates plus bounded reflection updates.")
      .def("reset", &AdaptiveLatticeLadderNLMS::reset, py::arg("value") = 0.0)
      .def("adapt_sample", &adapt_full_sample, py::arg("x"), py::arg("desired"))
      .def("adapt_block", &AdaptiveLatticeLadderNLMS::adapt_block, py::arg("x"), py::arg("desired"),
           "Legacy vector/list API returning only the error signal. Prefer process_adapt for NumPy arrays.")
      .def("process_adapt", &adapt_array<AdaptiveLatticeLadderNLMS>, py::arg("x"), py::arg("desired"),
           "Run the full adaptive loop in C++ for 1-D NumPy arrays. Returns (y, error).")
      .def_property_readonly("numerator", &AdaptiveLatticeLadderNLMS::numerator)
      .def_property_readonly("taps", &AdaptiveLatticeLadderNLMS::taps)
      .def_property_readonly("ladder", &AdaptiveLatticeLadderNLMS::ladder)
      .def_property_readonly("reflection", &AdaptiveLatticeLadderNLMS::reflection)
      .def_property_readonly("raw_reflection", &AdaptiveLatticeLadderNLMS::raw_reflection)
      .def_property_readonly("denominator", &AdaptiveLatticeLadderNLMS::denominator)
      .def_property_readonly("last_tap_gradient", &AdaptiveLatticeLadderNLMS::last_tap_gradient)
      .def_property_readonly("last_reflection_gradient", &AdaptiveLatticeLadderNLMS::last_reflection_gradient)
      .def_property_readonly("last_raw_gradient", &AdaptiveLatticeLadderNLMS::last_raw_gradient)
      .def_property("mu_taps", &AdaptiveLatticeLadderNLMS::mu_taps,
                    &AdaptiveLatticeLadderNLMS::set_mu_taps)
      .def_property("mu_reflection", &AdaptiveLatticeLadderNLMS::mu_reflection,
                    &AdaptiveLatticeLadderNLMS::set_mu_reflection)
      .def_property("freeze_reflection", &AdaptiveLatticeLadderNLMS::freeze_reflection,
                    &AdaptiveLatticeLadderNLMS::set_freeze_reflection)
      .def_property("reflection_update_period", &AdaptiveLatticeLadderNLMS::reflection_update_period,
                    &AdaptiveLatticeLadderNLMS::set_reflection_update_period)
      .def_property("scale_reflection_mu_by_period",
                    &AdaptiveLatticeLadderNLMS::scale_reflection_mu_by_period,
                    &AdaptiveLatticeLadderNLMS::set_scale_reflection_mu_by_period)
      .def_property("gradient_mode", &AdaptiveLatticeLadderNLMS::gradient_mode,
                    &AdaptiveLatticeLadderNLMS::set_gradient_mode)
      .def_property_readonly("margin", &AdaptiveLatticeLadderNLMS::margin);

  py::class_<LatticeLadderNLMS>(m, "LatticeLadderNLMS")
      .def(py::init<std::vector<double>, std::vector<double>, double, double>(),
           py::arg("reflection"), py::arg("initial_taps"),
           py::arg("mu") = 0.1, py::arg("epsilon") = 1e-8,
           "NLMS adaptation of the ladder taps for a fixed stable lattice denominator.")
      .def("reset", &LatticeLadderNLMS::reset, py::arg("value") = 0.0)
      .def("adapt_sample", &adapt_sample, py::arg("x"), py::arg("desired"))
      .def("adapt_block", &LatticeLadderNLMS::adapt_block, py::arg("x"), py::arg("desired"),
           "Legacy vector/list API returning only the error signal. Prefer process_adapt for NumPy arrays.")
      .def("process_adapt", &adapt_array<LatticeLadderNLMS>, py::arg("x"), py::arg("desired"),
           "Run the full adaptive loop in C++ for 1-D NumPy arrays. Returns (y, error).")
      .def_property_readonly("numerator", &LatticeLadderNLMS::numerator)
      .def_property_readonly("taps", &LatticeLadderNLMS::taps)
      .def_property_readonly("reflection", &LatticeLadderNLMS::reflection)
      .def_property_readonly("denominator", &LatticeLadderNLMS::denominator)
      .def_property("mu", &LatticeLadderNLMS::mu, &LatticeLadderNLMS::set_mu);


  py::class_<LatticeLadderRLS>(m, "LatticeLadderRLS")
      .def(py::init<std::vector<double>, std::vector<double>, double, double, double>(),
           py::arg("reflection"), py::arg("initial_taps"),
           py::arg("forgetting_factor") = 0.995,
           py::arg("initial_inverse_covariance") = 1000.0,
           py::arg("epsilon") = 1e-12,
           "RLS adaptation of numerator taps for a fixed stable reflection-parameterized denominator.")
      .def("reset", &LatticeLadderRLS::reset, py::arg("value") = 0.0)
      .def("reset_inverse_covariance", &LatticeLadderRLS::reset_inverse_covariance,
           py::arg("initial_inverse_covariance"))
      .def("adapt_sample", [](LatticeLadderRLS& adaptive, double x, double d) {
          auto [y, e] = adaptive.adapt_sample(x, d);
          return py::make_tuple(y, e);
        }, py::arg("x"), py::arg("desired"))
      .def("adapt_block", &LatticeLadderRLS::adapt_block, py::arg("x"), py::arg("desired"),
           "Legacy vector/list API returning only the error signal. Prefer process_adapt for NumPy arrays.")
      .def("process_adapt", &adapt_array<LatticeLadderRLS>, py::arg("x"), py::arg("desired"),
           "Run the RLS adaptive loop in C++ for 1-D NumPy arrays. Returns (y, error).")
      .def_property_readonly("numerator", &LatticeLadderRLS::numerator)
      .def_property_readonly("taps", &LatticeLadderRLS::taps)
      .def_property_readonly("reflection", &LatticeLadderRLS::reflection)
      .def_property_readonly("denominator", &LatticeLadderRLS::denominator)
      .def_property_readonly("inverse_covariance", &LatticeLadderRLS::inverse_covariance)
      .def_property_readonly("last_gain", &LatticeLadderRLS::last_gain)
      .def_property("forgetting_factor", &LatticeLadderRLS::forgetting_factor,
                    &LatticeLadderRLS::set_forgetting_factor)
      .def_property_readonly("epsilon", &LatticeLadderRLS::epsilon);

  py::class_<AdaptiveNotch>(m, "AdaptiveNotch")
      .def(py::init<double, double, double, double>(),
           py::arg("theta") = 0.25, py::arg("pole_radius") = 0.98,
           py::arg("mu") = 0.005, py::arg("epsilon") = 1e-8,
           "Adaptive second-order notch filter with normalized gradient angle updates.")
      .def("reset", &AdaptiveNotch::reset)
      .def("process_sample", &AdaptiveNotch::process_sample, py::arg("x"))
      .def("process", [](AdaptiveNotch& f, py::array_t<double, py::array::c_style | py::array::forcecast> x) {
          if (x.ndim() != 1) {
            throw std::invalid_argument("process expects a 1-D NumPy array");
          }
          auto xin = x.unchecked<1>();
          py::array_t<double> y(x.shape(0));
          auto yout = y.mutable_unchecked<1>();
          for (py::ssize_t i = 0; i < x.shape(0); ++i) {
            yout(i) = f.process_sample(xin(i));
          }
          return y;
        }, py::arg("x"))
      .def_property_readonly("theta", &AdaptiveNotch::theta)
      .def_property_readonly("coefficient", &AdaptiveNotch::coefficient)
      .def_property_readonly("pole_radius", &AdaptiveNotch::pole_radius)
      .def_property("mu", &AdaptiveNotch::mu, &AdaptiveNotch::set_mu)
      .def("set_theta", &AdaptiveNotch::set_theta, py::arg("theta"))
      .def("set_coefficient", &AdaptiveNotch::set_coefficient, py::arg("a"));

  m.def("process_batch", &process_batch,
        py::arg("reflection"), py::arg("taps"), py::arg("x"), py::arg("n_threads") = 0,
        py::arg("realization") = "direct",
        "Process one signal or a channel-by-sample matrix. Rows are parallelized with OpenMP. "
        "realization='direct' uses the reference direct-form core; realization='lattice' "
        "converts direct numerator taps to ladder taps and uses the synthesis lattice-ladder core.");

  m.def("adaptive_process_batch", &adaptive_process_batch,
        py::arg("initial_reflection"), py::arg("initial_taps"), py::arg("x"), py::arg("desired"),
        py::arg("mu_taps") = 0.05, py::arg("mu_reflection") = 0.001,
        py::arg("epsilon") = 1e-8, py::arg("margin") = 1e-4,
        py::arg("freeze_reflection") = false, py::arg("gradient_mode") = "analytic",
        py::arg("reflection_update_period") = 1, py::arg("scale_reflection_mu_by_period") = false,
        py::arg("n_threads") = 0,
        "Run independent adaptive IIR problems over a channel-by-sample matrix. Rows are parallelized "
        "with OpenMP. Returns (y, error, final_reflection, final_taps).");


  m.def("rls_process_batch", &rls_process_batch,
        py::arg("reflection"), py::arg("initial_taps"), py::arg("x"), py::arg("desired"),
        py::arg("forgetting_factor") = 0.995,
        py::arg("initial_inverse_covariance") = 1000.0,
        py::arg("epsilon") = 1e-12,
        py::arg("n_threads") = 0,
        "Run independent fixed-denominator RLS problems over a channel-by-sample matrix. "
        "Rows are parallelized with OpenMP. Returns (y, error, final_taps).");


  m.def("matrix_lattice_frequency_response", &matrix_lattice_frequency_response_cpp,
        py::arg("stage_blocks"), py::arg("residue"), py::arg("omega"), py::arg("n_threads") = 0,
        "Evaluate complex matrix lattice/all-pass frequency responses. stage_blocks has shape "
        "(stages, 4, dim, dim), residue has shape (dim, dim), and omega is 1-D. "
        "Frequencies are parallelized with OpenMP when available.");

  m.def("autocorrelation", &lattice_dsp::autocorrelation,
        py::arg("x"), py::arg("max_lag"), py::arg("biased") = true,
        "Compute autocorrelation r[0]...r[max_lag].");

  m.def("levinson_durbin_reflection", &lattice_dsp::levinson_durbin_reflection,
        py::arg("autocorr"), py::arg("order"), py::arg("regularization") = 1e-12,
        "Levinson-Durbin recursion returning stable PARCOR/reflection coefficients.");

  m.def("levinson_durbin_denominator", &lattice_dsp::levinson_durbin_denominator,
        py::arg("autocorr"), py::arg("order"), py::arg("regularization") = 1e-12,
        "Levinson-Durbin recursion returning a monic AR denominator.");

  m.def("levinson_durbin_error", &lattice_dsp::levinson_durbin_error,
        py::arg("autocorr"), py::arg("order"), py::arg("regularization") = 1e-12,
        "Final prediction error from the Levinson-Durbin recursion.");

  m.def("burg_reflection", &lattice_dsp::burg_reflection,
        py::arg("x"), py::arg("order"), py::arg("regularization") = 1e-12,
        "Burg AR estimator returning stable PARCOR/reflection coefficients.");

  m.def("burg_denominator", &lattice_dsp::burg_denominator,
        py::arg("x"), py::arg("order"), py::arg("regularization") = 1e-12,
        "Burg AR estimator returning a monic AR denominator.");

  m.def("reflection_to_denominator", &lattice_dsp::reflection_to_denominator,
        py::arg("reflection"),
        "Convert reflection/PARCOR coefficients to a monic denominator polynomial.");

  m.def("denominator_to_reflection", &lattice_dsp::denominator_to_reflection,
        py::arg("denominator"), py::arg("stability_tol") = 1e-12,
        "Convert a stable monic denominator polynomial to reflection/PARCOR coefficients.");

  m.def("bounded_reflection_from_raw", &lattice_dsp::bounded_reflection_from_raw,
        py::arg("raw"), py::arg("margin") = 1e-6,
        "Map unconstrained parameters to stable reflection coefficients using tanh.");

  m.def("denominator_reflection_jacobian", &lattice_dsp::denominator_reflection_jacobian,
        py::arg("reflection"),
        "Analytic Jacobian d[a1..aN]/d[k1..kN] for the reflection-to-denominator recursion.");

  m.def("denominator_raw_jacobian", &lattice_dsp::denominator_raw_jacobian,
        py::arg("raw"), py::arg("margin") = 1e-6,
        "Analytic Jacobian d[a1..aN]/d[raw1..rawN] through the bounded tanh map.");

  m.def("denominator_raw_jacobian_finite_difference",
        &lattice_dsp::denominator_raw_jacobian_finite_difference,
        py::arg("raw"), py::arg("margin") = 1e-6, py::arg("step_scale") = 1e-6,
        "Slow finite-difference Jacobian retained for testing and debugging.");


  m.def("iir_impulse_response", &lattice_dsp::iir_impulse_response,
        py::arg("denominator"), py::arg("numerator"), py::arg("n_samples"),
        "Compute the first n_samples of an IIR impulse response from direct numerator/denominator coefficients.");

  m.def("hankel_singular_values", &lattice_dsp::hankel_singular_values,
        py::arg("impulse_response"), py::arg("rows"), py::arg("cols"), py::arg("offset") = 1,
        "Finite-section Hankel singular values from an impulse response. This is a diagnostic for Hankel/Nehari/AAK-style model reduction.");

  auto finite_nehari_approximate_tail_py = [](const std::vector<double>& anticausal_tail,
                                              std::size_t rank,
                                              std::size_t rows,
                                              std::size_t cols,
                                              double regularization) {
          return finite_nehari_result_to_dict(lattice_dsp::finite_nehari_approximate_tail(
              anticausal_tail, rank, rows, cols, regularization));
        };

  m.def("finite_nehari_approximate_tail", finite_nehari_approximate_tail_py,
        py::arg("anticausal_tail"), py::arg("rank"), py::arg("rows"), py::arg("cols"),
        py::arg("regularization") = 1e-12,
        "Finite-section Nehari/AAK teaching helper for a SISO anticausal tail. Builds a Hankel matrix, "
        "computes the best unconstrained rank-r approximation, and Hankelizes it by anti-diagonal averaging. "
        "This is a finite-dimensional numerical baseline, not an exact infinite-dimensional Nehari or AAK solver.");

  auto finite_hankel_reduce_impulse_py = [](const std::vector<double>& impulse_response,
                                                std::size_t reduced_order,
                                                std::size_t rows,
                                                std::size_t cols,
                                                double regularization) {
          return finite_hankel_result_to_dict(lattice_dsp::finite_hankel_reduce_impulse(
              impulse_response, reduced_order, rows, cols, regularization));
        };

  auto finite_hankel_reduce_iir_py = [](const std::vector<double>& reflection,
                                            const std::vector<double>& numerator,
                                            std::size_t reduced_order,
                                            std::size_t n_impulse,
                                            std::size_t rows,
                                            std::size_t cols,
                                            double regularization) {
          return finite_hankel_result_to_dict(lattice_dsp::finite_hankel_reduce_iir(
              reflection, numerator, reduced_order, n_impulse, rows, cols, regularization));
        };

  m.def("finite_hankel_reduce_impulse", finite_hankel_reduce_impulse_py,
        py::arg("impulse_response"), py::arg("reduced_order"), py::arg("rows"), py::arg("cols"),
        py::arg("regularization") = 1e-12,
        "Finite-Hankel Ho-Kalman SISO reduction from an impulse response. Returns a dict with numerator, denominator, reflection, Hankel singular values, and diagnostics. This is a finite-section approximation, not an exact infinite-dimensional Nehari/AAK solver.");

  m.def("finite_hankel_reduce_iir", finite_hankel_reduce_iir_py,
        py::arg("reflection"), py::arg("numerator"), py::arg("reduced_order"),
        py::arg("n_impulse") = 512, py::arg("rows") = 64, py::arg("cols") = 64,
        py::arg("regularization") = 1e-12,
        "Finite-Hankel Ho-Kalman SISO reduction from a stable lattice IIR. Returns direct-form coefficients and diagnostics. This is a finite-section approximation, not an exact infinite-dimensional Nehari/AAK solver.");


  auto finite_hankel_reduce_mimo_py = [](py::array_t<double, py::array::c_style | py::array::forcecast> markov_parameters,
                                         std::size_t reduced_order,
                                         std::size_t block_rows,
                                         std::size_t block_cols,
                                         double regularization) {
          const auto flat = flatten_3d_markov(markov_parameters);
          return finite_hankel_mimo_result_to_dict(lattice_dsp::finite_hankel_reduce_mimo(
              flat,
              static_cast<std::size_t>(markov_parameters.shape(0)),
              static_cast<std::size_t>(markov_parameters.shape(1)),
              static_cast<std::size_t>(markov_parameters.shape(2)),
              reduced_order, block_rows, block_cols, regularization));
        };

  m.def("finite_hankel_reduce_mimo", finite_hankel_reduce_mimo_py,
        py::arg("markov_parameters"), py::arg("reduced_order"), py::arg("block_rows"), py::arg("block_cols"),
        py::arg("regularization") = 1e-12,
        "Finite block-Hankel Ho-Kalman reduction for MIMO Markov parameters. "
        "markov_parameters has shape (samples, outputs, inputs). Returns a state-space dict A, B, C, D and diagnostics. "
        "This is a finite-section baseline, not an exact matrix Nehari/AAK solver.");

  auto mimo_state_space_markov_response_py = [](py::array_t<double, py::array::c_style | py::array::forcecast> A,
                                                py::array_t<double, py::array::c_style | py::array::forcecast> B,
                                                py::array_t<double, py::array::c_style | py::array::forcecast> C,
                                                py::array_t<double, py::array::c_style | py::array::forcecast> D,
                                                std::size_t n_samples) {
          if (A.ndim() != 2 || A.shape(0) != A.shape(1)) {
            throw std::invalid_argument("A must have shape (state_order, state_order)");
          }
          if (D.ndim() != 2) {
            throw std::invalid_argument("D must have shape (outputs, inputs)");
          }
          const py::ssize_t state_order = A.shape(0);
          const py::ssize_t n_outputs = D.shape(0);
          const py::ssize_t n_inputs = D.shape(1);
          auto a_flat = flatten_2d(A, state_order, state_order, "A");
          auto b_flat = flatten_2d(B, state_order, n_inputs, "B");
          auto c_flat = flatten_2d(C, n_outputs, state_order, "C");
          auto d_flat = flatten_2d(D, n_outputs, n_inputs, "D");
          const auto out = lattice_dsp::mimo_state_space_markov_response(
              a_flat, b_flat, c_flat, d_flat,
              static_cast<std::size_t>(state_order),
              static_cast<std::size_t>(n_outputs),
              static_cast<std::size_t>(n_inputs),
              n_samples);
          return flat_to_array3(out, n_samples, static_cast<std::size_t>(n_outputs), static_cast<std::size_t>(n_inputs));
        };

  m.def("mimo_state_space_markov_response", mimo_state_space_markov_response_py,
        py::arg("A"), py::arg("B"), py::arg("C"), py::arg("D"), py::arg("n_samples"),
        "Return MIMO Markov parameters from a state-space realization. Output shape is (samples, outputs, inputs).");

  m.def("mimo_state_space_process_batch", &mimo_state_space_process_batch,
        py::arg("A"), py::arg("B"), py::arg("C"), py::arg("D"), py::arg("x"),
        py::arg("n_threads") = 0,
        "Process a batched MIMO state-space model. x has shape (batch, samples, inputs); "
        "the returned array has shape (batch, samples, outputs). The state update is x_state[n+1] = A x_state[n] + B u[n].");

  m.def("finite_hankel_aak_reduce_impulse", finite_hankel_reduce_impulse_py,
        py::arg("impulse_response"), py::arg("reduced_order"), py::arg("rows"), py::arg("cols"),
        py::arg("regularization") = 1e-12,
        "Deprecated compatibility alias for finite_hankel_reduce_impulse. The routine is finite-Hankel/Ho-Kalman, not an exact AAK solver.");

  m.def("finite_hankel_aak_reduce_iir", finite_hankel_reduce_iir_py,
        py::arg("reflection"), py::arg("numerator"), py::arg("reduced_order"),
        py::arg("n_impulse") = 512, py::arg("rows") = 64, py::arg("cols") = 64,
        py::arg("regularization") = 1e-12,
        "Deprecated compatibility alias for finite_hankel_reduce_iir. The routine is finite-Hankel/Ho-Kalman, not an exact AAK solver.");

  m.def("ladder_to_numerator", &lattice_dsp::ladder_to_numerator,
        py::arg("reflection"), py::arg("ladder_taps"),
        "Convert synthesis lattice-ladder taps to direct numerator coefficients.");

  m.def("numerator_to_ladder", &lattice_dsp::numerator_to_ladder,
        py::arg("reflection"), py::arg("numerator"),
        "Convert direct numerator coefficients to synthesis lattice-ladder taps.");
}
