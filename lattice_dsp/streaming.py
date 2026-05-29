"""Small stateful block helpers for streaming-style examples.

The heavy sample loops remain in the C++ extension. These wrappers mainly make
state ownership explicit for demos and notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._core import (
    AdaptiveLatticeLadderNLMS,
    LatticeIIR,
    LatticeLadderIIR,
    LatticeLadderNLMS,
    LatticeLadderRLS,
    numerator_to_ladder,
)

Realization = Literal["direct", "lattice"]
AdaptiveKind = Literal["nlms", "rls", "adaptive_iir"]


@dataclass
class BlockResult:
    """Output/error pair returned by adaptive block processors."""

    output: np.ndarray
    error: np.ndarray


class BlockProcessor:
    """Stateful block processor for a fixed stable lattice/IIR filter."""

    def __init__(
        self,
        reflection: list[float] | np.ndarray,
        taps: list[float] | np.ndarray,
        *,
        realization: Realization = "direct",
    ) -> None:
        reflection_list = np.asarray(reflection, dtype=float).tolist()
        taps_list = np.asarray(taps, dtype=float).tolist()
        if realization == "direct":
            self._filter = LatticeIIR(reflection_list, taps_list)
        elif realization == "lattice":
            self._filter = LatticeLadderIIR(
                reflection_list, numerator_to_ladder(reflection_list, taps_list)
            )
        else:
            raise ValueError("realization must be 'direct' or 'lattice'")
        self.realization = realization

    def reset(self, value: float = 0.0) -> None:
        self._filter.reset(value)

    def process(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._filter.process(np.asarray(x, dtype=float)), dtype=float)

    @property
    def filter(self):  # noqa: ANN201 - C++ extension type
        return self._filter


class AdaptiveBlockProcessor:
    """Stateful block processor for C++ adaptive filtering loops."""

    def __init__(
        self,
        reflection: list[float] | np.ndarray,
        initial_taps: list[float] | np.ndarray,
        *,
        kind: AdaptiveKind = "nlms",
        mu: float = 0.1,
        mu_taps: float = 0.05,
        mu_reflection: float = 0.001,
        forgetting_factor: float = 0.995,
        initial_inverse_covariance: float = 1000.0,
        epsilon: float = 1e-8,
        margin: float = 1e-4,
        reflection_update_period: int = 1,
        scale_reflection_mu_by_period: bool = False,
    ) -> None:
        reflection_list = np.asarray(reflection, dtype=float).tolist()
        taps_list = np.asarray(initial_taps, dtype=float).tolist()
        if kind == "nlms":
            self._adaptive = LatticeLadderNLMS(reflection_list, taps_list, mu, epsilon)
        elif kind == "rls":
            self._adaptive = LatticeLadderRLS(
                reflection_list,
                taps_list,
                forgetting_factor,
                initial_inverse_covariance,
                epsilon,
            )
        elif kind == "adaptive_iir":
            self._adaptive = AdaptiveLatticeLadderNLMS(
                reflection_list,
                taps_list,
                mu_taps,
                mu_reflection,
                epsilon,
                margin,
                False,
                "analytic",
                reflection_update_period,
                scale_reflection_mu_by_period,
            )
        else:
            raise ValueError("kind must be 'nlms', 'rls', or 'adaptive_iir'")
        self.kind = kind

    def reset(self, value: float = 0.0) -> None:
        self._adaptive.reset(value)

    def process_adapt(self, x: np.ndarray, desired: np.ndarray) -> BlockResult:
        y, e = self._adaptive.process_adapt(
            np.asarray(x, dtype=float), np.asarray(desired, dtype=float)
        )
        return BlockResult(output=np.asarray(y, dtype=float), error=np.asarray(e, dtype=float))

    @property
    def adaptive(self):  # noqa: ANN201 - C++ extension type
        return self._adaptive
