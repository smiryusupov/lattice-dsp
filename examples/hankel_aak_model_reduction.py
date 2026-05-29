"""Compatibility wrapper for the renamed finite-Hankel model-reduction tutorial.

Prefer running ``examples/finite_hankel_model_reduction.py``.  This wrapper is
kept so old commands do not break; the implementation is finite-Hankel/Ho-Kalman
reduction, not an exact AAK/Nehari solver.
"""

from finite_hankel_model_reduction import main


if __name__ == "__main__":
    main()
