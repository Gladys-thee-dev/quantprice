r"""Exact simulation of geometric Brownian motion at given monitoring dates.

Under the risk-neutral measure, :math:`dS = (r - q)S\,dt + \sigma S\,dW`. The log
price has independent Gaussian increments, so for dates
:math:`0 = t_0 < t_1 < \dots < t_m`:

.. math::

    S_{t_{k+1}} = S_{t_k} \exp\!\Big((r - q - \tfrac12\sigma^2)\Delta t_k
                  + \sigma\sqrt{\Delta t_k}\, Z_{k+1}\Big), \qquad Z_k \sim N(0, 1)\ \text{i.i.d.}

This is exact in distribution at the monitoring dates: there is no time-discretisation
error, however coarse the dates are (Glasserman 2004, section 3.2). Anything that
depends on the path *between* dates, such as a continuously monitored barrier, is
not captured. That is the discrete-monitoring bias handled in the barrier module.

Antithetic variates: the first half of the paths uses draws :math:`Z` and the second
half uses :math:`-Z`, so path ``i`` and path ``i + n_paths // 2`` form a pair
(Glasserman 2004, section 4.2). The pairs are not independent, so estimators must
average within each pair before computing statistics (see :mod:`quantprice.mc.estimate`).

Random numbers come from NumPy's ``Generator`` (PCG64 by default). A fixed integer
seed gives identical paths for a given NumPy version. NumPy does not promise identical
streams across versions (NEP 19), so tests compare against analytical values within
standard errors, never against stored random outputs.
"""

import numpy as np
from numpy.typing import ArrayLike

from quantprice._validation import FloatArray, validate_market_inputs

__all__ = ["SeedLike", "make_rng", "simulate_gbm"]

SeedLike = int | np.random.Generator | None


def make_rng(seed: SeedLike) -> np.random.Generator:
    """Return a Generator from an integer seed, an existing Generator, or None (OS entropy)."""
    if isinstance(seed, np.random.Generator):
        return seed
    if seed is None or (isinstance(seed, (int, np.integer)) and not isinstance(seed, bool)):
        if seed is not None and seed < 0:
            raise ValueError(f"seed must be a non-negative integer; got {seed}")
        return np.random.default_rng(seed)
    raise ValueError(f"seed must be an int, a numpy Generator or None; got {type(seed).__name__}")


def _validate_n_paths(n_paths: int, antithetic: bool) -> int:
    if isinstance(n_paths, bool) or not isinstance(n_paths, (int, np.integer)) or n_paths < 2:
        raise ValueError(f"n_paths must be an integer >= 2; got {n_paths!r}")
    if antithetic and n_paths % 2:
        raise ValueError(f"n_paths must be even with antithetic variates; got {n_paths}")
    return int(n_paths)


def _validate_times(times: ArrayLike) -> FloatArray:
    t = np.atleast_1d(np.asarray(times, dtype=np.float64))
    if t.ndim != 1 or t.size == 0:
        raise ValueError("times must be a non-empty 1-D sequence of monitoring dates")
    if not np.all(np.isfinite(t)) or t[0] <= 0 or np.any(np.diff(t) <= 0):
        raise ValueError("times must be finite, > 0 and strictly increasing")
    return t


def simulate_gbm(
    S: float,
    r: float,
    q: float,
    sigma: float,
    times: ArrayLike,
    n_paths: int,
    *,
    seed: SeedLike = None,
    antithetic: bool = True,
) -> FloatArray:
    """Simulate GBM prices at the monitoring dates ``times`` (years, ACT/365).

    Args:
        S: initial price, > 0.
        r, q: continuously compounded rate and dividend yield.
        sigma: annualised volatility, >= 0.
        times: strictly increasing dates t_1 < ... < t_m, all > 0.
        n_paths: number of paths (even if ``antithetic``).
        seed: int, ``numpy.random.Generator`` or None.
        antithetic: if True, rows ``i`` and ``i + n_paths // 2`` use opposite normals.

    Returns:
        Array of shape ``(n_paths, m)`` with S at each monitoring date (S_0 is not
        included).
    """
    S_, _, _, r_, sig, q_ = (float(a) for a in validate_market_inputs(S, 1.0, 0.0, r, sigma, q))
    t = _validate_times(times)
    n = _validate_n_paths(n_paths, antithetic)
    rng = make_rng(seed)

    dt = np.diff(t, prepend=0.0)
    n_draw = n // 2 if antithetic else n
    z = rng.standard_normal((n_draw, t.size))
    if antithetic:
        z = np.concatenate([z, -z], axis=0)
    log_increments = (r_ - q_ - 0.5 * sig * sig) * dt + sig * np.sqrt(dt) * z
    with np.errstate(over="raise"):
        try:
            paths: FloatArray = S_ * np.exp(np.cumsum(log_increments, axis=1))
        except FloatingPointError as exc:
            raise ValueError("simulated prices overflow double precision") from exc
    return paths
