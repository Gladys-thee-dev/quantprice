"""Input validation shared by all pricing models.

Every public pricing function sends its inputs through
:func:`validate_market_inputs`, so invalid inputs fail the same way everywhere:
a ``ValueError`` that names the parameter, the rule it broke and the offending
value (plus its index for array inputs).
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _describe(arr: FloatArray, bad: NDArray[np.bool_]) -> str:
    """Return the first offending value, plus its index when ``arr`` is an array."""
    idx = tuple(int(i) for i in np.argwhere(bad)[0])
    value = arr[idx] if arr.ndim else arr.item()
    return f"{float(value)!r}" + (f" at index {idx}" if arr.ndim else "")


def _as_finite_array(name: str, value: ArrayLike) -> FloatArray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number or array of real numbers") from exc
    bad = ~np.isfinite(arr)
    if bad.any():
        raise ValueError(f"{name} must be finite; got {_describe(arr, bad)}")
    return arr


def _require(
    name: str, arr: FloatArray, ok: Callable[[FloatArray], NDArray[np.bool_]], rule: str
) -> None:
    bad = ~ok(arr)
    if bad.any():
        raise ValueError(f"{name} must be {rule}; got {_describe(arr, bad)}")


def validate_market_inputs(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Check market inputs and broadcast them to one common shape.

    Rules:
        * spot ``S`` > 0 and strike ``K`` > 0, because the log-normal model is undefined
          at or below zero;
        * time to expiry ``T`` >= 0 (``T = 0`` means at expiry);
        * volatility ``sigma`` >= 0 (``sigma = 0`` means a deterministic forward);
        * rate ``r`` and dividend yield ``q`` can be any finite value. Negative rates
          are supported, and so is negative ``q`` (e.g. a borrow cost).

    All inputs must be finite (no NaN or infinity).

    Raises:
        ValueError: if a rule is broken or the shapes cannot be broadcast together.
    """
    S_ = _as_finite_array("spot S", S)
    K_ = _as_finite_array("strike K", K)
    T_ = _as_finite_array("time to expiry T", T)
    r_ = _as_finite_array("interest rate r", r)
    sig = _as_finite_array("volatility sigma", sigma)
    q_ = _as_finite_array("dividend yield q", q)

    _require("spot S", S_, lambda a: a > 0, "> 0")
    _require("strike K", K_, lambda a: a > 0, "> 0")
    _require("time to expiry T", T_, lambda a: a >= 0, ">= 0")
    _require("volatility sigma", sig, lambda a: a >= 0, ">= 0")

    try:
        S_, K_, T_, r_, sig, q_ = np.broadcast_arrays(S_, K_, T_, r_, sig, q_)
    except ValueError as exc:
        raise ValueError(f"input shapes cannot be broadcast together: {exc}") from exc
    return S_, K_, T_, r_, sig, q_
