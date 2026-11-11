r"""Monte Carlo for discretely monitored fixed-strike Asian options.

Paths are simulated exactly at the monitoring dates ``times`` (see
:mod:`quantprice.mc.paths`). These are the same dates the closed form
:func:`quantprice.asian.geometric_price` uses, so the geometric average here and in
the formula refer to the same random variable.

* ``average="geometric"``: payoff :math:`e^{-rT}\max(w(G - K), 0)`. It has the closed
  form above, so it validates the engine (the MC price should be within a few standard
  errors of the formula).
* ``average="arithmetic"``: payoff :math:`e^{-rT}\max(w(A - K), 0)` with
  :math:`A = \frac1n\sum_i S_{t_i}`. There is no closed form. By default, the
  discounted geometric payoff on the same path is the control variate, with its
  exactly known mean from :func:`quantprice.asian.geometric_price` (Kemna & Vorst 1990;
  Glasserman 2004, section 4.1). :math:`A \ge G` on every path (AM-GM) and the two are
  highly correlated, so the variance reduction is large. The measured factors are in
  the tests.

Conventions: annualised volatility; continuously compounded ``r`` and ``q``; dates in
years, ACT/365.
"""

import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from quantprice import asian
from quantprice._validation import validate_market_inputs, validate_times
from quantprice.bsm import OptionType, _parse_option_type
from quantprice.mc.estimate import estimate
from quantprice.mc.paths import SeedLike, _validate_n_paths, simulate_gbm
from quantprice.mc.result import MCResult

__all__ = ["AverageType", "price"]

AverageType = Literal["arithmetic", "geometric"]


def price(
    option_type: OptionType,
    S: float,
    K: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    times: ArrayLike,
    T: float | None = None,
    average: AverageType = "arithmetic",
    n_paths: int,
    seed: SeedLike = None,
    antithetic: bool = True,
    control_variate: bool | None = None,
) -> MCResult:
    """Monte Carlo price of a fixed-strike Asian call or put.

    Args:
        option_type: ``"call"`` or ``"put"``.
        S, K, r, sigma, q: scalar market inputs.
        times: monitoring dates in years (strictly increasing, > 0).
        T: payment date >= last monitoring date (default: last monitoring date).
        average: ``"arithmetic"`` or ``"geometric"``.
        n_paths: number of paths (even if ``antithetic``).
        seed: int, ``numpy.random.Generator`` or None.
        antithetic: use antithetic variates.
        control_variate: use the geometric Asian as a control. Defaults to True for
            arithmetic averages. It must be False (or None) for geometric averages,
            where the control would be the estimand itself.
    """
    w = 1.0 if _parse_option_type(option_type) else -1.0
    if average not in ("arithmetic", "geometric"):
        raise ValueError(f"average must be 'arithmetic' or 'geometric'; got {average!r}")
    use_cv = (average == "arithmetic") if control_variate is None else control_variate
    if use_cv and average == "geometric":
        raise ValueError("control_variate=True is meaningless for a geometric average")
    arrays = validate_market_inputs(S, K, 1.0, r, sigma, q)
    if any(a.ndim for a in arrays):
        raise ValueError("mc.asian.price takes scalar inputs")
    S_, K_, _, r_, sig, q_ = (float(a) for a in arrays)
    t = validate_times(times)
    T_ = asian._payment_date(t, T)
    n = _validate_n_paths(n_paths, antithetic)

    paths = simulate_gbm(S_, r_, q_, sig, t, n, seed=seed, antithetic=antithetic)
    disc = math.exp(-r_ * T_)
    geo = np.exp(np.log(paths).mean(axis=1))
    geo_payoff = disc * np.maximum(w * (geo - K_), 0.0)
    if average == "geometric":
        return estimate(geo_payoff, antithetic=antithetic)

    arith_payoff = disc * np.maximum(w * (paths.mean(axis=1) - K_), 0.0)
    if not use_cv:
        return estimate(arith_payoff, antithetic=antithetic)
    geo_exact = asian.geometric_price(option_type, S_, K_, r_, sig, q_, times=t, T=T_)
    return estimate(arith_payoff, antithetic=antithetic, control=geo_payoff, control_mean=geo_exact)
