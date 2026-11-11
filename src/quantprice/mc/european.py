r"""Monte Carlo pricing of European calls and puts, with a stock-price control variate.

Estimator
---------
Simulate :math:`S_T` exactly (one log-normal step; see :mod:`quantprice.mc.paths`) and
average the discounted payoffs :math:`Y = e^{-rT}\max(w(S_T - K), 0)`.

Control variate: :math:`X = e^{-rT} S_T`, the discounted terminal stock price. Its
expectation is known exactly under the risk-neutral measure,
:math:`E[X] = S e^{-qT}`. It does not use the Black-Scholes price, so the estimator
stays an independent check on the closed form. :math:`X` is strongly correlated with
in-the-money call payoffs and less so with out-of-the-money ones. The measured
reductions are in the tests. Source: Glasserman (2004), *Monte Carlo Methods in
Financial Engineering*, section 4.1 (the underlying asset as a control).

Conventions: annualised volatility; continuously compounded ``r`` and ``q``; ``T`` in
years, ACT/365.
"""

import math

import numpy as np

from quantprice._validation import validate_market_inputs
from quantprice.bsm import OptionType, _parse_option_type
from quantprice.mc.estimate import estimate
from quantprice.mc.paths import SeedLike, _validate_n_paths, simulate_gbm
from quantprice.mc.result import MCResult

__all__ = ["price"]


def price(
    option_type: OptionType,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    n_paths: int,
    seed: SeedLike = None,
    antithetic: bool = True,
    control_variate: bool = True,
) -> MCResult:
    """Monte Carlo price of a European call or put under BSM dynamics.

    Args:
        option_type: ``"call"`` or ``"put"``.
        S, K, T, r, sigma, q: scalar market inputs (validated as in :func:`quantprice.bsm.price`).
        n_paths: number of paths (even if ``antithetic``).
        seed: int, ``numpy.random.Generator`` or None. Use an int for reproducible results.
        antithetic: use antithetic variates.
        control_variate: use the discounted terminal stock price as a control.

    Returns:
        :class:`MCResult` with price, standard error and 95% confidence interval.
    """
    w = 1.0 if _parse_option_type(option_type) else -1.0
    arrays = validate_market_inputs(S, K, T, r, sigma, q)
    if any(a.ndim for a in arrays):
        raise ValueError("mc.european.price takes scalar inputs")
    S_, K_, T_, r_, sig, q_ = (float(a) for a in arrays)
    n = _validate_n_paths(n_paths, antithetic)

    if T_ == 0.0:  # payoff known today; no randomness
        v = max(w * (S_ - K_), 0.0)
        return MCResult(v, 0.0, v, v, n, n // 2 if antithetic else n, None)

    S_T = simulate_gbm(S_, r_, q_, sig, [T_], n, seed=seed, antithetic=antithetic)[:, 0]
    disc = math.exp(-r_ * T_)
    payoff = disc * np.maximum(w * (S_T - K_), 0.0)
    if control_variate:
        return estimate(
            payoff, antithetic=antithetic, control=disc * S_T, control_mean=S_ * math.exp(-q_ * T_)
        )
    return estimate(payoff, antithetic=antithetic)
