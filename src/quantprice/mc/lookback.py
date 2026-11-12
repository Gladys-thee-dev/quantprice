r"""Monte Carlo for discretely monitored lookback options.

The running extreme is taken over the spot at :math:`t = 0`, any ``running_extreme``
already observed, and the ``n_monitoring`` equally spaced dates :math:`t_i = iT/m`.
Paths are exact at those dates, so this prices the **discretely monitored** contract
without bias. It is worth less than the continuously monitored closed form in
:mod:`quantprice.lookback`, because a discrete maximum (minimum) is below (above) the
continuous one. The tests measure that gap as a function of :math:`m`.

Payoffs: floating call :math:`S_T - m_T`, floating put :math:`M_T - S_T`, fixed call
:math:`\max(M_T - K, 0)`, fixed put :math:`\max(K - m_T, 0)`, discounted at
:math:`e^{-rT}`. Paths are generated in batches (memory ``batch_size * n_monitoring``),
and statistics use antithetic pair averages as in :mod:`quantprice.mc.estimate`.

Conventions: annualised volatility; continuously compounded ``r`` and ``q``; ``T`` in
years, ACT/365.
"""

import dataclasses
import math
from typing import Literal

import numpy as np

from quantprice import lookback as closed_form
from quantprice._validation import FloatArray
from quantprice.bsm import OptionType, _parse_option_type
from quantprice.mc.estimate import estimate
from quantprice.mc.paths import SeedLike, _validate_n_paths, make_rng, simulate_gbm
from quantprice.mc.result import MCResult

__all__ = ["StrikeType", "price"]

StrikeType = Literal["floating", "fixed"]


def price(
    strike_type: StrikeType,
    option_type: OptionType,
    S: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    K: float | None = None,
    running_extreme: float | None = None,
    n_monitoring: int,
    n_paths: int,
    seed: SeedLike = None,
    antithetic: bool = True,
    batch_size: int = 50_000,
) -> MCResult:
    """Monte Carlo price of a lookback option monitored at ``n_monitoring`` equal dates.

    Args:
        strike_type: ``"floating"`` or ``"fixed"`` (``K`` required for fixed).
        option_type: ``"call"`` or ``"put"``.
        S, T, r, sigma, q: scalar market inputs.
        K: strike for fixed-strike options.
        running_extreme: observed minimum or maximum so far (default S), as in
            :mod:`quantprice.lookback`.
        n_monitoring, n_paths, seed, antithetic, batch_size: simulation controls.
    """
    is_call = _parse_option_type(option_type)
    if strike_type not in ("floating", "fixed"):
        raise ValueError(f"strike_type must be 'floating' or 'fixed'; got {strike_type!r}")
    if strike_type == "fixed" and K is None:
        raise ValueError("fixed-strike lookbacks need a strike K")
    S_, K_, T_, r_, sig, q_ = closed_form._inputs(S, 1.0 if K is None else K, T, r, sigma, q)
    if T_ == 0.0:
        raise ValueError("T must be > 0 for Monte Carlo (use the closed form at expiry)")
    if isinstance(n_monitoring, bool) or not isinstance(n_monitoring, int) or n_monitoring < 1:
        raise ValueError(f"n_monitoring must be an integer >= 1; got {n_monitoring!r}")
    n = _validate_n_paths(n_paths, antithetic)
    batch = _validate_n_paths(min(batch_size, n), antithetic)
    # Which extreme the payoff uses: minimum for floating call / fixed put.
    use_min = is_call if strike_type == "floating" else not is_call
    start = closed_form._extreme(running_extreme, S_, is_min=use_min)

    rng = make_rng(seed)
    times = T_ * np.arange(1, n_monitoring + 1) / n_monitoring
    disc = math.exp(-r_ * T_)
    samples: list[FloatArray] = []
    done = 0
    while done < n:
        size = min(batch, n - done)
        paths = simulate_gbm(S_, r_, q_, sig, times, size, seed=rng, antithetic=antithetic)
        if use_min:
            ext = np.minimum(paths.min(axis=1), start)
        else:
            ext = np.maximum(paths.max(axis=1), start)
        S_T = paths[:, -1]
        if strike_type == "floating":
            payoff = disc * ((S_T - ext) if is_call else (ext - S_T))
        else:
            payoff = disc * np.maximum((ext - K_) if is_call else (K_ - ext), 0.0)
        if antithetic:
            half = size // 2
            samples.append(0.5 * (payoff[:half] + payoff[half:]))
        else:
            samples.append(payoff)
        done += size

    res = estimate(np.concatenate(samples), antithetic=False)
    return dataclasses.replace(res, n_paths=n)
