r"""Monte Carlo for discretely monitored single-barrier options.

The barrier is checked only at ``n_monitoring`` equally spaced dates
:math:`t_i = iT/m` (plus the spot at :math:`t = 0`). Paths are exact at those dates
(see :mod:`quantprice.mc.paths`), so this estimator has **no discretisation bias for the
discretely monitored contract**. Compared with the continuously monitored closed form
in :mod:`quantprice.barrier`, it differs by the discrete-monitoring bias. The
Broadie-Glasserman-Kou shifted barrier (:func:`quantprice.barrier.discrete_price`)
approximates that bias, and the tests measure how well.

Payoffs (per path; :math:`\tau` = first monitoring date at which the barrier is
breached, :math:`\tau = 0` if breached at inception):

* knock-out: vanilla payoff at :math:`T` if never breached, otherwise the rebate,
  paid at :math:`\tau` (``rebate_at="hit"``) or at :math:`T` (``"expiry"``);
* knock-in: vanilla payoff at :math:`T` if breached, otherwise the rebate at :math:`T`.

Down barriers are breached when :math:`S_{t_i} \le H`, up barriers when
:math:`S_{t_i} \ge H`. Knock-in + knock-out = vanilla holds path by path when the
rebate is zero, which the tests use as an exact check.

Paths are generated in batches so memory stays bounded at
``batch_size * n_monitoring`` floats. Statistics are computed on antithetic pair
averages as in :mod:`quantprice.mc.estimate`.

Conventions: annualised volatility; continuously compounded ``r`` and ``q``; ``T`` in
years, ACT/365.
"""

import dataclasses
import math

import numpy as np

from quantprice import barrier as closed_form
from quantprice._validation import FloatArray
from quantprice.barrier import BarrierType, RebateTiming
from quantprice.bsm import OptionType
from quantprice.mc.estimate import estimate
from quantprice.mc.paths import SeedLike, _validate_n_paths, make_rng, simulate_gbm
from quantprice.mc.result import MCResult

__all__ = ["price"]


def price(
    option_type: OptionType,
    barrier_type: BarrierType,
    S: float,
    K: float,
    H: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    n_monitoring: int,
    n_paths: int,
    seed: SeedLike = None,
    antithetic: bool = True,
    rebate: float = 0.0,
    rebate_at: RebateTiming | None = None,
    batch_size: int = 50_000,
) -> MCResult:
    """Monte Carlo price of a barrier option monitored at ``n_monitoring`` equal dates.

    Arguments are as in :func:`quantprice.barrier.price`, plus the simulation controls.
    ``batch_size`` (even, if antithetic) bounds memory; it does not change the result
    for a given seed and batch size.
    """
    is_call, is_down, is_in, timing, (S_, K_, T_, r_, sig, q_) = closed_form._validate(
        option_type, barrier_type, S, K, H, T, r, sigma, q, rebate, rebate_at
    )
    n = _validate_n_paths(n_paths, antithetic)
    if isinstance(n_monitoring, bool) or not isinstance(n_monitoring, int) or n_monitoring < 1:
        raise ValueError(f"n_monitoring must be an integer >= 1; got {n_monitoring!r}")
    if T_ == 0.0:
        raise ValueError("T must be > 0 for Monte Carlo (use barrier.price at expiry)")
    batch = _validate_n_paths(min(batch_size, n), antithetic)

    rng = make_rng(seed)
    times = T_ * np.arange(1, n_monitoring + 1) / n_monitoring
    disc_T = math.exp(-r_ * T_)
    phi = 1.0 if is_call else -1.0
    breached_at_start = S_ <= H if is_down else S_ >= H

    samples: list[FloatArray] = []
    done = 0
    while done < n:
        size = min(batch, n - done)
        if antithetic and size % 2:
            size -= 1  # cannot happen: n and batch are both even
        paths = simulate_gbm(S_, r_, q_, sig, times, size, seed=rng, antithetic=antithetic)
        crossed = paths <= H if is_down else paths >= H
        hit = crossed.any(axis=1) | breached_at_start
        # First breached monitoring date; only used where hit is True.
        t_hit = np.zeros(size) if breached_at_start else times[np.argmax(crossed, axis=1)]
        vanilla = disc_T * np.maximum(phi * (paths[:, -1] - K_), 0.0)
        if is_in:
            payoff = np.where(hit, vanilla, rebate * disc_T)
        else:
            rebate_value = rebate * (np.exp(-r_ * t_hit) if timing == "hit" else disc_T)
            payoff = np.where(hit, rebate_value, vanilla)
        if antithetic:
            half = size // 2
            samples.append(0.5 * (payoff[:half] + payoff[half:]))
        else:
            samples.append(payoff)
        done += size

    res = estimate(np.concatenate(samples), antithetic=False)
    return dataclasses.replace(res, n_paths=n)
