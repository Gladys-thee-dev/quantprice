r"""Cox-Ross-Rubinstein (CRR) binomial tree for European and American options.

Model and conventions
---------------------
The same risk-neutral GBM as :mod:`quantprice.bsm`: annualised volatility
:math:`\sigma`, continuously compounded rate :math:`r` and dividend yield :math:`q`,
and :math:`T` in years (ACT/365). The interval :math:`[0, T]` is split into ``steps``
equal periods :math:`\Delta t = T/n`. Over each period the price moves up by
:math:`u` or down by :math:`d`:

.. math::

    u = e^{\sigma\sqrt{\Delta t}}, \qquad d = 1/u, \qquad
    p = \frac{e^{(r - q)\Delta t} - d}{u - d}.

:math:`p` is chosen so that :math:`E[S_{t+\Delta t}] = S_t e^{(r-q)\Delta t}`. The tree
therefore reproduces the forward exactly, and put-call parity holds exactly
(to rounding) for European options at every step count.

The value is computed by backward induction,

.. math:: V_{i,j} = e^{-r\Delta t}\,[p V_{i+1,j+1} + (1-p) V_{i+1,j}],

and for American exercise :math:`V_{i,j}` is replaced by
:math:`\max(V_{i,j}, \text{payoff}(S_{i,j}))` at every node. Exercise is only possible
at the ``steps + 1`` tree dates, so this is a Bermudan approximation that converges
to the American price as ``steps`` grows.

Sources:
    * J. C. Cox, S. A. Ross and M. Rubinstein (1979), "Option Pricing: A Simplified
      Approach", *Journal of Financial Economics* 7(3), 229-263.
    * J. C. Hull, *Options, Futures, and Other Derivatives*, chapters "Binomial Trees"
      and "Numerical Procedures" (dividend yield: growth rate r - q in p).

Validity: :math:`0 \le p \le 1` holds iff :math:`|r - q|\sqrt{\Delta t} \le \sigma`
exactly, since :math:`p \le 1 \iff e^{(r-q)\Delta t} \le e^{\sigma\sqrt{\Delta t}}`, and
likewise for :math:`p \ge 0`. Equivalently, ``steps >= (r - q)^2 T / sigma^2``. Otherwise the
"probabilities" are not probabilities, and the tree can produce arbitrage prices, so
:func:`price` raises ``ValueError`` and states the minimum step count.

Convergence: the European CRR price converges to Black-Scholes at rate O(1/n), but not
monotonically. The error oscillates with the position of the strike relative to the
terminal nodes, and in particular between odd and even ``n``. See D. P. J. Leisen and
M. Reimer (1996), "Binomial models for option valuation - examining and improving
convergence", *Applied Mathematical Finance* 3(4), 319-346. Averaging the prices for
``n`` and ``n + 1`` cancels most of the odd/even part (see the tests for measured
errors).

Limiting cases: ``T = 0`` returns the payoff. ``sigma = 0`` makes the tree a single
deterministic path :math:`S e^{(r-q)t_i}`. The European value is then the discounted
forward intrinsic (as in BSM), and the American value is the best discounted payoff
over the tree dates :math:`t_i = i\Delta t`.

Not vectorised over inputs: one call prices one option. The cost is O(steps^2).
"""

import math
from typing import Literal

import numpy as np

from quantprice._validation import FloatArray, validate_market_inputs
from quantprice.bsm import OptionType, _parse_option_type

__all__ = ["ExerciseStyle", "price"]

ExerciseStyle = Literal["european", "american"]


def _validate_steps(steps: int) -> None:
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError(f"steps must be an integer >= 1; got {steps!r}")


def _deterministic_value(
    w: float, american: bool, S: float, K: float, T: float, r: float, q: float, steps: int
) -> float:
    """sigma = 0: the stock follows S e^{(r-q)t} with certainty."""
    times = np.linspace(0.0, T, steps + 1) if american else np.array([T])
    discounted = np.exp(-r * times) * np.maximum(w * (S * np.exp((r - q) * times) - K), 0.0)
    return float(discounted.max())


def price(
    option_type: OptionType,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    steps: int,
    exercise: ExerciseStyle = "european",
) -> float:
    """Price a European or American call/put on a CRR binomial tree.

    Conventions: annualised volatility; continuously compounded ``r`` and ``q``; ``T``
    in years, ACT/365. See the module docstring for the model, its sources and its
    convergence behaviour.

    Args:
        option_type: ``"call"`` or ``"put"``.
        S, K, T, r, sigma, q: scalar market inputs, validated as in :func:`quantprice.bsm.price`.
        steps: number of time steps n >= 1.
        exercise: ``"european"`` or ``"american"`` (exercisable at every tree date).

    Raises:
        ValueError: for invalid inputs, non-scalar inputs, ``steps < 1``, an unknown
            exercise style, a risk-neutral probability outside [0, 1], or overflow in
            the node prices.
    """
    w = 1.0 if _parse_option_type(option_type) else -1.0
    if exercise not in ("european", "american"):
        raise ValueError(f"exercise must be 'european' or 'american'; got {exercise!r}")
    american = exercise == "american"
    _validate_steps(steps)
    arrays = validate_market_inputs(S, K, T, r, sigma, q)
    if any(a.ndim for a in arrays):
        raise ValueError("binomial.price takes scalar inputs; call it once per option")
    S_, K_, T_, r_, sig, q_ = (float(a) for a in arrays)
    n = int(steps)

    if T_ == 0.0:
        return max(w * (S_ - K_), 0.0)
    if sig == 0.0:
        return _deterministic_value(w, american, S_, K_, T_, r_, q_, n)

    dt = T_ / n
    log_u = sig * math.sqrt(dt)
    u, d = math.exp(log_u), math.exp(-log_u)
    p = (math.exp((r_ - q_) * dt) - d) / (u - d)
    if not 0.0 <= p <= 1.0:
        min_steps = math.ceil((r_ - q_) ** 2 * T_ / sig**2)
        raise ValueError(
            f"CRR risk-neutral probability p = {p:.6g} is outside [0, 1] "
            f"(needs |r - q| * sqrt(T / steps) <= sigma); "
            f"use at least {min_steps} steps"
        )
    disc = math.exp(-r_ * dt)
    disc_p, disc_1mp = disc * p, disc * (1.0 - p)

    with np.errstate(over="raise", invalid="raise"):
        try:
            # Node prices S u^j d^(i-j) = S exp(log_u * (2j - i)). exp() of the exact
            # exponent avoids the error that repeated multiplication by u accumulates.
            j = np.arange(n + 1, dtype=np.float64)
            values: FloatArray = np.maximum(w * (S_ * np.exp(log_u * (2.0 * j - n)) - K_), 0.0)
            for i in range(n - 1, -1, -1):
                values = disc_p * values[1:] + disc_1mp * values[:-1]
                if american:
                    stock = S_ * np.exp(log_u * (2.0 * j[: i + 1] - i))
                    np.maximum(values, w * (stock - K_), out=values)
        except FloatingPointError as exc:
            raise ValueError(
                "node prices overflow double precision (sigma * sqrt(T * steps) too large)"
            ) from exc
    return float(values[0])
