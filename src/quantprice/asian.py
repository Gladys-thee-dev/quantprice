r"""Closed form for a fixed-strike geometric-average Asian option with discrete monitoring.

Contract
--------
Monitoring dates :math:`0 < t_1 < \dots < t_n` (years, ACT/365) and payment date
:math:`T \ge t_n`. The geometric average is
:math:`G = \big(\prod_{i=1}^n S_{t_i}\big)^{1/n}`, and the payoff at :math:`T` is
:math:`\max(w(G - K), 0)`, with :math:`w = +1` for a call and :math:`-1` for a put.

Derivation (self-contained)
---------------------------
Under risk-neutral GBM with continuous dividend yield :math:`q`,
:math:`\ln S_t = \ln S + (r - q - \tfrac12\sigma^2)t + \sigma W_t`. The log of the
average, :math:`\ln G = \frac1n\sum_i \ln S_{t_i}`, is a linear combination of jointly
Gaussian variables, so it is Gaussian with

.. math::

    \mu_G = \ln S + (r - q - \tfrac12\sigma^2)\,\bar t, \qquad
    \bar t = \frac1n\sum_i t_i,

    v_G = \frac{\sigma^2}{n^2}\sum_{i=1}^n\sum_{j=1}^n \min(t_i, t_j)
        = \frac{\sigma^2}{n^2}\sum_{k=1}^n \big(2(n - k) + 1\big)\,t_k,

using :math:`\mathrm{Cov}(W_s, W_t) = \min(s, t)`. For ordered dates, :math:`t_k` is
the minimum for exactly :math:`(n-k+1)^2 - (n-k)^2 = 2(n-k)+1` ordered pairs, which
gives the :math:`O(n)` form. :math:`G` is therefore log-normal with
:math:`F_G = E[G] = e^{\mu_G + v_G/2}`, and the price is Black's formula on that forward:

.. math::

    V = e^{-rT}\big[w F_G N(w d_1) - w K N(w d_2)\big], \qquad
    d_1 = \frac{\ln(F_G/K) + v_G/2}{\sqrt{v_G}}, \quad d_2 = d_1 - \sqrt{v_G}.

This is evaluated with :func:`quantprice.bsm.price` (spot :math:`F_G`, ``T = 1``,
``r = q = 0``, ``sigma`` :math:`=\sqrt{v_G}`, then multiplied by :math:`e^{-rT}`). It
therefore inherits that function's tail accuracy and its :math:`\sigma = 0` limit.

Checks (see tests): ``n = 1`` with :math:`t_1 = T` reduces to Black-Scholes-Merton.
Equally spaced dates :math:`t_i = iT/n` give :math:`\bar t \to T/2` and
:math:`v_G \to \sigma^2 T/3`, the continuous-averaging result of Kemna & Vorst. The
formula agrees with Monte Carlo on the same dates.

Sources:
    * A. G. Z. Kemna and A. C. F. Vorst (1990), "A pricing method for options based on
      average asset values", *Journal of Banking and Finance* 14(1), 113-129
      (continuous geometric average; geometric option as a control variate for the
      arithmetic one).
    * P. Glasserman (2004), *Monte Carlo Methods in Financial Engineering*, section 4.1
      (discretely monitored geometric Asian option as a control variate).

Conventions: annualised volatility; continuously compounded ``r`` and ``q``; dates in
years, ACT/365.
"""

import math

import numpy as np
from numpy.typing import ArrayLike

from quantprice import bsm
from quantprice._validation import FloatArray, validate_market_inputs, validate_times
from quantprice.bsm import OptionType, _parse_option_type

__all__ = ["geometric_log_moments", "geometric_price"]


def _payment_date(t: FloatArray, T: float | None) -> float:
    if T is None:
        return float(t[-1])
    last = float(t[-1])
    if not (math.isfinite(T) and last <= T):
        raise ValueError(f"payment date T must be >= the last monitoring date {last}; got {T}")
    return float(T)


def geometric_log_moments(
    S: float, r: float, sigma: float, q: float, times: ArrayLike
) -> tuple[float, float]:
    """Mean and variance (mu_G, v_G) of ln G for the geometric average over ``times``."""
    S_, _, _, r_, sig, q_ = (float(a) for a in validate_market_inputs(S, 1.0, 0.0, r, sigma, q))
    t = validate_times(times)
    n = t.size
    weights = 2.0 * (n - np.arange(1, n + 1)) + 1.0
    mu = math.log(S_) + (r_ - q_ - 0.5 * sig * sig) * float(t.mean())
    v = sig * sig * float(np.dot(weights, t)) / (n * n)
    return mu, v


def geometric_price(
    option_type: OptionType,
    S: float,
    K: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    times: ArrayLike,
    T: float | None = None,
) -> float:
    """Fixed-strike geometric Asian call/put, averaging over ``times``, paid at ``T``.

    See the module docstring for the derivation and its sources.

    Args:
        option_type: ``"call"`` or ``"put"``.
        S, K, r, sigma, q: scalar market inputs (validated as in :func:`quantprice.bsm.price`).
        times: monitoring dates in years, strictly increasing and > 0.
        T: payment date, >= the last monitoring date (default: the last monitoring date).
    """
    _parse_option_type(option_type)
    arrays = validate_market_inputs(S, K, 1.0, r, sigma, q)
    if any(a.ndim for a in arrays):
        raise ValueError("asian.geometric_price takes scalar inputs")
    K_, r_ = float(arrays[1]), float(arrays[3])
    t = validate_times(times)
    T_ = _payment_date(t, T)
    mu, v = geometric_log_moments(S, r, sigma, q, t)
    forward = math.exp(mu + 0.5 * v)
    undiscounted = bsm.price(option_type, forward, K_, 1.0, 0.0, math.sqrt(v), 0.0)
    return math.exp(-r_ * T_) * undiscounted
