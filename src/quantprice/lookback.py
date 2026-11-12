r"""Closed forms for continuously monitored European lookback options.

Contracts (running minimum :math:`m_T` and maximum :math:`M_T` of the price over
:math:`[0, T]`, including any already observed extreme):

==========================  ==========================================
floating-strike call        :math:`S_T - m_T`
floating-strike put         :math:`M_T - S_T`
fixed-strike call           :math:`\max(M_T - K, 0)`
fixed-strike put            :math:`\max(K - m_T, 0)`
==========================  ==========================================

``running_extreme`` is the minimum (floating call, fixed put) or maximum (floating
put, fixed call) observed so far. It defaults to the spot, i.e. a newly issued option.
A seasoned option needs ``running_extreme <= S`` for a minimum and ``>= S`` for a
maximum.

Formulas
--------
Floating strike: Goldman, Sosin & Gatto (1979). Fixed strike: Conze & Viswanathan
(1991). Both are written here in the notation of Haug (2007), sections 4.15.1-4.15.2.
With :math:`b = r - q`, :math:`v = \sigma\sqrt T`, :math:`\hat S = S e^{-qT}`,
:math:`D = e^{-rT}` and, for a reference level :math:`Y`,
:math:`y(Y) = [\ln(S/Y) + (b + \sigma^2/2)T]/v`:

.. math::

    h^-(Y) &= \frac{\sigma^2}{2b}\Big[(S/Y)^{-2b/\sigma^2} N(-y + 2b\sqrt T/\sigma)
              - e^{bT} N(-y)\Big] \\
    h^+(Y) &= \frac{\sigma^2}{2b}\Big[-(S/Y)^{-2b/\sigma^2} N(y - 2b\sqrt T/\sigma)
              + e^{bT} N(y)\Big]

    \text{floating call}      &= \hat S N(y_m) - m D N(y_m - v) + S D\, h^-(m) \\
    \text{floating put}       &= M D N(-y_M + v) - \hat S N(-y_M) + S D\, h^+(M) \\
    \text{fixed call}, K > M  &= \hat S N(y_K) - K D N(y_K - v) + S D\, h^+(K) \\
    \text{fixed call}, K \le M &= D(M - K) + \hat S N(y_M) - M D N(y_M - v) + S D\, h^+(M) \\
    \text{fixed put}, K < m   &= K D N(-y_K + v) - \hat S N(-y_K) + S D\, h^-(K) \\
    \text{fixed put}, K \ge m &= D(K - m) - \hat S N(-y_m) + m D N(-y_m + v) + S D\, h^-(m)

with :math:`y_Y = y(Y)`, :math:`m` the running minimum and :math:`M` the running
maximum.

The b -> 0 singularity (our derivation)
---------------------------------------
:math:`\sigma^2/(2b)` multiplies a bracket that vanishes at :math:`b = 0`, so the
textbook form is 0/0 when :math:`r = q`, and it loses digits to cancellation when
:math:`b` is small. We rewrite, for :math:`h^-`, with :math:`A = -y + 2b\sqrt T/\sigma`,
:math:`B = -y`, :math:`L = \ln(S/Y)` and :math:`x = -b(2L/\sigma^2 + T)`:

.. math::

    h^- = \frac{\sigma^2}{2}\Big[-\Big(\frac{2L}{\sigma^2} + T\Big)\frac{e^x - 1}{x}\,
          e^{bT} N(A) + e^{bT}\,\frac{2\sqrt T}{\sigma}\,\bar\varphi(B, A)\Big],

where :math:`\bar\varphi(B, A) = (N(A) - N(B))/(A - B)` is the mean normal density on
:math:`[B, A]`, evaluated by 16-point Gauss-Legendre quadrature when the interval is
short. :math:`h^+` is analogous. Nothing here divides by :math:`b`, and at
:math:`b = 0` it reduces to :math:`h^- = \sigma\sqrt T[\varphi(y) - yN(-y)]` and
:math:`h^+ = \sigma\sqrt T[\varphi(y) + yN(y)]`. This rearrangement is ours, not from
the sources; the tests check it against the textbook form away from :math:`b = 0` and
by continuity through :math:`b = 0`.

Monitoring
----------
These are **continuous**-monitoring prices. With discrete monitoring the observed
maximum is lower and the minimum higher, so discretely monitored lookbacks are worth
less. Broadie, Glasserman & Kou (1999) relate the two. Our Monte Carlo
(:mod:`quantprice.mc.lookback`) is discrete, and the tests measure the gap.

Sources:
    * M. B. Goldman, H. B. Sosin and M. A. Gatto (1979), "Path dependent options: Buy at
      the low, sell at the high", *Journal of Finance* 34(5), 1111-1127.
    * A. Conze and R. Viswanathan (1991), "Path dependent options: The case of lookback
      options", *Journal of Finance* 46(5), 1893-1907.
    * E. G. Haug (2007), *The Complete Guide to Option Pricing Formulas*, 2nd ed.,
      sections 4.15.1-4.15.2.
    * M. Broadie, P. Glasserman and S. Kou (1999), "Connecting discrete and continuous
      path-dependent options", *Finance and Stochastics* 3(1), 55-82.

Conventions: annualised volatility; continuously compounded ``r`` and ``q``; ``T`` in
years, ACT/365.
"""

import math

import numpy as np
from scipy.special import log_ndtr, ndtr

from quantprice._validation import validate_market_inputs
from quantprice.bsm import OptionType, _parse_option_type

__all__ = ["fixed_price", "floating_price"]

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(16)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _mean_pdf(lo: float, hi: float) -> float:
    """(N(hi) - N(lo)) / (hi - lo): the mean standard normal density on [lo, hi]."""
    if lo > hi:
        lo, hi = hi, lo
    width = hi - lo
    if width == 0.0:
        return _INV_SQRT_2PI * math.exp(-0.5 * lo * lo)
    if width < 1.0:  # short interval: quadrature avoids cancellation in N(hi) - N(lo)
        x = 0.5 * (hi + lo) + 0.5 * width * _GL_NODES
        return float(0.5 * np.dot(_GL_WEIGHTS, _INV_SQRT_2PI * np.exp(-0.5 * x * x)))
    diff = float(ndtr(-lo) - ndtr(-hi)) if lo > 0 else float(ndtr(hi) - ndtr(lo))
    return diff / width


def _expm1_over_x_times(x: float, log_factor: float) -> float:
    """((e^x - 1) / x) * e^{log_factor}, without overflow for large x."""
    if x == 0.0:
        return math.exp(log_factor)
    if x > 50.0:
        return (math.exp(x + log_factor) - math.exp(log_factor)) / x
    return math.expm1(x) / x * math.exp(log_factor)


def _h(sign: float, S: float, Y: float, T: float, b: float, sigma: float) -> float:
    """h^- (sign = -1) or h^+ (sign = +1), stable for every b including b = 0."""
    v = sigma * math.sqrt(T)
    L = math.log(S / Y)
    y = (L + (b + 0.5 * sigma * sigma) * T) / v
    shift = 2.0 * b * math.sqrt(T) / sigma
    coef = 2.0 * L / (sigma * sigma) + T
    x = -b * coef
    if sign < 0:  # h^-: A = -y + shift, B = -y
        A, B = -y + shift, -y
        first = -coef * _expm1_over_x_times(x, b * T + float(log_ndtr(A)))
    else:  # h^+: A = y - shift, B = y
        A, B = y - shift, y
        first = coef * _expm1_over_x_times(x, b * T + float(log_ndtr(A)))
    second = math.exp(b * T) * (2.0 * math.sqrt(T) / sigma) * _mean_pdf(B, A)
    return 0.5 * sigma * sigma * (first + second)


def _inputs(
    S: float, K: float, T: float, r: float, sigma: float, q: float
) -> tuple[float, float, float, float, float, float]:
    arrays = validate_market_inputs(S, K, T, r, sigma, q)
    if any(a.ndim for a in arrays):
        raise ValueError("lookback pricing takes scalar inputs")
    S_, K_, T_, r_, sig, q_ = (float(a) for a in arrays)
    return S_, K_, T_, r_, sig, q_


def _extreme(running_extreme: float | None, S: float, is_min: bool) -> float:
    if running_extreme is None:
        return S
    e = float(running_extreme)
    if not (math.isfinite(e) and e > 0):
        raise ValueError(f"running_extreme must be finite and > 0; got {running_extreme!r}")
    if is_min and e > S:
        raise ValueError(f"running minimum must be <= S; got {e} > {S}")
    if not is_min and e < S:
        raise ValueError(f"running maximum must be >= S; got {e} < {S}")
    return e


def _deterministic_extreme(S: float, T: float, b: float, extreme: float, is_min: bool) -> float:
    """sigma = 0 or T = 0: the path S e^{bt} is monotone; its extreme is at an endpoint."""
    end = S * math.exp(b * T)
    return min(extreme, S, end) if is_min else max(extreme, S, end)


def floating_price(
    option_type: OptionType,
    S: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    running_extreme: float | None = None,
) -> float:
    """Floating-strike lookback (Goldman-Sosin-Gatto 1979), continuous monitoring.

    Call pays ``S_T - min``; put pays ``max - S_T``. ``running_extreme`` is the observed
    minimum (call) or maximum (put), and defaults to S (new option).
    """
    is_call = _parse_option_type(option_type)
    S_, _, T_, r_, sig, q_ = _inputs(S, 1.0, T, r, sigma, q)
    Y = _extreme(running_extreme, S_, is_min=is_call)
    b, D, S_hat = r_ - q_, math.exp(-r_ * T_), S_ * math.exp(-q_ * T_)
    if T_ == 0.0 or sig == 0.0:
        ext = _deterministic_extreme(S_, T_, b, Y, is_min=is_call)
        end = S_ * math.exp(b * T_)
        return D * ((end - ext) if is_call else (ext - end))
    v = sig * math.sqrt(T_)
    y = (math.log(S_ / Y) + (b + 0.5 * sig * sig) * T_) / v
    if is_call:
        value = S_hat * ndtr(y) - Y * D * ndtr(y - v) + S_ * D * _h(-1.0, S_, Y, T_, b, sig)
    else:
        value = Y * D * ndtr(-y + v) - S_hat * ndtr(-y) + S_ * D * _h(1.0, S_, Y, T_, b, sig)
    return max(float(value), 0.0)


def fixed_price(
    option_type: OptionType,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    running_extreme: float | None = None,
) -> float:
    """Fixed-strike lookback (Conze-Viswanathan 1991), continuous monitoring.

    Call pays ``max(max - K, 0)``; put pays ``max(K - min, 0)``. ``running_extreme`` is
    the observed maximum (call) or minimum (put), and defaults to S (new option).
    """
    is_call = _parse_option_type(option_type)
    S_, K_, T_, r_, sig, q_ = _inputs(S, K, T, r, sigma, q)
    E = _extreme(running_extreme, S_, is_min=not is_call)
    b, D, S_hat = r_ - q_, math.exp(-r_ * T_), S_ * math.exp(-q_ * T_)
    if T_ == 0.0 or sig == 0.0:
        ext = _deterministic_extreme(S_, T_, b, E, is_min=not is_call)
        return D * max((ext - K_) if is_call else (K_ - ext), 0.0)
    v = sig * math.sqrt(T_)

    def y(level: float) -> float:
        return (math.log(S_ / level) + (b + 0.5 * sig * sig) * T_) / v

    if is_call:
        if K_ > E:
            yk = y(K_)
            value = S_hat * ndtr(yk) - K_ * D * ndtr(yk - v) + S_ * D * _h(1.0, S_, K_, T_, b, sig)
        else:
            ye = y(E)
            value = (
                D * (E - K_)
                + S_hat * ndtr(ye)
                - E * D * ndtr(ye - v)
                + S_ * D * _h(1.0, S_, E, T_, b, sig)
            )
    elif K_ < E:
        yk = y(K_)
        value = K_ * D * ndtr(-yk + v) - S_hat * ndtr(-yk) + S_ * D * _h(-1.0, S_, K_, T_, b, sig)
    else:
        ye = y(E)
        value = (
            D * (K_ - E)
            - S_hat * ndtr(-ye)
            + E * D * ndtr(-ye + v)
            + S_ * D * _h(-1.0, S_, E, T_, b, sig)
        )
    return max(float(value), 0.0)
