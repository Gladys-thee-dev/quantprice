r"""Black-Scholes-Merton implied volatility: Newton-Raphson with a Brent fallback.

Problem
-------
Given a European option price :math:`p`, find :math:`\sigma \ge 0` with
:math:`V_{BSM}(\sigma) = p`, with the same conventions as :mod:`quantprice.bsm`
(annualised volatility, continuously compounded ``r`` and ``q``, ``T`` in years
ACT/365).

Existence and uniqueness
------------------------
With :math:`\hat S = S e^{-qT}` and :math:`\hat K = K e^{-rT}`, the BSM price is strictly
increasing in :math:`\sigma` (vega > 0 for T > 0). It runs from the discounted forward
intrinsic value at :math:`\sigma = 0` to an upper limit as :math:`\sigma \to \infty`:

=====  ===========================  ==============================
       lower bound (sigma = 0)      upper limit (sigma -> infinity)
=====  ===========================  ==============================
call   max(S_hat - K_hat, 0)        S_hat
put    max(K_hat - S_hat, 0)        K_hat
=====  ===========================  ==============================

These are the model-free no-arbitrage bounds for European options (Merton 1973). A
price strictly between them has exactly one implied volatility. We therefore:

* raise ``ValueError`` for prices below the lower bound or above the upper limit
  (they admit arbitrage), and for prices *at* the upper limit (sigma would be
  infinite);
* return ``sigma = 0`` for a price at the lower bound, or below it by no more than
  ``1e-14 * max(S_hat, K_hat)`` (rounding). That is the exact limit, not a failure.
  Any price strictly above the bound, however close, is solved for. Deep in or out
  of the money, an option with a tiny time value still has a well-defined implied
  vol.

Algorithm
---------
1. Bracket: ``lo = 0``, and ``hi`` doubled from 1 until :math:`V(hi) \ge p`. If
   :math:`\sigma\sqrt T` would exceed 50 first, raise: the bound is then unreachable in
   double precision. This is a defensive guard; in practice even a price one ulp below
   the upper limit is reached near :math:`\sigma\sqrt T \approx 16`. There the price is
   flat in sigma, so the returned sigma is not meaningful even though it reprices.
2. Newton-Raphson on :math:`f(\sigma) = V(\sigma) - p` with the analytic vega. Every
   iterate tightens the bracket (f is increasing, so the sign of f says which side
   the root is on). Start from the larger of the Manaster-Koehler point
   :math:`\sqrt{2|\ln(F/K)|/T}` and the Brenner-Subrahmanyam approximation
   :math:`p\sqrt{2\pi}/(\hat S\sqrt T)`, clipped into the bracket.
3. Brent fallback: if vega falls below ``1e-8`` of its maximum
   :math:`\hat S\sqrt{T/2\pi}`, if a Newton step leaves the bracket, or if Newton has
   not converged in 50 iterations, switch to Brent's method
   (``scipy.optimize.brentq``) on the current bracket. Brent needs no derivative and
   is guaranteed to converge on a bracket.

Convergence: :math:`|f| \le 10^{-12} p`, or the step falls below ``4 eps * sigma``
(the price has stopped changing at double precision).

Conditioning (why vol -> price -> vol is not always testable): near the bounds vega
is tiny, so a price error :math:`\delta p` maps to a vol error
:math:`\delta\sigma \approx \delta p / \text{vega}`. There, many different volatilities
give the *same* double-precision price, and the vol cannot be recovered even though
the price is reproduced exactly. The tests therefore check the round trip on price.

Sources:
    * S. Manaster and G. Koehler (1982), "The Calculation of Implied Variances from the
      Black-Scholes Model: A Note", *Journal of Finance* 37(1), 227-230.
    * M. Brenner and M. G. Subrahmanyam (1988), "A Simple Formula to Compute the Implied
      Standard Deviation", *Financial Analysts Journal* 44(5), 80-83.
    * R. P. Brent (1973), *Algorithms for Minimization without Derivatives*,
      Prentice-Hall, chapter 4 (the root finder behind ``scipy.optimize.brentq``).
    * R. C. Merton (1973), "Theory of Rational Option Pricing" (price bounds).
"""

import math
from dataclasses import dataclass
from typing import Literal

from scipy.optimize import brentq

from quantprice import bsm
from quantprice._validation import validate_market_inputs
from quantprice.bsm import OptionType, _parse_option_type

__all__ = ["IVResult", "implied_vol", "solve_implied_vol"]

_EPS = 2.220446049250313e-16
_MAX_NEWTON = 50
_MAX_S = 50.0  # largest sigma * sqrt(T) we search
_BOUND_TOL = 1e-14  # relative to scale, for "price is at the lower bound"


@dataclass(frozen=True, slots=True)
class IVResult:
    """Implied-volatility solution with diagnostics.

    Attributes:
        sigma: implied volatility (annualised).
        method: ``"newton"``, ``"brent"`` (fallback used) or ``"lower-bound"`` (sigma = 0).
        iterations: Newton iterations, plus Brent's function evaluations if it ran.
        price_error: ``V_BSM(sigma) - price``.
    """

    sigma: float
    method: Literal["newton", "brent", "lower-bound"]
    iterations: int
    price_error: float


def solve_implied_vol(
    option_type: OptionType,
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
) -> IVResult:
    """Implied volatility with diagnostics. See the module docstring for the method.

    Raises:
        ValueError: invalid inputs, ``T == 0`` (the price carries no volatility
            information), or a price outside, or at the upper end of, the no-arbitrage
            range.
    """
    is_call = _parse_option_type(option_type)
    if not (isinstance(price, (int, float)) and math.isfinite(price)):
        raise ValueError(f"price must be a finite number; got {price!r}")
    arrays = validate_market_inputs(S, K, T, r, 0.0, q)
    if any(a.ndim for a in arrays):
        raise ValueError("implied_vol takes scalar inputs")
    S_, K_, T_, r_, _, q_ = (float(a) for a in arrays)
    if T_ == 0.0:
        raise ValueError("T must be > 0: at expiry the price does not depend on volatility")

    S_hat, K_hat = S_ * math.exp(-q_ * T_), K_ * math.exp(-r_ * T_)
    scale = max(S_hat, K_hat)
    lower = max(S_hat - K_hat, 0.0) if is_call else max(K_hat - S_hat, 0.0)
    upper = S_hat if is_call else K_hat
    if price < lower - _BOUND_TOL * scale:
        raise ValueError(
            f"price {price!r} is below the no-arbitrage lower bound {lower!r} "
            f"(discounted forward intrinsic value)"
        )
    if price >= upper:
        raise ValueError(
            f"price {price!r} is at or above the no-arbitrage upper bound {upper!r} "
            f"({'S e^(-qT)' if is_call else 'K e^(-rT)'}); implied volatility would be infinite"
        )
    if price <= lower:  # at the bound, or below it by rounding only (checked above)
        return IVResult(
            0.0, "lower-bound", 0, bsm.price(option_type, S_, K_, T_, r_, 0.0, q_) - price
        )

    sqrt_T = math.sqrt(T_)

    def f(sig: float) -> float:
        return bsm.price(option_type, S_, K_, T_, r_, sig, q_) - price

    # 1. bracket
    lo, hi = 0.0, 1.0
    while f(hi) < 0.0:
        lo, hi = hi, 2.0 * hi
        if hi * sqrt_T > _MAX_S:
            raise ValueError(
                f"price {price!r} is too close to the upper bound {upper!r} to resolve an "
                "implied volatility in double precision"
            )

    # 2. safeguarded Newton
    mk = math.sqrt(2.0 * abs(math.log(S_hat / K_hat)) / T_)
    bs_atm = price * math.sqrt(2.0 * math.pi) / (S_hat * sqrt_T)
    sig = min(max(mk, bs_atm), hi)
    if not lo < sig < hi:
        sig = 0.5 * (lo + hi)
    max_vega = S_hat * sqrt_T / math.sqrt(2.0 * math.pi)
    iterations = 0
    while iterations < _MAX_NEWTON:
        iterations += 1
        g = bsm.greeks(option_type, S_, K_, T_, r_, sig, q_)
        err = g.price - price
        if abs(err) <= 1e-12 * price:
            return IVResult(sig, "newton", iterations, err)
        if err > 0:
            hi = sig
        else:
            lo = sig
        if g.vega < 1e-8 * max_vega:
            break  # flat: Newton steps are unreliable here
        step = err / g.vega
        new = sig - step
        if not lo < new < hi:
            break  # Newton left the bracket
        if abs(step) <= 4.0 * _EPS * new:
            return IVResult(new, "newton", iterations, f(new))
        sig = new

    # 3. Brent fallback on the current bracket
    root, info = brentq(f, lo, hi, xtol=1e-300, rtol=4.0 * _EPS, maxiter=500, full_output=True)
    return IVResult(float(root), "brent", iterations + int(info.function_calls), f(float(root)))


def implied_vol(
    option_type: OptionType,
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
) -> float:
    """Implied BSM volatility of a European option price. See :func:`solve_implied_vol`."""
    return solve_implied_vol(option_type, price, S, K, T, r, q).sigma
