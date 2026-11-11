r"""Black-Scholes-Merton prices for European options on an asset with a continuous dividend yield.

Model and conventions
---------------------
The underlying follows geometric Brownian motion under the risk-neutral measure,

.. math:: dS_t = (r - q) S_t\,dt + \sigma S_t\,dW_t,

with constant annualised volatility :math:`\sigma`, constant continuously compounded
rate :math:`r` and constant continuous dividend yield :math:`q`. Time to expiry
:math:`T` is in years under ACT/365. There are no transaction costs, and exercise is
only possible at expiry.

Formula (generalised BSM)
-------------------------
With :math:`\hat S = S e^{-qT}` (the dividend-adjusted spot), :math:`\hat K = K e^{-rT}`
(the discounted strike) and :math:`s = \sigma\sqrt{T}`,

.. math::

    d_1 = \frac{\ln(S/K) + (r - q)T}{s} + \frac{s}{2}, \qquad d_2 = d_1 - s,

    C = \hat S\,N(d_1) - \hat K\,N(d_2), \qquad
    P = \hat K\,N(-d_2) - \hat S\,N(-d_1).

Here :math:`d_1` is the textbook :math:`[\ln(S/K) + (r - q + \sigma^2/2)T]/(\sigma\sqrt T)`,
regrouped algebraically.

Sources:
    * F. Black and M. Scholes (1973), "The Pricing of Options and Corporate
      Liabilities", *Journal of Political Economy* 81(3), 637-654.
    * R. C. Merton (1973), "Theory of Rational Option Pricing", *Bell Journal of
      Economics and Management Science* 4(1), 141-183 (continuous dividend yield).
    * E. G. Haug (2007), *The Complete Guide to Option Pricing Formulas*, 2nd ed.,
      McGraw-Hill, section 1.1.6 ("generalized Black-Scholes-Merton").

Numerical choices
-----------------
* :math:`N` is evaluated with ``scipy.special.ndtr``, which stays accurate in the lower
  tail. The put therefore uses :math:`N(-d)` directly, never :math:`1 - N(d)`, which
  would lose every digit when :math:`N(d)` is close to 1.
* :math:`\ln(S/K)` is used instead of :math:`\ln S - \ln K`: near the money it keeps
  full relative precision.
* Deep out of the money, :math:`N(d_1)` and :math:`N(d_2)` underflow to 0 and the
  price is exactly 0. That is correct to double precision, so it is not an error.
* The result is floored at 0. Mathematically every price is positive. The floor only
  removes a negative value of order machine-epsilon times the scale, which floating-point
  cancellation could produce in extreme inputs (e.g. :math:`\sigma\sqrt T < 10^{-13}`).

Limiting case :math:`\sigma\sqrt{T} = 0`
----------------------------------------
If :math:`\sigma = 0` or :math:`T = 0` the terminal price is the forward
:math:`F = S e^{(r-q)T}` with certainty. The price is then the discounted
forward-intrinsic value

.. math:: C = \max(\hat S - \hat K, 0), \qquad P = \max(\hat K - \hat S, 0),

which is also the limit of the BSM formula as :math:`\sigma\sqrt{T} \to 0`. At
:math:`T = 0` this reduces to the payoff :math:`\max(S - K, 0)`.
"""

from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr

from quantprice._validation import FloatArray, validate_market_inputs

__all__ = ["OptionType", "call_price", "price", "put_price"]

OptionType = Literal["call", "put"]


def _parse_option_type(option_type: str) -> bool:
    """Return True for a call, False for a put; raise ValueError otherwise."""
    if option_type == "call":
        return True
    if option_type == "put":
        return False
    raise ValueError(f"option_type must be 'call' or 'put'; got {option_type!r}")


def _price_array(
    is_call: bool,
    S: FloatArray,
    K: FloatArray,
    T: FloatArray,
    r: FloatArray,
    sigma: FloatArray,
    q: FloatArray,
) -> FloatArray:
    """Vectorised BSM price for inputs that are already validated and broadcast."""
    w = 1.0 if is_call else -1.0  # +1 call, -1 put
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        try:
            S_hat = S * np.exp(-q * T)
            K_hat = K * np.exp(-r * T)
            s = sigma * np.sqrt(T)
            diffusive = s > 0
            # Where s == 0, use a dummy denominator so no 0/0 is evaluated; those
            # entries are replaced by the deterministic limit below.
            s_safe = np.where(diffusive, s, 1.0)
            d1 = (np.log(S / K) + (r - q) * T) / s_safe + 0.5 * s_safe
            d2 = d1 - s_safe
            bsm = w * (S_hat * ndtr(w * d1) - K_hat * ndtr(w * d2))
            limit = np.maximum(w * (S_hat - K_hat), 0.0)
        except FloatingPointError as exc:
            raise ValueError(
                "inputs overflow double precision (check the magnitudes of r*T, q*T and S/K)"
            ) from exc
    out: FloatArray = np.maximum(np.where(diffusive, bsm, limit), 0.0)
    return out


@overload
def price(
    option_type: OptionType, S: float, K: float, T: float, r: float, sigma: float, q: float = ...
) -> float: ...
@overload
def price(
    option_type: OptionType,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = ...,
) -> FloatArray: ...
def price(
    option_type: OptionType,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> float | FloatArray:
    """Price a European call or put under Black-Scholes-Merton with dividend yield ``q``.

    See the module docstring for the formula, its assumptions and the sigma*sqrt(T) = 0
    limit.

    Args:
        option_type: ``"call"`` or ``"put"``.
        S: spot price, > 0.
        K: strike, > 0.
        T: time to expiry in years (ACT/365), >= 0.
        r: continuously compounded annual risk-free rate. Can be negative.
        sigma: annualised volatility, >= 0.
        q: continuously compounded annual dividend yield (default 0).

    Inputs can be scalars or arrays and are broadcast together (NumPy rules). If every
    input is a scalar, the result is a Python ``float``; otherwise it is a float64 array.

    Raises:
        ValueError: for an unknown ``option_type``, inputs that are non-finite or out of
            range, shapes that cannot be broadcast, or magnitudes that overflow.
    """
    is_call = _parse_option_type(option_type)
    arrays = validate_market_inputs(S, K, T, r, sigma, q)
    out = _price_array(is_call, *arrays)
    return float(out) if out.ndim == 0 else out


@overload
def call_price(S: float, K: float, T: float, r: float, sigma: float, q: float = ...) -> float: ...
@overload
def call_price(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike = ...
) -> FloatArray: ...
def call_price(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike = 0.0
) -> float | FloatArray:
    """European call price under BSM; same as ``price("call", ...)``."""
    return price("call", S, K, T, r, sigma, q)


@overload
def put_price(S: float, K: float, T: float, r: float, sigma: float, q: float = ...) -> float: ...
@overload
def put_price(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike = ...
) -> FloatArray: ...
def put_price(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike = 0.0
) -> float | FloatArray:
    """European put price under BSM; same as ``price("put", ...)``."""
    return price("put", S, K, T, r, sigma, q)
