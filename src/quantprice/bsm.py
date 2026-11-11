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

import math
from dataclasses import dataclass
from typing import Generic, Literal, NamedTuple, TypeVar, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

from quantprice._validation import FloatArray, validate_market_inputs
from quantprice.conventions import DAYS_PER_YEAR, PERCENTAGE_POINT

__all__ = ["Greeks", "OptionType", "call_price", "greeks", "price", "put_price"]

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
# |d| beyond this makes phi(d) < 1e-300, which is 0 to double precision. Clipping avoids
# overflow in d*d when sigma*sqrt(T) is tiny (e.g. 1e-300) without changing any result.
_D_CLIP = 40.0

OptionType = Literal["call", "put"]


def _parse_option_type(option_type: str) -> bool:
    """Return True for a call, False for a put; raise ValueError otherwise."""
    if option_type == "call":
        return True
    if option_type == "put":
        return False
    raise ValueError(f"option_type must be 'call' or 'put'; got {option_type!r}")


_OVERFLOW_MSG = "inputs overflow double precision (check the magnitudes of r*T, q*T and S/K)"


class _Terms(NamedTuple):
    """Intermediate quantities shared by the price and the Greeks."""

    S_hat: FloatArray  # S e^{-qT}
    K_hat: FloatArray  # K e^{-rT}
    sqrt_T: FloatArray
    diffusive: NDArray[np.bool_]  # sigma*sqrt(T) > 0
    s: FloatArray  # sigma*sqrt(T), with 1.0 substituted where it is 0
    d1: FloatArray
    d2: FloatArray


def _terms(
    S: FloatArray, K: FloatArray, T: FloatArray, r: FloatArray, sigma: FloatArray, q: FloatArray
) -> _Terms:
    """Compute d1, d2 and the discounted spot/strike. Overflow is turned into ValueError."""
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        try:
            S_hat = S * np.exp(-q * T)
            K_hat = K * np.exp(-r * T)
            sqrt_T = np.sqrt(T)
            s = sigma * sqrt_T
            diffusive = s > 0
            # Where s == 0, use a dummy denominator so no 0/0 is evaluated; those
            # entries are replaced by the deterministic limit afterwards.
            s_safe = np.where(diffusive, s, 1.0)
            d1 = (np.log(S / K) + (r - q) * T) / s_safe + 0.5 * s_safe
            d2 = d1 - s_safe
        except FloatingPointError as exc:
            raise ValueError(_OVERFLOW_MSG) from exc
    return _Terms(S_hat, K_hat, sqrt_T, diffusive, s_safe, d1, d2)


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
    t = _terms(S, K, T, r, sigma, q)
    bsm = w * (t.S_hat * ndtr(w * t.d1) - t.K_hat * ndtr(w * t.d2))
    limit = np.maximum(w * (t.S_hat - t.K_hat), 0.0)
    out: FloatArray = np.maximum(np.where(t.diffusive, bsm, limit), 0.0)
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


# =============================================================================== Greeks

V = TypeVar("V", float, FloatArray)


@dataclass(frozen=True, slots=True)
class Greeks(Generic[V]):
    """BSM price and first-order Greeks, each reported in the units the project specifies.

    Attributes:
        price: option value.
        delta: dV/dS.
        gamma: d2V/dS2.
        vega: dV/dsigma for a change of 1.00 in volatility (e.g. 0.20 -> 1.20).
        vega_per_pct: change in value for +1 percentage point of volatility (0.20 -> 0.21),
            i.e. ``vega * 0.01``.
        theta: dV/dt per year of calendar time passing. The sign convention is
            theta = dV/dt = -dV/dT, so a long vanilla option usually has theta < 0.
        theta_per_day: ``theta / 365``, the change over one calendar day (ACT/365).
        rho: dV/dr for a change of 1.00 in the rate.
        rho_per_pct: change for +1 percentage point in r, i.e. ``rho * 0.01``.

    These per-pct and per-day figures are first-order (linear) rescalings of the
    derivatives. They are not repricings with bumped inputs.
    """

    price: V
    delta: V
    gamma: V
    vega: V
    vega_per_pct: V
    theta: V
    theta_per_day: V
    rho: V
    rho_per_pct: V


def _greeks_arrays(
    is_call: bool,
    S: FloatArray,
    K: FloatArray,
    T: FloatArray,
    r: FloatArray,
    sigma: FloatArray,
    q: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Vectorised (price, delta, gamma, vega, theta, rho) for validated, broadcast inputs."""
    w = 1.0 if is_call else -1.0
    t = _terms(S, K, T, r, sigma, q)
    price_ = _price_array(is_call, S, K, T, r, sigma, q)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        try:
            disc_q = np.exp(-q * T)
            sqrt_T_safe = np.where(t.diffusive, t.sqrt_T, 1.0)
            pdf_d1 = _INV_SQRT_2PI * np.exp(-0.5 * np.square(np.clip(t.d1, -_D_CLIP, _D_CLIP)))
            N_wd1 = ndtr(w * t.d1)
            N_wd2 = ndtr(w * t.d2)

            # ---- sigma*sqrt(T) > 0: closed forms (Haug 2007, section 2.1; Hull, BSM Greeks)
            delta = w * disc_q * N_wd1
            gamma = disc_q * pdf_d1 / (S * t.s)
            vega = t.S_hat * pdf_d1 * t.sqrt_T
            theta = -t.S_hat * pdf_d1 * sigma / (2.0 * sqrt_T_safe) + w * (
                q * t.S_hat * N_wd1 - r * t.K_hat * N_wd2
            )
            rho = w * T * t.K_hat * N_wd2

            # ---- sigma*sqrt(T) == 0: limits of the formulas above as sigma*sqrt(T) -> 0+.
            # N(w d1) and N(w d2) tend to 1 in the money, 0 out of the money, and 1/2 at
            # the kink S e^{-qT} == K e^{-rT}.
            kink = t.S_hat == t.K_hat
            itm = w * (t.S_hat - t.K_hat) > 0
            n_lim = np.where(kink, 0.5, np.where(itm, 1.0, 0.0))
            no_time_no_vol = (sigma == 0) & (T == 0)
            lim_delta = np.where(kink & no_time_no_vol, np.nan, w * disc_q * n_lim)
            lim_gamma = np.where(kink, np.nan, 0.0)
            lim_vega = np.where(kink, t.S_hat * t.sqrt_T * _INV_SQRT_2PI, 0.0)
            lim_theta = np.where(kink & (T == 0), np.nan, w * (q * t.S_hat - r * t.K_hat) * n_lim)
            lim_rho = w * T * t.K_hat * n_lim
        except FloatingPointError as exc:
            raise ValueError(_OVERFLOW_MSG) from exc

    d = t.diffusive
    return (
        price_,
        np.where(d, delta, lim_delta),
        np.where(d, gamma, lim_gamma),
        np.where(d, vega, lim_vega),
        np.where(d, theta, lim_theta),
        np.where(d, rho, lim_rho),
    )


@overload
def greeks(
    option_type: OptionType, S: float, K: float, T: float, r: float, sigma: float, q: float = ...
) -> Greeks[float]: ...
@overload
def greeks(
    option_type: OptionType,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = ...,
) -> Greeks[FloatArray]: ...
def greeks(
    option_type: OptionType,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> Greeks[float] | Greeks[FloatArray]:
    r"""Price and first-order Greeks of a European option under BSM with dividend yield ``q``.

    Conventions: annualised volatility; continuously compounded ``r`` and ``q``; ``T`` in
    years, ACT/365. Inputs, broadcasting and errors are the same as for :func:`price`.

    Formulas for :math:`\sigma\sqrt T > 0`, with :math:`w = +1` for a call and
    :math:`-1` for a put, :math:`\hat S = S e^{-qT}`, :math:`\hat K = K e^{-rT}`, and
    :math:`\varphi` the standard normal density:

    .. math::

        \Delta = w e^{-qT} N(w d_1), \qquad
        \Gamma = \frac{e^{-qT}\varphi(d_1)}{S\sigma\sqrt T}, \qquad
        \mathcal V = \hat S\,\varphi(d_1)\sqrt T,

        \Theta = -\frac{\hat S\,\varphi(d_1)\,\sigma}{2\sqrt T}
                 + w\left[q\hat S N(w d_1) - r\hat K N(w d_2)\right], \qquad
        \rho = w\,T\hat K N(w d_2).

    Sources: Haug (2007), *The Complete Guide to Option Pricing Formulas*, 2nd ed.,
    section 2.1 (Greeks of the generalised BSM formula); J. C. Hull, *Options, Futures,
    and Other Derivatives*, chapter "The Greek Letters".

    Limiting Greeks when sigma*sqrt(T) = 0
    --------------------------------------
    The value is the discounted forward intrinsic :math:`\max(w(\hat S - \hat K), 0)`.
    We report the limit of each formula above as :math:`\sigma\sqrt T \to 0^+`. Where
    that limit is infinite, or depends on how :math:`(\sigma, T)` reach 0, we report
    ``nan`` instead. "Kink" means :math:`\hat S = \hat K` exactly (at :math:`T = 0`:
    :math:`S = K`). ITM means in the money with respect to the forward.

    ======  ====================================  ====================================
    Greek   away from the kink                    at the kink
    ======  ====================================  ====================================
    delta   w e^{-qT} if ITM, else 0              w e^{-qT} / 2; ``nan`` if sigma = 0
                                                  and T = 0 (limit is path-dependent)
    gamma   0                                     ``nan`` (diverges like 1/(sigma sqrt T))
    vega    0                                     S_hat sqrt(T / (2 pi)) (0 at T = 0)
    theta   w (q S_hat - r K_hat) if ITM, else 0  w (q - r) S_hat / 2 if T > 0;
                                                  ``nan`` at T = 0 (diverges)
    rho     w T K_hat if ITM, else 0              w T K_hat / 2
    ======  ====================================  ====================================

    The kink test is an exact floating-point equality. One ulp away from it you get
    the "away" column, because the true Greeks are discontinuous there in the limit.

    Returns:
        :class:`Greeks` with Python floats if every input is a scalar, otherwise
        float64 arrays of the broadcast shape.
    """
    is_call = _parse_option_type(option_type)
    arrays = validate_market_inputs(S, K, T, r, sigma, q)
    p, delta, gamma, vega, theta, rho = _greeks_arrays(is_call, *arrays)
    vega_pct = vega * PERCENTAGE_POINT
    theta_day = theta / DAYS_PER_YEAR
    rho_pct = rho * PERCENTAGE_POINT
    if p.ndim == 0:
        return Greeks(
            *(float(x) for x in (p, delta, gamma, vega, vega_pct, theta, theta_day, rho, rho_pct))
        )
    return Greeks(p, delta, gamma, vega, vega_pct, theta, theta_day, rho, rho_pct)
