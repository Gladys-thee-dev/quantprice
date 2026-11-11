"""High-precision BSM reference, written independently of ``quantprice.bsm``. Used only in tests.

Prices use the textbook form of the formula
(d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt T)) and mpmath's normal CDF at
50 significant digits.

The Greeks are not closed forms. They are numerical derivatives of that
high-precision price (``mpmath.diff``, a finite-difference scheme carried out in
extended precision), so they share no formulas with the code under test. At 50 working
digits the differentiation error is far below double precision for these smooth
functions.

Reference: Haug (2007), *The Complete Guide to Option Pricing Formulas*, 2nd ed., section 1.1.6.
"""

from typing import Any

import mpmath as mp

_DPS = 50


def _mp_price(is_call: bool, S: Any, K: Any, T: Any, r: Any, sigma: Any, q: Any) -> Any:
    """BSM price as an mpf (call inside an mp.workdps block)."""
    disc_S = S * mp.exp(-q * T)
    disc_K = K * mp.exp(-r * T)
    if sigma == 0 or T == 0:
        fwd_intrinsic = disc_S - disc_K if is_call else disc_K - disc_S
        return max(fwd_intrinsic, mp.mpf(0))
    vol_sqrt_t = sigma * mp.sqrt(T)
    d1 = (mp.log(S / K) + (r - q + sigma**2 / 2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    if is_call:
        return disc_S * mp.ncdf(d1) - disc_K * mp.ncdf(d2)
    return disc_K * mp.ncdf(-d2) - disc_S * mp.ncdf(-d1)


def mp_bsm_price(
    is_call: bool, S: float, K: float, T: float, r: float, sigma: float, q: float
) -> float:
    """BSM price at 50 significant digits, rounded once to float at the end."""
    with mp.workdps(_DPS):
        args = (mp.mpf(x) for x in (S, K, T, r, sigma, q))
        return float(_mp_price(is_call, *args))


def mp_bsm_greeks(
    is_call: bool, S: float, K: float, T: float, r: float, sigma: float, q: float
) -> dict[str, float]:
    """delta, gamma, vega, theta (= -dV/dT, per year), rho by high-precision differentiation.

    Requires sigma > 0 and T > 0, since the price must be smooth around the point.
    """
    with mp.workdps(_DPS):
        S_, K_, T_, r_, sig, q_ = (mp.mpf(x) for x in (S, K, T, r, sigma, q))

        def f(**kw: Any) -> Any:
            a = {"S": S_, "K": K_, "T": T_, "r": r_, "sigma": sig, "q": q_} | kw
            return _mp_price(is_call, a["S"], a["K"], a["T"], a["r"], a["sigma"], a["q"])

        return {
            "delta": float(mp.diff(lambda x: f(S=x), S_)),
            "gamma": float(mp.diff(lambda x: f(S=x), S_, 2)),
            "vega": float(mp.diff(lambda x: f(sigma=x), sig)),
            "theta": float(-mp.diff(lambda x: f(T=x), T_)),
            "rho": float(mp.diff(lambda x: f(r=x), r_)),
        }
