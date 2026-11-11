"""High-precision BSM reference, written independently of ``quantprice.bsm``. Used only in tests.

It deliberately uses the textbook form of the formula
(d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt T)) and mpmath's normal CDF at
50 significant digits. A bug in the production code therefore cannot be copied here,
and the reference's own rounding error (about 1e-50 relative) is negligible next to
double precision.

Reference: Haug (2007), *The Complete Guide to Option Pricing Formulas*, 2nd ed., section 1.1.6.
"""

import mpmath as mp


def mp_bsm_price(
    is_call: bool, S: float, K: float, T: float, r: float, sigma: float, q: float, dps: int = 50
) -> float:
    """BSM price at ``dps`` significant digits, rounded once to float at the end."""
    with mp.workdps(dps):
        S_, K_, T_, r_, sig, q_ = (mp.mpf(x) for x in (S, K, T, r, sigma, q))
        disc_S = S_ * mp.exp(-q_ * T_)
        disc_K = K_ * mp.exp(-r_ * T_)
        if sig == 0 or T_ == 0:
            fwd_intrinsic = disc_S - disc_K if is_call else disc_K - disc_S
            return float(max(fwd_intrinsic, mp.mpf(0)))
        vol_sqrt_t = sig * mp.sqrt(T_)
        d1 = (mp.log(S_ / K_) + (r_ - q_ + sig**2 / 2) * T_) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
        if is_call:
            value = disc_S * mp.ncdf(d1) - disc_K * mp.ncdf(d2)
        else:
            value = disc_K * mp.ncdf(-d2) - disc_S * mp.ncdf(-d1)
        return float(value)
