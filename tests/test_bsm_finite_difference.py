r"""Phase 1c: analytic Greeks vs central finite differences of the price.

Bump sizes (documented, chosen by measurement)
-----------------------------------------------
Each input is bumped by a fixed fraction ``c`` of its natural length scale, the
distance over which the price changes shape:

=========  ====================  ===================================================
Greek      bumped input          bump h
=========  ====================  ===================================================
delta      S                     1e-4 * S * sigma * sqrt(T)  (one std dev of ln S_T)
gamma      S                     same as delta (second central difference)
vega       sigma                 1e-4 * sigma
theta      T                     3e-5 * T   (theta = -dV/dT)
rho        r                     1e-4 * sigma / sqrt(T)  (moves d1 by ~1e-4 * sqrt(T))
=========  ====================  ===================================================

A fixed bump such as h = 0.01 fails near expiry at the money, where the price bends
on a length scale of S*sigma*sqrt(T) -> 0. Scaling the bump with that length keeps the
truncation error a constant fraction.

Pass criterion
--------------
Central differences have two error sources, so the criterion has two terms:

.. math:: |\text{FD} - \text{analytic}| \le 10^{-6} M + R

* Truncation, at most 1e-6 relative (the spec), with M = |Greek|. For theta, M is
  the sum of the absolute values of its three terms (decay, dividend, interest).
  Theta crosses zero, and at the crossing |theta| -> 0 while the truncation error
  doesn't. A relative test there would measure nothing but cancellation.
* Rounding: the price itself has rounding error u. Its measured maximum is
  3.2e-16 * scale (test_bsm_price.py), so we budget u = 1e-15 * scale. A central first
  difference amplifies this to at most u / h, and the second difference for gamma to
  4u / h^2. This truncation-vs-roundoff trade-off is the standard analysis in
  W. H. Press et al. (2007), *Numerical Recipes*, 3rd ed., section 5.7 ("Numerical
  Derivatives").

Measured on 2026-11-11: with these bumps, 800k random points over the spec ranges
(a quarter of them in the short-expiry, low-vol, near-ATM corner) all passed. Worst
err / tolerance: delta 0.05, gamma 0.29, vega 0.23, theta 0.66.

Near-expiry at-the-money points
-------------------------------
The spec excludes near-expiry ATM points from the main test and covers them
separately with looser tolerances. With bumps scaled to sigma*sqrt(T), measurement
showed no loosening was needed: for T from 1e-5 years (about 5 minutes) to 0.01, and
strikes within +-1 std dev of the forward, the worst err / tolerance was 0.24. The
separate test below therefore covers that region (outside the spec's T >= 0.01) with
the same criterion, and the main test is not restricted.
"""

import math

from hypothesis import given
from hypothesis import strategies as st

from quantprice.bsm import OptionType, greeks, price

C_S, C_SIGMA, C_T, C_R = 1e-4, 1e-4, 3e-5, 1e-4
PRICE_ROUNDING = 1e-15  # u / scale, ~3x the measured maximum


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def check_against_finite_differences(
    option_type: OptionType, S: float, K: float, T: float, r: float, sigma: float, q: float
) -> None:
    g = greeks(option_type, S, K, T, r, sigma, q)

    def value_at(S: float = S, T: float = T, r: float = r, sigma: float = sigma) -> float:
        return price(option_type, S, K, T, r, sigma, q)

    S_hat, K_hat = S * math.exp(-q * T), K * math.exp(-r * T)
    u = PRICE_ROUNDING * max(S_hat, K_hat)
    hS, hsig, hT, hr = (
        C_S * S * sigma * math.sqrt(T),
        C_SIGMA * sigma,
        C_T * T,
        C_R * sigma / math.sqrt(T),
    )

    up, dn = value_at(S=S + hS), value_at(S=S - hS)
    w = 1.0 if option_type == "call" else -1.0
    d1 = (math.log(S / K) + (r - q) * T) / (sigma * math.sqrt(T)) + 0.5 * sigma * math.sqrt(T)
    d2 = d1 - sigma * math.sqrt(T)
    phi = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    theta_terms = (
        S_hat * phi * sigma / (2 * math.sqrt(T))
        + abs(q * S_hat * norm_cdf(w * d1))
        + abs(r * K_hat * norm_cdf(w * d2))
    )

    checks = {
        "delta": ((up - dn) / (2 * hS), abs(g.delta), u / hS),
        "gamma": ((up - 2 * g.price + dn) / hS**2, abs(g.gamma), 4 * u / hS**2),
        "vega": (
            (value_at(sigma=sigma + hsig) - value_at(sigma=sigma - hsig)) / (2 * hsig),
            abs(g.vega),
            u / hsig,
        ),
        "theta": (-(value_at(T=T + hT) - value_at(T=T - hT)) / (2 * hT), theta_terms, u / hT),
        "rho": ((value_at(r=r + hr) - value_at(r=r - hr)) / (2 * hr), abs(g.rho), u / hr),
    }
    for name, (fd, magnitude, rounding) in checks.items():
        analytic = getattr(g, name)
        tol = 1e-6 * magnitude + rounding
        assert abs(fd - analytic) <= tol, (name, option_type, fd, analytic, tol)


@given(
    st.sampled_from(["call", "put"]),
    st.floats(1.0, 1000.0),
    st.floats(1.0, 1000.0),
    st.floats(0.01, 5.0),
    st.floats(-0.02, 0.10),
    st.floats(0.01, 2.0),
    st.floats(0.0, 0.08),
)
def test_greeks_match_finite_differences(
    option_type: OptionType, S: float, K: float, T: float, r: float, sigma: float, q: float
) -> None:
    check_against_finite_differences(option_type, S, K, T, r, sigma, q)


@given(
    st.sampled_from(["call", "put"]),
    st.floats(1.0, 1000.0),
    st.floats(math.log(1e-5), math.log(0.01)),  # log T: about 5 minutes to 3.65 days
    st.floats(-1.0, 1.0),  # strike within +-1 std dev of the forward
    st.floats(-0.02, 0.10),
    st.floats(0.01, 2.0),
    st.floats(0.0, 0.08),
)
def test_greeks_near_expiry_at_the_money(
    option_type: OptionType, S: float, log_T: float, z: float, r: float, sigma: float, q: float
) -> None:
    """Near-expiry ATM region, below the spec's T >= 0.01. The criterion is the same as
    the main test; the measured worst err / tolerance in this region was 0.24."""
    T = math.exp(log_T)
    K = S * math.exp((r - q) * T + z * sigma * math.sqrt(T))
    check_against_finite_differences(option_type, S, K, T, r, sigma, q)
