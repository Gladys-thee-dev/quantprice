"""Phase 1c: Hypothesis property tests for BSM prices and Greeks over the spec's input ranges.

Ranges: S and K in [1, 1000], sigma in [0.01, 2.0], T in [0.01, 5], r in [-0.02, 0.10],
q in [0, 0.08]. Tolerances are relative to the problem's natural size,
scale = max(S e^{-qT}, K e^{-rT}). Without that, a deep out-of-the-money price of
1e-300 would be judged on relative error made entirely of rounding noise.
"""

import math

from hypothesis import given
from hypothesis import strategies as st

from quantprice.bsm import call_price, greeks, put_price

spot = st.floats(1.0, 1000.0)
strike = st.floats(1.0, 1000.0)
vol = st.floats(0.01, 2.0)
expiry = st.floats(0.01, 5.0)
rate = st.floats(-0.02, 0.10)
div_yield = st.floats(0.0, 0.08)


def scale(S: float, K: float, T: float, r: float, q: float) -> float:
    return max(S * math.exp(-q * T), K * math.exp(-r * T))


@given(spot, strike, expiry, rate, vol, div_yield)
def test_put_call_parity(S: float, K: float, T: float, r: float, sigma: float, q: float) -> None:
    """C - P = S e^{-qT} - K e^{-rT} (Hull, "Properties of Stock Options"; Merton 1973),
    within 1e-10 * scale."""
    lhs = call_price(S, K, T, r, sigma, q) - put_price(S, K, T, r, sigma, q)
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert abs(lhs - rhs) <= 1e-10 * scale(S, K, T, r, q)


@given(spot, strike, expiry, rate, vol, vol, div_yield)
def test_call_price_non_decreasing_in_volatility(
    S: float, K: float, T: float, r: float, sig_a: float, sig_b: float, q: float
) -> None:
    """vega >= 0, so sigma_lo <= sigma_hi implies C(sigma_lo) <= C(sigma_hi). The
    1e-13 * scale slack allows for rounding (measured price error <= 3.2e-16 * scale)."""
    lo, hi = sorted((sig_a, sig_b))
    c_lo, c_hi = call_price(S, K, T, r, lo, q), call_price(S, K, T, r, hi, q)
    assert c_hi >= c_lo - 1e-13 * scale(S, K, T, r, q)


@given(spot, strike, expiry, rate, vol, div_yield)
def test_delta_bounds(S: float, K: float, T: float, r: float, sigma: float, q: float) -> None:
    """0 <= delta_call <= e^{-qT} and -e^{-qT} <= delta_put <= 0.

    The factor (1 + 1e-15) allows for np.exp and math.exp differing by one ulp.
    """
    cap = math.exp(-q * T) * (1 + 1e-15)
    dc = greeks("call", S, K, T, r, sigma, q).delta
    dp = greeks("put", S, K, T, r, sigma, q).delta
    assert 0.0 <= dc <= cap
    assert -cap <= dp <= 0.0


@given(spot, strike, expiry, rate, vol, div_yield)
def test_delta_parity(S: float, K: float, T: float, r: float, sigma: float, q: float) -> None:
    """Differentiating put-call parity in S: delta_call - delta_put = e^{-qT}."""
    dc = greeks("call", S, K, T, r, sigma, q).delta
    dp = greeks("put", S, K, T, r, sigma, q).delta
    assert abs((dc - dp) - math.exp(-q * T)) <= 1e-14


@given(spot, strike, expiry, rate, vol, div_yield)
def test_gamma_and_vega_non_negative(
    S: float, K: float, T: float, r: float, sigma: float, q: float
) -> None:
    for option_type in ("call", "put"):
        g = greeks(option_type, S, K, T, r, sigma, q)
        assert g.gamma >= 0.0
        assert g.vega >= 0.0
