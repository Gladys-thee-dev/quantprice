"""Phase 4: implied volatility (Newton-Raphson with Brent fallback)."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantprice import bsm
from quantprice.bsm import OptionType
from quantprice.implied_vol import implied_vol, solve_implied_vol

spot = st.floats(1.0, 1000.0)
strike = st.floats(1.0, 1000.0)
vol = st.floats(0.01, 2.0)
expiry = st.floats(0.01, 5.0)
rate = st.floats(-0.02, 0.10)
div_yield = st.floats(0.0, 0.08)
kind = st.sampled_from(["call", "put"])


# ----------------------------------------------------------------------- round trips


@given(kind, spot, strike, expiry, rate, vol, div_yield)
def test_price_round_trip(
    option_type: OptionType, S: float, K: float, T: float, r: float, sigma: float, q: float
) -> None:
    """price -> implied vol -> price recovers the price to 1e-8 relative.

    Measured on 50,000 random points over these ranges (2026-11-12): max relative error
    1.4e-10, max absolute error 9.8e-13 * max(S e^{-qT}, K e^{-rT}). Fallback usage: Newton
    84%, Brent 13%, lower bound (sigma = 0) 3%.

    We do NOT test vol -> price -> vol over these ranges. Where vega is tiny (deep in or
    out of the money, short expiry), many different volatilities give the same
    double-precision price, so the original vol is unrecoverable even though the price
    is reproduced exactly (see test_vol_is_unrecoverable_where_vega_is_negligible). The
    inverse problem has condition number ~ price / (sigma * vega), which is unbounded
    near the no-arbitrage bounds.

    The 1e-300 absolute floor is for subnormal prices (below 2.2e-308), which carry fewer
    than 53 significant bits; relative error is meaningless there. Found by the 20k-example
    stress profile: a call priced at 4.6e-310 whose implied vol reprices to 0.0.
    """
    p = bsm.price(option_type, S, K, T, r, sigma, q)
    sig_hat = implied_vol(option_type, p, S, K, T, r, q)
    p_hat = bsm.price(option_type, S, K, T, r, sig_hat, q)
    assert abs(p_hat - p) <= 1e-8 * p + 1e-300


@given(kind, spot, strike, expiry, rate, vol, div_yield)
def test_vol_round_trip_where_well_conditioned(
    option_type: OptionType, S: float, K: float, T: float, r: float, sigma: float, q: float
) -> None:
    """Where vega >= 1% of its maximum S e^{-qT} sqrt(T / 2 pi), the vol itself is recovered:
    measured max relative error 1.8e-9; tolerance 1e-7."""
    g = bsm.greeks(option_type, S, K, T, r, sigma, q)
    if g.vega < 0.01 * S * math.exp(-q * T) * math.sqrt(T / (2 * math.pi)):
        return
    assert implied_vol(option_type, g.price, S, K, T, r, q) == pytest.approx(sigma, rel=1e-7)


def test_vol_is_unrecoverable_where_vega_is_negligible() -> None:
    """S = 100, K = 200, T = 0.1, sigma = 0.3. The OTM call (price 3.6e-13) gives back 0.3
    to 12 digits. The deep ITM put (price ~99) carries the same time value of ~3.6e-13 on
    top of 99, which is mostly lost to rounding in the price itself. Its implied vol comes
    back as ~0.2999, yet it reprices to the identical price. No solver can do better: the
    information is not in the input."""
    args = (100.0, 200.0, 0.1, 0.05)
    call = bsm.call_price(*args, 0.3)
    put = bsm.put_price(*args, 0.3)
    assert implied_vol("call", call, *args) == pytest.approx(0.3, rel=1e-10)
    put_vol = implied_vol("put", put, *args)
    assert abs(put_vol - 0.3) > 1e-6  # vol not recovered...
    assert bsm.put_price(*args, put_vol) == pytest.approx(put, rel=1e-15)  # ...price is


def test_known_volatility_is_recovered() -> None:
    """Hull's example price (S=42, K=40, r=10%, T=0.5) at sigma = 20% gives back 20%."""
    c = bsm.call_price(42.0, 40.0, 0.5, 0.10, 0.20)
    assert implied_vol("call", c, 42.0, 40.0, 0.5, 0.10) == pytest.approx(0.20, rel=1e-12)


@pytest.mark.parametrize("r", [-0.02, -0.005])
def test_negative_rates(r: float) -> None:
    p = bsm.put_price(100.0, 105.0, 2.0, r, 0.35, 0.01)
    assert implied_vol("put", p, 100.0, 105.0, 2.0, r, 0.01) == pytest.approx(0.35, rel=1e-10)


def test_call_and_put_give_the_same_vol_via_parity() -> None:
    S, K, T, r, q = 100.0, 110.0, 1.5, 0.03, 0.01
    c = bsm.call_price(S, K, T, r, 0.27, q)
    p = c - S * math.exp(-q * T) + K * math.exp(-r * T)  # put from parity
    assert implied_vol("call", c, S, K, T, r, q) == pytest.approx(
        implied_vol("put", p, S, K, T, r, q), rel=1e-10
    )


def test_implied_vol_increases_with_price() -> None:
    S, K, T, r = 100.0, 100.0, 1.0, 0.05
    prices = [bsm.call_price(S, K, T, r, s) for s in (0.1, 0.2, 0.4)]
    vols = [implied_vol("call", p, S, K, T, r) for p in prices]
    assert vols[0] < vols[1] < vols[2]


# ---------------------------------------------------------------- solver paths


def test_newton_converges_quickly_at_the_money() -> None:
    p = bsm.call_price(100.0, 100.0, 1.0, 0.05, 0.2)
    res = solve_implied_vol("call", p, 100.0, 100.0, 1.0, 0.05)
    assert res.method == "newton"
    assert res.iterations <= 6  # measured 4
    assert abs(res.price_error) <= 1e-12 * p


def test_brent_fallback_when_vega_is_tiny() -> None:
    """Deep OTM short-dated call (price 4.5e-20): vega is ~1e-17 of its maximum, so Newton
    steps are meaningless and Brent takes over. Measured: 31 iterations/evaluations."""
    args = (100.0, 150.0, 0.05, 0.02)
    p = bsm.call_price(*args, 0.2)
    res = solve_implied_vol("call", p, *args)
    assert res.method == "brent"
    assert res.sigma == pytest.approx(0.2, rel=1e-10)


# ---------------------------------------------------------- no-arbitrage bounds


@pytest.mark.parametrize(("option_type", "S", "K"), [("call", 120.0, 100.0), ("put", 80.0, 100.0)])
def test_price_below_lower_bound_raises(option_type: OptionType, S: float, K: float) -> None:
    T, r, q = 1.0, 0.05, 0.02
    intrinsic = abs(S * math.exp(-q * T) - K * math.exp(-r * T))
    with pytest.raises(ValueError, match="below the no-arbitrage lower bound"):
        implied_vol(option_type, intrinsic - 0.01, S, K, T, r, q)


@pytest.mark.parametrize(("option_type", "bound"), [("call", "S e"), ("put", "K e")])
def test_price_at_or_above_upper_bound_raises(option_type: OptionType, bound: str) -> None:
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.02
    upper = S * math.exp(-q * T) if option_type == "call" else K * math.exp(-r * T)
    for p in (upper, upper + 1.0):
        with pytest.raises(ValueError, match=f"at or above the no-arbitrage upper bound.*{bound}"):
            implied_vol(option_type, p, S, K, T, r, q)


def test_price_at_lower_bound_gives_zero_vol() -> None:
    """Exactly the discounted forward intrinsic value: sigma = 0, the limit, not a failure.
    One ulp below it (rounding) is treated the same way."""
    S, K, T, r = 120.0, 100.0, 1.0, 0.05
    lower = S - K * math.exp(-r * T)
    res = solve_implied_vol("call", lower, S, K, T, r)
    assert (res.sigma, res.method) == (0.0, "lower-bound")
    assert implied_vol("call", math.nextafter(lower, 0.0), S, K, T, r) == 0.0
    assert implied_vol("put", 0.0, S, K, T, r) == 0.0  # OTM put worth exactly 0


def test_price_one_ulp_below_upper_bound_still_reprices() -> None:
    """One ulp below S e^{-qT}: the price is reachable (at sigma * sqrt(T) ~ 16 the call's
    shortfall K e^{-rT} N(-s/2) is already below an ulp). The curve is flat there, so the
    returned sigma (measured 24.0) is effectively arbitrary, but the price is reproduced to
    an ulp. The solver's 'too close to the upper bound' error is a defensive guard for when
    rounding makes the bound unreachable; ordinary inputs don't trigger it."""
    upper = 100.0  # call upper bound S e^{-qT} with q = 0
    p = math.nextafter(upper, 0.0)
    sig = implied_vol("call", p, 100.0, 100.0, 1.0, 0.05)
    assert sig > 10.0
    assert bsm.call_price(100.0, 100.0, 1.0, 0.05, sig) == pytest.approx(p, rel=1e-15)


@pytest.mark.parametrize(
    ("price", "kwargs", "message"),
    [
        (5.0, {"T": 0.0}, "T must be > 0"),
        (float("nan"), {}, "price must be a finite number"),
        (-1.0, {}, "below the no-arbitrage lower bound"),
        (5.0, {"S": -100.0}, "spot S must be > 0"),
        (5.0, {"option_type": "straddle"}, "option_type must be"),
    ],
)
def test_invalid_inputs_raise(price: float, kwargs: dict[str, object], message: str) -> None:
    args: dict[str, object] = {"option_type": "call", "S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05}
    args |= kwargs
    with pytest.raises(ValueError, match=message):
        implied_vol(price=price, **args)  # type: ignore[arg-type]
