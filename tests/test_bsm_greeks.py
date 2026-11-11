"""Phase 1b: BSM Greeks, unit conventions and limiting Greeks at sigma*sqrt(T) = 0."""

import itertools
import math

import numpy as np
import pytest

from quantprice.bsm import OptionType, greeks, price
from reference_mpmath import mp_bsm_greeks

GREEKS = ("delta", "gamma", "vega", "theta", "rho")

GRID = list(
    itertools.product(
        [1.0, 100.0, 1000.0],  # S
        [1.0, 90.0, 100.0, 110.0, 1000.0],  # K
        [0.01, 0.5, 5.0],  # T
        [-0.02, 0.05],  # r
        [0.01, 0.3, 2.0],  # sigma
        [0.0, 0.08],  # q
    )
)


def natural_scales(
    S: float, K: float, T: float, r: float, sigma: float, q: float
) -> dict[str, float]:
    """Largest magnitude each Greek can reach for these inputs, up to O(1) factors.

    Errors are measured against these scales, so a Greek that is tiny because it is
    deep out of the money isn't judged on relative error made of rounding noise.
    """
    S_hat, K_hat, s = S * math.exp(-q * T), K * math.exp(-r * T), sigma * math.sqrt(T)
    return {
        "delta": 1.0,
        "gamma": math.exp(-q * T) / (S * s),
        "vega": S_hat * math.sqrt(T),
        "theta": S_hat * sigma / (2 * math.sqrt(T)) + abs(q) * S_hat + abs(r) * K_hat,
        "rho": T * K_hat,
    }


# ----------------------------------------------------------------- reference agreement


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_greeks_match_high_precision_differentiation(option_type: OptionType) -> None:
    """Closed-form Greeks vs mpmath.diff of the 50-digit price on 540 points per type.

    Measured on 2026-11-11 (both types): max |error| / scale <= 9.2e-16 and max relative
    error <= 2.8e-14 where |Greek| > 1e-8 * scale. Tolerances: 1e-13 * scale and 1e-11
    relative, about 100x headroom.
    """
    for S, K, T, r, sigma, q in GRID:
        ref = mp_bsm_greeks(option_type == "call", S, K, T, r, sigma, q)
        g = greeks(option_type, S, K, T, r, sigma, q)
        for name, sc in natural_scales(S, K, T, r, sigma, q).items():
            ours, exact = getattr(g, name), ref[name]
            point = (option_type, name, S, K, T, r, sigma, q, ours, exact)
            assert abs(ours - exact) <= 1e-13 * sc, point
            if abs(exact) > 1e-8 * sc:
                assert abs(ours - exact) <= 1e-11 * abs(exact), point


def test_hull_greeks_example() -> None:
    """Hull, *Options, Futures, and Other Derivatives*, chapter "The Greek Letters":
    call with S=49, K=50, r=5%, sigma=20%, T=20 weeks (20/52 years). Published values:
    price 2.40, delta 0.522, gamma 0.066, vega 12.1, theta -4.31 per year, rho 8.91.
    """
    g = greeks("call", 49.0, 50.0, 20 / 52, 0.05, 0.20)
    assert g.price == pytest.approx(2.40, abs=0.005)
    assert g.delta == pytest.approx(0.522, abs=0.0005)
    assert g.gamma == pytest.approx(0.066, abs=0.0005)
    assert g.vega == pytest.approx(12.1, abs=0.05)
    assert g.theta == pytest.approx(-4.31, abs=0.005)
    assert g.rho == pytest.approx(8.91, abs=0.005)


# -------------------------------------------------------------------- units & shape


def test_unit_conversions() -> None:
    g = greeks("put", 100.0, 105.0, 0.75, 0.04, 0.25, 0.01)
    assert g.vega_per_pct == g.vega * 0.01
    assert g.rho_per_pct == g.rho * 0.01
    assert g.theta_per_day == g.theta / 365.0
    assert g.price == price("put", 100.0, 105.0, 0.75, 0.04, 0.25, 0.01)


def test_theta_sign_convention_is_calendar_decay() -> None:
    """theta = dV/dt = -dV/dT: a long ATM call with q = 0 loses value as time passes."""
    g = greeks("call", 100.0, 100.0, 1.0, 0.05, 0.2)
    assert g.theta < 0
    # One-day repricing agrees with theta_per_day to first order.
    one_day_later = price("call", 100.0, 100.0, 1.0 - 1 / 365, 0.05, 0.2)
    assert one_day_later - g.price == pytest.approx(g.theta_per_day, rel=1e-3)


def test_scalar_inputs_give_floats_and_arrays_broadcast() -> None:
    g = greeks("call", 100.0, 100.0, 1.0, 0.05, 0.2)
    assert all(type(getattr(g, f)) is float for f in g.__dataclass_fields__)

    strikes = np.array([90.0, 100.0, 110.0])
    ga = greeks("call", 100.0, strikes, np.array([[0.5], [1.0]]), 0.05, 0.2)
    assert ga.delta.shape == (2, 3)
    gs = greeks("call", 100.0, 110.0, 1.0, 0.05, 0.2)
    assert ga.gamma[1, 2] == gs.gamma
    assert ga.theta_per_day[1, 2] == gs.theta_per_day


# -------------------------------------------------------------------- limiting Greeks


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("K", [90.0, 110.0])
def test_zero_vol_away_from_kink_matches_small_vol_limit(option_type: OptionType, K: float) -> None:
    """At sigma = 0 (T > 0, off the kink) the reported Greeks equal the BSM formulas at a
    tiny sigma, where N(d) has already saturated to exactly 0 or 1."""
    S, T, r, q = 100.0, 1.0, 0.05, 0.02
    lim = greeks(option_type, S, K, T, r, 0.0, q)
    near = greeks(option_type, S, K, T, r, 1e-6, q)
    for name in GREEKS:
        assert getattr(lim, name) == pytest.approx(getattr(near, name), abs=1e-12), name
    assert lim.gamma == 0.0
    assert lim.vega == 0.0


@pytest.mark.parametrize(("option_type", "w"), [("call", 1.0), ("put", -1.0)])
def test_zero_vol_at_kink(option_type: OptionType, w: float) -> None:
    """Kink with T > 0: S e^{-qT} == K e^{-rT} exactly (built with q = 0, S = K e^{-rT}).

    Expected limits: delta = w e^{-qT}/2, gamma = nan, vega = S_hat sqrt(T/(2 pi)),
    theta = w (q - r) S_hat / 2, rho = w T K_hat / 2. Each is checked against the BSM
    formulas at sigma = 1e-8, whose distance from the limit is O(sigma) (largest for
    theta: S_hat phi(0) sigma / (2 sqrt T) ~ 2e-7). A smaller sigma is not more accurate:
    the ~1e-17 rounding in S is divided by sigma sqrt(T) inside d1 (at sigma = 1e-10
    it moves delta by ~3e-8).
    """
    K, T, r, q = 100.0, 1.0, 0.05, 0.0
    S = float(K * np.exp(-r * T))  # same expression the library uses, so the kink is exact
    lim = greeks(option_type, S, K, T, r, 0.0, q)
    near = greeks(option_type, S, K, T, r, 1e-8, q)

    assert lim.delta == w * 0.5
    assert math.isnan(lim.gamma)
    assert lim.vega == pytest.approx(S * math.sqrt(T / (2 * math.pi)), rel=1e-15)
    assert lim.theta == pytest.approx(w * (q - r) * S / 2, rel=1e-15)
    assert lim.rho == pytest.approx(w * T * K * math.exp(-r * T) / 2, rel=1e-15)
    for name in ("delta", "vega", "theta", "rho"):
        assert getattr(lim, name) == pytest.approx(getattr(near, name), abs=1e-6), name


@pytest.mark.parametrize(("option_type", "w"), [("call", 1.0), ("put", -1.0)])
def test_at_expiry_at_the_money(option_type: OptionType, w: float) -> None:
    """T = 0, S = K, sigma > 0: delta -> w/2, gamma and theta diverge (nan), vega and
    rho are 0. delta is compared with the formula at T = 1e-14."""
    lim = greeks(option_type, 100.0, 100.0, 0.0, 0.05, 0.3, 0.02)
    near = greeks(option_type, 100.0, 100.0, 1e-14, 0.05, 0.3, 0.02)
    assert lim.delta == w * 0.5
    assert lim.delta == pytest.approx(near.delta, abs=1e-6)
    assert math.isnan(lim.gamma)
    assert math.isnan(lim.theta)
    assert lim.vega == 0.0
    assert lim.rho == 0.0


def test_zero_vol_and_zero_time_at_kink_delta_is_undefined() -> None:
    """With sigma = 0 and T = 0, the limit of delta depends on the path, so it's nan."""
    assert math.isnan(greeks("call", 100.0, 100.0, 0.0, 0.05, 0.0).delta)


@pytest.mark.parametrize(
    ("option_type", "S", "K", "delta", "theta"),
    [
        ("call", 110.0, 100.0, 1.0, 0.02 * 110 - 0.05 * 100),  # ITM call: theta = qS - rK
        ("call", 90.0, 100.0, 0.0, 0.0),  # OTM call
        ("put", 90.0, 100.0, -1.0, -(0.02 * 90 - 0.05 * 100)),  # ITM put: theta = rK - qS
        ("put", 110.0, 100.0, 0.0, 0.0),  # OTM put
    ],
)
def test_at_expiry_away_from_the_money(
    option_type: OptionType, S: float, K: float, delta: float, theta: float
) -> None:
    g = greeks(option_type, S, K, 0.0, 0.05, 0.3, 0.02)
    assert g.delta == delta
    assert g.theta == pytest.approx(theta, rel=1e-15, abs=1e-15)
    assert (g.gamma, g.vega, g.rho) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------- numerical stability


@pytest.mark.parametrize(
    ("S", "K", "T", "sigma"),
    [
        (1.0, 1000.0, 0.01, 0.01),
        (1000.0, 1.0, 0.01, 0.01),
        (1.0, 1000.0, 5.0, 2.0),
        (100.0, 100.0, 1.0, 1e-300),  # d*d would overflow without clipping
    ],
)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_extreme_inputs_give_finite_greeks(
    option_type: OptionType, S: float, K: float, T: float, sigma: float
) -> None:
    g = greeks(option_type, S, K, T, -0.02, sigma, 0.03)
    for name in GREEKS:
        assert math.isfinite(getattr(g, name)), name
    assert g.gamma >= 0.0
    assert g.vega >= 0.0


def test_invalid_input_raises() -> None:
    with pytest.raises(ValueError, match="volatility sigma must be >= 0"):
        greeks("call", 100.0, 100.0, 1.0, 0.05, -0.1)
