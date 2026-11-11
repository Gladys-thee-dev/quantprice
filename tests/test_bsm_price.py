"""Phase 1a: Black-Scholes-Merton prices, limiting cases and input validation."""

import itertools
import math
from typing import Any

import numpy as np
import pytest

from quantprice.bsm import OptionType, call_price, price, put_price
from reference_mpmath import mp_bsm_price

# Grid spanning the full ranges from the project spec, with negative rates included.
GRID = list(
    itertools.product(
        [1.0, 50.0, 100.0, 150.0, 1000.0],  # S
        [1.0, 80.0, 100.0, 120.0, 1000.0],  # K
        [0.01, 0.25, 1.0, 5.0],  # T
        [-0.02, 0.0, 0.05, 0.10],  # r
        [0.01, 0.2, 0.8, 2.0],  # sigma
        [0.0, 0.03, 0.08],  # q
    )
)


def scale(S: float, K: float, T: float, r: float, q: float) -> float:
    """Natural size of the problem: max(S e^{-qT}, K e^{-rT})."""
    return max(S * math.exp(-q * T), K * math.exp(-r * T))


# ----------------------------------------------------------------- reference agreement


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_matches_50_digit_reference_on_grid(option_type: OptionType) -> None:
    """Absolute error vs mpmath, measured at <= 3.2e-16 * scale on this grid (2026-11-11).

    The tolerance of 1e-14 * scale leaves about 30x headroom for platform differences
    in libm / scipy.
    """
    is_call = option_type == "call"
    params = np.array(GRID).T
    ours = price(option_type, *params)  # one vectorised call over all 4800 points
    for i, (S, K, T, r, sigma, q) in enumerate(GRID):
        ref = mp_bsm_price(is_call, S, K, T, r, sigma, q)
        assert abs(ours[i] - ref) <= 1e-14 * scale(S, K, T, r, q), (GRID[i], ours[i], ref)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_relative_accuracy_where_price_is_not_negligible(option_type: OptionType) -> None:
    """Relative error, measured at <= 1.9e-12 for prices > 1e-8 * scale; tolerance 1e-10.

    Deep out of the money, the price is a difference of two nearly equal tiny terms, so
    relative precision is lost to cancellation (see the bsm module docstring). Absolute
    precision is still covered by the test above.
    """
    is_call = option_type == "call"
    for S, K, T, r, sigma, q in GRID:
        ref = mp_bsm_price(is_call, S, K, T, r, sigma, q)
        if ref > 1e-8 * scale(S, K, T, r, q):
            ours = price(option_type, S, K, T, r, sigma, q)
            assert abs(ours - ref) / ref <= 1e-10


def test_hull_textbook_example() -> None:
    """Hull, *Options, Futures, and Other Derivatives*, worked BSM example:
    S=42, K=40, r=10%, sigma=20%, T=0.5 gives c = 4.76 and p = 0.81 (2 d.p.).

    It is only quoted to 2 d.p., so this catches gross errors only.
    """
    assert call_price(42.0, 40.0, 0.5, 0.10, 0.20) == pytest.approx(4.76, abs=0.005)
    assert put_price(42.0, 40.0, 0.5, 0.10, 0.20) == pytest.approx(0.81, abs=0.005)


def test_haug_merton_dividend_example() -> None:
    """Haug (2007), section 1.1.6 (Merton 1973, stock index): S=100, K=95, T=0.5, r=10%,
    q=5%, sigma=20%, put = 2.4648 (4 d.p.)."""
    assert put_price(100.0, 95.0, 0.5, 0.10, 0.20, 0.05) == pytest.approx(2.4648, abs=5e-5)


# ------------------------------------------------------------------- limiting cases


@pytest.mark.parametrize(
    ("S", "K", "T", "r", "q"),
    [
        (100.0, 90.0, 1.0, 0.05, 0.02),
        (100.0, 110.0, 1.0, 0.05, 0.02),
        (100.0, 100.0, 2.0, -0.02, 0.0),  # negative rate
        (100.0, 100.0 * math.exp(0.03), 1.0, 0.03, 0.0),  # strike exactly at the forward
    ],
)
def test_zero_volatility_is_discounted_forward_intrinsic(
    S: float, K: float, T: float, r: float, q: float
) -> None:
    S_hat, K_hat = S * math.exp(-q * T), K * math.exp(-r * T)
    assert call_price(S, K, T, r, 0.0, q) == max(S_hat - K_hat, 0.0)
    assert put_price(S, K, T, r, 0.0, q) == max(K_hat - S_hat, 0.0)


@pytest.mark.parametrize(("S", "K"), [(100.0, 90.0), (100.0, 110.0), (100.0, 100.0)])
def test_at_expiry_is_payoff(S: float, K: float) -> None:
    assert call_price(S, K, 0.0, 0.05, 0.3, 0.02) == max(S - K, 0.0)
    assert put_price(S, K, 0.0, 0.05, 0.3, 0.02) == max(K - S, 0.0)


@pytest.mark.parametrize("sigma", [1e-3, 1e-6, 1e-10, 1e-14])
@pytest.mark.parametrize("K", [90.0, 100.0 * math.exp(0.03), 110.0])
def test_price_is_continuous_as_volatility_goes_to_zero(sigma: float, K: float) -> None:
    """|C(sigma) - C(0)| <= S e^{-qT} * sigma * sqrt(T) / sqrt(2 pi).

    Proof: dC/dsigma = vega = S e^{-qT} phi(d1) sqrt(T) <= S e^{-qT} sqrt(T) / sqrt(2 pi).
    Integrate from 0 to sigma. The 1e-14 * scale term allows for rounding.
    """
    S, T, r, q = 100.0, 1.0, 0.03, 0.0
    bound = S * sigma * math.sqrt(T) / math.sqrt(2 * math.pi) + 1e-14 * scale(S, K, T, r, q)
    for ot in ("call", "put"):
        assert abs(price(ot, S, K, T, r, sigma, q) - price(ot, S, K, T, r, 0.0, q)) <= bound


# --------------------------------------------------------------- numerical stability


@pytest.mark.parametrize(
    ("S", "K", "T", "sigma"),
    [
        (1.0, 1000.0, 0.01, 0.01),  # d ~ -690: N(d) underflows
        (1000.0, 1.0, 0.01, 0.01),  # d ~ +690
        (1.0, 1000.0, 5.0, 2.0),
        (1000.0, 1.0, 5.0, 2.0),
        (100.0, 100.0, 1.0, 1e-300),  # sigma*sqrt(T) close to underflow
    ],
)
@pytest.mark.parametrize("r", [-0.02, 0.10])
def test_extreme_moneyness_is_finite_nonnegative_and_above_intrinsic(
    S: float, K: float, T: float, sigma: float, r: float
) -> None:
    q = 0.03
    S_hat, K_hat = S * math.exp(-q * T), K * math.exp(-r * T)
    c, p = call_price(S, K, T, r, sigma, q), put_price(S, K, T, r, sigma, q)
    tol = 1e-14 * scale(S, K, T, r, q)
    for v in (c, p):
        assert math.isfinite(v)
        assert v >= 0.0
    # No-arbitrage bounds: max(S_hat - K_hat, 0) <= C <= S_hat, and likewise for the put.
    assert max(S_hat - K_hat, 0.0) - tol <= c <= S_hat + tol
    assert max(K_hat - S_hat, 0.0) - tol <= p <= K_hat + tol


# ------------------------------------------------------------------- vectorisation


def test_scalar_inputs_return_python_float() -> None:
    assert type(call_price(100.0, 100.0, 1.0, 0.05, 0.2)) is float


def test_broadcasting_matches_scalar_loop() -> None:
    strikes = np.array([80.0, 100.0, 120.0])
    vols = np.array([[0.1], [0.3]])  # shape (2, 1) with (3,) broadcasts to (2, 3)
    out = put_price(100.0, strikes, 1.0, 0.05, vols, 0.01)
    assert out.shape == (2, 3)
    for i, j in itertools.product(range(2), range(3)):
        assert out[i, j] == put_price(100.0, float(strikes[j]), 1.0, 0.05, float(vols[i, 0]), 0.01)


def test_mixed_zero_and_positive_vol_in_one_array() -> None:
    """The s == 0 branch must not emit divide-by-zero warnings (warnings are errors here)."""
    out = call_price(100.0, 90.0, np.array([0.0, 1.0]), 0.05, np.array([0.2, 0.0]))
    np.testing.assert_array_equal(out, [10.0, 100.0 - 90.0 * math.exp(-0.05)])


# ------------------------------------------------------------------- input validation


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"S": 0.0}, r"spot S must be > 0; got 0\.0"),
        ({"S": -5.0}, r"spot S must be > 0; got -5\.0"),
        ({"K": 0.0}, r"strike K must be > 0"),
        ({"T": -0.1}, r"time to expiry T must be >= 0; got -0\.1"),
        ({"sigma": -0.2}, r"volatility sigma must be >= 0; got -0\.2"),
        ({"sigma": float("nan")}, r"volatility sigma must be finite"),
        ({"r": float("inf")}, r"interest rate r must be finite"),
        ({"q": float("-inf")}, r"dividend yield q must be finite"),
        ({"K": [100.0, -1.0]}, r"strike K must be > 0; got -1\.0 at index \(1,\)"),
        ({"S": "abc"}, r"spot S must be a real number"),
        ({"K": [1.0, 2.0], "sigma": [0.1, 0.2, 0.3]}, r"cannot be broadcast"),
        ({"r": -1000.0}, r"overflow"),  # exp(-r*T) = e^1000 overflows
    ],
)
def test_invalid_inputs_raise_value_error(kwargs: dict[str, object], message: str) -> None:
    args: dict[str, Any] = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "q": 0.0}
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        price("call", **args)


def test_unknown_option_type_raises() -> None:
    with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
        price("straddle", 100.0, 100.0, 1.0, 0.05, 0.2)  # type: ignore[call-overload]
