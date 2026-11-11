"""Phase 3b: discrete geometric Asian closed form and Asian Monte Carlo. Fixed seeds throughout."""

import math
from typing import Any

import numpy as np
import pytest
from numpy.typing import ArrayLike
from scipy.special import ndtr

from quantprice import asian, bsm
from quantprice.bsm import OptionType
from quantprice.mc import asian as mca

S, K, R, SIGMA, Q = 100.0, 100.0, 0.05, 0.3, 0.02
MONTHLY = np.arange(1, 13) / 12


# ------------------------------------------------------------------ closed form


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize(
    ("S_", "K_", "T", "r", "sigma", "q"),
    [(100.0, 100.0, 1.0, 0.05, 0.3, 0.02), (50.0, 80.0, 2.5, -0.01, 0.6, 0.0)],
)
def test_single_date_reduces_to_black_scholes(
    option_type: OptionType, S_: float, K_: float, T: float, r: float, sigma: float, q: float
) -> None:
    """n = 1 and t_1 = T: G = S_T, so the formula must equal BSM (measured gap 4.6e-14)."""
    geo = asian.geometric_price(option_type, S_, K_, r, sigma, q, times=[T])
    assert geo == pytest.approx(bsm.price(option_type, S_, K_, T, r, sigma, q), abs=1e-13 * S_)


def test_linear_time_variance_equals_double_sum() -> None:
    """The O(n) form sum_k (2(n-k)+1) t_k equals sum_ij min(t_i, t_j)."""
    t = np.sort(np.random.default_rng(0).uniform(0.01, 2.0, 300))
    _, v = asian.geometric_log_moments(S, R, SIGMA, Q, t)
    direct = SIGMA**2 * np.minimum.outer(t, t).sum() / t.size**2
    assert v == pytest.approx(direct, rel=1e-13)


def kemna_vorst_continuous(option_type: OptionType, T: float) -> float:
    """Continuous-average geometric Asian (Kemna & Vorst 1990), written independently:
    ln G ~ N(ln S + (r - q - sigma^2/2) T/2, sigma^2 T/3)."""
    mean = math.log(S) + (R - Q - 0.5 * SIGMA**2) * T / 2
    var = SIGMA**2 * T / 3
    fwd = math.exp(mean + var / 2)
    d1 = (math.log(fwd / K) + var / 2) / math.sqrt(var)
    d2 = d1 - math.sqrt(var)
    w = 1.0 if option_type == "call" else -1.0
    return math.exp(-R * T) * w * float(fwd * ndtr(w * d1) - K * ndtr(w * d2))


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_equally_spaced_dates_converge_to_continuous_limit(option_type: OptionType) -> None:
    """With t_i = iT/n, discrete - continuous ~ c/n. Measured for the call: n * gap =
    5.7429, 5.7204, 5.7184, 5.7182 for n = 10..10^4. Test: n * gap is stable to 1e-3
    between n = 1000 and n = 10000, and the gap at n = 10^5 is below 1e-4."""

    def gap(n: int) -> float:
        times = np.arange(1, n + 1) / n
        return asian.geometric_price(option_type, S, K, R, SIGMA, Q, times=times) - (
            kemna_vorst_continuous(option_type, 1.0)
        )

    assert 1000 * gap(1000) == pytest.approx(10_000 * gap(10_000), rel=1e-3)
    assert abs(gap(100_000)) < 1e-4


def test_geometric_put_call_parity() -> None:
    """C - P = e^{-rT}(E[G] - K), with E[G] = exp(mu_G + v_G / 2)."""
    times, T = [0.1, 0.15, 0.6, 0.61, 0.9], 1.5
    c = asian.geometric_price("call", S, K, R, SIGMA, Q, times=times, T=T)
    p = asian.geometric_price("put", S, K, R, SIGMA, Q, times=times, T=T)
    mu, v = asian.geometric_log_moments(S, R, SIGMA, Q, times)
    assert c - p == pytest.approx(math.exp(-R * T) * (math.exp(mu + v / 2) - K), abs=1e-12 * S)


def test_zero_volatility_is_deterministic_average() -> None:
    """sigma = 0: G = S exp((r - q) * mean(t)) exactly."""
    times = [0.25, 0.5, 1.0]
    G = S * math.exp((R - Q) * np.mean(times))
    for ot, w in (("call", 1.0), ("put", -1.0)):
        for strike in (90.0, 110.0):
            expected = math.exp(-R * 1.0) * max(w * (G - strike), 0.0)
            got = asian.geometric_price(ot, S, strike, R, 0.0, Q, times=times)  # type: ignore[arg-type]
            assert got == pytest.approx(expected, rel=1e-14, abs=1e-14)


# ------------------------------------------------------- geometric MC validates the engine

DATE_SETS: dict[str, tuple[ArrayLike, float | None]] = {
    "monthly": (MONTHLY, None),
    "uneven dates, late payment": ([0.1, 0.15, 0.6, 0.61, 0.9], 1.5),
    "two dates": ([0.5, 1.0], None),
}


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("dates", DATE_SETS)
def test_geometric_mc_within_three_se_of_closed_form(option_type: OptionType, dates: str) -> None:
    """The MC monitoring dates are the formula's dates. 400k paths, seed 7. Measured z:
    monthly -0.88/-0.89, uneven -2.52/-2.74, two dates -1.15/-1.33 (call/put). The uneven
    case was re-run with 8 more seeds x 2M paths: pooled z = -0.10 / -0.12, so no bias."""
    times, T = DATE_SETS[dates]
    cf = asian.geometric_price(option_type, S, K, R, SIGMA, Q, times=times, T=T)
    res = mca.price(
        option_type,
        S,
        K,
        R,
        SIGMA,
        Q,
        times=times,
        T=T,
        average="geometric",
        n_paths=400_000,
        seed=7,
    )
    assert res.within(cf, 3.0), (res, cf)


def test_geometric_mc_confidence_interval_coverage() -> None:
    """400 seeds x 2000 paths, monthly dates: about 380 of 400 CIs should contain the
    closed form; we accept 366..394 (about +-3 binomial SD)."""
    cf = asian.geometric_price("call", S, K, R, SIGMA, Q, times=MONTHLY)
    hits = sum(
        mca.price(
            "call", S, K, R, SIGMA, Q, times=MONTHLY, average="geometric", n_paths=2000, seed=s
        ).contains(cf)
        for s in range(400)
    )
    assert 366 <= hits <= 394, hits


# ------------------------------------------------------------------- arithmetic Asian


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_arithmetic_control_variate_agrees_with_plain_mc(option_type: OptionType) -> None:
    """There is no closed form, so check that two independent estimators agree: plain MC
    (seed 3) and geometric-control MC (seed 4), within 3 combined standard errors."""
    plain = mca.price(
        option_type,
        S,
        K,
        R,
        SIGMA,
        Q,
        times=MONTHLY,
        n_paths=400_000,
        seed=3,
        antithetic=False,
        control_variate=False,
    )
    cv = mca.price(option_type, S, K, R, SIGMA, Q, times=MONTHLY, n_paths=100_000, seed=4)
    assert abs(plain.price - cv.price) <= 3 * math.hypot(plain.std_error, cv.std_error)


def test_arithmetic_respects_am_gm_and_jensen_bounds() -> None:
    """A >= G on every path, so arithmetic call >= geometric call and arithmetic put <=
    geometric put. Convexity, max(A - K, 0) <= mean_i max(S_i - K, 0), gives
    call <= e^{-rT} mean_i e^{r t_i} C_BS(t_i). Each is checked with 3 SE of slack."""
    T = 1.0
    call = mca.price("call", S, K, R, SIGMA, Q, times=MONTHLY, n_paths=100_000, seed=5)
    put = mca.price("put", S, K, R, SIGMA, Q, times=MONTHLY, n_paths=100_000, seed=6)
    assert call.price + 3 * call.std_error >= asian.geometric_price(
        "call", S, K, R, SIGMA, Q, times=MONTHLY
    )
    assert put.price - 3 * put.std_error <= asian.geometric_price(
        "put", S, K, R, SIGMA, Q, times=MONTHLY
    )
    upper = math.exp(-R * T) * np.mean(
        [math.exp(R * t) * bsm.call_price(S, K, float(t), R, SIGMA, Q) for t in MONTHLY]
    )
    assert call.price - 3 * call.std_error <= upper


@pytest.mark.parametrize(("option_type", "measured"), [("call", 599.1), ("put", 673.8)])
def test_geometric_control_variate_reduction(option_type: OptionType, measured: float) -> None:
    """Variance(plain) / Variance(geometric control), 200k paths, seed 3, no antithetics.
    Measured 599x (call) and 674x (put) on 2026-11-11; the test requires >= 200x."""
    kw: dict[str, Any] = {"times": MONTHLY, "n_paths": 200_000, "seed": 3, "antithetic": False}
    plain = mca.price(option_type, S, K, R, SIGMA, Q, control_variate=False, **kw)
    cv = mca.price(option_type, S, K, R, SIGMA, Q, control_variate=True, **kw)
    assert (plain.std_error / cv.std_error) ** 2 >= 200


def test_arithmetic_zero_volatility_is_exact() -> None:
    times = [0.25, 0.5, 1.0]
    A = np.mean([S * math.exp((R - Q) * t) for t in times])
    res = mca.price("call", S, 95.0, R, 0.0, Q, times=times, n_paths=100, seed=0)
    assert res.price == pytest.approx(math.exp(-R) * (A - 95.0), rel=1e-13)
    assert res.std_error <= 1e-12 * res.price


# ------------------------------------------------------------------------ validation


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"average": "geometric", "control_variate": True}, "meaningless for a geometric"),
        ({"average": "harmonic"}, "average must be"),
        ({"T": 0.5}, "payment date T must be >="),
        ({"times": [0.5, 0.2]}, "strictly increasing"),
        ({"n_paths": 7}, "must be even"),
    ],
)
def test_invalid_mc_inputs_raise(kwargs: dict[str, Any], message: str) -> None:
    args: dict[str, Any] = {"times": MONTHLY, "n_paths": 100, "seed": 0} | kwargs
    with pytest.raises(ValueError, match=message):
        mca.price("call", S, K, R, SIGMA, Q, **args)


def test_invalid_closed_form_inputs_raise() -> None:
    with pytest.raises(ValueError, match="payment date T must be >="):
        asian.geometric_price("call", S, K, R, SIGMA, Q, times=[0.5, 1.0], T=0.9)
    with pytest.raises(ValueError, match="volatility sigma must be >= 0"):
        asian.geometric_price("call", S, K, R, -0.1, Q, times=[1.0])
