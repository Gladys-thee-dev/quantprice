"""Phase 2: Cox-Ross-Rubinstein binomial tree (European and American exercise)."""

import itertools
import math

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from quantprice import binomial, bsm
from quantprice.bsm import OptionType


def scale(S: float, K: float, T: float, r: float, q: float) -> float:
    return max(S * math.exp(-q * T), K * math.exp(-r * T))


# ----------------------------------------------------------- convergence to Black-Scholes

CONVERGENCE_GRID = list(
    itertools.product(
        [50.0, 100.0, 150.0],  # S (K = 100)
        [0.1, 1.0, 5.0],  # T
        [-0.02, 0.10],  # r
        [0.05, 0.2, 1.0],  # sigma
        [0.0, 0.05],  # q
    )
)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_european_converges_to_black_scholes(option_type: OptionType) -> None:
    """At n = 5000 and n = 5001 steps, and for their average, the CRR price is within
    1e-4 * scale of Black-Scholes. Where the price exceeds 1% of scale, it is also within
    5e-3 relative.

    Measured on 2026-11-11 over a superset of this grid: max |error| / scale = 4.1e-5,
    and max relative error = 1.9e-3 where the price exceeds 1% of scale. CRR error is
    O(1/n) with an oscillating constant (error * n ranged from -2.0 to +1.5 in the
    cases inspected), so these bounds are specific to n ~ 5000.
    """
    K = 100.0
    for S, T, r, sigma, q in CONVERGENCE_GRID:
        bs = bsm.price(option_type, S, K, T, r, sigma, q)
        sc = scale(S, K, T, r, q)
        p_even = binomial.price(option_type, S, K, T, r, sigma, q, steps=5000)
        p_odd = binomial.price(option_type, S, K, T, r, sigma, q, steps=5001)
        for v in (p_even, p_odd, 0.5 * (p_even + p_odd)):
            assert abs(v - bs) <= 1e-4 * sc, (S, T, r, sigma, q, v, bs)
            if bs > 0.01 * sc:
                assert abs(v - bs) <= 5e-3 * bs, (S, T, r, sigma, q, v, bs)


def test_odd_even_oscillation_and_averaging() -> None:
    """With S = K the strike is a terminal node only for even n, so the error flips sign
    between even and odd n. Averaging n and n + 1 cancels most of it.

    Measured: errors for n = 5000..5003 alternate (-4.0e-4, +3.5e-4, ...); the average
    of n = 5000 and 5001 has error -2.5e-5 (16x smaller). Test: alternating signs and at
    least a 5x improvement.
    """
    args = (100.0, 100.0, 1.0, 0.05, 0.2)
    bs = bsm.call_price(*args)
    errs = [binomial.price("call", *args, steps=n) - bs for n in range(5000, 5004)]
    assert all(e1 * e2 < 0 for e1, e2 in itertools.pairwise(errs))
    assert abs(0.5 * (errs[0] + errs[1])) * 5 <= min(abs(errs[0]), abs(errs[1]))


def test_error_envelope_shrinks_like_one_over_n() -> None:
    """The max |error| over n = 5000..5009 is about 1/10 of that over n = 500..509 (the
    measured ratio is 0.0995), which is the O(1/n) rate. Test: ratio <= 0.15."""
    args = (100.0, 110.0, 1.0, 0.05, 0.2)  # off-ATM strike: oscillating error
    bs = bsm.call_price(*args)

    def envelope(lo: int) -> float:
        return max(abs(binomial.price("call", *args, steps=n) - bs) for n in range(lo, lo + 10))

    assert envelope(5000) / envelope(500) <= 0.15


# ----------------------------------------------------------------- published example


def test_hull_five_step_american_put() -> None:
    """Hull, *Options, Futures, and Other Derivatives*, chapter "Binomial Trees" /
    "Numerical Procedures": American put S=50, K=50, r=10%, sigma=40%, T=5 months,
    5-step tree gives 4.49 (2 d.p.)."""
    v = binomial.price("put", 50.0, 50.0, 5 / 12, 0.10, 0.40, steps=5, exercise="american")
    assert v == pytest.approx(4.49, abs=0.005)


# --------------------------------------------------------- properties at any step count

steps_st = st.integers(50, 200)
params = (
    st.floats(1.0, 1000.0),
    st.floats(1.0, 1000.0),
    st.floats(0.01, 5.0),
    st.floats(-0.02, 0.10),
    st.floats(0.01, 2.0),
    st.floats(0.0, 0.08),
)


def valid_tree(T: float, r: float, sigma: float, q: float, steps: int) -> bool:
    """p in [0, 1] (with a 1% margin): |r - q| sqrt(T / steps) < sigma."""
    return abs(r - q) * math.sqrt(T / steps) < 0.99 * sigma


@given(*params, steps_st)
def test_tree_put_call_parity_holds_exactly(
    S: float, K: float, T: float, r: float, sigma: float, q: float, steps: int
) -> None:
    """p matches the forward exactly, so European C - P = S e^{-qT} - K e^{-rT} at every
    n, not only as n -> infinity."""
    assume(valid_tree(T, r, sigma, q, steps))
    c = binomial.price("call", S, K, T, r, sigma, q, steps=steps)
    p = binomial.price("put", S, K, T, r, sigma, q, steps=steps)
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert abs((c - p) - rhs) <= 1e-10 * scale(S, K, T, r, q)


@given(*params, steps_st)
def test_american_put_at_least_european_and_intrinsic(
    S: float, K: float, T: float, r: float, sigma: float, q: float, steps: int
) -> None:
    assume(valid_tree(T, r, sigma, q, steps))
    eu = binomial.price("put", S, K, T, r, sigma, q, steps=steps)
    am = binomial.price("put", S, K, T, r, sigma, q, steps=steps, exercise="american")
    assert am >= eu
    assert am >= K - S


@given(
    st.floats(1.0, 1000.0),
    st.floats(1.0, 1000.0),
    st.floats(0.01, 5.0),
    st.floats(0.0, 0.10),  # r >= 0 is required: see the negative-rate test below
    st.floats(0.01, 2.0),
    steps_st,
)
def test_american_call_without_dividends_equals_european(
    S: float, K: float, T: float, r: float, sigma: float, steps: int
) -> None:
    """With q = 0 and r >= 0, every tree node satisfies continuation value >= S - K e^{-r tau}
    >= S - K, so early exercise never happens (Merton 1973). Only floating-point ties can
    differ, hence 1e-12 relative."""
    assume(valid_tree(T, r, sigma, 0.0, steps))
    eu = binomial.price("call", S, K, T, r, sigma, 0.0, steps=steps)
    am = binomial.price("call", S, K, T, r, sigma, 0.0, steps=steps, exercise="american")
    assert am == pytest.approx(eu, rel=1e-12, abs=1e-12 * scale(S, K, T, r, 0.0))


def test_american_call_can_exceed_european_with_negative_rates() -> None:
    """The q = 0 equivalence needs r >= 0. With r < 0, paying K later costs K e^{-rT} > K
    in today's money. A deep ITM call is then worth more exercised now (S - K = 50)
    than held: the European value is about 46.15 here, its lower bound being
    S - K e^{-rT} ~ 44.74."""
    S, K, T, r, sigma = 100.0, 50.0, 5.0, -0.02, 0.2
    eu = binomial.price("call", S, K, T, r, sigma, steps=500)
    am = binomial.price("call", S, K, T, r, sigma, steps=500, exercise="american")
    assert S - K * math.exp(-r * T) <= eu < S - K
    assert am == pytest.approx(S - K)  # exercise immediately


# ---------------------------------------------------------------------- limiting cases


@pytest.mark.parametrize("exercise", ["european", "american"])
def test_at_expiry_returns_payoff(exercise: binomial.ExerciseStyle) -> None:
    assert binomial.price("call", 110.0, 100.0, 0.0, 0.05, 0.2, steps=10, exercise=exercise) == 10.0
    assert binomial.price("put", 110.0, 100.0, 0.0, 0.05, 0.2, steps=10, exercise=exercise) == 0.0


def test_zero_volatility_european_matches_bsm_limit() -> None:
    for ot in ("call", "put"):
        tree = binomial.price(ot, 100.0, 95.0, 1.0, 0.03, 0.0, 0.01, steps=100)
        assert tree == pytest.approx(bsm.price(ot, 100.0, 95.0, 1.0, 0.03, 0.0, 0.01), rel=1e-14)


def test_zero_volatility_american_exercises_at_best_tree_date() -> None:
    # Put, stock drifting up at r - q = 5%: the best date is now, value K - S = 10.
    assert binomial.price(
        "put", 100.0, 110.0, 1.0, 0.05, 0.0, steps=100, exercise="american"
    ) == pytest.approx(10.0)
    # Call, stock decaying at r - q = -5% with r = 0: exercise now (10) beats the
    # European value 100 e^{-0.05} - 90 ~ 5.12.
    am = binomial.price("call", 100.0, 90.0, 1.0, 0.0, 0.0, 0.05, steps=100, exercise="american")
    eu = binomial.price("call", 100.0, 90.0, 1.0, 0.0, 0.0, 0.05, steps=100)
    assert am == pytest.approx(10.0)
    assert eu == pytest.approx(100.0 * math.exp(-0.05) - 90.0)


# ------------------------------------------------------------------------- validation


def test_probability_outside_unit_interval_raises_with_step_advice() -> None:
    """|r - q| sqrt(T/n) > sigma: 0.10 * sqrt(5/1) = 0.22 > 0.01."""
    with pytest.raises(
        ValueError, match=r"probability p = .* outside \[0, 1\].*at least 50[01] steps"
    ):
        binomial.price("call", 100.0, 100.0, 5.0, 0.10, 0.01, steps=1)
    # The advised step count (n >= (r - q)^2 T / sigma^2 = 500) is enough.
    binomial.price("call", 100.0, 100.0, 5.0, 0.10, 0.01, steps=501)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"steps": 0}, "steps must be an integer >= 1"),
        ({"steps": 2.5}, "steps must be an integer >= 1"),
        ({"steps": True}, "steps must be an integer >= 1"),
        ({"exercise": "bermudan"}, "exercise must be 'european' or 'american'"),
        ({"sigma": -0.1}, "volatility sigma must be >= 0"),
        ({"S": np.array([100.0, 110.0])}, "takes scalar inputs"),
        ({"steps": 1_000_000, "sigma": 2.0, "T": 5.0}, "overflow"),
    ],
)
def test_invalid_inputs_raise(kwargs: dict[str, object], message: str) -> None:
    args: dict[str, object] = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2}
    args |= kwargs
    steps = args.pop("steps", 10)
    exercise = args.pop("exercise", "european")
    with pytest.raises(ValueError, match=message):
        binomial.price("call", steps=steps, exercise=exercise, **args)  # type: ignore[arg-type]
