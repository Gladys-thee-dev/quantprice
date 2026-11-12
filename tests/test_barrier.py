"""Phase 3c: barrier options. Closed forms (continuous monitoring), Monte Carlo (discrete
monitoring), in-out parity and the Broadie-Glasserman-Kou continuity correction."""

import itertools
import math
from typing import Any, cast

import pytest

from quantprice import barrier, bsm
from quantprice.barrier import BarrierType
from quantprice.bsm import OptionType
from quantprice.mc import barrier as mcb


def bt(kind: str, leg: str) -> BarrierType:
    """Barrier type from ('down' | 'up', 'in' | 'out')."""
    return cast(BarrierType, f"{kind}-and-{leg}")


# ------------------------------------------------------------ published reference values

# Haug (2007), section 4.17.1, standard barrier option table: S = 100, T = 0.5, r = 8%,
# cost of carry b = r - q = 4% (so q = 4%), rebate 3, sigma = 25%. Out options pay the
# rebate at the hit; in options pay it at expiry. These are 4 d.p. values. All 36
# matched on first run, and they are independently cross-checked below by in-out parity
# and Monte Carlo, so a transcription error would be caught.
HAUG = [
    ("call", "down-and-out", 95.0, {90.0: 9.0246, 100.0: 6.7924, 110.0: 4.8759}),
    ("call", "down-and-out", 100.0, {90.0: 3.0000, 100.0: 3.0000, 110.0: 3.0000}),
    ("call", "up-and-out", 105.0, {90.0: 2.6789, 100.0: 2.3580, 110.0: 2.3453}),
    ("call", "down-and-in", 95.0, {90.0: 7.7627, 100.0: 4.0109, 110.0: 2.0576}),
    ("call", "down-and-in", 100.0, {90.0: 13.8333, 100.0: 7.8494, 110.0: 3.9795}),
    ("call", "up-and-in", 105.0, {90.0: 14.1112, 100.0: 8.4482, 110.0: 4.5910}),
    ("put", "down-and-in", 95.0, {90.0: 2.9586, 100.0: 6.5677, 110.0: 11.9752}),
    ("put", "down-and-in", 100.0, {90.0: 2.2845, 100.0: 5.9085, 110.0: 11.6465}),
    ("put", "up-and-in", 105.0, {90.0: 1.4653, 100.0: 3.3721, 110.0: 7.0846}),
    ("put", "down-and-out", 95.0, {90.0: 2.2798, 100.0: 2.2947, 110.0: 2.6252}),
    ("put", "down-and-out", 100.0, {90.0: 3.0000, 100.0: 3.0000, 110.0: 3.0000}),
    ("put", "up-and-out", 105.0, {90.0: 3.7760, 100.0: 5.4932, 110.0: 7.5187}),
]


@pytest.mark.parametrize(("option_type", "barrier_type", "H", "values"), HAUG)
def test_haug_reference_table(
    option_type: OptionType, barrier_type: BarrierType, H: float, values: dict[float, float]
) -> None:
    for K, expected in values.items():
        got = barrier.price(
            option_type, barrier_type, 100.0, K, H, 0.5, 0.08, 0.25, 0.04, rebate=3.0
        )
        assert got == pytest.approx(expected, abs=5e-5), (K, got, expected)


# ------------------------------------------------------------------- in-out parity

PARITY_GRID = list(
    itertools.product(
        ["call", "put"],
        [80.0, 95.0, 100.0, 105.0, 120.0],  # K
        [90.0, 110.0],  # H (below / above S = 100)
        [0.01, 0.1, 0.3, 0.8],  # sigma (0.01 stresses the large-mu powers)
        [(-0.02, 0.0), (0.08, 0.04)],  # (r, q)
    )
)


@pytest.mark.parametrize(("option_type", "K", "H", "sigma", "rq"), PARITY_GRID)
def test_in_plus_out_equals_vanilla(
    option_type: OptionType, K: float, H: float, sigma: float, rq: tuple[float, float]
) -> None:
    """Knock-in + knock-out = vanilla (zero rebate). The two sides use different A-F
    combinations, so this cross-checks the case table. Tolerance 1e-12 * S (measured
    worst 1.4e-14 on a similar grid)."""
    r, q = rq
    kind = "down" if H < 100.0 else "up"
    ki = barrier.price(option_type, bt(kind, "in"), 100.0, K, H, 1.0, r, sigma, q)
    ko = barrier.price(option_type, bt(kind, "out"), 100.0, K, H, 1.0, r, sigma, q)
    assert math.isfinite(ki) and math.isfinite(ko)
    assert ki >= 0 and ko >= 0
    assert ki + ko == pytest.approx(
        bsm.price(option_type, 100.0, K, 1.0, r, sigma, q), abs=1e-12 * 100
    )


@pytest.mark.parametrize("kind", ["down", "up"])
def test_parity_with_rebate_paid_at_expiry(kind: str) -> None:
    """Rebate R at expiry on both legs: exactly one leg pays it, so
    in + out = vanilla + R e^{-rT}."""
    S, K, H, T, r, sigma, q, R = (
        100.0,
        100.0,
        90.0 if kind == "down" else 110.0,
        1.0,
        0.05,
        0.3,
        0.01,
        2.5,
    )
    ki = barrier.price("call", bt(kind, "in"), S, K, H, T, r, sigma, q, rebate=R)
    ko = barrier.price(
        "call", bt(kind, "out"), S, K, H, T, r, sigma, q, rebate=R, rebate_at="expiry"
    )
    expected = bsm.call_price(S, K, T, r, sigma, q) + R * math.exp(-r * T)
    assert ki + ko == pytest.approx(expected, abs=1e-12 * S)


def test_rebate_paid_at_hit_is_worth_more_than_at_expiry_when_rates_positive() -> None:
    hit = barrier.price(
        "put", "up-and-out", 100.0, 100.0, 110.0, 2.0, 0.05, 0.3, rebate=5.0, rebate_at="hit"
    )
    exp = barrier.price(
        "put", "up-and-out", 100.0, 100.0, 110.0, 2.0, 0.05, 0.3, rebate=5.0, rebate_at="expiry"
    )
    assert hit > exp


# ---------------------------------------------------------------------- limiting cases


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_distant_barrier_behaves_like_no_barrier(option_type: OptionType) -> None:
    """A barrier that can't realistically be reached: out -> vanilla, in -> 0."""
    vanilla = bsm.price(option_type, 100.0, 100.0, 1.0, 0.05, 0.2, 0.01)
    for H, kind in ((1e-3, "down"), (1e4, "up")):
        out = barrier.price(option_type, bt(kind, "out"), 100.0, 100.0, H, 1.0, 0.05, 0.2, 0.01)
        inn = barrier.price(option_type, bt(kind, "in"), 100.0, 100.0, H, 1.0, 0.05, 0.2, 0.01)
        assert out == pytest.approx(vanilla, rel=1e-12)
        assert inn == pytest.approx(0.0, abs=1e-12)


def test_breached_at_inception() -> None:
    vanilla = bsm.put_price(100.0, 105.0, 1.0, 0.05, 0.2)
    assert barrier.price("put", "down-and-in", 100.0, 105.0, 100.0, 1.0, 0.05, 0.2) == vanilla
    assert (
        barrier.price("put", "down-and-out", 100.0, 105.0, 101.0, 1.0, 0.05, 0.2, rebate=2.0) == 2.0
    )
    assert barrier.price(
        "put", "down-and-out", 100.0, 105.0, 101.0, 1.0, 0.05, 0.2, rebate=2.0, rebate_at="expiry"
    ) == pytest.approx(2.0 * math.exp(-0.05))


def test_up_and_out_call_struck_above_barrier_is_only_its_rebate() -> None:
    """K >= H: the call can only finish in the money after crossing H, so it's knocked out."""
    assert barrier.price("call", "up-and-out", 100.0, 120.0, 110.0, 1.0, 0.05, 0.3) == 0.0
    with_rebate = barrier.price(
        "call", "up-and-out", 100.0, 120.0, 110.0, 1.0, 0.05, 0.3, rebate=1.0
    )
    hit_value = barrier.price("put", "up-and-out", 100.0, 1e-9, 110.0, 1.0, 0.05, 0.3, rebate=1.0)
    assert with_rebate == pytest.approx(hit_value, rel=1e-9)  # both equal F alone


def test_zero_volatility_follows_deterministic_path() -> None:
    """sigma = 0, r - q = 5%: the path rises from 100 to 100 e^{0.05}, crossing H = 103 at
    t* = ln(1.03) / 0.05."""
    S, K, H, T, r, q = 100.0, 95.0, 103.0, 1.0, 0.05, 0.0
    t_star = math.log(H / S) / (r - q)
    out = barrier.price("call", "up-and-out", S, K, H, T, r, 0.0, q, rebate=1.0)
    assert out == pytest.approx(math.exp(-r * t_star))
    inn = barrier.price("call", "up-and-in", S, K, H, T, r, 0.0, q)
    assert inn == pytest.approx(bsm.call_price(S, K, T, r, 0.0, q))


def test_at_expiry_not_breached() -> None:
    assert barrier.price("call", "down-and-out", 100.0, 90.0, 80.0, 0.0, 0.05, 0.2) == 10.0
    assert (
        barrier.price("call", "down-and-in", 100.0, 90.0, 80.0, 0.0, 0.05, 0.2, rebate=1.5) == 1.5
    )


# --------------------------------------------------------- Monte Carlo, discrete monitoring

S0, T0, R0, Q0, SIG0 = 100.0, 0.5, 0.08, 0.04, 0.25
MC_CASES: dict[str, tuple[Any, ...]] = {
    # name: (option, barrier type, K, H, rebate, rebate_at)
    "DO call": ("call", "down-and-out", 100.0, 95.0, 0.0, None),
    "UO call": ("call", "up-and-out", 100.0, 120.0, 0.0, None),
    "DI put": ("put", "down-and-in", 100.0, 90.0, 0.0, None),
    "UO put, rebate at hit": ("put", "up-and-out", 110.0, 105.0, 3.0, "hit"),
    "DO call, rebate at expiry": ("call", "down-and-out", 100.0, 95.0, 3.0, "expiry"),
    "UI call, rebate": ("call", "up-and-in", 90.0, 105.0, 3.0, None),
}


def run_mc(case: str, m: int, n_paths: int = 400_000, seed: int = 1) -> tuple[Any, float, float]:
    ot, bt, K, H, R, ra = MC_CASES[case]
    res = mcb.price(
        ot,
        bt,
        S0,
        K,
        H,
        T0,
        R0,
        SIG0,
        Q0,
        n_monitoring=m,
        n_paths=n_paths,
        seed=seed,
        rebate=R,
        rebate_at=ra,
    )
    cont = barrier.price(ot, bt, S0, K, H, T0, R0, SIG0, Q0, rebate=R, rebate_at=ra)
    bgk = barrier.discrete_price(
        ot, bt, S0, K, H, T0, R0, SIG0, Q0, n_monitoring=m, rebate=R, rebate_at=ra
    )
    return res, cont, bgk


@pytest.mark.parametrize("case", MC_CASES)
def test_daily_monitoring_matches_bgk_corrected_formula(case: str) -> None:
    """m = 252: MC on the discrete contract vs the BGK-shifted continuous formula.
    Measured z (MC - BGK) / SE on 2026-11-12: +1.11, -1.28, +1.71, +1.21, +1.07, +1.37."""
    res, _, bgk = run_mc(case, 252)
    assert res.within(bgk, 3.0), (res, bgk)


@pytest.mark.parametrize("case", [c for c in MC_CASES if c != "UO call"])
def test_weekly_monitoring_matches_bgk_corrected_formula(case: str) -> None:
    """m = 52. Measured z: -0.13, -0.16, -0.14, -0.06, -0.04. The up-and-out call is
    excluded: there BGK misses by 0.033 (z = -6.1); see the next test."""
    res, _, bgk = run_mc(case, 52)
    assert res.within(bgk, 3.0), (res, bgk)


def test_bgk_error_shrinks_with_monitoring_frequency_where_it_is_worst() -> None:
    """Up-and-out call, K = 100, H = 120: the payoff drops from 20 to 0 at the barrier, the
    hardest case for an asymptotic correction. Measured |MC - BGK| = 0.103 (m = 12),
    0.033 (m = 52), 0.0067 (m = 252, within MC noise). Test: strictly decreasing, and
    m = 252 within 3 SE."""
    errs = []
    for m in (12, 52, 252):
        res, _, bgk = run_mc("UO call", m)
        errs.append(abs(res.price - bgk))
    assert errs[0] > errs[1] > errs[2]
    assert res.within(bgk, 3.0)  # m = 252


@pytest.mark.parametrize("case", ["DO call", "UO call", "UO put, rebate at hit"])
def test_discrete_knock_out_is_worth_more_than_continuous(case: str) -> None:
    """Discrete monitoring misses crossings between dates, so knock-outs are worth more
    than the continuous price. Measured with m = 52: z = +51.6, +56.5, +73.1."""
    res, cont, _ = run_mc(case, 52)
    assert res.price - cont > 10 * res.std_error


@pytest.mark.parametrize(
    ("option_type", "kind", "H"),
    [("call", "down", 95.0), ("put", "up", 105.0), ("call", "up", 130.0)],
)
def test_mc_in_plus_out_equals_unbarriered_on_same_paths(
    option_type: OptionType, kind: str, H: float
) -> None:
    """With zero rebate, 1{hit} + 1{no hit} = 1 on every path. The same seed gives the same
    paths, and an out option with an unreachable barrier is the vanilla on those paths, so
    the equality must hold to rounding."""
    kw: dict[str, Any] = {"n_monitoring": 50, "n_paths": 20_000, "seed": 5}
    args = (100.0, 100.0)
    ki = mcb.price(option_type, bt(kind, "in"), *args, H, 0.5, 0.05, 0.25, 0.01, **kw)
    ko = mcb.price(option_type, bt(kind, "out"), *args, H, 0.5, 0.05, 0.25, 0.01, **kw)
    far = 1e-12 if kind == "down" else 1e12
    van = mcb.price(option_type, bt(kind, "out"), *args, far, 0.5, 0.05, 0.25, 0.01, **kw)
    assert ki.price + ko.price == pytest.approx(van.price, rel=1e-12)


def test_mc_result_does_not_depend_on_batch_size_parity_or_layout() -> None:
    """Different batch sizes split the random stream differently, so prices differ, but both
    must agree within their combined standard errors."""
    a = mcb.price(
        "call",
        "down-and-out",
        100.0,
        100.0,
        95.0,
        0.5,
        0.05,
        0.25,
        n_monitoring=50,
        n_paths=40_000,
        seed=3,
        batch_size=40_000,
    )
    b = mcb.price(
        "call",
        "down-and-out",
        100.0,
        100.0,
        95.0,
        0.5,
        0.05,
        0.25,
        n_monitoring=50,
        n_paths=40_000,
        seed=3,
        batch_size=10_000,
    )
    assert a.n_paths == b.n_paths == 40_000
    assert abs(a.price - b.price) <= 3 * math.hypot(a.std_error, b.std_error)


# ------------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"barrier_type": "sideways-and-out"}, "barrier_type must be one of"),
        ({"H": -1.0}, "barrier H must be finite and > 0"),
        ({"rebate": -1.0}, "rebate must be finite and >= 0"),
        ({"barrier_type": "down-and-in", "rebate_at": "hit"}, "knock-in rebate is paid at expiry"),
        ({"rebate_at": "sometime"}, "rebate_at must be"),
        ({"sigma": -0.1}, "volatility sigma must be >= 0"),
    ],
)
def test_invalid_inputs_raise(kwargs: dict[str, Any], message: str) -> None:
    args: dict[str, Any] = {
        "barrier_type": "down-and-out",
        "S": 100.0,
        "K": 100.0,
        "H": 90.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.2,
        "q": 0.0,
    } | kwargs
    bt = args.pop("barrier_type")
    with pytest.raises(ValueError, match=message):
        barrier.price("call", bt, **args)


def test_rebate_at_hit_with_complex_lambda_raises() -> None:
    """mu^2 + 2r/sigma^2 < 0 needs r < 0 and small mu: r = -0.05, q = -0.05, sigma = 0.4
    gives mu = -0.5, so the discriminant is 0.25 - 0.625 < 0."""
    with pytest.raises(ValueError, match="not supported; use rebate_at='expiry'"):
        barrier.price(
            "call", "down-and-out", 100.0, 100.0, 90.0, 1.0, -0.05, 0.4, -0.05, rebate=1.0
        )


def test_mc_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="n_monitoring must be an integer >= 1"):
        mcb.price(
            "call", "down-and-out", 100.0, 100.0, 90.0, 1.0, 0.05, 0.2, n_monitoring=0, n_paths=10
        )
    with pytest.raises(ValueError, match="T must be > 0"):
        mcb.price(
            "call", "down-and-out", 100.0, 100.0, 90.0, 0.0, 0.05, 0.2, n_monitoring=5, n_paths=10
        )
