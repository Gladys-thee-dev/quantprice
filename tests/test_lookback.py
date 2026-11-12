"""Phase 3d: lookback options. Closed forms (continuous monitoring) and Monte Carlo
(discrete monitoring). Fixed seeds throughout."""

import itertools
import math
from typing import Any

import pytest
from scipy.special import ndtr

from quantprice import bsm, lookback
from quantprice.bsm import OptionType
from quantprice.mc import lookback as mcl

S, T, R, Q, SIG = 100.0, 1.0, 0.05, 0.02, 0.3
BAD_STRIKE_TYPE: Any = "hybrid"  # deliberately invalid, for the validation test


def textbook_h(sign: float, S_: float, Y: float, T_: float, b: float, sigma: float) -> float:
    """The h-terms exactly as written in Haug (2007), sections 4.15.1-4.15.2, valid for b != 0.
    Written independently of the stable rewrite in quantprice.lookback."""
    v = sigma * math.sqrt(T_)
    y = (math.log(S_ / Y) + (b + sigma * sigma / 2) * T_) / v
    power = (S_ / Y) ** (-2 * b / sigma**2)
    shift = 2 * b * math.sqrt(T_) / sigma
    if sign < 0:
        bracket = power * ndtr(-y + shift) - math.exp(b * T_) * ndtr(-y)
    else:
        bracket = -power * ndtr(y - shift) + math.exp(b * T_) * ndtr(y)
    return float(sigma**2 / (2 * b) * bracket)


# ----------------------------------------------------- stable rewrite of the b-singularity


@pytest.mark.parametrize(
    ("sign", "S_", "Y", "T_", "b", "sigma"),
    [
        (sign, S_, Y, T_, b, sigma)
        for sign, (S_, Y), T_, b, sigma in itertools.product(
            [-1.0, 1.0],
            [(100.0, 100.0), (100.0, 80.0), (100.0, 120.0), (120.0, 100.0), (80.0, 100.0)],
            [0.1, 1.0, 5.0],
            [-0.08, -0.01, 0.03, 0.1],
            [0.1, 0.3, 1.0],
        )
        if not (sign < 0 and Y > S_)
    ],
)
def test_stable_h_matches_textbook_form_away_from_b_zero(
    sign: float, S_: float, Y: float, T_: float, b: float, sigma: float
) -> None:
    """Measured worst relative difference 2.7e-12 on this grid; tolerance 1e-10."""
    assert lookback._h(sign, S_, Y, T_, b, sigma) == pytest.approx(
        textbook_h(sign, S_, Y, T_, b, sigma), rel=1e-10
    )


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("kind", ["floating", "fixed"])
def test_price_is_smooth_through_r_equals_q(kind: str, option_type: OptionType) -> None:
    """At r = q the textbook formula is 0/0. The stable form must be continuous there:
    V(b = 0) equals the average of V(b = +-1e-6) to second order (|V''| * 1e-12)."""

    def v(q: float) -> float:
        if kind == "floating":
            return lookback.floating_price(option_type, S, T, R, SIG, q)
        return lookback.fixed_price(option_type, S, 105.0, T, R, SIG, q)

    mid, up, dn = v(R), v(R - 1e-6), v(R + 1e-6)
    assert math.isfinite(mid)
    assert mid == pytest.approx(0.5 * (up + dn), rel=1e-10)
    assert min(up, dn) <= mid <= max(up, dn)


# ------------------------------------------------------------ identities and bounds


@pytest.mark.parametrize(("K", "M"), [(80.0, None), (100.0, None), (100.0, 115.0), (110.0, 115.0)])
def test_fixed_call_in_the_money_equals_floating_put_identity(K: float, M: float | None) -> None:
    """For K <= running max, max(M_T - K, 0) = M_T - K, so
    fixed call = floating put + S e^{-qT} - K e^{-rT}. This links the Conze-Viswanathan
    and Goldman-Sosin-Gatto formulas, which are coded separately."""
    fixed = lookback.fixed_price("call", S, K, T, R, SIG, Q, running_extreme=M)
    floating = lookback.floating_price("put", S, T, R, SIG, Q, running_extreme=M)
    assert fixed == pytest.approx(
        floating + S * math.exp(-Q * T) - K * math.exp(-R * T), abs=1e-12 * S
    )


@pytest.mark.parametrize(("K", "m"), [(120.0, None), (100.0, None), (100.0, 85.0), (90.0, 85.0)])
def test_fixed_put_in_the_money_equals_floating_call_identity(K: float, m: float | None) -> None:
    """For K >= running min: fixed put = floating call + K e^{-rT} - S e^{-qT}."""
    fixed = lookback.fixed_price("put", S, K, T, R, SIG, Q, running_extreme=m)
    floating = lookback.floating_price("call", S, T, R, SIG, Q, running_extreme=m)
    assert fixed == pytest.approx(
        floating + K * math.exp(-R * T) - S * math.exp(-Q * T), abs=1e-12 * S
    )


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_fixed_strike_continuous_across_branch_boundary(option_type: OptionType) -> None:
    """The formula switches branch at K = running extreme. The price is Lipschitz in K with
    constant e^{-rT}, so |V(K + d) - V(K - d)| <= 2 d e^{-rT}."""
    d = 1e-7 * S
    gap = abs(
        lookback.fixed_price(option_type, S, S + d, T, R, SIG, Q)
        - lookback.fixed_price(option_type, S, S - d, T, R, SIG, Q)
    )
    assert gap <= 2 * d * math.exp(-R * T) * (1 + 1e-6)


@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_lookbacks_dominate_european_options(K: float) -> None:
    """Pathwise M_T >= S_T >= m_T, so: fixed call >= European call, fixed put >= European
    put. And S_T - m_T >= (S_T - S)^+, so floating call >= the ATM European call."""
    assert lookback.fixed_price("call", S, K, T, R, SIG, Q) >= bsm.call_price(S, K, T, R, SIG, Q)
    assert lookback.fixed_price("put", S, K, T, R, SIG, Q) >= bsm.put_price(S, K, T, R, SIG, Q)
    assert lookback.floating_price("call", S, T, R, SIG, Q) >= bsm.call_price(S, S, T, R, SIG, Q)
    assert lookback.floating_price("put", S, T, R, SIG, Q) >= bsm.put_price(S, S, T, R, SIG, Q)


def test_fixed_call_decreases_in_strike() -> None:
    prices = [lookback.fixed_price("call", S, K, T, R, SIG, Q) for K in (80, 95, 100, 105, 130)]
    assert all(a > b for a, b in itertools.pairwise(prices))


def test_zero_volatility_and_expiry_are_deterministic() -> None:
    """sigma = 0, r - q = 3%: the path rises monotonically from 100 to 100 e^{0.03}."""
    end = S * math.exp(0.03)
    D = math.exp(-0.05)
    assert lookback.floating_price("call", S, 1.0, 0.05, 0.0, 0.02) == pytest.approx(D * (end - S))
    assert lookback.floating_price("put", S, 1.0, 0.05, 0.0, 0.02) == pytest.approx(0.0, abs=1e-12)
    assert lookback.fixed_price("call", S, 101.0, 1.0, 0.05, 0.0, 0.02) == pytest.approx(
        D * (end - 101.0)
    )
    assert lookback.floating_price("put", S, 0.0, 0.05, 0.3, running_extreme=120.0) == 20.0
    assert lookback.fixed_price("put", S, 95.0, 0.0, 0.05, 0.3, running_extreme=90.0) == 5.0


# ------------------------------------------------------- Monte Carlo, discrete monitoring

MC_CASES: dict[str, dict[str, Any]] = {
    "floating call": {"strike_type": "floating", "option_type": "call"},
    "floating put": {"strike_type": "floating", "option_type": "put"},
    "fixed call": {"strike_type": "fixed", "option_type": "call", "K": 110.0},
    "fixed put": {"strike_type": "fixed", "option_type": "put", "K": 90.0},
    "seasoned floating call": {
        "strike_type": "floating",
        "option_type": "call",
        "running_extreme": 90.0,
    },
    "seasoned fixed call": {
        "strike_type": "fixed",
        "option_type": "call",
        "K": 100.0,
        "running_extreme": 115.0,
    },
}


def closed_form(case: dict[str, Any]) -> float:
    if case["strike_type"] == "floating":
        return lookback.floating_price(
            case["option_type"], S, T, R, SIG, Q, running_extreme=case.get("running_extreme")
        )
    return lookback.fixed_price(
        case["option_type"], S, case["K"], T, R, SIG, Q, running_extreme=case.get("running_extreme")
    )


def mc(case: dict[str, Any], m: int, n_paths: int, seed: int) -> Any:
    kw = {k: v for k, v in case.items() if k not in ("strike_type", "option_type")}
    return mcl.price(
        case["strike_type"],
        case["option_type"],
        S,
        T,
        R,
        SIG,
        Q,
        n_monitoring=m,
        n_paths=n_paths,
        seed=seed,
        **kw,
    )


@pytest.mark.parametrize("name", MC_CASES)
def test_richardson_extrapolated_mc_matches_continuous_formula(name: str) -> None:
    """The discrete-monitoring gap behaves like c / sqrt(m) (next test), so
    V_cont ~ 2 V_{4m} - V_m. With m = 100 / 400 and 100k paths (seeds 31 / 32), measured
    z = -1.21, -2.05, -1.65, -1.57, -1.20, -1.56 on 2026-11-12. A 4-seed x 200k check on
    the floating put found no residual bias at +-0.02 precision (z = -1.70 at m = 100/400,
    +0.96 at m = 200/800, opposite signs). Test: within 3 combined SE."""
    case = MC_CASES[name]
    a, b = mc(case, 100, 100_000, 31), mc(case, 400, 100_000, 32)
    extrapolated = 2 * b.price - a.price
    se = math.hypot(2 * b.std_error, a.std_error)
    assert abs(extrapolated - closed_form(case)) <= 3 * se


def test_discrete_monitoring_gap_scales_like_inverse_sqrt_m() -> None:
    """Fresh floating put, 200k paths, seed 2. Measured gap * sqrt(m) = 20.05, 20.60, 20.66
    for m = 50, 200, 800. Test: the gap is positive (discrete < continuous) by > 10 SE at
    m = 50, and gap * sqrt(m) agrees within 15% between m = 50 and m = 200."""
    case = MC_CASES["floating put"]
    cf = closed_form(case)
    g50 = mc(case, 50, 200_000, 2)
    g200 = mc(case, 200, 200_000, 2)
    gap50, gap200 = cf - g50.price, cf - g200.price
    assert gap50 > 10 * g50.std_error
    assert gap200 * math.sqrt(200) == pytest.approx(gap50 * math.sqrt(50), rel=0.15)


# ------------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: lookback.floating_price("call", S, T, R, SIG, running_extreme=110.0),
            "running minimum must be <= S",
        ),
        (
            lambda: lookback.floating_price("put", S, T, R, SIG, running_extreme=90.0),
            "running maximum must be >= S",
        ),
        (
            lambda: lookback.fixed_price("call", S, 100.0, T, R, SIG, running_extreme=-1.0),
            "finite and > 0",
        ),
        (
            lambda: lookback.fixed_price("call", S, 100.0, T, R, -0.2),
            "volatility sigma must be >= 0",
        ),
        (
            lambda: mcl.price("fixed", "call", S, T, R, SIG, n_monitoring=10, n_paths=10),
            "need a strike K",
        ),
        (
            lambda: mcl.price(BAD_STRIKE_TYPE, "call", S, T, R, SIG, n_monitoring=10, n_paths=10),
            "strike_type must be",
        ),
        (
            lambda: mcl.price("floating", "call", S, 0.0, R, SIG, n_monitoring=10, n_paths=10),
            "T must be > 0",
        ),
    ],
)
def test_invalid_inputs_raise(call: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()
