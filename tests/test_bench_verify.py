"""Phase 5: every generated benchmark answer is re-derived by an independent method.

The generators compute answers with quantprice itself. Here each record is recomputed
from its *inputs field only* (what the solver sees), with code that does not call the
function that produced the answer:

* ``bs_price``: 50-digit mpmath price (tests/reference_mpmath.py);
* ``greek_*``: 50-digit numerical derivatives of that mpmath price;
* ``put_call_parity``: the quote's implied vol, then the mpmath price of the other option;
* ``binomial_american_put``: a plain-loop CRR tree written here;
* ``implied_vol``: mpmath root finding on the 50-digit price, plus a bracket check: the
  price is below the quote at ``answer - tol`` and above it at ``answer + tol``;
* ``geometric_asian``: the variance of ln G from the explicit double sum over
  min(t_i, t_j), and the price in mpmath;
* ``barrier_in_out_parity``: the mpmath vanilla minus the quote, and separately the
  Reiner-Rubinstein knock-in closed form (within the quote's rounding).

The check is ``|answer - reference| <= VERIFY_FRACTION * tolerance``: the verification
error must be negligible next to the grading band.
"""

import math
from typing import Any

import mpmath as mp
import pytest

from quantprice import barrier
from quantprice.benchmarks import Problem, generate_all
from quantprice.conventions import DAYS_PER_YEAR, PERCENTAGE_POINT
from quantprice.implied_vol import implied_vol
from reference_mpmath import mp_bsm_greeks, mp_bsm_price

SEED = 2026
PER_TOPIC = 30
VERIFY_FRACTION = 1e-3

PROBLEMS = generate_all(SEED, PER_TOPIC)


def market(p: Problem) -> tuple[bool, float, float, float, float, float]:
    i: dict[str, Any] = p.inputs
    assert i["T"] == i["T_days"] / DAYS_PER_YEAR
    return i["option_type"] == "call", i["S"], i["K"], i["T"], i["r"], i["q"]


def reference(p: Problem) -> float:
    is_call, S, K, T, r, q = market(p)
    if p.topic == "bs_price":
        return mp_bsm_price(is_call, S, K, T, r, p.inputs["sigma"], q)  # type: ignore[arg-type]
    if p.topic.startswith("greek_"):
        name = p.topic.removeprefix("greek_")
        raw = mp_bsm_greeks(is_call, S, K, T, r, p.inputs["sigma"], q)[name]  # type: ignore[arg-type]
        hard = p.difficulty == "hard"
        factor = {
            "vega": 1.0 if hard else PERCENTAGE_POINT,
            "rho": 1.0 if hard else PERCENTAGE_POINT,
            "theta": 1.0 if hard else 1.0 / DAYS_PER_YEAR,
        }.get(name, 1.0)
        return raw * factor
    if p.topic == "put_call_parity":
        given = p.inputs["given_type"]
        quote = p.inputs["given_price"]
        assert given != p.inputs["option_type"]
        sigma = implied_vol(given, quote, S, K, T, r, q)  # type: ignore[arg-type]
        return mp_bsm_price(is_call, S, K, T, r, sigma, q)
    if p.topic == "binomial_american_put":
        return loop_tree_american_put(S, K, T, r, p.inputs["sigma"], q, p.inputs["steps"])  # type: ignore[arg-type]
    if p.topic == "implied_vol":
        return mp_implied_vol(is_call, p.inputs["price"], S, K, T, r, q, p.verified_answer)  # type: ignore[arg-type]
    if p.topic == "geometric_asian":
        return mp_geometric_asian(p)
    if p.topic == "barrier_in_out_parity":
        vanilla = mp_bsm_price(is_call, S, K, T, r, p.inputs["sigma"], q)  # type: ignore[arg-type]
        return vanilla - p.inputs["knock_out_price"]  # type: ignore[operator]
    raise AssertionError(f"no reference for topic {p.topic}")


def loop_tree_american_put(
    S: float, K: float, T: float, r: float, sigma: float, q: float, n: int
) -> float:
    """CRR American put by explicit nested loops (no NumPy), per Hull's textbook layout."""
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    assert 0 <= p <= 1
    disc = math.exp(-r * dt)
    v = [max(K - S * u**j * d ** (n - j), 0.0) for j in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            cont = disc * (p * v[j + 1] + (1 - p) * v[j])
            v[j] = max(cont, K - S * u**j * d ** (i - j))
    return v[0]


def mp_implied_vol(
    is_call: bool, quote: float, S: float, K: float, T: float, r: float, q: float, start: float
) -> float:
    """Root of the 50-digit price minus the quote, by mpmath's secant solver."""
    with mp.workdps(50):
        Sm, Km, Tm, rm, qm = (mp.mpf(x) for x in (S, K, T, r, q))

        def f(sig: Any) -> Any:
            d1 = (mp.log(Sm / Km) + (rm - qm + sig**2 / 2) * Tm) / (sig * mp.sqrt(Tm))
            d2 = d1 - sig * mp.sqrt(Tm)
            fs, fk = Sm * mp.exp(-qm * Tm), Km * mp.exp(-rm * Tm)
            if is_call:
                v = fs * mp.ncdf(d1) - fk * mp.ncdf(d2)
            else:
                v = fk * mp.ncdf(-d2) - fs * mp.ncdf(-d1)
            return v - mp.mpf(quote)

        return float(mp.findroot(f, (mp.mpf(start) * 0.9, mp.mpf(start) * 1.1), solver="secant"))


def mp_geometric_asian(p: Problem) -> float:
    """ln G ~ Normal(mu, v): mean from the average fixing time, variance from the explicit
    double sum sigma^2 / n^2 * sum_ij min(t_i, t_j); then Black-76 in mpmath."""
    i: dict[str, Any] = p.inputs
    n, gap = i["n_fixings"], i["fixing_interval_days"]
    with mp.workdps(50):
        S, K, r, q, sig = (mp.mpf(i[k]) for k in ("S", "K", "r", "q", "sigma"))
        t = [mp.mpf(gap * (k + 1)) / 365 for k in range(n)]
        assert float(t[-1]) == i["T"]
        mu = mp.log(S) + (r - q - sig**2 / 2) * mp.fsum(t) / n
        v = sig**2 / n**2 * mp.fsum(min(a, b) for a in t for b in t)
        F = mp.exp(mu + v / 2)
        d1 = (mp.log(F / K) + v / 2) / mp.sqrt(v)
        d2 = d1 - mp.sqrt(v)
        disc = mp.exp(-r * t[-1])
        if i["option_type"] == "call":
            return float(disc * (F * mp.ncdf(d1) - K * mp.ncdf(d2)))
        return float(disc * (K * mp.ncdf(-d2) - F * mp.ncdf(-d1)))


@pytest.mark.parametrize("p", PROBLEMS, ids=[p.id for p in PROBLEMS])
def test_answer_matches_independent_reference(p: Problem) -> None:
    ref = reference(p)
    assert abs(p.verified_answer - ref) <= VERIFY_FRACTION * p.tolerance.abs, (
        p.verified_answer,
        ref,
    )


def test_unit_conventions_are_distinguishable() -> None:
    """The grading band separates the stated unit from the common alternatives
    (vega per 1.00 vs per pct; theta per day vs per year vs per 252 trading days)."""
    for p in PROBLEMS:
        name = p.topic.removeprefix("greek_")
        if name not in ("vega", "rho", "theta") or p.difficulty == "hard":
            continue
        v, tol = p.verified_answer, p.tolerance
        wrong = [v / PERCENTAGE_POINT] if name != "theta" else [v * 365, v * 365 / 252]
        assert not any(tol.accepts(w, v) for w in wrong), p.id


def test_ignoring_the_dividend_yield_fails() -> None:
    """For medium problems (q >= 0.25%), pricing with q = 0 must fall outside the band."""
    caught = total = 0
    for p in PROBLEMS:
        if p.topic != "bs_price" or p.difficulty != "medium":
            continue
        is_call, S, K, T, r, _ = market(p)
        wrong = mp_bsm_price(is_call, S, K, T, r, p.inputs["sigma"], 0.0)  # type: ignore[arg-type]
        total += 1
        caught += not p.tolerance.accepts(wrong, p.verified_answer)
    assert total == PER_TOPIC // 3
    assert caught == total


def test_quoted_prices_have_six_decimals() -> None:
    for p in PROBLEMS:
        if "given_price" in p.inputs:
            quote = p.inputs["given_price"]
            assert isinstance(quote, float)
            assert f"{quote:.6f}" in p.problem
            assert math.isclose(float(f"{quote:.6f}"), quote, rel_tol=0, abs_tol=0)


def test_implied_vol_band_brackets_the_root() -> None:
    """Price is increasing in sigma, so if V(answer - tol) < quote < V(answer + tol) in
    50-digit arithmetic, the exact implied vol of the quote is inside the band."""
    for p in PROBLEMS:
        if p.topic != "implied_vol":
            continue
        is_call, S, K, T, r, q = market(p)
        quote, ans, tol = p.inputs["price"], p.verified_answer, p.tolerance.abs
        lo = mp_bsm_price(is_call, S, K, T, r, ans - tol, q)
        hi = mp_bsm_price(is_call, S, K, T, r, ans + tol, q)
        assert lo < quote < hi, p.id  # type: ignore[operator]


def test_barrier_answer_matches_reiner_rubinstein_knock_in() -> None:
    """Parity answer vs the knock-in closed form (formula terms A-E, not parity). They
    differ only by the 6-decimal rounding of the quoted knock-out price."""
    for p in PROBLEMS:
        if p.topic != "barrier_in_out_parity":
            continue
        _, S, K, T, r, q = market(p)
        i: dict[str, Any] = p.inputs
        ki = barrier.price(i["option_type"], i["barrier_type"], S, K, i["H"], T, r, i["sigma"], q)
        assert abs(p.verified_answer - ki) <= 5e-7 + 1e-12 * S, p.id


def test_binomial_problems_need_the_tree() -> None:
    """The value beats intrinsic by >= 10 tolerances, so K - S never passes."""
    for p in PROBLEMS:
        if p.topic == "binomial_american_put":
            i: dict[str, Any] = p.inputs
            intrinsic = max(i["K"] - i["S"], 0.0)
            assert p.verified_answer - intrinsic >= 10 * p.tolerance.abs, p.id
