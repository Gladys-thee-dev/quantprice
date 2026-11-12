"""Phase 5: every generated benchmark answer is re-derived by an independent method.

The generators compute answers with quantprice itself. Here each record is recomputed
from its *inputs field only* (what the solver sees), with code that does not call the
function that produced the answer:

* ``bs_price``: 50-digit mpmath price (tests/reference_mpmath.py);
* ``greek_*``: 50-digit numerical derivatives of that mpmath price;
* ``put_call_parity``: the quote's implied vol, then the mpmath price of the other option.

The check is ``|answer - reference| <= VERIFY_FRACTION * tolerance``: the verification
error must be negligible next to the grading band.
"""

import math
from typing import Any

import pytest

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
    raise AssertionError(f"no reference for topic {p.topic}")


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
