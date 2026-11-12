"""Optional cross-validation against QuantLib-Python.

Run with::

    pip install -e ".[dev,quantlib]"
    pytest -m quantlib -s tests/crossval      # -s prints the error table

Skipped when QuantLib is not installed (CI does not install it).

Conventions are aligned with quantprice:

* ``Actual365Fixed`` day count and ``NullCalendar`` (no holidays);
* flat, continuously compounded ``FlatForward`` curves for r and q;
* ``BlackConstantVol`` volatility;
* expiry ``today + days``, so QuantLib's year fraction is exactly ``days / 365``.

Each model is compared over a grid, and ``test_zz_report`` prints three numbers per
model:

* max absolute error;
* max relative error over all points, ``|ours - QL| / max(|QL|, 1e-12 * S)``;
* max relative error over points where ``|QL| >= 1e-6 * S`` (values that are not
  negligible).

The tests assert on absolute error scaled by spot (``abs <= 1e-12 * S`` for analytic
models), plus the non-negligible relative error. Relative error over *all* points is
dominated by options worth ~1e-10. There the result is cancellation noise of ~1e-13
in absolute terms. For the worst such BSM and geometric-Asian cases, the 50-digit
mpmath reference matches quantprice (to ~1e-13 relative), and QuantLib differs from it
by up to 1.8e-4 (BSM put worth 7e-11) and 9.5e-6 (Asian call worth 8e-9). For the
worst barrier and fixed-lookback cases (values 2e-8 and 3e-10, absolute differences
~1e-13), no 50-digit reference was built, so no claim is made about which is closer.

Measured with QuantLib 1.44 on 2026-11-12 (Windows, Python 3.14, NumPy 2.5.4):

==============================  =====  ===========  =============  ==================
model                               n  max abs err  max rel (all)  max rel (material)
==============================  =====  ===========  =============  ==================
bsm price                        1458    4.974e-14      1.062e-04           3.019e-11
bsm delta                        1458    4.052e-15      6.121e-07           4.375e-13
bsm gamma                        1458    3.123e-16      5.237e-13           1.352e-13
bsm vega                         1458    1.279e-13      5.729e-13           2.868e-13
bsm theta                        1458    3.046e-13      1.505e-07           5.776e-13
bsm rho                          1458    1.990e-13      2.279e-06           1.923e-11
implied vol                      1234    7.740e-12      7.740e-11           7.740e-11
american put (2000-step trees)    144    2.359e-03      2.581e-03           2.581e-03
barrier                          3456    8.282e-14      3.811e-06           1.991e-10
geometric asian                  1458    1.137e-13      9.520e-06           4.050e-11
lookback floating                 486    2.132e-13      1.186e-14           1.186e-14
lookback fixed                   1458    2.274e-13      1.502e-07           1.403e-10
==============================  =====  ===========  =============  ==================

One known model difference: QuantLib's ``"crr"`` tree is the additive (log-space)
version, with up-probability ``1/2 + (r - q - sigma^2/2) sqrt(dt) / (2 sigma)``. Ours
uses the multiplicative ``(e^{(r-q) dt} - d)/(u - d)``. They agree only as
``steps -> infinity``, so the American comparison uses 2000 steps in both, and its
tolerance reflects the O(1/n) tree error, not a coding error.
"""

import itertools
import math
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import pytest

from quantprice import asian, barrier, binomial, bsm, lookback
from quantprice.barrier import BarrierType
from quantprice.bsm import OptionType
from quantprice.implied_vol import implied_vol

ql: Any = pytest.importorskip("QuantLib")

pytestmark = pytest.mark.quantlib

TODAY = ql.Date(12, 11, 2026)
# model -> [max abs err, max rel err (all), max rel err (|QL| >= 1e-6 S), max abs / S, n]
ERRORS: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])


def record(model: str, ours: float, theirs: float, S: float) -> None:
    abs_err = abs(ours - theirs)
    rel_err = abs_err / max(abs(theirs), 1e-12 * S)
    e = ERRORS[model]
    e[0] = max(e[0], abs_err)
    e[1] = max(e[1], rel_err)
    if abs(theirs) >= 1e-6 * S:
        e[2] = max(e[2], rel_err)
    e[3] = max(e[3], abs_err / S)
    e[4] += 1


def check(model: str, abs_over_s: float, rel_material: float) -> None:
    e = ERRORS[model]
    assert e[3] <= abs_over_s, (model, e)
    assert e[2] <= rel_material, (model, e)


def process(S: float, r: float, q: float, sigma: float) -> Any:
    ql.Settings.instance().evaluationDate = TODAY
    dc = ql.Actual365Fixed()
    return ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(S)),
        ql.YieldTermStructureHandle(ql.FlatForward(TODAY, q, dc, ql.Continuous)),
        ql.YieldTermStructureHandle(ql.FlatForward(TODAY, r, dc, ql.Continuous)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(TODAY, ql.NullCalendar(), sigma, dc)),
    )


KINDS: list[OptionType] = ["call", "put"]


def ql_type(kind: OptionType) -> Any:
    return ql.Option.Call if kind == "call" else ql.Option.Put


SPOTS = [80.0, 100.0, 120.0]
STRIKES = [90.0, 100.0, 110.0]
VOLS = [0.1, 0.3, 0.6]
RATES = [-0.01, 0.03, 0.08]
YIELDS = [0.0, 0.02, 0.05]
DAYS = [30, 182, 730]


def full_grid() -> Iterator[tuple[OptionType, float, float, int, float, float, float]]:
    yield from itertools.product(KINDS, SPOTS, STRIKES, DAYS, RATES, VOLS, YIELDS)


# --------------------------------------------------------------------- European BSM


def test_bsm_prices_and_greeks() -> None:
    """1458 options (2 types x 729 market points). Price, delta, gamma, vega, theta (per
    year) and rho against AnalyticEuropeanEngine."""
    for kind, S, K, days, r, sigma, q in full_grid():
        T = days / 365
        opt = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql_type(kind), K), ql.EuropeanExercise(TODAY + days)
        )
        opt.setPricingEngine(ql.AnalyticEuropeanEngine(process(S, r, q, sigma)))
        g = bsm.greeks(kind, S, K, T, r, sigma, q)
        pairs = {
            "bsm price": (g.price, opt.NPV()),
            "bsm delta": (g.delta, opt.delta()),
            "bsm gamma": (g.gamma, opt.gamma()),
            "bsm vega": (g.vega, opt.vega()),
            "bsm theta": (g.theta, opt.theta()),
            "bsm rho": (g.rho, opt.rho()),
        }
        for model, (ours, theirs) in pairs.items():
            record(model, ours, theirs, S)
    for model in ("bsm price", "bsm delta", "bsm gamma", "bsm vega", "bsm theta", "bsm rho"):
        check(model, 1e-12, 1e-9)


def test_implied_vol() -> None:
    """Recover sigma from our BSM price with both solvers, where vega >= 1% of max."""
    for kind, S, K, days, r, sigma, q in full_grid():
        T = days / 365
        g = bsm.greeks(kind, S, K, T, r, sigma, q)
        if g.vega < 0.01 * S * math.exp(-q * T) * math.sqrt(T / (2 * math.pi)):
            continue
        opt = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql_type(kind), K), ql.EuropeanExercise(TODAY + days)
        )
        theirs = opt.impliedVolatility(g.price, process(S, r, q, sigma), 1e-14, 1000, 1e-7, 5.0)
        ours = implied_vol(kind, g.price, S, K, T, r, q)
        record("implied vol", ours, theirs, 1.0)
    check("implied vol", 1e-9, 1e-9)


# ------------------------------------------------------------------ American (tree)


def test_american_put_binomial_2000_steps() -> None:
    """American puts on 2000-step trees: our multiplicative CRR vs QuantLib's additive
    "crr". The difference is the O(1/n) tree discretisation (see the module docstring).
    Measured for the worst grid point (S=80, K=90, 182 days, r=3%, sigma=30%, q=2%): the
    gap is 4.8e-3, 2.2e-3, 2.4e-3, 1.1e-3 and 1.8e-4 at 500, 1000, 2000, 4000 and 8000
    steps. It shrinks with n, with the usual odd/even oscillation."""
    for S, K, days, r, sigma, q in itertools.product(
        SPOTS, STRIKES, [182, 730], [0.03, 0.08], [0.1, 0.3], [0.0, 0.02]
    ):
        opt = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Put, K), ql.AmericanExercise(TODAY, TODAY + days)
        )
        opt.setPricingEngine(ql.BinomialVanillaEngine(process(S, r, q, sigma), "crr", 2000))
        ours = binomial.price("put", S, K, days / 365, r, sigma, q, steps=2000, exercise="american")
        record("american put (2000-step trees)", ours, opt.NPV(), S)
    check("american put (2000-step trees)", 1e-4, 1e-2)


# --------------------------------------------------------------------------- exotics


def test_barriers() -> None:
    """All eight single-barrier types, with and without a rebate (knock-out rebate paid
    at the hit, knock-in at expiry, as in both libraries), against AnalyticBarrierEngine."""
    kinds: dict[BarrierType, Any] = {
        "down-and-out": ql.Barrier.DownOut,
        "down-and-in": ql.Barrier.DownIn,
        "up-and-out": ql.Barrier.UpOut,
        "up-and-in": ql.Barrier.UpIn,
    }
    for (btype, qlb), kind, K, days, r, sigma, q, rebate in itertools.product(
        kinds.items(),
        KINDS,
        STRIKES,
        [30, 182, 730],
        [0.03, 0.08],
        VOLS,
        [0.0, 0.02],
        [0.0, 3.0],
    ):
        S = 100.0
        for H in [85.0, 95.0] if btype.startswith("down") else [105.0, 115.0]:
            opt = ql.BarrierOption(
                qlb,
                H,
                rebate,
                ql.PlainVanillaPayoff(ql_type(kind), K),
                ql.EuropeanExercise(TODAY + days),
            )
            opt.setPricingEngine(ql.AnalyticBarrierEngine(process(S, r, q, sigma)))
            ours = barrier.price(
                kind,
                btype,
                S,
                K,
                H,
                days / 365,
                r,
                sigma,
                q,
                rebate=rebate,
            )
            record("barrier", ours, opt.NPV(), S)
    check("barrier", 1e-12, 1e-9)


def test_geometric_asian() -> None:
    """Discrete geometric average-price Asians against
    AnalyticDiscreteGeometricAveragePriceAsianEngine (fresh issue, payment at the last
    fixing)."""
    for kind, S, K, r, sigma, q, (n, gap) in itertools.product(
        KINDS, SPOTS, STRIKES, RATES, VOLS, YIELDS, [(4, 91), (12, 30), (52, 7)]
    ):
        fixing_dates = [TODAY + gap * (i + 1) for i in range(n)]
        opt = ql.DiscreteAveragingAsianOption(
            ql.Average.Geometric,
            1.0,
            0,
            fixing_dates,
            ql.PlainVanillaPayoff(ql_type(kind), K),
            ql.EuropeanExercise(fixing_dates[-1]),
        )
        opt.setPricingEngine(
            ql.AnalyticDiscreteGeometricAveragePriceAsianEngine(process(S, r, q, sigma))
        )
        times = [gap * (i + 1) / 365 for i in range(n)]
        ours = asian.geometric_price(kind, S, K, r, sigma, q, times=times)
        record("geometric asian", ours, opt.NPV(), S)
    check("geometric asian", 1e-12, 1e-9)


def test_lookbacks() -> None:
    """Fresh-issue continuous lookbacks against AnalyticContinuousFloatingLookbackEngine
    and AnalyticContinuousFixedLookbackEngine. The grid avoids r = q, where the textbook
    formula QuantLib implements has a 0/0 (ours has a stable rewrite there)."""
    for kind, S, days, r, sigma, q in itertools.product(KINDS, SPOTS, DAYS, RATES, VOLS, YIELDS):
        if abs(r - q) < 1e-12:
            continue
        T = days / 365
        proc = process(S, r, q, sigma)
        ex = ql.EuropeanExercise(TODAY + days)
        fl = ql.ContinuousFloatingLookbackOption(S, ql.FloatingTypePayoff(ql_type(kind)), ex)
        fl.setPricingEngine(ql.AnalyticContinuousFloatingLookbackEngine(proc))
        ours = lookback.floating_price(kind, S, T, r, sigma, q)
        record("lookback floating", ours, fl.NPV(), S)
        for K in STRIKES:
            fx = ql.ContinuousFixedLookbackOption(S, ql.PlainVanillaPayoff(ql_type(kind), K), ex)
            fx.setPricingEngine(ql.AnalyticContinuousFixedLookbackEngine(proc))
            ours = lookback.fixed_price(kind, S, K, T, r, sigma, q)
            record("lookback fixed", ours, fx.NPV(), S)
    check("lookback floating", 1e-12, 1e-9)
    check("lookback fixed", 1e-12, 1e-9)


def test_zz_report() -> None:
    """Print the per-model error table (visible with ``pytest -s``)."""
    print("\nQuantLib", ql.__version__, "cross-validation")
    print(
        f"{'model':32s} {'n':>5s} {'max abs err':>12s} {'max rel (all)':>14s} "
        f"{'max rel (|V|>=1e-6 S)':>22s}"
    )
    for model, (a, r_all, r_mat, _, n) in ERRORS.items():
        print(f"{model:32s} {int(n):5d} {a:12.3e} {r_all:14.3e} {r_mat:22.3e}")
    assert ERRORS, "run the whole module so the table has rows"
