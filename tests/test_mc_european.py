"""Phase 3a: Monte Carlo European pricer. Every statistical test uses fixed seeds, so it
is deterministic for a given NumPy version."""

import math
from typing import Any

import pytest

from quantprice import bsm
from quantprice.mc import MCResult
from quantprice.mc import european as mce
from quantprice.mc.result import Z_95

Case = tuple[bsm.OptionType, float, float, float, float, float, float]
CASES: dict[str, Case] = {
    "ATM call": ("call", 100.0, 100.0, 1.0, 0.05, 0.2, 0.02),
    "ITM call": ("call", 120.0, 100.0, 1.0, 0.05, 0.2, 0.02),
    "OTM call": ("call", 80.0, 100.0, 1.0, 0.05, 0.2, 0.02),
    "ATM put": ("put", 100.0, 100.0, 1.0, 0.05, 0.2, 0.02),
    "high-vol put, neg rate": ("put", 100.0, 60.0, 2.0, -0.01, 0.8, 0.0),
}
CONFIGS: dict[str, dict[str, Any]] = {
    "plain": {"antithetic": False, "control_variate": False},
    "antithetic": {"antithetic": True, "control_variate": False},
    "control": {"antithetic": False, "control_variate": True},
    "both": {"antithetic": True, "control_variate": True},
}


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("case", CASES)
def test_within_three_standard_errors_of_black_scholes(case: str, config: str) -> None:
    args = CASES[case]
    res = mce.price(*args, n_paths=200_000, seed=20261111, **CONFIGS[config])
    assert res.within(bsm.price(*args), 3.0), (res, bsm.price(*args))


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("case", ["ATM call", "high-vol put, neg rate"])
def test_confidence_interval_coverage_is_about_95_percent(case: str, config: str) -> None:
    """Checks that the standard error is honest, including antithetic pairing and the
    control-variate residual SE. Over 400 independent seeds (2000 paths each), the 95% CI
    should contain the BS price about 380 times. Binomial SD = sqrt(400 * .95 * .05) ~ 4.4,
    so we accept 366..394 (about +-3 SD).

    Measured on 2026-11-11: 373..380 for every case and config.
    """
    args = CASES[case]
    bs = bsm.price(*args)
    hits = sum(
        mce.price(*args, n_paths=2000, seed=s, **CONFIGS[config]).contains(bs) for s in range(400)
    )
    assert 366 <= hits <= 394, hits


def variance_ratio(case: str, config: str) -> float:
    """Var(plain) / Var(config) at equal path count (100k, seed 1). Equal paths means equal
    simulation cost: antithetic draws half as many normals, and the control is free."""
    args = CASES[case]
    se = {
        c: mce.price(*args, n_paths=100_000, seed=1, **CONFIGS[c]).std_error
        for c in ("plain", config)
    }
    return (se["plain"] / se[config]) ** 2


@pytest.mark.parametrize(
    ("case", "config", "at_least", "measured"),
    [
        ("ITM call", "antithetic", 3.0, 5.82),
        ("ITM call", "control", 15.0, 31.5),
        ("ITM call", "both", 200.0, 447.0),
        ("ATM call", "both", 10.0, 22.9),
        ("OTM call", "both", 8.0, 19.5),
        ("ATM put", "both", 5.0, 10.1),
    ],
)
def test_variance_reduction_is_real(
    case: str, config: str, at_least: float, measured: float
) -> None:
    """Variance-reduction factors measured on 2026-11-11 (``measured``). The test requires at
    least about half of each, leaving room for other NumPy versions or platforms. The
    control (discounted S_T) helps most when the payoff is nearly linear in S_T (ITM) and
    little for a far-from-linear payoff: for the high-vol put case the control alone gave
    only 1.23x, so we make no claim there."""
    assert variance_ratio(case, config) >= at_least


def test_control_coefficient_is_a_valid_regression_slope() -> None:
    """Without antithetics, b = Cov(Y, X) / Var(X) with Y = e^{-rT}(S_T - K)^+ and
    X = e^{-rT} S_T. The call payoff is increasing with slope <= 1 in S_T, so
    0 <= Cov(Y, X) <= Var(X) and b lies in [0, 1]. (With antithetics, b is fitted to pair
    averages and is not bounded this way; see mc.estimate.)"""
    res = mce.price(
        "call", 100.0, 100.0, 1.0, 0.05, 0.2, 0.02, n_paths=10_000, seed=0, antithetic=False
    )
    assert res.control_coef is not None
    assert 0.0 <= res.control_coef <= 1.0
    plain = mce.price(
        "call", 100.0, 100.0, 1.0, 0.05, 0.2, 0.02, n_paths=10_000, seed=0, control_variate=False
    )
    assert plain.control_coef is None


# ----------------------------------------------------------------- result object, limits


def test_result_reports_price_se_and_interval() -> None:
    res = mce.price("put", 100.0, 95.0, 0.5, 0.03, 0.25, n_paths=10_000, seed=4)
    assert isinstance(res, MCResult)
    assert res.std_error > 0
    assert res.ci_high - res.price == pytest.approx(Z_95 * res.std_error)
    assert res.price - res.ci_low == pytest.approx(Z_95 * res.std_error)
    assert (res.n_paths, res.n_samples) == (10_000, 5_000)  # antithetic pairs
    assert "95% CI" in str(res)


def test_same_seed_same_result() -> None:
    a = mce.price("call", 100.0, 100.0, 1.0, 0.05, 0.2, n_paths=1000, seed=9)
    b = mce.price("call", 100.0, 100.0, 1.0, 0.05, 0.2, n_paths=1000, seed=9)
    c = mce.price("call", 100.0, 100.0, 1.0, 0.05, 0.2, n_paths=1000, seed=10)
    assert a == b
    assert a != c


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_zero_volatility_is_exact_with_zero_error(option_type: bsm.OptionType) -> None:
    """sigma = 0: every path is identical. The SE is zero up to rounding in the mean, and the
    control is switched off (b = 0) rather than fitted to rounding noise."""
    res = mce.price(option_type, 100.0, 95.0, 1.0, 0.03, 0.0, 0.01, n_paths=100, seed=0)
    assert res.std_error <= 1e-12 * max(res.price, 1.0)
    assert res.control_coef == 0.0
    assert res.price == pytest.approx(
        bsm.price(option_type, 100.0, 95.0, 1.0, 0.03, 0.0, 0.01), rel=1e-14
    )


def test_at_expiry_returns_payoff() -> None:
    res = mce.price("call", 110.0, 100.0, 0.0, 0.05, 0.2, n_paths=100, seed=0)
    assert (res.price, res.std_error) == (10.0, 0.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_paths": 101}, "must be even with antithetic"),
        ({"n_paths": 1, "antithetic": False}, "integer >= 2"),
        ({"seed": -3}, "non-negative"),
        ({"K": [100.0, 110.0]}, "scalar inputs"),
        ({"sigma": -0.2}, "volatility sigma must be >= 0"),
    ],
)
def test_invalid_inputs_raise(kwargs: dict[str, object], message: str) -> None:
    args: dict[str, Any] = {
        "S": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.2,
        "n_paths": 100,
    }
    args |= kwargs
    with pytest.raises(ValueError, match=message):
        mce.price("call", **args)


def test_control_variate_needs_enough_samples() -> None:
    with pytest.raises(ValueError, match="at least 3 independent samples"):
        mce.price("call", 100.0, 100.0, 1.0, 0.05, 0.2, n_paths=4, seed=0)


def test_discounting_uses_continuous_compounding() -> None:
    """Deep ITM call with sigma tiny: price ~ S e^{-qT} - K e^{-rT} (continuous compounding)."""
    res = mce.price("call", 200.0, 100.0, 2.0, 0.05, 1e-6, 0.03, n_paths=1000, seed=0)
    assert res.price == pytest.approx(200 * math.exp(-0.06) - 100 * math.exp(-0.10), rel=1e-6)
