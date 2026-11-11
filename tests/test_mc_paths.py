"""Phase 3a: exact GBM path simulation and antithetic layout. All tests use fixed seeds."""

import math
from typing import Any

import numpy as np
import pytest

from quantprice.mc.paths import make_rng, simulate_gbm

S0, R, Q, SIGMA = 100.0, 0.05, 0.02, 0.3
TIMES = [0.25, 0.5, 1.0, 2.0]


def test_shape_and_positivity() -> None:
    paths = simulate_gbm(S0, R, Q, SIGMA, TIMES, 1000, seed=0)
    assert paths.shape == (1000, 4)
    assert np.all(paths > 0)


def test_mean_matches_forward_at_each_date() -> None:
    """E[S_t] = S0 e^{(r-q)t}. Plain sampling with 200k paths; each date within 3 SE."""
    paths = simulate_gbm(S0, R, Q, SIGMA, TIMES, 200_000, seed=11, antithetic=False)
    for k, t in enumerate(TIMES):
        col = paths[:, k]
        se = col.std(ddof=1) / math.sqrt(col.size)
        assert abs(col.mean() - S0 * math.exp((R - Q) * t)) <= 3 * se, t


def test_log_increments_have_correct_variance_and_are_uncorrelated() -> None:
    """Var[ln(S_{t_k} / S_{t_{k-1}})] = sigma^2 dt_k, within 3 SE. For normal data the SE of a
    sample variance v is v sqrt(2 / (n - 1)). Consecutive increments are uncorrelated:
    |corr| <= 3 / sqrt(n)."""
    n = 200_000
    paths = simulate_gbm(S0, R, Q, SIGMA, TIMES, n, seed=12, antithetic=False)
    logs = np.log(np.column_stack([np.full(n, S0), paths]))
    inc = np.diff(logs, axis=1)
    dts = np.diff([0.0, *TIMES])
    for k, dt in enumerate(dts):
        v = SIGMA**2 * dt
        assert abs(inc[:, k].var(ddof=1) - v) <= 3 * v * math.sqrt(2 / (n - 1))
    for k in range(len(dts) - 1):
        assert abs(np.corrcoef(inc[:, k], inc[:, k + 1])[0, 1]) <= 3 / math.sqrt(n)


def test_antithetic_rows_mirror_each_other() -> None:
    """Row i uses Z and row i + n/2 uses -Z, so their log increments sum to exactly
    2 (r - q - sigma^2/2) dt."""
    n = 1000
    paths = simulate_gbm(S0, R, Q, SIGMA, TIMES, n, seed=3, antithetic=True)
    logs = np.log(np.column_stack([np.full(n, S0), paths]))
    inc = np.diff(logs, axis=1)
    drift = 2 * (R - Q - 0.5 * SIGMA**2) * np.diff([0.0, *TIMES])
    np.testing.assert_allclose(
        inc[: n // 2] + inc[n // 2 :], np.broadcast_to(drift, (n // 2, 4)), atol=1e-12
    )


def test_same_seed_is_reproducible_and_generator_is_accepted() -> None:
    a = simulate_gbm(S0, R, Q, SIGMA, TIMES, 100, seed=5)
    b = simulate_gbm(S0, R, Q, SIGMA, TIMES, 100, seed=5)
    c = simulate_gbm(S0, R, Q, SIGMA, TIMES, 100, seed=6)
    d = simulate_gbm(S0, R, Q, SIGMA, TIMES, 100, seed=np.random.default_rng(5))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, d)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_paths": 3}, "must be even with antithetic"),
        ({"n_paths": 1, "antithetic": False}, "integer >= 2"),
        ({"times": [0.5, 0.25]}, "strictly increasing"),
        ({"times": [0.0, 1.0]}, "> 0"),
        ({"times": []}, "non-empty"),
        ({"seed": -1}, "non-negative"),
        ({"seed": "abc"}, "seed must be"),
        ({"S": -1.0}, "spot S must be > 0"),
    ],
)
def test_invalid_inputs_raise(kwargs: dict[str, object], message: str) -> None:
    args: dict[str, Any] = {
        "S": S0,
        "r": R,
        "q": Q,
        "sigma": SIGMA,
        "times": TIMES,
        "n_paths": 10,
        "seed": 0,
    }
    args |= kwargs
    with pytest.raises(ValueError, match=message):
        simulate_gbm(**args)


def test_make_rng_passes_through_generator() -> None:
    g = np.random.default_rng(0)
    assert make_rng(g) is g
