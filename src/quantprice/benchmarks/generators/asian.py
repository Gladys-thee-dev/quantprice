r"""Topic ``geometric_asian``: discretely sampled, fixed-strike geometric Asian option.

Payoff at the last fixing date :math:`T = t_n`: :math:`\max(w(G - K), 0)` with
:math:`G = (\prod_{i=1}^n S_{t_i})^{1/n}`, equally spaced fixings :math:`t_i = i\,\Delta`
(today's spot is not a fixing).

:math:`\ln G` is normal with (Kemna and Vorst 1990; Haug 2007, section 4.20.1; discrete
form as in :mod:`quantprice.asian`)

.. math::

    \mu = \ln S + (r - q - \sigma^2/2)\,\bar t,\qquad
    v = \frac{\sigma^2}{n^2}\sum_{i,j}\min(t_i, t_j)
      = \frac{\sigma^2}{n^2}\sum_{k=1}^n (2(n-k)+1)\,t_k,

so the price is Black-76 on the forward :math:`F = e^{\mu + v/2}`:
:math:`e^{-rT}[w F N(w d_1) - w K N(w d_2)]`, :math:`d_{1,2} = (\ln(F/K) \pm v/2)/\sqrt v`.

Fixings: easy 2-4 fixings every 30 or 91 days (q = 0); medium 6 or 12 monthly
(30-day) fixings; hard 26 or 52 weekly fixings.

Verified answer: :func:`quantprice.asian.geometric_price`. Independent check in the
tests: v from the explicit double sum over min(t_i, t_j), and the price in mpmath.
"""

import math
import random
from dataclasses import replace

from quantprice import asian
from quantprice.benchmarks.generators._common import (
    Draft,
    N,
    assemble,
    band,
    conventions,
    g,
    market_lines,
    sample_market,
)
from quantprice.benchmarks.schema import Difficulty
from quantprice.bsm import OptionType
from quantprice.conventions import DAYS_PER_YEAR

_FIXINGS: dict[Difficulty, tuple[list[int], list[int]]] = {
    # (number of fixings, days between fixings)
    "easy": ([2, 3, 4], [30, 91]),
    "medium": ([6, 12], [30]),
    "hard": ([26, 52], [7]),
}


def make(rng: random.Random, difficulty: Difficulty) -> Draft:
    m0 = sample_market(rng, difficulty)
    counts, gaps = _FIXINGS[difficulty]
    n, gap = rng.choice(counts), rng.choice(gaps)
    m = replace(m0, days=n * gap)
    kind: OptionType = rng.choice(["call", "put"])
    S, K, T, r, sigma, q = m.args()
    fixing_days = [gap * (i + 1) for i in range(n)]
    times = [day / DAYS_PER_YEAR for day in fixing_days]
    answer = asian.geometric_price(kind, S, K, r, sigma, q, times=times)
    mu, v = asian.geometric_log_moments(S, r, sigma, q, times)
    F = math.exp(mu + 0.5 * v)
    d1 = (math.log(F / K) + 0.5 * v) / math.sqrt(v)
    d2 = d1 - math.sqrt(v)
    tol = band(m.scale)
    steps = [
        f"Fixing times t_i = {gap} i / 365 years, i = 1..{n}; T = t_{n} = {g(T)}.",
        f"Mean fixing time t_bar = {g(sum(times) / n)}.",
        f"mu = ln S + (r - q - sigma^2/2) t_bar = {g(mu)}.",
        f"v = (sigma^2 / n^2) sum_(i,j) min(t_i, t_j) = {g(v)}.",
        f"Forward of G: F = exp(mu + v/2) = {g(F)}.",
        f"d1 = (ln(F/K) + v/2) / sqrt(v) = {g(d1)}; d2 = d1 - sqrt(v) = {g(d2)}.",
        (
            f"Price = e^(-rT) [F N(d1) - K N(d2)] = e^(-rT) [{g(F)} x {g(N(d1))} - "
            f"{g(K)} x {g(N(d2))}] = {g(answer)}."
            if kind == "call"
            else f"Price = e^(-rT) [K N(-d2) - F N(-d1)] = e^(-rT) [{g(K)} x {g(N(-d2))} - "
            f"{g(F)} x {g(N(-d1))}] = {g(answer)}."
        ),
    ]
    conv = conventions(
        "Black-Scholes-Merton dynamics: geometric Brownian motion under the risk-neutral "
        "measure with constant r, q and sigma. The option is European-style and pays at "
        "the last fixing date. Today's spot is not one of the fixings."
    )
    payoff = "max(G - K, 0)" if kind == "call" else "max(K - G, 0)"
    facts = [
        f"Spot price S = {S:g}.",
        f"Strike K = {K:g}.",
        f"Fixings: {n} dates, every {gap} calendar days from today (days "
        + ", ".join(str(day) for day in fixing_days[:4])
        + (", ..., " + str(fixing_days[-1]) if n > 4 else "")
        + f"), so fixing i is at t_i = {gap} i / 365 years (ACT/365).",
        f"G is the geometric average of the {n} fixing prices: G = (S(t_1) ... S(t_{n}))^(1/{n}).",
        f"The {kind} pays {payoff} at the last fixing date T = {n * gap}/365 years.",
        *market_lines(m)[3:],
    ]
    text = assemble(
        f"A geometric-average Asian {kind} option (fixed strike) is written on a stock.",
        facts,
        conv,
        f"What is the fair value of the Asian {kind} today?",
        tol,
        "the option value in currency units",
    )
    inputs = m.inputs(kind) | {"n_fixings": n, "fixing_interval_days": gap}
    basis = "1e-4 * max(S e^(-qT), K e^(-rT)) with T the last fixing date, floor 1e-12"
    return Draft(text, conv, inputs, answer, tol, basis, steps)
