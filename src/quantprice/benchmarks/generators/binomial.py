r"""Topic ``binomial_american_put``: value an American put on a stated CRR tree.

The answer is the value on the tree with the stated number of steps, not the
continuous-time American price, so it is exact. Tree (Cox, Ross and Rubinstein 1979;
Hull, chapter "Binomial Trees", with dividend yield q):

    dt = T/n,  u = e^(sigma sqrt(dt)),  d = 1/u,  p = (e^((r-q) dt) - d)/(u - d),

with value :math:`V = \max(K - S_{node}, e^{-r\,dt}[p V_{up} + (1-p) V_{down}])` at every
node, including today. Steps: easy 2-4 (feasible by hand, node values shown in the
solution), medium 10-30, hard 100-250 (needs code).

Draws are rejected where p falls outside [0, 1] (an invalid tree), and where the value
is within 10 tolerances of the intrinsic value ``max(K - S, 0)``. There a solver could
answer without building the tree.

Verified answer: :func:`quantprice.binomial.price`. Independent check in the tests: a
plain-loop tree written separately.
"""

import math
import random
from dataclasses import replace

from quantprice import binomial
from quantprice.benchmarks.generators._common import (
    MIN_ANSWER_IN_TOLS,
    Draft,
    assemble,
    conventions,
    g,
    market_lines,
    price_tolerance,
    sample_market,
)
from quantprice.benchmarks.schema import Difficulty

_STEPS: dict[Difficulty, list[int]] = {
    "easy": [2, 3, 4],
    "medium": [10, 15, 20, 25, 30],
    "hard": [100, 150, 200, 250],
}
_SHOW_NODES_UP_TO = 4


def make(rng: random.Random, difficulty: Difficulty) -> Draft | None:
    m = sample_market(rng, difficulty)
    if difficulty == "easy":  # short trees: keep expiry short too, so dt is not huge
        m = replace(m, days=rng.choice([91, 182]))
    n = rng.choice(_STEPS[difficulty])
    try:
        answer = binomial.price("put", *m.args(), steps=n, exercise="american")
        european = binomial.price("put", *m.args(), steps=n, exercise="european")
    except ValueError:
        return None  # p outside [0, 1] for this draw
    S, K, T, r, sigma, q = m.args()
    tol, basis = price_tolerance(m)
    if answer - max(K - S, 0.0) < MIN_ANSWER_IN_TOLS * tol:
        return None  # (near-)immediate exercise: K - S would answer it without the tree
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    steps = [
        f"T = {m.days}/365 = {g(T)} years; dt = T/{n} = {g(dt)}.",
        f"u = e^(sigma sqrt(dt)) = {g(u)}; d = 1/u = {g(d)}.",
        f"p = (e^((r - q) dt) - d) / (u - d) = {g(p)}; one-step discount e^(-r dt) = {g(disc)}.",
    ]
    if n <= _SHOW_NODES_UP_TO:
        values = [max(K - S * u**j * d ** (n - j), 0.0) for j in range(n + 1)]
        steps.append(f"Payoffs at step {n} (j = 0..{n} up-moves): {[g(v) for v in values]}.")
        for i in range(n - 1, -1, -1):
            values = [
                max(K - S * u**j * d ** (i - j), disc * (p * values[j + 1] + (1 - p) * values[j]))
                for j in range(i + 1)
            ]
            steps.append(f"Step {i} (max of exercise and continuation): {[g(v) for v in values]}.")
    else:
        steps.append(
            f"Backward induction over {n} steps, taking max(K - S_node, continuation) at "
            "every node."
        )
    steps.append(
        f"American put on the tree = {g(answer)} (European on the same tree {g(european)}; "
        f"early-exercise premium {g(answer - european)})."
    )
    conv = conventions(
        "American exercise: the put may be exercised at any of the tree's dates, including "
        f"today. Use a Cox-Ross-Rubinstein binomial tree with n = {n} equal steps: "
        "dt = T/n, u = exp(sigma sqrt(dt)), d = 1/u, risk-neutral up-probability "
        "p = (exp((r - q) dt) - d)/(u - d), and discount factor exp(-r dt) per step."
    )
    text = assemble(
        "An American put option is written on a stock.",
        market_lines(m),
        conv,
        f"What is the value of the American put on this {n}-step tree?",
        tol,
        "the option value in currency units",
    )
    inputs = m.inputs("put") | {"steps": n, "exercise": "american"}
    return Draft(text, conv, inputs, answer, tol, basis, steps)
