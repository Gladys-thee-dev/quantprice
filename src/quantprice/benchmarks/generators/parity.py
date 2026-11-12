"""Topic ``put_call_parity``: infer a European put (call) price from the call (put).

Put-call parity for European options on an asset with continuous yield q (Merton 1973;
Hull, chapter "Properties of Stock Options"):

    C - P = S e^(-qT) - K e^(-rT).

It is model-free: no volatility is given, and none is needed. The quoted price is a BSM
price at a hidden volatility, rounded to 6 decimals. The verified answer is computed by
parity from the rounded quote, which is exactly what the solver sees.

Independent check in the tests: the answer equals the mpmath BSM price of the other
option at the quote's implied volatility, to within the quote's rounding.
"""

import random

from quantprice import bsm
from quantprice.benchmarks.generators._common import (
    Draft,
    assemble,
    conventions,
    g,
    market_lines,
    price_tolerance,
    sample_market,
    shown,
)
from quantprice.benchmarks.schema import Difficulty
from quantprice.bsm import OptionType


def make(rng: random.Random, difficulty: Difficulty) -> Draft:
    m = sample_market(rng, difficulty)
    given: OptionType = rng.choice(["call", "put"])
    asked: OptionType = "put" if given == "call" else "call"
    quote = shown(bsm.price(given, *m.args()))
    forward_gap = m.S_hat - m.K_hat  # C - P
    answer = quote - forward_gap if given == "call" else quote + forward_gap
    tol, basis = price_tolerance(m)
    conv = conventions(
        "European exercise (at expiry only). No arbitrage; frictionless market. The call "
        "and put have the same underlying, strike and expiry."
    )
    inputs = m.inputs(asked, sigma=False) | {"given_type": given, "given_price": quote}
    text = assemble(
        f"A European {given} and a European {asked} are written on the same stock.",
        [*market_lines(m, sigma=False), f"The {given} trades at {quote:.6f}."],
        conv,
        f"What is the arbitrage-free price of the {asked}?",
        tol,
        f"the {asked} price in currency units",
    )
    steps = [
        f"T = {m.days}/365 = {g(m.T)} years.",
        f"S e^(-qT) = {g(m.S_hat)}; K e^(-rT) = {g(m.K_hat)}.",
        f"Put-call parity: C - P = S e^(-qT) - K e^(-rT) = {g(forward_gap)}.",
        (
            f"P = C - (S e^(-qT) - K e^(-rT)) = {g(answer)}."
            if given == "call"
            else f"C = P + (S e^(-qT) - K e^(-rT)) = {g(answer)}."
        ),
    ]
    return Draft(text, conv, inputs, answer, tol, basis, steps)
