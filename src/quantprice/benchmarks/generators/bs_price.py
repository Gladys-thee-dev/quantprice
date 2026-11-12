"""Topic ``bs_price``: price a European call or put under Black-Scholes-Merton.

Formula (generalised BSM with continuous yield q; Merton 1973, as in Hull, *Options,
Futures, and Other Derivatives*, chapter "Black-Scholes-Merton model", and Haug 2007,
section 1.1.6):

    C = S e^(-qT) N(d1) - K e^(-rT) N(d2),    P = K e^(-rT) N(-d2) - S e^(-qT) N(-d1).

Verified answer: :func:`quantprice.bsm.price`. Independent check in the tests: a 50-digit
mpmath evaluation of the textbook formula.
"""

import random

from quantprice import bsm
from quantprice.benchmarks.generators._common import (
    EUROPEAN_BSM,
    Draft,
    N,
    assemble,
    bsm_terms,
    conventions,
    g,
    market_lines,
    price_tolerance,
    sample_market,
)
from quantprice.benchmarks.schema import Difficulty
from quantprice.bsm import OptionType


def make(rng: random.Random, difficulty: Difficulty) -> Draft:
    m = sample_market(rng, difficulty)
    kind: OptionType = rng.choice(["call", "put"])
    answer = bsm.price(kind, *m.args())
    tol, basis = price_tolerance(m)
    t = bsm_terms(m)
    if kind == "call":
        last = [
            f"N(d1) = {g(N(t.d1))}; N(d2) = {g(N(t.d2))}.",
            f"C = S e^(-qT) N(d1) - K e^(-rT) N(d2) = {g(answer)}.",
        ]
    else:
        last = [
            f"N(-d1) = {g(N(-t.d1))}; N(-d2) = {g(N(-t.d2))}.",
            f"P = K e^(-rT) N(-d2) - S e^(-qT) N(-d1) = {g(answer)}.",
        ]
    conv = conventions(EUROPEAN_BSM)
    text = assemble(
        f"A European {kind} option is written on a stock.",
        market_lines(m),
        conv,
        f"What is the fair value of the {kind} today?",
        tol,
        "the option value in currency units",
    )
    return Draft(text, conv, m.inputs(kind), answer, tol, basis, t.lines + last)
