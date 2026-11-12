"""Topic ``barrier_in_out_parity``: knock-in value from a knock-out quote.

With no rebate and continuous monitoring, exactly one of the knock-in and the knock-out
with the same terms (type, strike, barrier, expiry) is alive at expiry, so (Reiner and
Rubinstein 1991; Haug 2007, section 4.17.1)

    knock-in + knock-out = vanilla European option.

The knock-out quote is the Reiner-Rubinstein price from :func:`quantprice.barrier.price`,
rounded to 6 decimals. The answer is ``BSM vanilla - quote``. The barrier is strictly on
the far side of the spot (down: ``H < S``; up: ``H > S``), so neither option has been
triggered yet.

Independent checks in the tests: the vanilla is recomputed in 50-digit mpmath, and the
answer is compared with the Reiner-Rubinstein knock-in closed form, which shares no
code path with parity. It agrees to within the quote's rounding.
"""

import random

from quantprice import barrier, bsm
from quantprice.benchmarks.generators._common import (
    EUROPEAN_BSM,
    Draft,
    N,
    assemble,
    bsm_terms,
    conventions,
    g,
    market_lines,
    num,
    price_tolerance,
    sample_market,
    shown,
)
from quantprice.benchmarks.schema import Difficulty
from quantprice.bsm import OptionType

_DOWN = [0.75, 0.8, 0.85, 0.9, 0.95]
_UP = [1.05, 1.1, 1.15, 1.2, 1.25]


def make(rng: random.Random, difficulty: Difficulty) -> Draft | None:
    m = sample_market(rng, difficulty)
    kind: OptionType = rng.choice(["call", "put"])
    direction = rng.choice(["down", "up"])
    H = float(round(m.S * rng.choice(_DOWN if direction == "down" else _UP)))
    if (direction == "down" and not H < m.S) or (direction == "up" and not H > m.S):
        return None
    ko_type, ki_type = f"{direction}-and-out", f"{direction}-and-in"
    S, K, T, r, sigma, q = m.args()
    quote = shown(barrier.price(kind, ko_type, S, K, H, T, r, sigma, q))  # type: ignore[arg-type]
    vanilla = bsm.price(kind, *m.args())
    answer = vanilla - quote
    tol, basis = price_tolerance(m)
    t = bsm_terms(m)
    vanilla_line = (
        f"Vanilla call = S e^(-qT) N(d1) - K e^(-rT) N(d2) with N(d1) = {g(N(t.d1))}, "
        f"N(d2) = {g(N(t.d2))}: {g(vanilla)}."
        if kind == "call"
        else f"Vanilla put = K e^(-rT) N(-d2) - S e^(-qT) N(-d1) with N(-d1) = {g(N(-t.d1))}, "
        f"N(-d2) = {g(N(-t.d2))}: {g(vanilla)}."
    )
    steps = [
        "In-out parity (no rebate, same terms, continuous monitoring): exactly one of the "
        "two options is alive at expiry, so knock-in + knock-out = vanilla.",
        *t.lines,
        vanilla_line,
        f"{ki_type} {kind} = {g(vanilla)} - {quote:.6f} = {g(answer)}.",
    ]
    conv = conventions(
        EUROPEAN_BSM,
        "The barrier is monitored continuously from today to expiry. Neither option pays "
        "a rebate. A knock-out option becomes worthless the first time the spot touches "
        "the barrier; a knock-in option becomes a vanilla option at that moment and "
        "otherwise expires worthless.",
    )
    text = assemble(
        f"Two European barrier {kind} options are written on a stock with identical "
        f"terms: a {ko_type} {kind} and a {ki_type} {kind}.",
        [
            *market_lines(m),
            f"Barrier H = {num(H)} ({'below' if direction == 'down' else 'above'} the spot; "
            "not yet touched).",
            f"The {ko_type} {kind} is worth {quote:.6f}.",
        ],
        conv,
        f"What is the value of the {ki_type} {kind}?",
        tol,
        "the option value in currency units",
    )
    inputs = m.inputs(kind) | {"barrier_type": ki_type, "H": H, "knock_out_price": quote}
    return Draft(text, conv, inputs, answer, tol, basis, steps)
