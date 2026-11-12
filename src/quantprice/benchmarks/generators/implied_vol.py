"""Topic ``implied_vol``: back out the Black-Scholes-Merton volatility from a price.

The quote is a BSM price at a hidden volatility, rounded to 6 decimals. The answer is
the implied volatility of the *rounded* quote, from :func:`quantprice.implied_vol.implied_vol`
(Newton-Raphson with a Brent fallback).

Well-posedness: implied volatility is ill-conditioned where vega is tiny (see
:mod:`quantprice.implied_vol`). Draws are kept only if vega is at least 10% of its
maximum ``S e^(-qT) sqrt(T / (2 pi))``. Then the 6-decimal rounding moves sigma by less
than 1e-6, far inside the band of 1e-4 absolute (0.01 volatility points). Draws are also
rejected if the recovered sigma differs from the hidden one by more than 1/100 of the band.

Independent check in the tests: the 50-digit mpmath price is below the quote at
``answer - tol`` and above it at ``answer + tol``. Price increases with sigma, so the
true implied vol lies inside the band.
"""

import math
import random

from quantprice import bsm
from quantprice.benchmarks.generators._common import (
    EUROPEAN_BSM,
    Draft,
    assemble,
    band,
    conventions,
    g,
    market_lines,
    sample_market,
    shown,
)
from quantprice.benchmarks.schema import Difficulty
from quantprice.bsm import OptionType
from quantprice.implied_vol import solve_implied_vol

MIN_VEGA_FRACTION = 0.1
_SHOW_ITERATES = 12


def make(rng: random.Random, difficulty: Difficulty) -> Draft | None:
    m = sample_market(rng, difficulty)
    kind: OptionType = rng.choice(["call", "put"])
    S, K, T, r, sigma, q = m.args()
    max_vega = m.S_hat * math.sqrt(T / (2 * math.pi))
    if bsm.greeks(kind, *m.args()).vega < MIN_VEGA_FRACTION * max_vega:
        return None
    quote = shown(bsm.price(kind, *m.args()))
    try:
        res = solve_implied_vol(kind, quote, S, K, T, r, q)
    except ValueError:
        return None
    tol = band(1.0)
    if abs(res.sigma - sigma) > tol / 100:
        return None
    lower = max(m.S_hat - m.K_hat, 0.0) if kind == "call" else max(m.K_hat - m.S_hat, 0.0)
    upper = m.S_hat if kind == "call" else m.K_hat
    steps = [
        f"T = {m.days}/365 = {g(T)} years; S e^(-qT) = {g(m.S_hat)}; K e^(-rT) = {g(m.K_hat)}.",
        f"No-arbitrage range for the {kind}: ({g(lower)}, {g(upper)}). The quote {quote:.6f} "
        "lies inside it, and the BSM price increases strictly with sigma, so exactly one "
        "sigma matches.",
    ]
    # Plain Newton from the Brenner-Subrahmanyam start, shown when it converges cleanly.
    sig = math.sqrt(2 * math.pi / T) * quote / m.S_hat
    iterates: list[str] = []
    for _ in range(_SHOW_ITERATES):
        if not 0.0 < sig < 10.0:
            break
        res_k = bsm.greeks(kind, S, K, T, r, sig, q)
        iterates.append(f"sigma = {g(sig)}: V = {g(res_k.price)}, vega = {g(res_k.vega)}")
        step = (res_k.price - quote) / res_k.vega
        sig -= step
        if abs(step) <= 1e-10 * sig:
            break  # converged: further iterates repeat the same digits
    if iterates and abs(sig - res.sigma) <= 1e-9:
        steps.append(
            "Newton-Raphson, sigma <- sigma - (V(sigma) - quote) / vega(sigma), started at the "
            "Brenner-Subrahmanyam approximation sqrt(2 pi / T) * quote / (S e^(-qT)):"
        )
        steps.extend(iterates)
    else:
        steps.append(
            "Solve V(sigma) = quote with a bracketing root finder (e.g. Brent's method), "
            "since the price is monotone in sigma."
        )
    steps.append(f"Implied volatility sigma = {g(res.sigma)}.")
    conv = conventions(EUROPEAN_BSM, "Report sigma as a decimal per year (0.25 means 25%).")
    text = assemble(
        f"A European {kind} option is written on a stock.",
        [*market_lines(m, sigma=False), f"The {kind} trades at {quote:.6f}."],
        conv,
        "What volatility sigma, used in the Black-Scholes-Merton formula, reproduces this price?",
        tol,
        "sigma",
    )
    inputs = m.inputs(kind, sigma=False) | {"price": quote}
    return Draft(
        text, conv, inputs, res.sigma, tol, "1e-4 absolute in sigma (0.01 vol points)", steps
    )
