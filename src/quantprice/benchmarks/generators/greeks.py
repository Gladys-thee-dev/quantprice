r"""Topics ``greek_delta``, ``greek_gamma``, ``greek_vega``, ``greek_theta``, ``greek_rho``.

Generalised BSM Greeks with continuous yield q (Hull, chapter "The Greek Letters";
Haug 2007, section 2.1), with :math:`\hat S = S e^{-qT}`, :math:`\hat K = K e^{-rT}`,
:math:`w = +1` (call) or :math:`-1` (put):

=====  ==============================================================================
delta  :math:`w\,e^{-qT} N(w d_1)`
gamma  :math:`e^{-qT}\varphi(d_1) / (S\sigma\sqrt T)`
vega   :math:`\hat S\varphi(d_1)\sqrt T`
theta  :math:`-\hat S\varphi(d_1)\sigma/(2\sqrt T) + w q\hat S N(w d_1) - w r\hat K N(w d_2)`
       (= dV/dt, calendar time passing)
rho    :math:`w\,K T e^{-rT} N(w d_2)`
=====  ==============================================================================

Units are part of the problem. Easy and medium problems ask for vega and rho per 1
percentage point and theta per calendar day (per-year value / 365). Hard problems ask
for the raw derivatives (per 1.00 change; theta per year). Both are stated in the text.

Tolerance magnitude M is the largest the Greek can be at these S, T, r, q and sigma,
each rescaled by the unit factor:

* delta: ``e^(-qT)``;
* gamma: ``e^(-qT) / (S sigma sqrt(2 pi T))``;
* vega: ``S e^(-qT) sqrt(T / (2 pi))``;
* theta: the sum of the absolute values of its three terms;
* rho: ``K T e^(-rT)``.

Verified answer: :func:`quantprice.bsm.greeks`. Independent check in the tests:
50-digit numerical differentiation of an mpmath price (``tests/reference_mpmath.py``).
"""

import math
import random
from collections.abc import Callable

from quantprice import bsm
from quantprice.benchmarks.generators._common import (
    EUROPEAN_BSM,
    Draft,
    N,
    assemble,
    band,
    bsm_terms,
    conventions,
    g,
    market_lines,
    phi,
    sample_market,
)
from quantprice.benchmarks.schema import Difficulty
from quantprice.bsm import OptionType
from quantprice.conventions import DAYS_PER_YEAR, PERCENTAGE_POINT

GREEKS = ("delta", "gamma", "vega", "theta", "rho")

_DEFINITION = {
    "delta": "delta = dV/dS, the sensitivity of the option value to the spot price",
    "gamma": "gamma = d^2V/dS^2, the sensitivity of delta to the spot price",
    "vega": "vega = dV/dsigma, the sensitivity of the option value to volatility",
    "theta": (
        "theta = dV/dt, the change in option value as calendar time passes with all else "
        "fixed (so a long option usually has negative theta)"
    ),
    "rho": "rho = dV/dr, the sensitivity of the option value to the interest rate",
}

# (factor applied to the raw derivative, description of the unit), for raw=False / True.
_UNITS: dict[str, tuple[tuple[float, str], tuple[float, str]]] = {
    "vega": (
        (PERCENTAGE_POINT, "per 1 percentage point of volatility (0.01 x dV/dsigma)"),
        (1.0, "per 1.00 change in sigma (e.g. 0.20 -> 1.20)"),
    ),
    "rho": (
        (PERCENTAGE_POINT, "per 1 percentage point of r (0.01 x dV/dr)"),
        (1.0, "per 1.00 change in r"),
    ),
    "theta": (
        (1.0 / DAYS_PER_YEAR, "per calendar day (the per-year value divided by 365)"),
        (1.0, "per year"),
    ),
}


def make_greek(name: str) -> Callable[[random.Random, Difficulty], Draft]:
    """Generator for one Greek (one of :data:`GREEKS`)."""
    if name not in GREEKS:
        raise ValueError(f"unknown Greek {name!r}")

    def make(rng: random.Random, difficulty: Difficulty) -> Draft:
        m = sample_market(rng, difficulty)
        kind: OptionType = rng.choice(["call", "put"])
        w = 1.0 if kind == "call" else -1.0
        S, K, T, r, sigma, q = m.args()
        res = bsm.greeks(kind, S, K, T, r, sigma, q)
        factor, unit = _UNITS[name][difficulty == "hard"] if name in _UNITS else (1.0, "")
        t = bsm_terms(m)
        nd1, nd2, pdf = N(w * t.d1), N(w * t.d2), phi(t.d1)
        steps = list(t.lines)
        if name == "delta":
            answer = res.delta
            magnitude = math.exp(-q * T)
            steps.append(f"delta = w e^(-qT) N(w d1) with w = {w:+g}: {g(answer)}.")
        elif name == "gamma":
            answer = res.gamma
            magnitude = math.exp(-q * T) / (S * sigma * math.sqrt(2 * math.pi * T))
            steps.append(f"phi(d1) = {g(pdf)}.")
            steps.append(f"gamma = e^(-qT) phi(d1) / (S sigma sqrt(T)) = {g(answer)}.")
        elif name == "vega":
            answer = res.vega * factor
            magnitude = m.S_hat * math.sqrt(T / (2 * math.pi)) * factor
            steps.append(f"phi(d1) = {g(pdf)}.")
            steps.append(f"dV/dsigma = S e^(-qT) phi(d1) sqrt(T) = {g(res.vega)}.")
        elif name == "theta":
            answer = res.theta * factor
            terms = (
                -m.S_hat * pdf * sigma / (2 * math.sqrt(T)),
                w * q * m.S_hat * nd1,
                -w * r * m.K_hat * nd2,
            )
            magnitude = sum(abs(x) for x in terms) * factor
            steps.append(f"phi(d1) = {g(pdf)}; N(w d1) = {g(nd1)}; N(w d2) = {g(nd2)}; w = {w:+g}.")
            steps.append(
                "theta per year = -S e^(-qT) phi(d1) sigma / (2 sqrt(T)) + w q S e^(-qT) N(w d1)"
                f" - w r K e^(-rT) N(w d2) = {g(terms[0])} + {g(terms[1])} + {g(terms[2])}"
                f" = {g(res.theta)}."
            )
        else:
            answer = res.rho * factor
            magnitude = K * T * math.exp(-r * T) * factor
            steps.append(f"N(w d2) = {g(nd2)} with w = {w:+g}.")
            steps.append(f"dV/dr = w K T e^(-rT) N(w d2) = {g(res.rho)}.")
        if factor != 1.0:
            steps.append(f"In the requested unit ({unit}): {g(answer)}.")
        tol = band(magnitude)
        basis = f"1e-4 * (largest possible |{name}| at these inputs, in the requested unit)"
        conv = conventions(EUROPEAN_BSM, *([f"Report {name} {unit}."] if unit else []))
        text = assemble(
            f"A European {kind} option is written on a stock.",
            market_lines(m),
            conv,
            f"Compute the option's {_DEFINITION[name]}" + (f", {unit}." if unit else "."),
            tol,
            name,
        )
        return Draft(text, conv, m.inputs(kind), answer, tol, basis, steps)

    return make
