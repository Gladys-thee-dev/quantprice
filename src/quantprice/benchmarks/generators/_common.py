"""Shared pieces for the problem generators: parameter sampling, text and tolerances.

Randomness
----------
Each problem is drawn from its own ``random.Random`` seeded with the string
``"quantprice-bench/<seed>/<topic>/<index>"``. Python guarantees that string seeds
(version 2, via SHA-512) and the integer methods used here (``randrange`` and
``choice``) give the same sequence across versions and platforms. NumPy's
``Generator`` makes no such promise between releases. Only integer draws are used, so
every input is a short decimal (e.g. ``r = 13/400 = 3.25%``) and the problem text shows
it exactly.

Difficulty
----------
Problems cycle easy, medium, hard by index. The tiers set the parameter ranges:

======  ==========================================================================
easy    S in [80, 120], K within 10% of S, 91-365 days, r in [0, 6%], q = 0,
        sigma in [15%, 40%]
medium  S in [20, 300], K within 20% of S, 30-730 days, r in [0, 7%],
        q in [0.25%, 5%], sigma in [10%, 60%]
hard    S in [5, 1000], K within 30% of S, 7-1825 days, r in [-2%, 8%] (negative
        rates included), q in [0, 8%], sigma in [5%, 100%]
======  ==========================================================================

Topic generators may add their own difficulty knobs (tree steps, number of fixings).

Tolerances
----------
Each answer is graded against an absolute band ``REL * M``, floored at 1e-12, where
``REL = 1e-4`` and ``M`` is the quantity's natural magnitude:

* prices: ``max(S e^{-qT}, K e^{-rT})`` (about 0.01 on a 100 stock);
* Greeks: the largest value the Greek can take for the given S, T, r, q and sigma
  (see each generator).

This scale-relative band is tight enough to separate convention errors (ignoring q, or
using the wrong day count or compounding) from rounding of intermediate values to 5-6
significant figures. A problem is resampled if ``|answer| < 10 * tol``, so a solver can't
pass by answering 0.
"""

import math
import random
from dataclasses import dataclass
from typing import NamedTuple

from scipy.special import ndtr

from quantprice.benchmarks.schema import Difficulty, JSONValue
from quantprice.bsm import OptionType
from quantprice.conventions import CONVENTIONS_TEXT, DAYS_PER_YEAR

REL = 1e-4
FLOOR = 1e-12
MIN_ANSWER_IN_TOLS = 10.0
SHOWN_DECIMALS = 6  # prices quoted inside a problem (parity, implied vol, barrier)


@dataclass(frozen=True, slots=True)
class Draft:
    """A problem before the registry adds id, topic, difficulty and version."""

    problem: str
    conventions: str
    inputs: dict[str, JSONValue]
    answer: float
    tol_abs: float
    tol_basis: str
    steps: list[str]


@dataclass(frozen=True, slots=True)
class Market:
    S: float
    K: float
    days: int
    r: float
    sigma: float
    q: float

    @property
    def T(self) -> float:
        return self.days / DAYS_PER_YEAR

    @property
    def S_hat(self) -> float:
        return self.S * math.exp(-self.q * self.T)

    @property
    def K_hat(self) -> float:
        return self.K * math.exp(-self.r * self.T)

    @property
    def scale(self) -> float:
        return max(self.S_hat, self.K_hat)

    def args(self) -> tuple[float, float, float, float, float, float]:
        return self.S, self.K, self.T, self.r, self.sigma, self.q

    def inputs(self, option_type: OptionType, *, sigma: bool = True) -> dict[str, JSONValue]:
        d: dict[str, JSONValue] = {
            "option_type": option_type,
            "S": self.S,
            "K": self.K,
            "T_days": self.days,
            "T": self.T,
            "r": self.r,
            "q": self.q,
        }
        if sigma:
            d["sigma"] = self.sigma
        return d


def _strike(S: int, moneyness: float) -> float:
    return float(max(1, round(S * moneyness)))


def sample_market(rng: random.Random, difficulty: Difficulty) -> Market:
    """Draw market inputs for a difficulty tier (see the module docstring)."""
    if difficulty == "easy":
        S = rng.randrange(80, 121)
        K = _strike(S, rng.choice([0.9, 0.95, 1.0, 1.05, 1.1]))
        days = rng.choice([91, 182, 273, 365])
        r, q = rng.randrange(0, 25) / 400, 0.0
        sigma = rng.randrange(15, 41) / 100
    elif difficulty == "medium":
        S = rng.randrange(20, 301)
        K = _strike(S, rng.choice([0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]))
        days = rng.choice([30, 61, 91, 182, 273, 365, 547, 730])
        r, q = rng.randrange(0, 29) / 400, rng.randrange(1, 21) / 400
        sigma = rng.randrange(10, 61) / 100
    else:
        S = rng.randrange(5, 1001)
        K = _strike(S, rng.choice([0.7 + 0.05 * i for i in range(13)]))
        days = rng.choice([7, 14, 30, 61, 91, 182, 365, 547, 730, 1095, 1825])
        r, q = rng.randrange(-8, 33) / 400, rng.randrange(0, 33) / 400
        sigma = rng.randrange(5, 101) / 100
    return Market(float(S), K, days, r, sigma, q)


# ------------------------------------------------------------------------ formatting


def num(x: float) -> str:
    """An exact short decimal input (S, K, H), e.g. 100 or 97.5."""
    return f"{x:g}"


def pct(x: float) -> str:
    """A rate or volatility as a percentage, exactly: 13/400 -> '3.25%'."""
    return f"{round(100.0 * x, 8):g}%"


def g(x: float) -> str:
    """An intermediate value in solution steps, to 10 significant figures."""
    return f"{x:.10g}"


def shown(x: float) -> float:
    """Round a price quoted in a problem to SHOWN_DECIMALS places. Answers are computed
    from the rounded value, which is exactly what the solver sees."""
    return float(f"{x:.{SHOWN_DECIMALS}f}")


def market_lines(m: Market, *, sigma: bool = True) -> list[str]:
    lines = [
        f"Spot price S = {num(m.S)}.",
        f"Strike K = {num(m.K)}.",
        f"Time to expiry: {m.days} calendar days, so T = {m.days}/365 years (ACT/365).",
        f"Risk-free rate r = {pct(m.r)} per year, continuously compounded.",
        (
            "The underlying pays no dividends (q = 0)."
            if m.q == 0.0
            else f"The underlying pays a continuous dividend yield q = {pct(m.q)} per year."
        ),
    ]
    if sigma:
        lines.append(f"Volatility sigma = {pct(m.sigma)} per year.")
    return lines


def conventions(*extra: str) -> str:
    return " ".join([CONVENTIONS_TEXT, *extra])


EUROPEAN_BSM = (
    "European exercise (at expiry only). Black-Scholes-Merton model: the underlying "
    "follows geometric Brownian motion under the risk-neutral measure with constant r, "
    "q and sigma."
)


def answer_line(tol: float, what: str) -> str:
    return (
        f"Give {what} as a single number. "
        f"Answers within {tol:.3g} of the verified value are correct."
    )


def assemble(lead: str, facts: list[str], conv: str, question: str, tol: float, what: str) -> str:
    return "\n".join([lead, *facts, f"Conventions: {conv}", question, answer_line(tol, what)])


# --------------------------------------------------------------------- tolerances


def band(magnitude: float) -> float:
    return max(REL * magnitude, FLOOR)


def price_tolerance(m: Market) -> tuple[float, str]:
    return band(m.scale), "1e-4 * max(S e^(-qT), K e^(-rT)), floor 1e-12"


# ------------------------------------------------------------ BSM worked-solution terms


class BSMTerms(NamedTuple):
    d1: float
    d2: float
    lines: list[str]


def bsm_terms(m: Market, sigma: float | None = None) -> BSMTerms:
    """d1, d2 and the solution lines leading to them (generalised BSM with yield q)."""
    sig = m.sigma if sigma is None else sigma
    sqrt_T = math.sqrt(m.T)
    d1 = (math.log(m.S / m.K) + (m.r - m.q + 0.5 * sig * sig) * m.T) / (sig * sqrt_T)
    d2 = d1 - sig * sqrt_T
    return BSMTerms(
        d1,
        d2,
        [
            f"T = {m.days}/365 = {g(m.T)} years.",
            f"S e^(-qT) = {g(m.S_hat)}; K e^(-rT) = {g(m.K_hat)}.",
            f"d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T)) = {g(d1)}.",
            f"d2 = d1 - sigma sqrt(T) = {g(d2)}.",
        ],
    )


def N(x: float) -> float:
    return float(ndtr(x))


def phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
