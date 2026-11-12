"""Problem generators, one module per topic, and the registry that runs them.

``generate(topic, n, seed)`` returns ``n`` problems. Problem ``i`` has difficulty
``("easy", "medium", "hard")[i % 3]`` and its own seeded RNG (see ``_common``), so the
records are reproducible one at a time. Adding problems or topics never changes the
ones already generated.
"""

import random
from collections.abc import Callable, Iterable

from quantprice._version import __version__
from quantprice.benchmarks.generators import (
    asian,
    barrier,
    binomial,
    bs_price,
    greeks,
    implied_vol,
    parity,
)
from quantprice.benchmarks.generators._common import MIN_ANSWER_IN_TOLS, Draft
from quantprice.benchmarks.schema import Difficulty, Problem, Tolerance

__all__ = ["DIFFICULTIES", "TOPICS", "generate", "generate_all"]

Generator = Callable[[random.Random, Difficulty], Draft | None]

DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard")

TOPICS: dict[str, Generator] = {
    "bs_price": bs_price.make,
    **{f"greek_{name}": greeks.make_greek(name) for name in greeks.GREEKS},
    "put_call_parity": parity.make,
    "binomial_american_put": binomial.make,
    "implied_vol": implied_vol.make,
    "geometric_asian": asian.make,
    "barrier_in_out_parity": barrier.make,
}

_MAX_TRIES = 1000


def _check_count(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be an integer >= 0; got {value!r}")


def generate(topic: str, n: int, seed: int) -> list[Problem]:
    """``n`` problems for ``topic`` from ``seed``.

    A draw is rejected and redrawn (from the same RNG) in two cases: the generator
    returns ``None`` (inputs it can't use, e.g. a tree with p outside [0, 1]), or the
    answer is smaller than ``MIN_ANSWER_IN_TOLS`` tolerances, so answering 0 can't score.
    """
    if topic not in TOPICS:
        raise ValueError(f"unknown topic {topic!r}; choose from {sorted(TOPICS)}")
    _check_count("n", n)
    _check_count("seed", seed)
    make = TOPICS[topic]
    out = []
    for index in range(n):
        difficulty = DIFFICULTIES[index % len(DIFFICULTIES)]
        rng = random.Random(f"quantprice-bench/{seed}/{topic}/{index}")
        for _ in range(_MAX_TRIES):
            draft = make(rng, difficulty)
            if draft is not None and abs(draft.answer) >= MIN_ANSWER_IN_TOLS * draft.tol_abs:
                break
        else:  # pragma: no cover - would mean a generator's ranges are broken
            raise RuntimeError(f"{topic}: no usable draw in {_MAX_TRIES} tries")
        out.append(
            Problem(
                id=f"{topic}-{seed}-{index:04d}",
                topic=topic,
                difficulty=difficulty,
                problem=draft.problem,
                conventions=draft.conventions,
                inputs=draft.inputs,
                verified_answer=float(draft.answer),
                tolerance=Tolerance(float(draft.tol_abs), draft.tol_basis),
                solution_steps=draft.steps,
                library_version=__version__,
            )
        )
    return out


def generate_all(seed: int, per_topic: int, topics: Iterable[str] | None = None) -> list[Problem]:
    """``per_topic`` problems for each topic (all topics by default), topic by topic."""
    chosen = list(TOPICS) if topics is None else list(topics)
    return [p for topic in chosen for p in generate(topic, per_topic, seed)]
