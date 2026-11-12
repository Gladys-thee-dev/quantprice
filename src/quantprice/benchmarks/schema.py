"""Benchmark problem record and its JSONL encoding.

One problem per line, as a JSON object with exactly these keys:

==================  =====================================================================
``id``              unique string, ``"<topic>-<seed>-<index>"``
``topic``           generator name, e.g. ``"bs_price"`` or ``"greek_vega"``
``difficulty``      ``"easy"``, ``"medium"`` or ``"hard"``
``problem``         self-contained question text. It states every convention needed for
                    a unique answer, so it can be given to a solver on its own.
``conventions``     the conventions stated in the problem, repeated as a separate field
``inputs``          the numbers given in the problem (JSON numbers, strings or lists)
``verified_answer`` the answer computed by quantprice from exactly those inputs
``tolerance``       ``{"abs": float, "basis": str}``: graded correct iff
                    ``|answer - verified_answer| <= abs``. ``basis`` says how ``abs`` was set.
``solution_steps``  list of strings: a worked solution with intermediate values
``library_version`` quantprice version that produced the record
==================  =====================================================================

"Verified" means two things. The answer comes from the library's tested
implementation. In addition, the test suite (``tests/test_bench_verify.py``) recomputes
every record from the pinned seeds, and the committed dataset, with an independent
reference: a 50-digit mpmath formula, a plain-loop binomial tree, a covariance-matrix
Asian moment calculation, or the Reiner-Rubinstein knock-in closed form.

Floats are written with Python's ``repr``, which round-trips exactly.
"""

import json
import math
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias, get_args

__all__ = [
    "FIELDS",
    "Difficulty",
    "JSONValue",
    "Problem",
    "Tolerance",
    "read_jsonl",
    "write_jsonl",
]

Difficulty = Literal["easy", "medium", "hard"]
JSONValue: TypeAlias = str | int | float | bool | list[Any] | dict[str, Any] | None

FIELDS: tuple[str, ...] = (
    "id",
    "topic",
    "difficulty",
    "problem",
    "conventions",
    "inputs",
    "verified_answer",
    "tolerance",
    "solution_steps",
    "library_version",
)


@dataclass(frozen=True, slots=True)
class Tolerance:
    """Absolute grading band around the verified answer, and how it was chosen."""

    abs: float
    basis: str

    def __post_init__(self) -> None:
        if not (isinstance(self.abs, float) and math.isfinite(self.abs) and self.abs > 0):
            raise ValueError(f"tolerance.abs must be a finite float > 0; got {self.abs!r}")
        if not self.basis:
            raise ValueError("tolerance.basis must be a non-empty string")

    def accepts(self, answer: float, verified: float) -> bool:
        return math.isfinite(answer) and abs(answer - verified) <= self.abs


@dataclass(frozen=True, slots=True)
class Problem:
    """One benchmark problem. See the module docstring for the meaning of each field."""

    id: str
    topic: str
    difficulty: Difficulty
    problem: str
    conventions: str
    inputs: dict[str, JSONValue]
    verified_answer: float
    tolerance: Tolerance
    solution_steps: list[str] = field(default_factory=list)
    library_version: str = ""

    def __post_init__(self) -> None:
        for name in ("id", "topic", "problem", "conventions", "library_version"):
            value = getattr(self, name)
            if not (isinstance(value, str) and value):
                raise ValueError(f"{name} must be a non-empty string; got {value!r}")
        if self.difficulty not in get_args(Difficulty):
            raise ValueError(f"difficulty must be easy, medium or hard; got {self.difficulty!r}")
        if not isinstance(self.inputs, dict):
            raise ValueError("inputs must be a dict")
        answer = self.verified_answer
        if not (isinstance(answer, float) and math.isfinite(answer)):
            raise ValueError(f"verified_answer must be a finite float; got {answer!r}")
        if not isinstance(self.tolerance, Tolerance):
            raise ValueError("tolerance must be a Tolerance")
        steps = self.solution_steps
        if not (isinstance(steps, list) and steps and all(isinstance(s, str) for s in steps)):
            raise ValueError("solution_steps must be a non-empty list of strings")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: d[k] for k in FIELDS}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Problem":
        if not isinstance(d, dict):
            raise ValueError("a problem record must be a JSON object")
        missing = [k for k in FIELDS if k not in d]
        extra = sorted(set(d) - set(FIELDS))
        if missing or extra:
            raise ValueError(f"bad problem keys: missing {missing}, unexpected {extra}")
        tol = d["tolerance"]
        if not (isinstance(tol, dict) and set(tol) == {"abs", "basis"}):
            raise ValueError("tolerance must be an object with keys 'abs' and 'basis'")
        answer = d["verified_answer"]
        if isinstance(answer, int) and not isinstance(answer, bool):
            answer = float(answer)  # JSON has one number type; 3 and 3.0 are the same value
        abs_tol = tol["abs"]
        if isinstance(abs_tol, int) and not isinstance(abs_tol, bool):
            abs_tol = float(abs_tol)
        return cls(
            id=d["id"],
            topic=d["topic"],
            difficulty=d["difficulty"],
            problem=d["problem"],
            conventions=d["conventions"],
            inputs=d["inputs"],
            verified_answer=answer,
            tolerance=Tolerance(abs_tol, tol["basis"]),
            solution_steps=d["solution_steps"],
            library_version=d["library_version"],
        )


def write_jsonl(problems: Iterable[Problem], path: str | Path) -> int:
    """Write problems one per line (UTF-8, ``\\n`` endings). Returns the count written.

    Raises:
        ValueError: if two problems share an id.
    """
    seen: set[str] = set()
    n = 0
    with Path(path).open("w", encoding="utf-8", newline="\n") as fh:
        for p in problems:
            if p.id in seen:
                raise ValueError(f"duplicate problem id {p.id!r}")
            seen.add(p.id)
            fh.write(p.to_json() + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[Problem]:
    """Read and validate problems written by :func:`write_jsonl`. Blank lines are skipped.

    Raises:
        ValueError: naming the line number, for invalid JSON, an invalid record or a
            duplicate id.
    """
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                problem = Problem.from_dict(json.loads(line))
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
            if problem.id in seen:
                raise ValueError(f"{path}:{lineno}: duplicate problem id {problem.id!r}")
            seen.add(problem.id)
            yield problem
