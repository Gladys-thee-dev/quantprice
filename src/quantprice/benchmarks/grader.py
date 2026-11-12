"""Score submitted answers against verified benchmark answers.

Answers file: JSONL, one object per line, ``{"id": "<problem id>", "answer": <number>}``.
Other keys (e.g. a model's reasoning) are allowed and ignored.

Each problem gets exactly one status:

=========  ==================================================================
correct    a finite JSON number with ``|answer - verified_answer| <= tolerance.abs``
wrong      a finite number outside the band
invalid    present but not a finite number (a string, null, bool or non-finite)
missing    no line with this problem's id
=========  ==================================================================

Accuracy is ``correct / total``, so missing and invalid answers count against the
score. Answers whose ids match no problem are listed as ``unknown_ids`` and are not
scored. A duplicate id in the answers file is an error, since it is ambiguous which
answer to grade.
"""

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from quantprice.benchmarks.schema import Difficulty, Problem

__all__ = ["GradedProblem", "Report", "Score", "grade", "load_answers"]

Status = Literal["correct", "wrong", "invalid", "missing"]


@dataclass(frozen=True, slots=True)
class GradedProblem:
    id: str
    topic: str
    difficulty: Difficulty
    verified_answer: float
    tolerance: float
    answer: Any
    status: Status


@dataclass(slots=True)
class Score:
    total: int = 0
    correct: int = 0
    wrong: int = 0
    invalid: int = 0
    missing: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else math.nan

    def add(self, status: Status) -> None:
        self.total += 1
        setattr(self, status, getattr(self, status) + 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "correct": self.correct,
            "wrong": self.wrong,
            "invalid": self.invalid,
            "missing": self.missing,
            "accuracy": None if self.total == 0 else self.accuracy,
        }


@dataclass(slots=True)
class Report:
    results: list[GradedProblem]
    overall: Score
    by_topic: dict[str, Score]
    by_difficulty: dict[str, Score]
    unknown_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall.to_dict(),
            "by_topic": {k: v.to_dict() for k, v in self.by_topic.items()},
            "by_difficulty": {k: v.to_dict() for k, v in self.by_difficulty.items()},
            "unknown_ids": self.unknown_ids,
        }

    def format_table(self) -> str:
        width = max([len("overall"), *(len(t) for t in self.by_topic)])
        head = (
            f"{'topic':<{width}}  {'total':>5}  {'correct':>7}  {'wrong':>5}  "
            f"{'invalid':>7}  {'missing':>7}  {'accuracy':>8}"
        )

        def row(name: str, s: Score) -> str:
            acc = "n/a" if s.total == 0 else f"{100 * s.accuracy:.1f}%"
            return (
                f"{name:<{width}}  {s.total:>5}  {s.correct:>7}  {s.wrong:>5}  "
                f"{s.invalid:>7}  {s.missing:>7}  {acc:>8}"
            )

        lines = [head, "-" * len(head)]
        lines += [row(t, s) for t, s in self.by_topic.items()]
        lines += ["-" * len(head)]
        lines += [row(d, s) for d, s in self.by_difficulty.items()]
        lines += ["-" * len(head), row("overall", self.overall)]
        if self.unknown_ids:
            lines.append(f"{len(self.unknown_ids)} answer(s) with unknown ids were ignored.")
        return "\n".join(lines)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def grade(problems: Iterable[Problem], answers: Mapping[str, Any]) -> Report:
    """Grade ``answers`` (id -> submitted value) against ``problems``."""
    results: list[GradedProblem] = []
    overall = Score()
    by_topic: dict[str, Score] = {}
    by_difficulty: dict[str, Score] = {d: Score() for d in ("easy", "medium", "hard")}
    ids: set[str] = set()
    for p in problems:
        if p.id in ids:
            raise ValueError(f"duplicate problem id {p.id!r}")
        ids.add(p.id)
        status: Status
        if p.id not in answers:
            status, value = "missing", None
        else:
            value = answers[p.id]
            if not _is_number(value):
                status = "invalid"
            elif p.tolerance.accepts(float(value), p.verified_answer):
                status = "correct"
            else:
                status = "wrong"
        results.append(
            GradedProblem(
                p.id, p.topic, p.difficulty, p.verified_answer, p.tolerance.abs, value, status
            )
        )
        overall.add(status)
        by_topic.setdefault(p.topic, Score()).add(status)
        by_difficulty[p.difficulty].add(status)
    unknown = sorted(set(answers) - ids)
    return Report(results, overall, by_topic, by_difficulty, unknown)


def load_answers(path: str | Path) -> dict[str, Any]:
    """Read an answers JSONL file into ``{id: answer}``. Values are not checked here;
    :func:`grade` marks non-numeric ones invalid.

    Raises:
        ValueError: naming the line, for invalid JSON, a line that is not an object with
            a string ``id`` and an ``answer`` key, or a duplicate id.
    """
    out: dict[str, Any] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
            if not (isinstance(obj, dict) and isinstance(obj.get("id"), str) and "answer" in obj):
                raise ValueError(
                    f"{path}:{lineno}: expected an object with a string 'id' and an 'answer'"
                )
            if obj["id"] in out:
                raise ValueError(f"{path}:{lineno}: duplicate answer for id {obj['id']!r}")
            out[obj["id"]] = obj["answer"]
    return out
