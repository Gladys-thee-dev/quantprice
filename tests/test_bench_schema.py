"""Phase 5: benchmark problem schema, JSONL round trip, determinism of the generators."""

import json
import math
from pathlib import Path
from typing import Any

import pytest

from quantprice import __version__
from quantprice.benchmarks import (
    TOPICS,
    Problem,
    Tolerance,
    generate,
    generate_all,
    read_jsonl,
    write_jsonl,
)
from quantprice.benchmarks.generators._common import MIN_ANSWER_IN_TOLS
from quantprice.benchmarks.schema import FIELDS
from quantprice.conventions import CONVENTIONS_TEXT

SEED = 2026
PER_TOPIC = 12


@pytest.fixture(scope="module")
def problems() -> list[Problem]:
    return generate_all(SEED, PER_TOPIC)


def record(**overrides: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "id": "t-1-0000",
        "topic": "t",
        "difficulty": "easy",
        "problem": "What is 1 + 1?",
        "conventions": "none",
        "inputs": {"a": 1},
        "verified_answer": 2.0,
        "tolerance": {"abs": 0.5, "basis": "test"},
        "solution_steps": ["1 + 1 = 2"],
        "library_version": "0",
    }
    return d | overrides


# ------------------------------------------------------------------- record shape


def test_every_record_has_exactly_the_specified_fields(problems: list[Problem]) -> None:
    assert len(problems) == PER_TOPIC * len(TOPICS)
    for p in problems:
        assert tuple(json.loads(p.to_json())) == FIELDS
        assert p.library_version == __version__


def test_ids_are_unique_and_topics_balanced(problems: list[Problem]) -> None:
    assert len({p.id for p in problems}) == len(problems)
    for topic in TOPICS:
        mine = [p for p in problems if p.topic == topic]
        assert len(mine) == PER_TOPIC
        assert [p.difficulty for p in mine[:3]] == ["easy", "medium", "hard"]


def test_problem_text_states_conventions_and_tolerance(problems: list[Problem]) -> None:
    """Every problem must be answerable from its text alone."""
    for p in problems:
        assert CONVENTIONS_TEXT in p.conventions
        assert f"Conventions: {p.conventions}" in p.problem
        assert f"within {p.tolerance.abs:.3g}" in p.problem
        assert f"{p.inputs['T_days']} calendar days" in p.problem


def test_answers_are_not_degenerate(problems: list[Problem]) -> None:
    """Resampling guarantees |answer| >= 10 tolerances, so answering 0 always fails."""
    for p in problems:
        assert abs(p.verified_answer) >= MIN_ANSWER_IN_TOLS * p.tolerance.abs
        assert not p.tolerance.accepts(0.0, p.verified_answer)


# ---------------------------------------------------------------- determinism


def test_generation_is_deterministic() -> None:
    a = [p.to_json() for p in generate_all(SEED, 6)]
    b = [p.to_json() for p in generate_all(SEED, 6)]
    assert a == b


def test_problems_are_independent_of_count_and_other_topics() -> None:
    """Problem i depends only on (seed, topic, i)."""
    long = generate("bs_price", 9, SEED)
    short = generate("bs_price", 4, SEED)
    assert [p.to_json() for p in long[:4]] == [p.to_json() for p in short]


def test_different_seeds_give_different_problems() -> None:
    a = generate("bs_price", 6, 1)
    b = generate("bs_price", 6, 2)
    assert {p.problem for p in a}.isdisjoint({p.problem for p in b})


# ------------------------------------------------------------------- JSONL I/O


def test_jsonl_round_trip_is_exact(problems: list[Problem], tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    assert write_jsonl(problems, path) == len(problems)
    back = list(read_jsonl(path))
    assert back == problems
    assert path.read_bytes().count(b"\n") == len(problems)
    assert b"\r" not in path.read_bytes()


def test_write_rejects_duplicate_ids(problems: list[Problem], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate problem id"):
        write_jsonl([problems[0], problems[0]], tmp_path / "dup.jsonl")


def test_read_reports_line_numbers(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(record()) + "\n\n" + "{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:3"):
        list(read_jsonl(path))
    path.write_text((json.dumps(record()) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match=r":2: duplicate problem id"):
        list(read_jsonl(path))


def test_integer_json_numbers_are_accepted() -> None:
    p = Problem.from_dict(record(verified_answer=2, tolerance={"abs": 1, "basis": "b"}))
    assert (p.verified_answer, p.tolerance.abs) == (2.0, 1.0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"extra": 1}, "unexpected"),
        ({"difficulty": "brutal"}, "difficulty"),
        ({"verified_answer": math.nan}, "verified_answer"),
        ({"verified_answer": "2"}, "verified_answer"),
        ({"verified_answer": True}, "verified_answer"),
        ({"tolerance": {"abs": 0.0, "basis": "b"}}, "tolerance.abs"),
        ({"tolerance": {"abs": 1.0}}, "tolerance must be an object"),
        ({"solution_steps": []}, "solution_steps"),
        ({"id": ""}, "id must be"),
    ],
)
def test_invalid_records_are_rejected(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Problem.from_dict(record(**overrides))


def test_missing_field_is_rejected() -> None:
    d = record()
    del d["tolerance"]
    with pytest.raises(ValueError, match=r"missing \['tolerance'\]"):
        Problem.from_dict(d)


def test_tolerance_band_is_inclusive_and_rejects_non_finite() -> None:
    tol = Tolerance(0.5, "b")
    assert tol.accepts(2.5, 2.0)
    assert not tol.accepts(2.5000001, 2.0)
    assert not tol.accepts(math.nan, 2.0)
    assert not tol.accepts(math.inf, 2.0)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("nope", 1, 0), "unknown topic"),
        (("bs_price", -1, 0), "n must be"),
        (("bs_price", 1, -3), "seed must be"),
        (("bs_price", True, 0), "n must be"),
    ],
)
def test_generate_validates_arguments(args: tuple[Any, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        generate(*args)
