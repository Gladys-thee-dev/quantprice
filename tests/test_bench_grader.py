"""Phase 5: the grader and the quantprice-bench command line."""

import json
import math
from pathlib import Path
from typing import Any

import pytest

from quantprice.benchmarks import (
    TOPICS,
    Problem,
    generate_all,
    grade,
    load_answers,
    read_jsonl,
    write_jsonl,
)
from quantprice.benchmarks.cli import main

PROBLEMS = generate_all(11, 3)


def answers_with(**changes: Any) -> dict[str, Any]:
    out: dict[str, Any] = {p.id: p.verified_answer for p in PROBLEMS}
    for k, v in changes.items():
        if v is ...:
            del out[k]
        else:
            out[k] = v
    return out


def test_verified_answers_score_100_percent() -> None:
    rep = grade(PROBLEMS, answers_with())
    assert rep.overall.correct == rep.overall.total == len(PROBLEMS)
    assert rep.overall.accuracy == 1.0
    assert set(rep.by_topic) == set(TOPICS)
    assert all(s.total == 3 and s.correct == 3 for s in rep.by_topic.values())
    assert [s.total for s in rep.by_difficulty.values()] == [len(TOPICS)] * 3


def test_each_status_is_counted_once() -> None:
    a, b, c, d = PROBLEMS[:4]
    rep = grade(
        PROBLEMS,
        {
            **answers_with(),
            a.id: a.verified_answer + 2 * a.tolerance.abs,  # wrong
            b.id: "12.3",  # invalid: string
            c.id: None,  # invalid: null
            d.id: True,  # invalid: bools are not numbers
            "no-such-id": 1.0,
        },
    )
    status = {r.id: r.status for r in rep.results}
    assert status[a.id] == "wrong"
    assert status[b.id] == status[c.id] == status[d.id] == "invalid"
    assert rep.unknown_ids == ["no-such-id"]
    t = rep.overall
    assert t.correct + t.wrong + t.invalid + t.missing == t.total == len(PROBLEMS)


def test_missing_answers_count_against_accuracy() -> None:
    p = PROBLEMS[0]
    rep = grade(PROBLEMS, answers_with(**{p.id: ...}))
    assert rep.overall.missing == 1
    assert rep.by_topic[p.topic].accuracy == pytest.approx(2 / 3)


def test_band_edges_and_integer_answers() -> None:
    p = PROBLEMS[0]
    edge = p.verified_answer + p.tolerance.abs
    assert grade([p], {p.id: edge}).overall.correct == 1
    assert grade([p], {p.id: math.nextafter(edge, math.inf) + 1e-9 * abs(edge)}).overall.wrong == 1
    whole = Problem.from_dict(p.to_dict() | {"verified_answer": 3.0})
    assert grade([whole], {p.id: 3}).overall.correct == 1


def test_duplicate_problem_ids_raise() -> None:
    with pytest.raises(ValueError, match="duplicate problem id"):
        grade([PROBLEMS[0], PROBLEMS[0]], {})


def test_report_serialises_to_json() -> None:
    rep = grade(PROBLEMS, {})
    d = json.loads(json.dumps(rep.to_dict()))
    assert d["overall"]["missing"] == len(PROBLEMS)
    assert d["overall"]["accuracy"] == 0.0
    assert "overall" in rep.format_table()


# ---------------------------------------------------------------- answers file


def write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_answers_allows_extra_keys_and_blank_lines(tmp_path: Path) -> None:
    f = write(tmp_path / "a.jsonl", ['{"id": "x", "answer": 1.5, "reasoning": "..."}', "", "  "])
    assert load_answers(f) == {"x": 1.5}


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (['{"id": "x", "answer": 1}', "{oops"], r"a\.jsonl:2: invalid JSON"),
        (['{"answer": 1}'], "string 'id'"),
        (['{"id": 7, "answer": 1}'], "string 'id'"),
        (['{"id": "x"}'], "'answer'"),
        (["[1, 2]"], "expected an object"),
        (['{"id": "x", "answer": 1}', '{"id": "x", "answer": 2}'], ":2: duplicate answer"),
    ],
)
def test_load_answers_rejects_malformed_files(
    tmp_path: Path, lines: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_answers(write(tmp_path / "a.jsonl", lines))


# ------------------------------------------------------------------------- CLI


def test_cli_generate_answer_key_grade(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    probs, ans = tmp_path / "p.jsonl", tmp_path / "a.jsonl"
    assert main(["generate", "--seed", "5", "--per-topic", "2", "-o", str(probs)]) == 0
    assert len(list(read_jsonl(probs))) == 2 * len(TOPICS)
    assert main(["answer-key", str(probs), "-o", str(ans)]) == 0
    capsys.readouterr()
    assert main(["grade", str(probs), str(ans), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["overall"]["accuracy"] == 1.0


def test_cli_details_and_topic_filter(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    probs, ans = tmp_path / "p.jsonl", tmp_path / "a.jsonl"
    main(
        ["generate", "--seed", "5", "--per-topic", "2", "--topics", "implied_vol", "-o", str(probs)]
    )
    problems = list(read_jsonl(probs))
    assert {p.topic for p in problems} == {"implied_vol"}
    write(ans, [json.dumps({"id": problems[0].id, "answer": 0.0})])
    capsys.readouterr()
    assert main(["grade", str(probs), str(ans), "--details"]) == 0
    out = capsys.readouterr().out
    assert "wrong" in out and "missing" in out and problems[1].id in out


def test_cli_reports_errors_with_exit_code_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    probs = tmp_path / "p.jsonl"
    write_jsonl(PROBLEMS[:1], probs)
    assert main(["grade", str(probs), str(tmp_path / "missing.jsonl")]) == 2
    assert "quantprice-bench: error" in capsys.readouterr().err


def test_cli_lists_topics(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["topics"]) == 0
    assert capsys.readouterr().out.split() == list(TOPICS)
