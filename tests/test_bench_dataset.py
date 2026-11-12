"""The committed benchmark dataset: loads, is fully verified, and is reproducible.

Regenerating from the same seed must give the same problems. Answers and tolerance
bands are compared to 1e-12 relative rather than bit for bit: they come from
``math.exp`` and SciPy special functions, whose last bits differ between platforms
(CI on Linux found a 1-ulp difference against the Windows-generated file) and between
library versions. The text and inputs must match exactly.
"""

from collections import Counter
from pathlib import Path

import pytest

from quantprice.benchmarks import TOPICS, Problem, generate_all, read_jsonl
from test_bench_verify import VERIFY_FRACTION, reference

DATASET = (
    Path(__file__).resolve().parents[1] / "datasets" / "quantprice-bench-v0.1.0-seed2026.jsonl"
)
SEED, PER_TOPIC = 2026, 30

PROBLEMS = list(read_jsonl(DATASET))


def test_dataset_shape() -> None:
    assert len(PROBLEMS) == PER_TOPIC * len(TOPICS)
    assert Counter(p.topic for p in PROBLEMS) == dict.fromkeys(TOPICS, PER_TOPIC)
    assert Counter(p.difficulty for p in PROBLEMS) == dict.fromkeys(
        ("easy", "medium", "hard"), PER_TOPIC * len(TOPICS) // 3
    )
    assert {p.library_version for p in PROBLEMS} == {"0.1.0"}


def test_dataset_is_reproducible_from_its_seed() -> None:
    fresh = generate_all(SEED, PER_TOPIC)
    assert [p.id for p in fresh] == [p.id for p in PROBLEMS]
    for new, old in zip(fresh, PROBLEMS, strict=True):
        assert new.problem == old.problem, old.id
        assert new.inputs == old.inputs, old.id
        assert new.tolerance.basis == old.tolerance.basis, old.id
        assert new.tolerance.abs == pytest.approx(old.tolerance.abs, rel=1e-12), old.id
        assert new.verified_answer == pytest.approx(old.verified_answer, rel=1e-12), old.id


@pytest.mark.parametrize("p", PROBLEMS, ids=[p.id for p in PROBLEMS])
def test_dataset_answers_are_independently_verified(p: Problem) -> None:
    assert abs(p.verified_answer - reference(p)) <= VERIFY_FRACTION * p.tolerance.abs
