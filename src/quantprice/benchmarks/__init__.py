"""Ground-truth benchmark problems for evaluating mathematical reasoning about options.

* :mod:`~quantprice.benchmarks.schema`: the problem record and its JSONL encoding.
* :mod:`~quantprice.benchmarks.generators`: seeded, parameterised generators, one per topic.
* :mod:`~quantprice.benchmarks.grader`: per-topic scoring of submitted answers.
* :mod:`~quantprice.benchmarks.cli`: the ``quantprice-bench`` command.
"""

from quantprice.benchmarks.generators import DIFFICULTIES, TOPICS, generate, generate_all
from quantprice.benchmarks.grader import Report, grade, load_answers
from quantprice.benchmarks.schema import Problem, Tolerance, read_jsonl, write_jsonl

__all__ = [
    "DIFFICULTIES",
    "TOPICS",
    "Problem",
    "Report",
    "Tolerance",
    "generate",
    "generate_all",
    "grade",
    "load_answers",
    "read_jsonl",
    "write_jsonl",
]
