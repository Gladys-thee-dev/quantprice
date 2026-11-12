"""Ground-truth benchmark problems for evaluating mathematical reasoning about options.

* :mod:`~quantprice.benchmarks.schema`: the problem record and its JSONL encoding.
* :mod:`~quantprice.benchmarks.generators`: seeded, parameterised generators, one per topic.
"""

from quantprice.benchmarks.generators import DIFFICULTIES, TOPICS, generate, generate_all
from quantprice.benchmarks.schema import Problem, Tolerance, read_jsonl, write_jsonl

__all__ = [
    "DIFFICULTIES",
    "TOPICS",
    "Problem",
    "Tolerance",
    "generate",
    "generate_all",
    "read_jsonl",
    "write_jsonl",
]
