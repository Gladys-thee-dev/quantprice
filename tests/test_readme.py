"""The README's examples are doctests: the outputs shown there must be what the code prints."""

import doctest
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_examples_run() -> None:
    result = doctest.testfile(
        str(README), module_relative=False, optionflags=doctest.NORMALIZE_WHITESPACE
    )
    assert result.attempted >= 10
    assert result.failed == 0
