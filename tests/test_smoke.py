"""Phase 0 smoke test: the package installs, imports and reports one consistent version."""

from importlib.metadata import version

import quantprice


def test_version_matches_installed_metadata() -> None:
    assert quantprice.__version__ == version("quantprice")
