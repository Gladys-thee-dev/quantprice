"""Shared pytest configuration.

Hypothesis profiles:

* ``dev`` (default): random exploration, 200 examples per property.
* ``ci``: ``derandomize=True`` so every CI run checks the same examples and a
  failure is reproducible from the log alone. Select with
  ``HYPOTHESIS_PROFILE=ci``.
"""

import os

from hypothesis import settings

settings.register_profile("dev", max_examples=200, deadline=None)
settings.register_profile("ci", max_examples=500, deadline=None, derandomize=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
