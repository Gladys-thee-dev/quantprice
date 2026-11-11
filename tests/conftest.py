"""Shared pytest configuration.

Hypothesis profiles:

* ``dev`` (default): random exploration, 200 examples per property.
* ``ci``: ``derandomize=True`` so every CI run checks the same examples and a
  failure is reproducible from the log alone. Select with
  ``HYPOTHESIS_PROFILE=ci``.
* ``stress``: 20,000 examples per property, for occasional deep searches
  (``HYPOTHESIS_PROFILE=stress pytest tests/test_bsm_properties.py ...``).
"""

import os

from hypothesis import settings

settings.register_profile("dev", max_examples=200, deadline=None)
settings.register_profile("ci", max_examples=500, deadline=None, derandomize=True)
settings.register_profile("stress", max_examples=20_000, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
