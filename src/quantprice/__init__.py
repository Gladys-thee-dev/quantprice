"""quantprice: options pricing with verified ground-truth problem generation.

Conventions used throughout the library (stated again in each public docstring):

* Volatility is annualised.
* Rates ``r`` and dividend yield ``q`` are continuously compounded, annualised.
* Time to expiry ``T`` is in years under ACT/365 (calendar days / 365).
* Continuous dividend yield ``q`` is supported in every model (default ``q = 0``).
"""

from quantprice._version import __version__

__all__ = ["__version__"]
