"""Market and reporting conventions shared by every model in quantprice.

Every exported problem and public docstring restates these, because an option
price is only well defined once the conventions are fixed.
"""

from typing import Final

DAYS_PER_YEAR: Final[float] = 365.0
"""ACT/365 (fixed): time to expiry ``T`` in years = calendar days / 365."""

PERCENTAGE_POINT: Final[float] = 0.01
"""One percentage point, used to rescale vega and rho (e.g. 20% -> 21% volatility)."""

CONVENTIONS_TEXT: Final[str] = (
    "Volatility is annualised. Interest rate r and dividend yield q are annualised and "
    "continuously compounded. Time to expiry T is in years using ACT/365 "
    "(calendar days / 365). The underlying pays a continuous dividend yield q."
)
