"""Monte Carlo engine: exact GBM paths, antithetic variates, control variates.

Every pricer returns an :class:`MCResult` (price, standard error, 95% CI).
"""

from quantprice.mc.result import MCResult

__all__ = ["MCResult"]
