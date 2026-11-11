r"""Monte Carlo estimate with its standard error and 95% confidence interval.

Every Monte Carlo pricer in quantprice returns an :class:`MCResult`, never a bare
number. A point estimate without its sampling error cannot be checked.

Statistics
----------
Given independent, identically distributed samples :math:`Y_1, \dots, Y_m` (for
antithetic sampling, each :math:`Y_i` is the average of one antithetic pair), the
estimate is :math:`\bar Y`. Its standard error is :math:`s/\sqrt m`, with :math:`s` the
sample standard deviation (divisor :math:`m - 1`). The 95% interval
:math:`\bar Y \pm z_{0.975}\,s/\sqrt m` with :math:`z_{0.975} = 1.959963...` relies on
the central limit theorem, so it is asymptotic: accurate for large :math:`m`, and
not exact for heavy-tailed payoffs or small samples.

Source: P. Glasserman (2004), *Monte Carlo Methods in Financial Engineering*,
Springer, section 1.1 (estimators and confidence intervals).
"""

from dataclasses import dataclass

from scipy.stats import norm

__all__ = ["Z_95", "MCResult"]

Z_95: float = float(norm.ppf(0.975))
"""Two-sided 95% standard-normal quantile, about 1.959964."""


@dataclass(frozen=True, slots=True)
class MCResult:
    """A Monte Carlo price estimate.

    Attributes:
        price: point estimate.
        std_error: estimated standard error of ``price``.
        ci_low, ci_high: asymptotic 95% confidence interval, ``price -+ 1.96 * std_error``.
        n_paths: number of simulated paths, counting both members of each antithetic pair.
        n_samples: number of independent samples the statistics are computed from
            (``n_paths // 2`` with antithetic variates, otherwise ``n_paths``).
        control_coef: estimated control-variate coefficient b, or ``None`` if no control
            variate was used.
    """

    price: float
    std_error: float
    ci_low: float
    ci_high: float
    n_paths: int
    n_samples: int
    control_coef: float | None = None

    def contains(self, value: float) -> bool:
        """True if ``value`` lies inside the 95% confidence interval."""
        return self.ci_low <= value <= self.ci_high

    def within(self, value: float, n_std_errors: float) -> bool:
        """True if ``|price - value| <= n_std_errors * std_error``."""
        return abs(self.price - value) <= n_std_errors * self.std_error

    def __str__(self) -> str:
        return (
            f"{self.price:.6g} (SE {self.std_error:.3g}, "
            f"95% CI [{self.ci_low:.6g}, {self.ci_high:.6g}], {self.n_paths} paths)"
        )
