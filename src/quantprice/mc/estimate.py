r"""Turn per-path discounted payoffs into an :class:`MCResult`, with antithetic pairing and
an optional control variate.

Antithetic pairing
------------------
With antithetic paths, row ``i`` and row ``i + n/2`` are dependent. We first form the
pair averages :math:`\tilde Y_i = (Y_i + Y_{i+n/2})/2`, which are i.i.d., and compute all
statistics from these :math:`n/2` samples (Glasserman 2004, section 4.2). Treating the
:math:`n` paths as independent would misstate the standard error.

Control variate
---------------
Given a control :math:`X` with known mean :math:`\mu_X`, the estimator is

.. math:: \hat V = \bar Y - \hat b(\bar X - \mu_X), \qquad
          \hat b = \frac{\widehat{\mathrm{Cov}}(Y, X)}{\widehat{\mathrm{Var}}(X)},

and its standard error is the sample standard deviation of the residuals
:math:`Y_i - \hat b(X_i - \mu_X)` divided by :math:`\sqrt m`, using divisor
:math:`m - 2`. The estimator is from Glasserman (2004), section 4.1. The divisor
:math:`m - 2` is our choice: the usual regression correction for two fitted parameters
(the mean and :math:`\hat b`); at the sample sizes used here it differs from
:math:`m - 1` by a negligible amount. Estimating :math:`\hat b` from the same samples
introduces a bias of order :math:`1/m`, small against the :math:`1/\sqrt m` standard
error (Glasserman 2004, section 4.1 discusses this bias). If the
spread of :math:`X` is at rounding level (e.g. sigma = 0, where every path is
identical), the control carries no information and :math:`\hat b = 0`.

With antithetic sampling, the regression is on pair averages. Their :math:`X`
varies only through the convexity of :math:`e^{\sigma\sqrt T Z}`, so :math:`\hat b`
is not comparable to a delta and can exceed 1.
"""

import math

import numpy as np

from quantprice._validation import FloatArray
from quantprice.mc.result import Z_95, MCResult

__all__ = ["estimate"]


def _pair_average(x: FloatArray) -> FloatArray:
    half = x.shape[0] // 2
    out: FloatArray = 0.5 * (x[:half] + x[half:])
    return out


def estimate(
    discounted_payoffs: FloatArray,
    *,
    antithetic: bool,
    control: FloatArray | None = None,
    control_mean: float | None = None,
) -> MCResult:
    """Build an MCResult from per-path discounted payoffs (and optional control values).

    Args:
        discounted_payoffs: shape ``(n_paths,)``, laid out as :func:`simulate_gbm` returns
            paths (antithetic partner of row i is row i + n_paths // 2).
        antithetic: whether the rows form antithetic pairs.
        control: per-path control variate values X, same layout, or None.
        control_mean: the exact E[X]; required if ``control`` is given.
    """
    n_paths = discounted_payoffs.shape[0]
    y = _pair_average(discounted_payoffs) if antithetic else discounted_payoffs
    m = y.shape[0]

    coef: float | None = None
    if control is not None:
        if control_mean is None:
            raise ValueError("control_mean is required when a control variate is given")
        x = _pair_average(control) if antithetic else control
        if m < 3:
            raise ValueError("a control variate needs at least 3 independent samples")
        xc = x - x.mean()
        var_x = float(np.dot(xc, xc))
        # A control whose spread is at rounding level (e.g. sigma = 0: every path equal)
        # carries no information; regressing on rounding noise would give a meaningless b.
        noise_floor = m * (1e-12 * float(np.max(np.abs(x)))) ** 2
        coef = float(np.dot(y - y.mean(), xc) / var_x) if var_x > noise_floor else 0.0
        y = y - coef * (x - control_mean)
        ddof = 2
    else:
        if m < 2:
            raise ValueError("need at least 2 independent samples for a standard error")
        ddof = 1

    est = float(y.mean())
    resid = y - est
    se = math.sqrt(float(np.dot(resid, resid)) / (m - ddof) / m)
    return MCResult(
        price=est,
        std_error=se,
        ci_low=est - Z_95 * se,
        ci_high=est + Z_95 * se,
        n_paths=n_paths,
        n_samples=m,
        control_coef=coef,
    )
