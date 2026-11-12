# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-12

First release.

### Added
- **Black-Scholes-Merton** (`bsm`): call and put prices with continuous dividend
  yield, vectorised over inputs.
  - First-order Greeks with per-year and per-day theta, and per-1.00 and per-point
    vega and rho.
  - Limiting values at zero volatility or zero time.
  - Input validation with clear `ValueError` messages.
- **Binomial** (`binomial`): Cox-Ross-Rubinstein tree, European and American
  exercise, with an exact check that the risk-neutral probability lies in [0, 1].
- **Monte Carlo** (`mc`): exact GBM paths, antithetic variates and control variates.
  - Results (`MCResult`) carry a price, standard error and 95% confidence interval.
  - Pricers for European, Asian, barrier and lookback options.
- **Asian** (`asian`): discrete geometric Asian closed form. The arithmetic Asian
  Monte Carlo uses it as a control variate.
- **Barrier** (`barrier`): Reiner-Rubinstein single-barrier closed forms with rebates,
  and the Broadie-Glasserman-Kou discrete-monitoring correction.
- **Lookback** (`lookback`): floating-strike (Goldman-Sosin-Gatto) and fixed-strike
  (Conze-Viswanathan) closed forms, kept numerically stable at `r = q`.
- **Implied volatility** (`implied_vol`): safeguarded Newton with a Brent fallback,
  and errors for prices outside the no-arbitrage bounds.
- **Benchmarks** (`benchmarks`, `quantprice-bench` CLI): seeded JSONL problem
  generators for 11 topics, and a grader that reports accuracy by topic and
  difficulty.
  - Committed dataset `datasets/quantprice-bench-v0.1.0-seed2026.jsonl` with 330
    problems.
- **Tests:**
  - 50-digit mpmath references and Hypothesis property tests;
  - finite-difference Greek checks and Monte Carlo checks within 3 standard errors;
  - an independent re-derivation of every benchmark answer;
  - README doctests.
- **CI:** ruff, strict mypy and pytest on Python 3.11 and 3.12.
