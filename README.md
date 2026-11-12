# quantprice

Options pricing in Python, built for two jobs:

1. **Pricing.** Black-Scholes-Merton prices and Greeks, a CRR binomial tree (European
   and American), Monte Carlo with variance reduction, barrier, lookback and Asian
   options, and an implied-volatility solver. Every formula cites its source, and every
   numerical claim is checked by a test.
2. **Ground truth for evaluating LLM mathematical reasoning.** `quantprice-bench`
   generates seeded, self-contained options problems as JSONL, each with a verified
   answer, a tolerance band and a worked solution. It then grades submitted answers
   per topic.

Status: version 0.1.0, a research and teaching library. It is **not for production
trading** (see [Limitations](#limitations)).

## Quickstart

```bash
pip install "quantprice @ git+https://github.com/Gladys-thee-dev/quantprice"
```

This needs Python 3.11 or newer; the dependencies are NumPy and SciPy. For
development, run `pip install -e ".[dev]"` from a clone.

## Conventions

These apply throughout, and every exported benchmark problem restates them:

* volatility `sigma` is annualised;
* the rate `r` and dividend yield `q` are annualised and continuously compounded;
* time to expiry `T` is in years under ACT/365 (calendar days / 365);
* the underlying pays a continuous dividend yield `q` (default 0) in every model.

Greek units:

* `theta = dV/dt` as calendar time passes, reported per year and per calendar day
  (`/365`);
* `vega` and `rho` are reported per 1.00 and per 1 percentage point.

Invalid inputs (negative volatility or time, non-positive spot or strike) raise
`ValueError` with a message naming the input.

## Examples

These examples run as doctests in the test suite (`tests/test_readme.py`), so the
outputs shown are real.

Black-Scholes-Merton price and Greeks, and implied volatility:

```python
>>> from quantprice import bsm
>>> from quantprice.implied_vol import implied_vol
>>> g = bsm.greeks("call", S=100, K=105, T=0.5, r=0.03, sigma=0.25, q=0.01)
>>> print(f"{g.price:.4f} {g.delta:.4f} {g.vega_per_pct:.4f} {g.theta_per_day:.4f}")
5.3498 0.4456 0.2783 -0.0211
>>> round(implied_vol("call", g.price, 100, 105, 0.5, 0.03, 0.01), 10)
0.25

```

American put on a CRR tree, and exotic options:

```python
>>> from quantprice import asian, barrier, binomial, lookback
>>> round(binomial.price("put", 100, 105, 0.5, 0.03, 0.25, 0.01, steps=500, exercise="american"), 4)
9.4248
>>> round(barrier.price("call", "down-and-out", 100, 105, 90, 0.5, 0.03, 0.25, 0.01), 4)
4.7521
>>> # the same barrier checked daily (126 dates), Broadie-Glasserman-Kou corrected
>>> round(barrier.discrete_price("call", "down-and-out", 100, 105, 90, 0.5, 0.03, 0.25, 0.01,
...                              n_monitoring=126), 4)
4.8794
>>> round(lookback.floating_price("call", 100, 0.5, 0.03, 0.25, 0.01), 4)
13.7117
>>> monthly = [i / 12 for i in range(1, 7)]
>>> round(asian.geometric_price("call", 100, 100, 0.03, 0.25, 0.01, times=monthly), 4)
4.6791

```

Monte Carlo returns a price with its standard error and a 95% confidence interval.
The exact digits depend on NumPy's random stream, so this example checks the
3-standard-error band against the closed form instead of printing digits:

```python
>>> from quantprice.mc import european
>>> res = european.price("call", 100, 105, 0.5, 0.03, 0.25, 0.01, n_paths=200_000, seed=42)
>>> abs(res.price - g.price) <= 3 * res.std_error
True

```

## Benchmark problems for LLM evaluation

```bash
quantprice-bench generate --seed 2026 --per-topic 30 -o problems.jsonl
quantprice-bench answer-key problems.jsonl -o key.jsonl   # verified answers; grades 100%
quantprice-bench grade problems.jsonl my_answers.jsonl    # per-topic accuracy table
```

A pre-generated set is committed at
[`datasets/quantprice-bench-v0.1.0-seed2026.jsonl`](datasets/quantprice-bench-v0.1.0-seed2026.jsonl):
330 problems, 30 per topic, split evenly into easy, medium and hard.

**Topics:**
* `bs_price`;
* `greek_delta`, `greek_gamma`, `greek_vega`, `greek_theta`, `greek_rho`;
* `put_call_parity`;
* `binomial_american_put`;
* `implied_vol`;
* `geometric_asian`;
* `barrier_in_out_parity`.

**Record** (one per line):

`{id, topic, difficulty, problem, conventions, inputs, verified_answer, tolerance,
solution_steps, library_version}`

* The `problem` text states every convention needed for a unique answer. It also
  states the units and the grading band.
* `tolerance` is `{"abs": ..., "basis": ...}`. An answer is correct iff
  `|answer - verified_answer| <= abs`.
* Bands are `1e-4` times the quantity's natural magnitude:
  * prices: `max(S e^(-qT), K e^(-rT))`;
  * Greeks: the largest value the Greek can take at the stated inputs;
  * implied vol: `1e-4` absolute.
* Problems whose answer is under 10 bands (or, for American puts, within 10 bands of
  intrinsic value) are redrawn, so trivial answers can't score.

**Answers file:** JSONL `{"id": ..., "answer": <number>}`; extra keys are ignored.
Non-numbers are graded *invalid*, and absent ids are graded *missing*. Both count
against accuracy.

**How the answers are verified:** the library computes each answer. The tests then
re-derive every answer in a 330-problem set, and in the committed dataset, from the
record's `inputs` alone, using code that doesn't share the producing function:

| Topic | Independent check |
|---|---|
| prices, parity, barrier vanilla | 50-digit `mpmath` evaluation |
| Greeks | `mpmath` numerical differentiation |
| American put | plain-loop CRR tree |
| implied vol | `mpmath` root finding, plus a check that the band brackets the true root |
| geometric Asian | explicit double-sum variance |
| barrier parity | the Reiner-Rubinstein knock-in closed form |

The largest disagreement measured was 3.4e-9 of the grading band (implied vol). Tests
also confirm that common convention mistakes are graded wrong:
* vega per 1.00 instead of per point;
* theta per year, or per 252 trading days, instead of per calendar day;
* ignoring the dividend yield.

Generation uses Python's string-seeded `random.Random` (stable across Python versions)
rather than NumPy's generator, whose streams may change between releases.

## Models and sources

| Module | Method | Sources |
|---|---|---|
| `bsm` | Generalised BSM prices and Greeks; limiting values at `sigma sqrt(T) = 0` | Black & Scholes (1973); Merton (1973); Haug (2007) §1.1.6, §2.1; Hull, *Options, Futures, and Other Derivatives* |
| `binomial` | CRR tree, European/American, with yield `q` | Cox, Ross & Rubinstein (1979); Leisen & Reimer (1996) on convergence |
| `mc` | Exact GBM steps, antithetic variates, control variates (discounted `S_T`; geometric Asian for arithmetic Asians), SE and 95% CI | Glasserman (2004), *Monte Carlo Methods in Financial Engineering* §4.1-4.2 |
| `asian` | Discrete geometric Asian closed form | Kemna & Vorst (1990); Haug (2007) §4.20 |
| `barrier` | Single-barrier closed forms with rebates; BGK discrete-monitoring correction | Reiner & Rubinstein (1991); Haug (2007) §4.17; Broadie, Glasserman & Kou (1997) |
| `lookback` | Floating and fixed strike, rewritten to stay stable at `r = q` | Goldman, Sosin & Gatto (1979); Conze & Viswanathan (1991); Haug (2007) §4.15 |
| `implied_vol` | Safeguarded Newton with Brent fallback; no-arbitrage bound checks | Manaster & Koehler (1982); Brenner & Subrahmanyam (1988); Brent (1973) |

Each module's docstring gives the formulas, assumptions and full citations.

## Testing

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy && pytest
HYPOTHESIS_PROFILE=stress pytest tests/test_bsm_properties.py   # 20k examples per property
```

The suite combines:
* a 50-digit `mpmath` reference for prices and Greeks;
* Hypothesis property tests (bounds, monotonicity, parity);
* finite-difference Greek checks with rounding-aware tolerances;
* Monte Carlo checks against closed forms within 3 standard errors, with fixed seeds;
* tests that measure binomial and discrete-monitoring convergence rates.

CI runs ruff, mypy (strict) and pytest on Python 3.11 and 3.12. Test coverage has not
been measured, so no coverage figure is claimed.

### Optional: cross-validation against QuantLib

```bash
pip install -e ".[dev,quantlib]"
pytest -m quantlib -s tests/crossval
```

Conventions are aligned: `Actual365Fixed`, `NullCalendar`, flat continuously
compounded curves, and expiry `today + days`. The suite is not run in CI. Measured
with QuantLib 1.44 on 2026-11-12 (Python 3.14, Windows), over grids of spot, strike,
vol, rate, dividend yield and expiry:

| Model | n | Max abs error | Max rel error, values >= 1e-6 S |
|---|---|---|---|
| BSM price | 1458 | 5.0e-14 | 3.0e-11 |
| BSM delta / gamma / vega / theta / rho | 1458 each | <= 3.0e-13 | <= 1.9e-11 |
| Implied vol | 1234 | 7.7e-12 | 7.7e-11 |
| Barrier (8 types, with/without rebate) | 3456 | 8.3e-14 | 2.0e-10 |
| Geometric Asian (discrete) | 1458 | 1.1e-13 | 4.1e-11 |
| Lookback floating / fixed | 486 / 1458 | 2.3e-13 | 1.4e-10 |
| American put, 2000-step trees | 144 | 2.4e-3 | 2.6e-3 |

**Relative error on tiny values.** Over *all* points, relative error reaches 1e-4. It
comes from options worth about 1e-10, where both libraries are at rounding level (about
1e-13 absolute). For the worst BSM and Asian cases, a 50-digit mpmath evaluation matches
quantprice, not QuantLib.

**The American gap is a model difference, not a bug.** QuantLib's `"crr"` engine is the
additive (log-space) tree; ours uses the multiplicative probability. The gap shrinks as
steps increase: 4.8e-3 at 500 steps, 1.8e-4 at 8000.

## Limitations

Read these before using the numbers for anything with money attached.

* **Constant volatility.** Every model assumes one flat `sigma`: there is no smile,
  skew, term structure, local volatility or stochastic volatility. Implied vol
  inverts this single-vol model and can't reproduce a market surface.
* **Continuous dividend yield only.** Discrete cash or proportional dividends aren't
  modelled. For single stocks with known dividends, American exercise and forwards
  around ex-dates will be wrong.
* **No stochastic or term-structure rates.** `r` is one flat, continuously compounded
  rate. There are no curves, no rate volatility, and no funding or collateral
  (OIS/CSA) discounting.
* **Monte Carlo error.** Results carry sampling error, reported as a standard error
  and 95% confidence interval. The interval assumes the CLT holds at the chosen path
  count. Paths are exact at the simulated dates, but path-dependent payoffs see only
  those dates (discretisation). Fixed seeds make runs reproducible, not more accurate.
* **Discrete-monitoring bias for barriers (and lookbacks).** The closed forms assume
  continuous monitoring, but real contracts usually check daily. The
  Broadie-Glasserman-Kou shift is an asymptotic correction: it degrades when the
  barrier is close to the spot or there are few monitoring dates.
* **American exercise only through the binomial tree.** That is a Bermudan
  approximation with O(1/n), oscillating convergence. There is no PDE solver and no
  regression-based (Longstaff-Schwartz) Monte Carlo, and no American barriers, Asians
  or lookbacks.
* **Scope.** Single-asset equity-style options only: no FX quanto, baskets, spreads,
  credit or rates products. The Greeks are BSM only (no closed-form exotic Greeks).
  Lookback `running_extreme` (seasoned contracts) is implemented but lightly tested.
* **No market data connectivity.** There are no feeds, calendars, holidays or
  business-day conventions, no settlement lags and no bid/ask. Day count is ACT/365
  only, and inputs are taken as given.
* **Numerical edges.** Extreme inputs (very large `sigma sqrt(T)`, huge trees) raise
  `ValueError` rather than return garbage. Where vega is negligible, implied vol can't
  be recovered from a double-precision price (documented and tested). Performance has
  not been benchmarked.
* **Benchmark caveats.**
  * Answers are model values under the stated assumptions, not market prices.
  * Problems are template-generated, so the wording varies little.
  * Scale-relative bands are loose for options worth far less than the underlying.
  * Verification uses independent code written in this repository, not an external
    audit.
  * Once published, the committed dataset may leak into model training data. Generate
    a fresh seed for held-out evaluation.
* **Not for production trading.** It has no risk controls, no audit trail and no
  model-validation sign-off. Use it for learning, research and evaluation.

## Changelog and license

See [CHANGELOG.md](CHANGELOG.md). Released under the [MIT License](LICENSE).
