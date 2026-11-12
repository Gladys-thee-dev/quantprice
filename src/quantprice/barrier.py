r"""Closed forms for single-barrier European options with continuous monitoring.

Contract
--------
A vanilla call or put (strike :math:`K`, expiry :math:`T`) that is switched on
("in") or off ("out") the first time the price touches the barrier :math:`H`,
monitored **continuously** on :math:`[0, T]`:

* down-and-out / down-and-in: barrier below the spot (:math:`H < S`);
* up-and-out / up-and-in: barrier above the spot (:math:`H > S`).

Optional cash rebate :math:`R`:

* knock-in options pay :math:`R` at expiry if the barrier was never touched;
* knock-out options pay :math:`R` either at the first touch (``rebate_at="hit"``, the
  Reiner-Rubinstein convention and the default) or at expiry if touched
  (``rebate_at="expiry"``).

If the barrier is already breached at :math:`t = 0` (:math:`S \le H` for down,
:math:`S \ge H` for up), an "in" option is the vanilla option, and an "out" option is
worth the rebate: :math:`R` paid now, or :math:`Re^{-rT}` if paid at expiry.

Formulas
--------
Reiner & Rubinstein (1991), in the notation of Haug (2007). Let
:math:`b = r - q`, :math:`v = \sigma\sqrt T`,
:math:`\mu = (b - \sigma^2/2)/\sigma^2`, :math:`\lambda = \sqrt{\mu^2 + 2r/\sigma^2}`,

.. math::

    x_1 = \tfrac{\ln(S/K)}{v} + (1+\mu)v,\quad
    x_2 = \tfrac{\ln(S/H)}{v} + (1+\mu)v,\quad
    y_1 = \tfrac{\ln(H^2/(SK))}{v} + (1+\mu)v,\quad
    y_2 = \tfrac{\ln(H/S)}{v} + (1+\mu)v,\quad
    z = \tfrac{\ln(H/S)}{v} + \lambda v,

and, with :math:`\phi = \pm1` for call/put and :math:`\eta = \pm1` for down/up barriers,

.. math::

    A &= \phi S e^{-qT} N(\phi x_1) - \phi K e^{-rT} N(\phi x_1 - \phi v) \\
    B &= \phi S e^{-qT} N(\phi x_2) - \phi K e^{-rT} N(\phi x_2 - \phi v) \\
    C &= \phi S e^{-qT} (H/S)^{2(\mu+1)} N(\eta y_1)
         - \phi K e^{-rT} (H/S)^{2\mu} N(\eta y_1 - \eta v) \\
    D &= \phi S e^{-qT} (H/S)^{2(\mu+1)} N(\eta y_2)
         - \phi K e^{-rT} (H/S)^{2\mu} N(\eta y_2 - \eta v) \\
    E &= R e^{-rT}\big[N(\eta x_2 - \eta v) - (H/S)^{2\mu} N(\eta y_2 - \eta v)\big] \\
    F &= R\big[(H/S)^{\mu+\lambda} N(\eta z) + (H/S)^{\mu-\lambda} N(\eta z - 2\eta\lambda v)\big]

:math:`E` is :math:`R e^{-rT}\,P(\text{no touch})` and :math:`F` is
:math:`R\,E[e^{-r\tau}1_{\tau \le T}]` for the first-touch time :math:`\tau`.

=================  ===============================  ================================
option             K > H                            K <= H
=================  ===============================  ================================
down-and-in call   C + E                            A - B + D + E
up-and-in call     A + E                            B - C + D + E
down-and-in put    B - C + D + E                    A + E
up-and-in put      A - B + D + E                    C + E
down-and-out call  A - C + F                        B - D + F
up-and-out call    F                                A - B + C - D + F
down-and-out put   A - B + C - D + F                F
up-and-out put     B - D + F                        A - C + F
=================  ===============================  ================================

(Up-and-out call with K > H is worthless apart from the rebate, and so is the
down-and-out put with K <= H, because the option can only end in the money after
the barrier has been touched.)

Numerical choices: the powers :math:`(H/S)^{a}` and :math:`N(\cdot)` are combined in log
space (:math:`\exp(a\ln(H/S) + \log N(\cdot))`, via ``scipy.special.log_ndtr``), so low
volatility (huge :math:`\mu`) can't produce ``inf * 0 = nan``.

Monitoring
----------
These formulas assume **continuous** monitoring. Real contracts are monitored at
discrete dates (e.g. daily closes), and a discretely monitored knock-out is worth
*more* than the continuous one, because the price can cross the barrier between
monitoring dates unseen. Broadie, Glasserman & Kou (1997) show that the discrete
price with :math:`m` equally spaced dates is approximately the continuous price with
the barrier shifted away from the spot,

.. math:: H \to H e^{\pm\beta\sigma\sqrt{T/m}}, \qquad
          \beta = -\zeta(1/2)/\sqrt{2\pi} \approx 0.5826,

(+ for up barriers, - for down barriers), with an error of :math:`o(1/\sqrt m)`. Use
:func:`discrete_price` for that approximation. It is an asymptotic correction, not
an exact price; measured errors are in the tests.

Sources:
    * E. Reiner and M. Rubinstein (1991), "Breaking down the barriers", *Risk* 4(8), 28-35.
    * E. G. Haug (2007), *The Complete Guide to Option Pricing Formulas*, 2nd ed.,
      section 4.17.1 (standard barrier options; the A-F notation above).
    * M. Broadie, P. Glasserman and S. Kou (1997), "A continuity correction for discrete
      barrier options", *Mathematical Finance* 7(4), 325-349.

Conventions: annualised volatility; continuously compounded ``r`` and ``q``; ``T`` in
years, ACT/365.
"""

import math
from typing import Literal

from scipy.special import log_ndtr, ndtr, zeta

from quantprice import bsm
from quantprice._validation import validate_market_inputs
from quantprice.bsm import OptionType, _parse_option_type

__all__ = ["BGK_BETA", "BarrierType", "RebateTiming", "discrete_price", "price"]

BarrierType = Literal["down-and-out", "down-and-in", "up-and-out", "up-and-in"]
RebateTiming = Literal["hit", "expiry"]

BGK_BETA: float = float(-zeta(0.5) / math.sqrt(2.0 * math.pi))
"""Broadie-Glasserman-Kou constant -zeta(1/2)/sqrt(2 pi), about 0.5826."""

_BARRIER_TYPES = ("down-and-out", "down-and-in", "up-and-out", "up-and-in")


def _scaled_ncdf(log_power_base: float, power: float, x: float) -> float:
    """(H/S)^power * N(x), computed as exp(power * ln(H/S) + log N(x))."""
    return math.exp(power * log_power_base + float(log_ndtr(x)))


def _validate(
    option_type: str,
    barrier_type: str,
    S: float,
    K: float,
    H: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    rebate: float,
    rebate_at: str | None,
) -> tuple[bool, bool, bool, str, tuple[float, ...]]:
    is_call = _parse_option_type(option_type)
    if barrier_type not in _BARRIER_TYPES:
        raise ValueError(f"barrier_type must be one of {_BARRIER_TYPES}; got {barrier_type!r}")
    arrays = validate_market_inputs(S, K, T, r, sigma, q)
    if any(a.ndim for a in arrays):
        raise ValueError("barrier pricing takes scalar inputs")
    if not (math.isfinite(H) and H > 0):
        raise ValueError(f"barrier H must be finite and > 0; got {H!r}")
    if not (math.isfinite(rebate) and rebate >= 0):
        raise ValueError(f"rebate must be finite and >= 0; got {rebate!r}")
    is_down = barrier_type.startswith("down")
    is_in = barrier_type.endswith("in")
    timing = rebate_at if rebate_at is not None else ("expiry" if is_in else "hit")
    if timing not in ("hit", "expiry"):
        raise ValueError(f"rebate_at must be 'hit' or 'expiry'; got {rebate_at!r}")
    if is_in and timing == "hit":
        raise ValueError("a knock-in rebate is paid at expiry (if never touched); use 'expiry'")
    return is_call, is_down, is_in, timing, tuple(float(a) for a in arrays)


def _deterministic(
    is_call: bool,
    is_down: bool,
    is_in: bool,
    timing: str,
    S: float,
    K: float,
    H: float,
    T: float,
    r: float,
    q: float,
    rebate: float,
) -> float:
    """sigma = 0 (T > 0): the path S e^{(r-q)t} is monotone, so the touch time is explicit."""
    fwd = S * math.exp((r - q) * T)
    touched = (min(S, fwd) <= H) if is_down else (max(S, fwd) >= H)
    vanilla = math.exp(-r * T) * max((fwd - K) if is_call else (K - fwd), 0.0)
    if is_in:
        return vanilla if touched else rebate * math.exp(-r * T)
    if not touched:
        return vanilla
    if timing == "expiry":
        return rebate * math.exp(-r * T)
    t_hit = math.log(H / S) / (r - q)  # touched with S on the other side, so r != q
    return rebate * math.exp(-r * t_hit)


def price(
    option_type: OptionType,
    barrier_type: BarrierType,
    S: float,
    K: float,
    H: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    rebate: float = 0.0,
    rebate_at: RebateTiming | None = None,
) -> float:
    """Continuously monitored single-barrier option price (Reiner-Rubinstein 1991).

    Args:
        option_type: ``"call"`` or ``"put"``.
        barrier_type: ``"down-and-out"``, ``"down-and-in"``, ``"up-and-out"`` or
            ``"up-and-in"``.
        S, K, T, r, sigma, q: scalar market inputs (validated as in :func:`quantprice.bsm.price`).
        H: barrier level, > 0.
        rebate: cash rebate R >= 0 (see the module docstring for when it is paid).
        rebate_at: ``"hit"`` or ``"expiry"`` for knock-outs (default ``"hit"``). Knock-ins
            always pay at expiry (default ``"expiry"``; ``"hit"`` raises).

    Raises:
        ValueError: for invalid inputs, or for a rebate paid at the hit time when
            :math:`\\mu^2 + 2r/\\sigma^2 < 0` (possible only for negative r), where the
            closed form for F needs complex arithmetic and is not implemented.
    """
    is_call, is_down, is_in, timing, (S, K, T, r, sigma, q) = _validate(
        option_type, barrier_type, S, K, H, T, r, sigma, q, rebate, rebate_at
    )
    vanilla_type: OptionType = "call" if is_call else "put"

    # Barrier already breached at t = 0: "in" is vanilla, "out" pays its rebate.
    if (is_down and S <= H) or (not is_down and S >= H):
        if is_in:
            return bsm.price(vanilla_type, S, K, T, r, sigma, q)
        return rebate if timing == "hit" else rebate * math.exp(-r * T)
    if T == 0.0:  # not breached, expiring now: "out" pays the vanilla payoff, "in" its rebate
        payoff = max((S - K) if is_call else (K - S), 0.0)
        return rebate if is_in else payoff
    if sigma == 0.0:
        return _deterministic(is_call, is_down, is_in, timing, S, K, H, T, r, q, rebate)

    phi = 1.0 if is_call else -1.0
    eta = 1.0 if is_down else -1.0
    v = sigma * math.sqrt(T)
    mu = (r - q - 0.5 * sigma * sigma) / (sigma * sigma)
    ln_hs = math.log(H / S)
    dS, dK = S * math.exp(-q * T), K * math.exp(-r * T)

    x1 = math.log(S / K) / v + (1 + mu) * v
    x2 = -ln_hs / v + (1 + mu) * v
    y1 = math.log(H * H / (S * K)) / v + (1 + mu) * v
    y2 = ln_hs / v + (1 + mu) * v

    A = phi * dS * ndtr(phi * x1) - phi * dK * ndtr(phi * x1 - phi * v)
    B = phi * dS * ndtr(phi * x2) - phi * dK * ndtr(phi * x2 - phi * v)
    C = phi * dS * _scaled_ncdf(ln_hs, 2 * (mu + 1), eta * y1) - phi * dK * _scaled_ncdf(
        ln_hs, 2 * mu, eta * y1 - eta * v
    )
    D = phi * dS * _scaled_ncdf(ln_hs, 2 * (mu + 1), eta * y2) - phi * dK * _scaled_ncdf(
        ln_hs, 2 * mu, eta * y2 - eta * v
    )

    E = 0.0
    F = 0.0
    if rebate > 0:
        no_touch = ndtr(eta * x2 - eta * v) - _scaled_ncdf(ln_hs, 2 * mu, eta * y2 - eta * v)
        E = rebate * math.exp(-r * T) * no_touch
        if not is_in:
            if timing == "expiry":
                F = rebate * math.exp(-r * T) - E  # R e^{-rT} P(touch)
            else:
                disc = mu * mu + 2 * r / (sigma * sigma)
                if disc < 0:
                    raise ValueError(
                        "rebate paid at hit with mu^2 + 2r/sigma^2 < 0 (negative r) is not "
                        "supported; use rebate_at='expiry'"
                    )
                lam = math.sqrt(disc)
                z = ln_hs / v + lam * v
                F = rebate * (
                    _scaled_ncdf(ln_hs, mu + lam, eta * z)
                    + _scaled_ncdf(ln_hs, mu - lam, eta * z - 2 * eta * lam * v)
                )

    strike_above = K > H
    value: float
    if barrier_type == "down-and-in":
        value = (
            (C + E if strike_above else A - B + D + E)
            if is_call
            else (B - C + D + E if strike_above else A + E)
        )
    elif barrier_type == "up-and-in":
        value = (
            (A + E if strike_above else B - C + D + E)
            if is_call
            else (A - B + D + E if strike_above else C + E)
        )
    elif barrier_type == "down-and-out":
        value = (
            (A - C + F if strike_above else B - D + F)
            if is_call
            else (A - B + C - D + F if strike_above else F)
        )
    else:  # up-and-out
        value = (
            (F if strike_above else A - B + C - D + F)
            if is_call
            else (B - D + F if strike_above else A - C + F)
        )
    return max(float(value), 0.0)


def discrete_price(
    option_type: OptionType,
    barrier_type: BarrierType,
    S: float,
    K: float,
    H: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    n_monitoring: int,
    rebate: float = 0.0,
    rebate_at: RebateTiming | None = None,
) -> float:
    """Approximate price with ``n_monitoring`` equally spaced monitoring dates on (0, T]:
    the continuous formula with the Broadie-Glasserman-Kou shifted barrier
    H exp(+-0.5826 sigma sqrt(T/m)).

    This is an asymptotic approximation (error o(1/sqrt(m))), not an exact price. The
    breach-at-inception check uses the *unshifted* barrier, because the contract
    barrier is what matters at t = 0.
    """
    if isinstance(n_monitoring, bool) or not isinstance(n_monitoring, int) or n_monitoring < 1:
        raise ValueError(f"n_monitoring must be an integer >= 1; got {n_monitoring!r}")
    if barrier_type not in _BARRIER_TYPES:
        raise ValueError(f"barrier_type must be one of {_BARRIER_TYPES}; got {barrier_type!r}")
    is_down = barrier_type.startswith("down")
    if (is_down and S <= H) or (not is_down and S >= H) or T == 0.0 or sigma == 0.0:
        return price(
            option_type, barrier_type, S, K, H, T, r, sigma, q, rebate=rebate, rebate_at=rebate_at
        )
    shift = BGK_BETA * sigma * math.sqrt(T / n_monitoring)
    H_shifted = H * math.exp(-shift if is_down else shift)
    return price(
        option_type,
        barrier_type,
        S,
        K,
        H_shifted,
        T,
        r,
        sigma,
        q,
        rebate=rebate,
        rebate_at=rebate_at,
    )
