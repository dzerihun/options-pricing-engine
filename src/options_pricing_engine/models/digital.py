"""
Digital (binary) option pricing.

This module provides functions for pricing cash-or-nothing digital options
using both closed-form Black-Scholes formulas and Monte Carlo simulation.
"""

import math
from typing import Tuple

import numpy as np
from scipy.stats import norm

from ..core.option_types import Option, OptionType, ExerciseStyle


def _compute_d2(option: Option) -> float:
    """
    Compute d2 for Black-Scholes digital option formula.

    d2 = [ln(S/K) + (r - sigma^2/2) * T] / (sigma * sqrt(T))
    """
    S = option.spot
    K = option.strike
    r = option.rate
    sigma = option.volatility
    T = option.time_to_maturity

    sqrt_T = math.sqrt(T)
    d2 = (math.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)

    return d2


def price_digital_black_scholes(
    option: Option,
    payout: float = 1.0,
) -> float:
    """
    Price a European cash-or-nothing digital option using Black-Scholes.

    A cash-or-nothing digital option pays a fixed amount (payout) at expiration
    if the option finishes in-the-money, and nothing otherwise.

    For calls: pays `payout` if S_T > K
    For puts:  pays `payout` if S_T < K

    Args:
        option: The option contract (European only)
        payout: The fixed cash payout if option expires ITM (default: 1.0)

    Returns:
        The theoretical price of the digital option

    Raises:
        ValueError: If option is not European style
        ValueError: If payout is not positive

    Formula:
        Digital Call: payout * exp(-rT) * N(d2)
        Digital Put:  payout * exp(-rT) * N(-d2)

    Example:
        >>> option = Option(100, 100, 0.05, 0.20, 1.0, OptionType.CALL, ExerciseStyle.EUROPEAN)
        >>> price = price_digital_black_scholes(option, payout=100)
        >>> 40 < price < 60  # ATM digital is roughly 0.5 * payout * discount
        True
    """
    if option.exercise_style != ExerciseStyle.EUROPEAN:
        raise ValueError(
            f"Digital option pricing only supports European options, "
            f"got {option.exercise_style.value}"
        )

    if payout <= 0:
        raise ValueError(f"Payout must be positive, got {payout}")

    r = option.rate
    T = option.time_to_maturity

    d2 = _compute_d2(option)
    discount = math.exp(-r * T)

    if option.option_type == OptionType.CALL:
        # Digital call: pays if S_T > K
        return payout * discount * norm.cdf(d2)
    else:
        # Digital put: pays if S_T < K
        return payout * discount * norm.cdf(-d2)


def price_digital_monte_carlo(
    option: Option,
    payout: float = 1.0,
    num_paths: int = 100_000,
    seed: int | None = None,
) -> Tuple[float, float]:
    """
    Price a cash-or-nothing digital option using Monte Carlo simulation.

    Simulates terminal asset prices and computes the expected discounted payoff.

    Args:
        option: The option contract (European only)
        payout: The fixed cash payout if option expires ITM (default: 1.0)
        num_paths: Number of simulation paths (default: 100,000)
        seed: Random seed for reproducibility (default: None)

    Returns:
        Tuple of (price, standard_error):
            - price: Monte Carlo estimate of the digital option price
            - standard_error: Standard error of the estimate

    Raises:
        ValueError: If option is not European style
        ValueError: If payout is not positive
        ValueError: If num_paths is not positive

    Example:
        >>> option = Option(100, 100, 0.05, 0.20, 1.0, OptionType.CALL, ExerciseStyle.EUROPEAN)
        >>> price, se = price_digital_monte_carlo(option, payout=100, seed=42)
        >>> se < 1.0  # Standard error should be small
        True
    """
    if option.exercise_style != ExerciseStyle.EUROPEAN:
        raise ValueError(
            f"Digital option pricing only supports European options, "
            f"got {option.exercise_style.value}"
        )

    if payout <= 0:
        raise ValueError(f"Payout must be positive, got {payout}")

    if num_paths <= 0:
        raise ValueError(f"Number of paths must be positive, got {num_paths}")

    # Extract parameters
    S = option.spot
    K = option.strike
    r = option.rate
    sigma = option.volatility
    T = option.time_to_maturity
    is_call = option.option_type == OptionType.CALL

    # Initialize RNG
    rng = np.random.default_rng(seed)

    # Simulate terminal prices using GBM
    Z = rng.standard_normal(num_paths)
    drift = (r - 0.5 * sigma ** 2) * T
    diffusion = sigma * math.sqrt(T)
    S_T = S * np.exp(drift + diffusion * Z)

    # Compute digital payoffs
    if is_call:
        # Pay if S_T > K
        payoffs = np.where(S_T > K, payout, 0.0)
    else:
        # Pay if S_T < K
        payoffs = np.where(S_T < K, payout, 0.0)

    # Discount factor
    discount = math.exp(-r * T)

    # Compute price and standard error
    price = discount * np.mean(payoffs)
    std_dev = np.std(payoffs, ddof=1)
    std_error = discount * std_dev / math.sqrt(num_paths)

    return float(price), float(std_error)


def digital_delta(option: Option, payout: float = 1.0) -> float:
    """
    Compute the delta of a digital option.

    Note: Digital options have discontinuous payoffs, so delta can be
    very large near the strike at expiration.

    Args:
        option: The option contract
        payout: The fixed cash payout

    Returns:
        The delta of the digital option
    """
    if option.exercise_style != ExerciseStyle.EUROPEAN:
        raise ValueError("Only European options supported")

    S = option.spot
    K = option.strike
    r = option.rate
    sigma = option.volatility
    T = option.time_to_maturity

    d2 = _compute_d2(option)
    discount = math.exp(-r * T)

    # Delta = d(Price)/dS = payout * exp(-rT) * N'(d2) * (d(d2)/dS)
    # d(d2)/dS = 1 / (S * sigma * sqrt(T))
    dd2_dS = 1 / (S * sigma * math.sqrt(T))

    if option.option_type == OptionType.CALL:
        return payout * discount * norm.pdf(d2) * dd2_dS
    else:
        return -payout * discount * norm.pdf(d2) * dd2_dS
