"""
Monte Carlo option pricing model implementation.

This module provides functions for pricing European options using
Monte Carlo simulation under geometric Brownian motion.
"""

import math
from typing import Tuple

import numpy as np

from ..core.option_types import Option, OptionType, ExerciseStyle


def price_monte_carlo(
    option: Option,
    num_paths: int = 100_000,
    antithetic: bool = True,
    seed: int | None = None,
) -> Tuple[float, float]:
    """
    Price a European option using Monte Carlo under geometric Brownian motion.

    Simulates the terminal asset price using:
        S_T = S_0 * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)

    where Z ~ N(0, 1).

    Args:
        option: The European option to price
        num_paths: Number of simulation paths (default: 100,000)
        antithetic: Use antithetic variates for variance reduction (default: True)
        seed: Random seed for reproducibility (default: None)

    Returns:
        Tuple of (price, standard_error):
            - price: Discounted expected payoff
            - standard_error: Standard error of the price estimate

    Raises:
        ValueError: If the option is not European style
        ValueError: If num_paths is not positive
    """
    # Validate inputs
    if option.exercise_style != ExerciseStyle.EUROPEAN:
        raise ValueError(
            f"Monte Carlo pricing only supports European options, "
            f"got {option.exercise_style.value}"
        )

    if num_paths <= 0:
        raise ValueError(f"Number of paths must be positive, got {num_paths}")

    # Extract option parameters
    S = option.spot
    K = option.strike
    r = option.rate
    sigma = option.volatility
    T = option.time_to_maturity
    is_call = option.option_type == OptionType.CALL

    # Initialize random number generator
    rng = np.random.default_rng(seed)

    # Generate standard normal random variables
    if antithetic:
        # With antithetic variates, generate half the paths
        # and use both Z and -Z
        num_samples = num_paths // 2
        Z = rng.standard_normal(num_samples)

        # Compute terminal prices for both Z and -Z
        drift = (r - 0.5 * sigma ** 2) * T
        diffusion = sigma * math.sqrt(T)

        S_T_pos = S * np.exp(drift + diffusion * Z)
        S_T_neg = S * np.exp(drift - diffusion * Z)

        # Compute payoffs
        if is_call:
            payoffs_pos = np.maximum(S_T_pos - K, 0.0)
            payoffs_neg = np.maximum(S_T_neg - K, 0.0)
        else:
            payoffs_pos = np.maximum(K - S_T_pos, 0.0)
            payoffs_neg = np.maximum(K - S_T_neg, 0.0)

        # Average the antithetic pairs
        payoffs = 0.5 * (payoffs_pos + payoffs_neg)
        effective_paths = num_samples

    else:
        # Standard Monte Carlo without variance reduction
        Z = rng.standard_normal(num_paths)

        # Compute terminal prices
        drift = (r - 0.5 * sigma ** 2) * T
        diffusion = sigma * math.sqrt(T)
        S_T = S * np.exp(drift + diffusion * Z)

        # Compute payoffs
        if is_call:
            payoffs = np.maximum(S_T - K, 0.0)
        else:
            payoffs = np.maximum(K - S_T, 0.0)

        effective_paths = num_paths

    # Discount factor
    discount = math.exp(-r * T)

    # Compute price as discounted expected payoff
    price = discount * np.mean(payoffs)

    # Compute standard error
    # SE = (sample std dev) / sqrt(n)
    std_dev = np.std(payoffs, ddof=1)  # Sample standard deviation
    std_error = discount * std_dev / math.sqrt(effective_paths)

    return float(price), float(std_error)


def price_monte_carlo_with_greeks(
    option: Option,
    num_paths: int = 100_000,
    seed: int | None = None,
) -> dict:
    """
    Price a European option and estimate Greeks using Monte Carlo.

    Uses finite difference methods to estimate delta and gamma.

    Args:
        option: The European option to price
        num_paths: Number of simulation paths
        seed: Random seed for reproducibility

    Returns:
        Dictionary with keys: 'price', 'std_error', 'delta', 'gamma'
    """
    # Get base price
    price, std_error = price_monte_carlo(option, num_paths, antithetic=True, seed=seed)

    # Estimate delta using finite differences
    dS = option.spot * 0.01  # 1% bump

    option_up = Option(
        spot=option.spot + dS,
        strike=option.strike,
        rate=option.rate,
        volatility=option.volatility,
        time_to_maturity=option.time_to_maturity,
        option_type=option.option_type,
        exercise_style=option.exercise_style
    )

    option_down = Option(
        spot=option.spot - dS,
        strike=option.strike,
        rate=option.rate,
        volatility=option.volatility,
        time_to_maturity=option.time_to_maturity,
        option_type=option.option_type,
        exercise_style=option.exercise_style
    )

    price_up, _ = price_monte_carlo(option_up, num_paths, antithetic=True, seed=seed)
    price_down, _ = price_monte_carlo(option_down, num_paths, antithetic=True, seed=seed)

    delta = (price_up - price_down) / (2 * dS)
    gamma = (price_up - 2 * price + price_down) / (dS ** 2)

    return {
        'price': price,
        'std_error': std_error,
        'delta': delta,
        'gamma': gamma,
    }
