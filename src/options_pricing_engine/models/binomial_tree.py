"""
Binomial tree option pricing model implementation.

This module provides functions for pricing European and American options
using the Cox-Ross-Rubinstein (CRR) binomial tree model.
"""

import math

import numpy as np

from ..core.option_types import ExerciseStyle, Option, OptionType


def price_binomial(option: Option, steps: int = 100) -> float:
    """
    Price an option using a Cox-Ross-Rubinstein binomial tree.

    Supports:
    - European and American exercise styles
    - Call and put options

    Args:
        option: The option to price
        steps: Number of time steps in the binomial tree (default: 100)

    Returns:
        The theoretical price of the option

    Raises:
        ValueError: If steps is not positive

    The CRR model uses:
        - dt = T / steps (time step)
        - u = exp(sigma * sqrt(dt)) (up factor)
        - d = 1 / u (down factor)
        - p = (exp(r * dt) - d) / (u - d) (risk-neutral probability)
    """
    if steps <= 0:
        raise ValueError(f"Number of steps must be positive, got {steps}")

    # Extract option parameters
    S = option.spot
    K = option.strike
    r = option.rate
    sigma = option.volatility
    T = option.time_to_maturity
    is_call = option.option_type == OptionType.CALL
    is_american = option.exercise_style == ExerciseStyle.AMERICAN

    # CRR parameters
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    discount = math.exp(-r * dt)
    p = (math.exp(r * dt) - d) / (u - d)

    # Build asset prices at maturity (final nodes)
    # At step n, there are n+1 nodes
    # Asset price at node (n, j) is S * u^j * d^(n-j)
    asset_prices = np.array([S * (u**j) * (d ** (steps - j)) for j in range(steps + 1)])

    # Compute option values at maturity
    if is_call:
        option_values = np.maximum(asset_prices - K, 0.0)
    else:
        option_values = np.maximum(K - asset_prices, 0.0)

    # Backward induction through the tree
    for i in range(steps - 1, -1, -1):
        # Asset prices at this time step
        asset_prices_at_step = np.array([S * (u**j) * (d ** (i - j)) for j in range(i + 1)])

        # Continuation value (discounted expected value under risk-neutral measure)
        continuation_values = discount * (
            p * option_values[1 : i + 2] + (1 - p) * option_values[0 : i + 1]
        )

        if is_american:
            # For American options, take max of intrinsic and continuation
            if is_call:
                intrinsic_values = np.maximum(asset_prices_at_step - K, 0.0)
            else:
                intrinsic_values = np.maximum(K - asset_prices_at_step, 0.0)

            option_values = np.maximum(intrinsic_values, continuation_values)
        else:
            # For European options, only use continuation value
            option_values = continuation_values

    return float(option_values[0])


def _get_early_exercise_boundary(option: Option, steps: int = 100) -> list[float]:
    """
    Compute the early exercise boundary for an American option.

    This is useful for understanding when early exercise is optimal.

    Args:
        option: The American option to analyze
        steps: Number of time steps

    Returns:
        List of critical asset prices at each time step where early exercise
        becomes optimal. Returns empty list for European options.
    """
    if option.exercise_style != ExerciseStyle.AMERICAN:
        return []

    S = option.spot
    K = option.strike
    r = option.rate
    sigma = option.volatility
    T = option.time_to_maturity
    is_call = option.option_type == OptionType.CALL

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    discount = math.exp(-r * dt)
    p = (math.exp(r * dt) - d) / (u - d)

    # Build the tree
    asset_prices = np.array([S * (u**j) * (d ** (steps - j)) for j in range(steps + 1)])

    if is_call:
        option_values = np.maximum(asset_prices - K, 0.0)
    else:
        option_values = np.maximum(K - asset_prices, 0.0)

    boundary = []

    for i in range(steps - 1, -1, -1):
        asset_prices_at_step = np.array([S * (u**j) * (d ** (i - j)) for j in range(i + 1)])
        continuation_values = discount * (
            p * option_values[1 : i + 2] + (1 - p) * option_values[0 : i + 1]
        )

        if is_call:
            intrinsic_values = np.maximum(asset_prices_at_step - K, 0.0)
        else:
            intrinsic_values = np.maximum(K - asset_prices_at_step, 0.0)

        # Find the boundary (where early exercise becomes optimal)
        exercise_optimal = intrinsic_values > continuation_values

        if np.any(exercise_optimal):
            if is_call:
                # For calls, boundary is the lowest asset price where exercise is optimal
                boundary_price = asset_prices_at_step[exercise_optimal].min()
            else:
                # For puts, boundary is the highest asset price where exercise is optimal
                boundary_price = asset_prices_at_step[exercise_optimal].max()
            boundary.append(boundary_price)
        else:
            boundary.append(float("nan"))

        option_values = np.maximum(intrinsic_values, continuation_values)

    return list(reversed(boundary))
