"""
Implied volatility solver.

This module provides functions for computing implied volatility from
market option prices using root-finding methods.
"""

from typing import Optional

from scipy.optimize import brentq

from ..core.option_types import Option, OptionType, ExerciseStyle
from .black_scholes import price as bs_price, vega as bs_vega


def implied_volatility(
    option: Option,
    market_price: float,
    initial_guess: float = 0.2,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Compute the implied volatility that matches a given market option price.

    Uses Brent's method to find the volatility that makes the Black-Scholes
    price equal to the market price.

    Args:
        option: The option contract (volatility field is ignored)
        market_price: The observed market price of the option
        initial_guess: Initial volatility guess (default: 0.2, not used by Brent)
        tol: Convergence tolerance for the root finder (default: 1e-6)
        max_iter: Maximum iterations for the root finder (default: 100)

    Returns:
        The implied volatility as a decimal (e.g., 0.20 for 20%)

    Raises:
        ValueError: If the option is not European style
        ValueError: If market_price is not positive
        ValueError: If no valid implied volatility exists in the search range
        ValueError: If the market price violates arbitrage bounds

    Example:
        >>> option = Option(
        ...     spot=100.0, strike=100.0, rate=0.05, volatility=0.20,
        ...     time_to_maturity=1.0, option_type=OptionType.CALL,
        ...     exercise_style=ExerciseStyle.EUROPEAN
        ... )
        >>> market_price = 10.4506  # Black-Scholes price at 20% vol
        >>> iv = implied_volatility(option, market_price)
        >>> abs(iv - 0.20) < 1e-4
        True
    """
    # Validate inputs
    if option.exercise_style != ExerciseStyle.EUROPEAN:
        raise ValueError(
            f"Implied volatility solver only supports European options, "
            f"got {option.exercise_style.value}"
        )

    if market_price <= 0:
        raise ValueError(f"Market price must be positive, got {market_price}")

    # Check arbitrage bounds
    S = option.spot
    K = option.strike
    r = option.rate
    T = option.time_to_maturity

    import math
    discount = math.exp(-r * T)

    if option.option_type == OptionType.CALL:
        # Call price bounds: max(0, S - K*e^(-rT)) <= C <= S
        lower_bound = max(0.0, S - K * discount)
        upper_bound = S

        if market_price < lower_bound:
            raise ValueError(
                f"Market price {market_price:.4f} is below the lower arbitrage bound "
                f"{lower_bound:.4f} for this call option"
            )
        if market_price > upper_bound:
            raise ValueError(
                f"Market price {market_price:.4f} exceeds the upper arbitrage bound "
                f"{upper_bound:.4f} (spot price) for this call option"
            )
    else:  # PUT
        # Put price bounds: max(0, K*e^(-rT) - S) <= P <= K*e^(-rT)
        lower_bound = max(0.0, K * discount - S)
        upper_bound = K * discount

        if market_price < lower_bound:
            raise ValueError(
                f"Market price {market_price:.4f} is below the lower arbitrage bound "
                f"{lower_bound:.4f} for this put option"
            )
        if market_price > upper_bound:
            raise ValueError(
                f"Market price {market_price:.4f} exceeds the upper arbitrage bound "
                f"{upper_bound:.4f} for this put option"
            )

    # Define the objective function: BS_price(sigma) - market_price = 0
    def objective(sigma: float) -> float:
        """Compute the difference between BS price and market price."""
        # Create option with test volatility
        test_option = Option(
            spot=option.spot,
            strike=option.strike,
            rate=option.rate,
            volatility=sigma,
            time_to_maturity=option.time_to_maturity,
            option_type=option.option_type,
            exercise_style=option.exercise_style
        )
        return bs_price(test_option) - market_price

    # Search range for implied volatility
    vol_min = 1e-4  # 0.01%
    vol_max = 5.0   # 500%

    # Check if solution exists within bounds
    try:
        f_min = objective(vol_min)
        f_max = objective(vol_max)
    except Exception as e:
        raise ValueError(f"Failed to evaluate objective function: {e}")

    # Check for sign change (required for Brent's method)
    if f_min * f_max > 0:
        if f_min > 0 and f_max > 0:
            raise ValueError(
                f"Market price {market_price:.4f} is too low; "
                f"no valid implied volatility exists in [{vol_min}, {vol_max}]"
            )
        else:
            raise ValueError(
                f"Market price {market_price:.4f} is too high; "
                f"no valid implied volatility exists in [{vol_min}, {vol_max}]"
            )

    # Use Brent's method to find the root
    try:
        iv = brentq(objective, vol_min, vol_max, xtol=tol, maxiter=max_iter)
    except Exception as e:
        raise ValueError(f"Root finding failed: {e}")

    return float(iv)


def implied_volatility_newton(
    option: Option,
    market_price: float,
    initial_guess: float = 0.2,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Compute implied volatility using Newton-Raphson method with vega.

    This is faster than Brent's method when it converges, but may fail
    for extreme cases. Falls back to Brent's method if Newton fails.

    Args:
        option: The option contract (volatility field is ignored)
        market_price: The observed market price of the option
        initial_guess: Initial volatility guess (default: 0.2)
        tol: Convergence tolerance (default: 1e-6)
        max_iter: Maximum iterations (default: 100)

    Returns:
        The implied volatility as a decimal
    """
    if option.exercise_style != ExerciseStyle.EUROPEAN:
        raise ValueError(
            f"Implied volatility solver only supports European options, "
            f"got {option.exercise_style.value}"
        )

    if market_price <= 0:
        raise ValueError(f"Market price must be positive, got {market_price}")

    sigma = initial_guess

    for i in range(max_iter):
        # Create option with current volatility estimate
        test_option = Option(
            spot=option.spot,
            strike=option.strike,
            rate=option.rate,
            volatility=sigma,
            time_to_maturity=option.time_to_maturity,
            option_type=option.option_type,
            exercise_style=option.exercise_style
        )

        # Compute price and vega
        price_diff = bs_price(test_option) - market_price
        vega_val = bs_vega(test_option)

        # Check for convergence
        if abs(price_diff) < tol:
            return sigma

        # Check for near-zero vega (can cause instability)
        if abs(vega_val) < 1e-10:
            # Fall back to Brent's method
            return implied_volatility(option, market_price, initial_guess, tol, max_iter)

        # Newton update
        sigma = sigma - price_diff / vega_val

        # Keep sigma in reasonable bounds
        sigma = max(1e-4, min(5.0, sigma))

    # If Newton didn't converge, fall back to Brent
    return implied_volatility(option, market_price, initial_guess, tol, max_iter)
