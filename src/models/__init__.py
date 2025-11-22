"""
Options pricing models.

This module provides the public API for pricing options using three methods:

Pricing Functions:
    price(option) -> float
        Black-Scholes closed-form solution for European options.
        Fastest method, provides exact analytical prices.

    price_binomial(option, steps=100) -> float
        Cox-Ross-Rubinstein binomial tree model.
        Supports both European and American exercise styles.

    price_monte_carlo(option, num_paths=100_000, antithetic=True, seed=None) -> tuple[float, float]
        Monte Carlo simulation under geometric Brownian motion.
        Returns (price, standard_error) for European options.

Greeks (Black-Scholes only):
    delta(option) -> float  # dPrice/dSpot
    gamma(option) -> float  # d²Price/dSpot²
    vega(option) -> float   # dPrice/dVolatility
    theta(option) -> float  # dPrice/dTime
    rho(option) -> float    # dPrice/dRate

Example:
    >>> from src.core.option_types import Option, OptionType, ExerciseStyle
    >>> from src.models import price, price_binomial, price_monte_carlo
    >>>
    >>> option = Option(
    ...     spot=100.0, strike=100.0, rate=0.05, volatility=0.20,
    ...     time_to_maturity=1.0, option_type=OptionType.CALL,
    ...     exercise_style=ExerciseStyle.EUROPEAN
    ... )
    >>>
    >>> price(option)  # Black-Scholes
    10.450583572185565
    >>> price_binomial(option, steps=200)  # Binomial tree
    10.449..
    >>> price_monte_carlo(option, seed=42)  # Monte Carlo
    (10.45..., 0.03...)
"""

from .black_scholes import price, delta, gamma, vega, theta, rho
from .binomial_tree import price_binomial
from .monte_carlo import price_monte_carlo

__all__ = [
    # Pricing functions
    "price",
    "price_binomial",
    "price_monte_carlo",
    # Greeks
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
]
