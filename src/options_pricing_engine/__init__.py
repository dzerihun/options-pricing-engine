"""
Options Pricing Engine

A comprehensive library for pricing financial options using multiple methods:
- Black-Scholes closed-form solutions
- Binomial tree (Cox-Ross-Rubinstein)
- Monte Carlo simulation

Example:
    >>> from options_pricing_engine import Option, OptionType, ExerciseStyle
    >>> from options_pricing_engine import price, delta, price_binomial, price_monte_carlo
    >>>
    >>> option = Option(
    ...     spot=100.0, strike=100.0, rate=0.05, volatility=0.20,
    ...     time_to_maturity=1.0, option_type=OptionType.CALL,
    ...     exercise_style=ExerciseStyle.EUROPEAN
    ... )
    >>>
    >>> price(option)  # Black-Scholes
    10.450583572185565
"""

__version__ = "0.1.0"

# Core types
from .core.option_types import Option, OptionType, ExerciseStyle
from .core.portfolio import Position, Portfolio, portfolio_price, portfolio_greeks, scenario_pnl

# Pricing models
from .models import (
    price,
    price_binomial,
    price_monte_carlo,
    implied_volatility,
    delta,
    gamma,
    vega,
    theta,
    rho,
    price_digital_black_scholes,
    price_digital_monte_carlo,
)

# Analysis tools
from .analysis.volatility_smile import (
    generate_synthetic_call_prices,
    recover_implied_vols_for_strikes,
    generate_vol_smile_data,
)
from .analysis.convergence import binomial_convergence, monte_carlo_convergence

__all__ = [
    # Version
    "__version__",
    # Core types
    "Option",
    "OptionType",
    "ExerciseStyle",
    "Position",
    "Portfolio",
    # Portfolio functions
    "portfolio_price",
    "portfolio_greeks",
    "scenario_pnl",
    # Pricing functions
    "price",
    "price_binomial",
    "price_monte_carlo",
    "implied_volatility",
    # Digital options
    "price_digital_black_scholes",
    "price_digital_monte_carlo",
    # Greeks
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    # Analysis
    "generate_synthetic_call_prices",
    "recover_implied_vols_for_strikes",
    "generate_vol_smile_data",
    "binomial_convergence",
    "monte_carlo_convergence",
]
