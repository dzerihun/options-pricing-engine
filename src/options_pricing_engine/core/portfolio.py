"""
Portfolio management and risk analysis.

This module provides data structures and functions for managing portfolios
of options and computing aggregate risk metrics.
"""

from dataclasses import dataclass
from typing import List, Dict
import copy

from .option_types import Option, OptionType, ExerciseStyle
from ..models.black_scholes import (
    price as bs_price,
    delta as bs_delta,
    gamma as bs_gamma,
    vega as bs_vega,
    theta as bs_theta,
    rho as bs_rho,
)


@dataclass
class Position:
    """
    A position in a single option.

    Attributes:
        option: The option contract
        quantity: Number of contracts (positive for long, negative for short)
    """
    option: Option
    quantity: float


@dataclass
class Portfolio:
    """
    A portfolio of option positions.

    Attributes:
        positions: List of Position objects
    """
    positions: List[Position]

    def __post_init__(self):
        """Validate portfolio."""
        if not self.positions:
            raise ValueError("Portfolio must have at least one position")


def portfolio_price(portfolio: Portfolio) -> float:
    """
    Compute the total price of a portfolio.

    The portfolio price is the sum of (quantity * option_price) for each position.

    Args:
        portfolio: The portfolio to price

    Returns:
        The total portfolio value

    Example:
        >>> from options_pricing_engine.core.option_types import Option, OptionType, ExerciseStyle
        >>> call = Option(100, 100, 0.05, 0.20, 1.0, OptionType.CALL, ExerciseStyle.EUROPEAN)
        >>> portfolio = Portfolio([Position(call, 10)])
        >>> price = portfolio_price(portfolio)
        >>> price > 0
        True
    """
    total = 0.0
    for position in portfolio.positions:
        option_price = bs_price(position.option)
        total += position.quantity * option_price
    return total


def portfolio_greeks(portfolio: Portfolio) -> Dict[str, float]:
    """
    Compute the aggregate Greeks for a portfolio.

    Each Greek is computed as the sum of (quantity * greek_value) for each position.

    Args:
        portfolio: The portfolio to analyze

    Returns:
        Dictionary with keys: 'delta', 'gamma', 'vega', 'theta', 'rho'

    Example:
        >>> from options_pricing_engine.core.option_types import Option, OptionType, ExerciseStyle
        >>> call = Option(100, 100, 0.05, 0.20, 1.0, OptionType.CALL, ExerciseStyle.EUROPEAN)
        >>> portfolio = Portfolio([Position(call, 10)])
        >>> greeks = portfolio_greeks(portfolio)
        >>> 'delta' in greeks and 'gamma' in greeks
        True
    """
    total_delta = 0.0
    total_gamma = 0.0
    total_vega = 0.0
    total_theta = 0.0
    total_rho = 0.0

    for position in portfolio.positions:
        qty = position.quantity
        opt = position.option

        total_delta += qty * bs_delta(opt)
        total_gamma += qty * bs_gamma(opt)
        total_vega += qty * bs_vega(opt)
        total_theta += qty * bs_theta(opt)
        total_rho += qty * bs_rho(opt)

    return {
        'delta': total_delta,
        'gamma': total_gamma,
        'vega': total_vega,
        'theta': total_theta,
        'rho': total_rho,
    }


def scenario_pnl(
    portfolio: Portfolio,
    spot_shocks: List[float],
    vol_shocks: List[float],
) -> List[Dict[str, float]]:
    """
    Compute portfolio P&L across a grid of spot and volatility shocks.

    For each combination of spot and vol shock, recomputes the portfolio price
    and calculates the P&L relative to the base (unshocked) portfolio price.

    Args:
        portfolio: The portfolio to analyze
        spot_shocks: List of absolute spot price changes (e.g., [-10, 0, 10])
        vol_shocks: List of absolute volatility changes (e.g., [-0.05, 0, 0.05])

    Returns:
        List of dictionaries, each containing:
            - 'spot_shock': The spot shock applied
            - 'vol_shock': The volatility shock applied
            - 'new_spot': The shocked spot price
            - 'new_vol': The shocked volatility
            - 'price': The portfolio price after shocks
            - 'pnl': The P&L relative to base price

    Example:
        >>> from options_pricing_engine.core.option_types import Option, OptionType, ExerciseStyle
        >>> call = Option(100, 100, 0.05, 0.20, 1.0, OptionType.CALL, ExerciseStyle.EUROPEAN)
        >>> portfolio = Portfolio([Position(call, 1)])
        >>> results = scenario_pnl(portfolio, [0], [0])
        >>> abs(results[0]['pnl']) < 1e-10
        True
    """
    # Compute base portfolio price
    base_price = portfolio_price(portfolio)

    results = []

    for spot_shock in spot_shocks:
        for vol_shock in vol_shocks:
            # Create shocked portfolio
            shocked_positions = []

            for position in portfolio.positions:
                # Create a copy of the option with shocked parameters
                old_opt = position.option
                new_spot = old_opt.spot + spot_shock
                new_vol = old_opt.volatility + vol_shock

                # Validate shocked parameters
                if new_spot <= 0:
                    # Skip invalid scenarios or use minimum
                    new_spot = 0.01
                if new_vol <= 0:
                    new_vol = 0.001

                shocked_option = Option(
                    spot=new_spot,
                    strike=old_opt.strike,
                    rate=old_opt.rate,
                    volatility=new_vol,
                    time_to_maturity=old_opt.time_to_maturity,
                    option_type=old_opt.option_type,
                    exercise_style=old_opt.exercise_style,
                )

                shocked_positions.append(Position(shocked_option, position.quantity))

            # Compute shocked portfolio price
            shocked_portfolio = Portfolio(shocked_positions)
            shocked_price = portfolio_price(shocked_portfolio)

            # Calculate P&L
            pnl = shocked_price - base_price

            results.append({
                'spot_shock': spot_shock,
                'vol_shock': vol_shock,
                'new_spot': portfolio.positions[0].option.spot + spot_shock,
                'new_vol': portfolio.positions[0].option.volatility + vol_shock,
                'price': shocked_price,
                'pnl': pnl,
            })

    return results


def create_synthetic_forward(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
) -> Portfolio:
    """
    Create a synthetic forward position (long call + short put at same strike).

    A synthetic forward replicates a forward contract:
    Long Call + Short Put = Forward

    The portfolio value should equal S - K*exp(-rT).

    Args:
        spot: Current spot price
        strike: Strike price for both options
        rate: Risk-free rate
        volatility: Volatility (same for both options)
        time_to_maturity: Time to expiration

    Returns:
        Portfolio containing long call and short put
    """
    call = Option(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        time_to_maturity=time_to_maturity,
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.EUROPEAN,
    )

    put = Option(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        time_to_maturity=time_to_maturity,
        option_type=OptionType.PUT,
        exercise_style=ExerciseStyle.EUROPEAN,
    )

    return Portfolio([
        Position(call, 1),   # Long call
        Position(put, -1),   # Short put
    ])
