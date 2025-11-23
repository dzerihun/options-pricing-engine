"""
Convergence analysis tools for numerical pricing methods.

This module provides functions for studying the convergence of
binomial tree and Monte Carlo methods toward Black-Scholes prices.
"""

from collections.abc import Sequence
from typing import Tuple, List

from ..core.option_types import Option
from ..models.black_scholes import price as bs_price
from ..models.binomial_tree import price_binomial
from ..models.monte_carlo import price_monte_carlo


def binomial_convergence(
    option: Option,
    steps_list: Sequence[int],
) -> Tuple[float, List[float], List[float]]:
    """
    Analyze convergence of binomial tree pricing to Black-Scholes.

    Computes binomial tree prices for various numbers of time steps
    and compares them to the analytical Black-Scholes price.

    Args:
        option: The option to price
        steps_list: Sequence of step counts to test

    Returns:
        Tuple of (bs_price, binomial_prices, errors):
            - bs_price: The Black-Scholes analytical price
            - binomial_prices: List of binomial prices for each step count
            - errors: List of absolute errors (binomial - BS) for each step count

    Example:
        >>> from options_pricing_engine.core.option_types import Option, OptionType, ExerciseStyle
        >>> option = Option(
        ...     spot=100, strike=100, rate=0.05, volatility=0.20,
        ...     time_to_maturity=1.0, option_type=OptionType.CALL,
        ...     exercise_style=ExerciseStyle.EUROPEAN
        ... )
        >>> bs, prices, errors = binomial_convergence(option, [10, 50, 100])
        >>> errors[-1] < errors[0]  # Error decreases with more steps
        True
    """
    # Get Black-Scholes benchmark price
    bs = bs_price(option)

    binomial_prices = []
    errors = []

    for steps in steps_list:
        bin_price = price_binomial(option, steps=steps)
        binomial_prices.append(bin_price)
        errors.append(bin_price - bs)

    return bs, binomial_prices, errors


def monte_carlo_convergence(
    option: Option,
    paths_list: Sequence[int],
    seed: int = 42,
) -> Tuple[float, List[float], List[float]]:
    """
    Analyze convergence of Monte Carlo pricing to Black-Scholes.

    Computes Monte Carlo prices and standard errors for various
    numbers of simulation paths.

    Args:
        option: The European option to price
        paths_list: Sequence of path counts to test
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (bs_price, mc_prices, std_errors):
            - bs_price: The Black-Scholes analytical price
            - mc_prices: List of Monte Carlo prices for each path count
            - std_errors: List of standard errors for each path count

    Example:
        >>> from options_pricing_engine.core.option_types import Option, OptionType, ExerciseStyle
        >>> option = Option(
        ...     spot=100, strike=100, rate=0.05, volatility=0.20,
        ...     time_to_maturity=1.0, option_type=OptionType.CALL,
        ...     exercise_style=ExerciseStyle.EUROPEAN
        ... )
        >>> bs, prices, ses = monte_carlo_convergence(option, [1000, 10000])
        >>> ses[-1] < ses[0]  # SE decreases with more paths
        True
    """
    # Get Black-Scholes benchmark price
    bs = bs_price(option)

    mc_prices = []
    std_errors = []

    for num_paths in paths_list:
        mc_price, std_error = price_monte_carlo(
            option, num_paths=num_paths, antithetic=True, seed=seed
        )
        mc_prices.append(mc_price)
        std_errors.append(std_error)

    return bs, mc_prices, std_errors


def compute_convergence_stats(
    option: Option,
    steps_list: Sequence[int] = (10, 25, 50, 100, 200, 500),
    paths_list: Sequence[int] = (1000, 5000, 10000, 50000, 100000),
    seed: int = 42,
) -> dict:
    """
    Compute comprehensive convergence statistics for both methods.

    Args:
        option: The option to analyze
        steps_list: Step counts for binomial tree
        paths_list: Path counts for Monte Carlo
        seed: Random seed for Monte Carlo

    Returns:
        Dictionary with keys:
            - 'bs_price': Black-Scholes price
            - 'binomial': {'steps', 'prices', 'errors'}
            - 'monte_carlo': {'paths', 'prices', 'std_errors'}
    """
    bs, bin_prices, bin_errors = binomial_convergence(option, steps_list)
    _, mc_prices, mc_std_errors = monte_carlo_convergence(option, paths_list, seed)

    return {
        'bs_price': bs,
        'binomial': {
            'steps': list(steps_list),
            'prices': bin_prices,
            'errors': bin_errors,
        },
        'monte_carlo': {
            'paths': list(paths_list),
            'prices': mc_prices,
            'std_errors': mc_std_errors,
        }
    }
