"""
Volatility smile and term structure analysis tools.

This module provides functions for generating synthetic option prices
and recovering implied volatilities to study volatility surfaces.
"""

from collections.abc import Sequence

import numpy as np

from ..core.option_types import ExerciseStyle, Option, OptionType
from ..models.black_scholes import price as bs_price
from ..models.implied_volatility import implied_volatility


def generate_synthetic_call_prices(
    spot: float,
    rate: float,
    time_to_maturity: float,
    strikes: Sequence[float],
    true_vol: float,
) -> list[float]:
    """
    Generate synthetic European call prices using Black-Scholes.

    Creates option prices for a grid of strikes at a fixed maturity,
    assuming a flat 'true' volatility across all strikes.

    Args:
        spot: Current price of the underlying asset
        rate: Risk-free interest rate (annualized)
        time_to_maturity: Time to expiration in years
        strikes: Sequence of strike prices
        true_vol: The 'true' volatility to use for pricing

    Returns:
        List of Black-Scholes call prices for each strike

    Example:
        >>> strikes = [90, 100, 110]
        >>> prices = generate_synthetic_call_prices(100, 0.05, 1.0, strikes, 0.20)
        >>> len(prices) == 3
        True
    """
    prices = []

    for K in strikes:
        option = Option(
            spot=spot,
            strike=K,
            rate=rate,
            volatility=true_vol,
            time_to_maturity=time_to_maturity,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN,
        )
        prices.append(bs_price(option))

    return prices


def recover_implied_vols_for_strikes(
    spot: float,
    rate: float,
    time_to_maturity: float,
    strikes: Sequence[float],
    prices: Sequence[float],
) -> list[float]:
    """
    Recover implied volatilities from option prices for multiple strikes.

    For each strike and corresponding price, constructs an Option and
    computes the implied volatility using the library's solver.

    Args:
        spot: Current price of the underlying asset
        rate: Risk-free interest rate (annualized)
        time_to_maturity: Time to expiration in years
        strikes: Sequence of strike prices
        prices: Sequence of observed market prices (same length as strikes)

    Returns:
        List of implied volatilities for each strike

    Raises:
        ValueError: If strikes and prices have different lengths
        ValueError: If implied volatility cannot be recovered for any option

    Example:
        >>> strikes = [90, 100, 110]
        >>> prices = [15.0, 10.5, 6.5]
        >>> ivs = recover_implied_vols_for_strikes(100, 0.05, 1.0, strikes, prices)
        >>> len(ivs) == 3
        True
    """
    if len(strikes) != len(prices):
        raise ValueError(
            f"strikes and prices must have same length, got {len(strikes)} and {len(prices)}"
        )

    implied_vols = []

    for K, price in zip(strikes, prices):
        option = Option(
            spot=spot,
            strike=K,
            rate=rate,
            volatility=0.20,  # Initial guess, will be overwritten
            time_to_maturity=time_to_maturity,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN,
        )
        iv = implied_volatility(option, price)
        implied_vols.append(iv)

    return implied_vols


def generate_vol_smile_data(
    spot: float,
    rate: float,
    time_to_maturity: float,
    strikes: Sequence[float],
    true_vol: float,
    apply_noise: bool = False,
    noise_std: float = 0.05,
    seed: int | None = None,
) -> tuple[list[float], list[float]]:
    """
    Generate volatility smile data by creating prices and recovering IVs.

    This helper function:
    1. Generates synthetic call prices using Black-Scholes
    2. Optionally perturbs prices with noise (to mimic market data)
    3. Recovers implied volatilities from the (noisy) prices

    Args:
        spot: Current price of the underlying asset
        rate: Risk-free interest rate (annualized)
        time_to_maturity: Time to expiration in years
        strikes: Sequence of strike prices
        true_vol: The 'true' volatility used for generating prices
        apply_noise: Whether to add random noise to prices (default: False)
        noise_std: Standard deviation of noise as fraction of price (default: 0.05)
        seed: Random seed for reproducibility (default: None)

    Returns:
        Tuple of (strikes_list, implied_vols):
            - strikes_list: List of strikes (same as input)
            - implied_vols: List of recovered implied volatilities

    Example:
        >>> strikes = [90, 100, 110]
        >>> strikes_out, ivs = generate_vol_smile_data(
        ...     100, 0.05, 1.0, strikes, 0.20, apply_noise=False
        ... )
        >>> all(abs(iv - 0.20) < 1e-4 for iv in ivs)  # Should recover true vol
        True
    """
    # Generate clean prices
    prices = generate_synthetic_call_prices(spot, rate, time_to_maturity, strikes, true_vol)

    # Optionally add noise
    if apply_noise:
        rng = np.random.default_rng(seed)
        noisy_prices = []
        for p in prices:
            # Add relative noise, ensure price stays positive
            noise = rng.normal(0, noise_std * p)
            noisy_price = max(p + noise, 0.01)  # Ensure positive
            noisy_prices.append(noisy_price)
        prices = noisy_prices

    # Recover implied volatilities
    implied_vols = recover_implied_vols_for_strikes(spot, rate, time_to_maturity, strikes, prices)

    return list(strikes), implied_vols


def generate_term_structure_data(
    spot: float,
    strike: float,
    rate: float,
    maturities: Sequence[float],
    true_vols: Sequence[float],
) -> tuple[list[float], list[float]]:
    """
    Generate volatility term structure data.

    Creates option prices for different maturities with corresponding
    volatilities, then recovers implied volatilities.

    Args:
        spot: Current price of the underlying asset
        strike: Strike price (fixed for all maturities)
        rate: Risk-free interest rate (annualized)
        maturities: Sequence of times to maturity in years
        true_vols: Sequence of true volatilities for each maturity

    Returns:
        Tuple of (maturities_list, implied_vols):
            - maturities_list: List of maturities (same as input)
            - implied_vols: List of recovered implied volatilities

    Raises:
        ValueError: If maturities and true_vols have different lengths
    """
    if len(maturities) != len(true_vols):
        raise ValueError(
            f"maturities and true_vols must have same length, "
            f"got {len(maturities)} and {len(true_vols)}"
        )

    implied_vols = []

    for T, vol in zip(maturities, true_vols):
        # Generate price at this maturity and vol
        option = Option(
            spot=spot,
            strike=strike,
            rate=rate,
            volatility=vol,
            time_to_maturity=T,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN,
        )
        price = bs_price(option)

        # Recover implied vol
        iv = implied_volatility(option, price)
        implied_vols.append(iv)

    return list(maturities), implied_vols
