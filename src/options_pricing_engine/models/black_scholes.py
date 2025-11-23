"""
Black-Scholes option pricing model implementation.

This module provides functions for pricing European options and computing
their Greeks using the Black-Scholes closed-form solution.
"""

import math

from scipy.stats import norm

from ..core.option_types import ExerciseStyle, Option, OptionType


def _compute_d1_d2(option: Option) -> tuple[float, float]:
    """
    Compute the d1 and d2 parameters used in Black-Scholes formulas.

    Args:
        option: The option to compute d1 and d2 for

    Returns:
        Tuple of (d1, d2) values

    The formulas are:
        d1 = [ln(S/K) + (r + sigma^2/2) * T] / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
    """
    S = option.spot
    K = option.strike
    r = option.rate
    sigma = option.volatility
    T = option.time_to_maturity

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    return d1, d2


def _validate_european(option: Option) -> None:
    """
    Validate that the option is European style.

    Args:
        option: The option to validate

    Raises:
        ValueError: If the option is not European style
    """
    if option.exercise_style != ExerciseStyle.EUROPEAN:
        raise ValueError(
            f"Black-Scholes model only supports European options, got {option.exercise_style.value}"
        )


def price(option: Option) -> float:
    """
    Calculate the Black-Scholes price of a European option.

    Args:
        option: The option to price

    Returns:
        The theoretical price of the option

    Raises:
        ValueError: If the option is not European style

    For a call:
        C = S * N(d1) - K * e^(-rT) * N(d2)

    For a put:
        P = K * e^(-rT) * N(-d2) - S * N(-d1)
    """
    _validate_european(option)

    S = option.spot
    K = option.strike
    r = option.rate
    T = option.time_to_maturity

    d1, d2 = _compute_d1_d2(option)
    discount = math.exp(-r * T)

    if option.option_type == OptionType.CALL:
        return S * norm.cdf(d1) - K * discount * norm.cdf(d2)
    else:  # PUT
        return K * discount * norm.cdf(-d2) - S * norm.cdf(-d1)


def delta(option: Option) -> float:
    """
    Calculate the delta of a European option.

    Delta measures the rate of change of option price with respect to
    changes in the underlying asset's price.

    Args:
        option: The option to compute delta for

    Returns:
        The delta value

    For a call: delta = N(d1)
    For a put: delta = N(d1) - 1
    """
    _validate_european(option)

    d1, _ = _compute_d1_d2(option)

    if option.option_type == OptionType.CALL:
        return norm.cdf(d1)
    else:  # PUT
        return norm.cdf(d1) - 1


def gamma(option: Option) -> float:
    """
    Calculate the gamma of a European option.

    Gamma measures the rate of change of delta with respect to changes
    in the underlying asset's price. It is the same for calls and puts.

    Args:
        option: The option to compute gamma for

    Returns:
        The gamma value

    Formula: gamma = N'(d1) / (S * sigma * sqrt(T))
    """
    _validate_european(option)

    S = option.spot
    sigma = option.volatility
    T = option.time_to_maturity

    d1, _ = _compute_d1_d2(option)

    return norm.pdf(d1) / (S * sigma * math.sqrt(T))


def vega(option: Option) -> float:
    """
    Calculate the vega of a European option.

    Vega measures the sensitivity of the option price to changes in
    volatility. It is the same for calls and puts.

    Args:
        option: The option to compute vega for

    Returns:
        The vega value (per 1 unit change in volatility)

    Formula: vega = S * sqrt(T) * N'(d1)

    Note: Often reported per 1% change in volatility (divide by 100)
    """
    _validate_european(option)

    S = option.spot
    T = option.time_to_maturity

    d1, _ = _compute_d1_d2(option)

    return S * math.sqrt(T) * norm.pdf(d1)


def theta(option: Option) -> float:
    """
    Calculate the theta of a European option.

    Theta measures the rate of change of option price with respect to
    time (time decay). Returns the change per year; divide by 365 for daily theta.

    Args:
        option: The option to compute theta for

    Returns:
        The theta value (per year)

    For a call:
        theta = -[S * N'(d1) * sigma / (2 * sqrt(T))] - r * K * e^(-rT) * N(d2)

    For a put:
        theta = -[S * N'(d1) * sigma / (2 * sqrt(T))] + r * K * e^(-rT) * N(-d2)
    """
    _validate_european(option)

    S = option.spot
    K = option.strike
    r = option.rate
    sigma = option.volatility
    T = option.time_to_maturity

    d1, d2 = _compute_d1_d2(option)
    sqrt_T = math.sqrt(T)
    discount = math.exp(-r * T)

    # First term is the same for calls and puts
    first_term = -(S * norm.pdf(d1) * sigma) / (2 * sqrt_T)

    if option.option_type == OptionType.CALL:
        return first_term - r * K * discount * norm.cdf(d2)
    else:  # PUT
        return first_term + r * K * discount * norm.cdf(-d2)


def rho(option: Option) -> float:
    """
    Calculate the rho of a European option.

    Rho measures the sensitivity of the option price to changes in
    the risk-free interest rate.

    Args:
        option: The option to compute rho for

    Returns:
        The rho value (per 1 unit change in rate)

    For a call: rho = K * T * e^(-rT) * N(d2)
    For a put: rho = -K * T * e^(-rT) * N(-d2)

    Note: Often reported per 1% change in rate (divide by 100)
    """
    _validate_european(option)

    K = option.strike
    r = option.rate
    T = option.time_to_maturity

    _, d2 = _compute_d1_d2(option)
    discount = math.exp(-r * T)

    if option.option_type == OptionType.CALL:
        return K * T * discount * norm.cdf(d2)
    else:  # PUT
        return -K * T * discount * norm.cdf(-d2)
