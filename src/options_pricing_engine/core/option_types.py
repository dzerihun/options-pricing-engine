"""Core option data types and enumerations."""

from dataclasses import dataclass
from enum import Enum


class OptionType(Enum):
    """Type of option - call or put."""

    CALL = "call"
    PUT = "put"


class ExerciseStyle(Enum):
    """Exercise style of the option."""

    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass
class Option:
    """
    Represents an option contract with all necessary parameters for pricing.

    Attributes:
        spot: Current price of the underlying asset (S)
        strike: Strike price of the option (K)
        rate: Risk-free interest rate (r), annualized as a decimal (e.g., 0.05 for 5%)
        volatility: Volatility of the underlying asset (sigma), annualized as a decimal
        time_to_maturity: Time to expiration in years (T)
        option_type: Type of option (CALL or PUT)
        exercise_style: Exercise style (EUROPEAN or AMERICAN)
    """

    spot: float
    strike: float
    rate: float
    volatility: float
    time_to_maturity: float
    option_type: OptionType
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN

    def __post_init__(self) -> None:
        """Validate option parameters after initialization."""
        if self.spot <= 0:
            raise ValueError(f"Spot price must be positive, got {self.spot}")
        if self.strike <= 0:
            raise ValueError(f"Strike price must be positive, got {self.strike}")
        if self.volatility <= 0:
            raise ValueError(f"Volatility must be positive, got {self.volatility}")
        if self.time_to_maturity <= 0:
            raise ValueError(f"Time to maturity must be positive, got {self.time_to_maturity}")
