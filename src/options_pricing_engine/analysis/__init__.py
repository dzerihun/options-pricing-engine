"""Analysis tools for volatility surfaces and term structures."""

from .convergence import (
    binomial_convergence,
    monte_carlo_convergence,
)
from .volatility_smile import (
    generate_synthetic_call_prices,
    generate_vol_smile_data,
    recover_implied_vols_for_strikes,
)

__all__ = [
    "generate_synthetic_call_prices",
    "recover_implied_vols_for_strikes",
    "generate_vol_smile_data",
    "binomial_convergence",
    "monte_carlo_convergence",
]
