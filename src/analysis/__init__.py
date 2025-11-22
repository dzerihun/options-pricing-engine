"""Analysis tools for volatility surfaces and term structures."""

from .volatility_smile import (
    generate_synthetic_call_prices,
    recover_implied_vols_for_strikes,
    generate_vol_smile_data,
)
from .convergence import (
    binomial_convergence,
    monte_carlo_convergence,
)

__all__ = [
    "generate_synthetic_call_prices",
    "recover_implied_vols_for_strikes",
    "generate_vol_smile_data",
    "binomial_convergence",
    "monte_carlo_convergence",
]
