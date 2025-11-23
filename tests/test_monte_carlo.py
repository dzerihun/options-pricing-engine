"""
Tests for the Monte Carlo option pricing model.

Tests include:
- Comparison to Black-Scholes prices
- Standard error behavior
- Antithetic variates variance reduction
- Reproducibility with seeds
"""

import math
import pytest

from options_pricing_engine.core.option_types import Option, OptionType, ExerciseStyle
from options_pricing_engine.models import black_scholes
from options_pricing_engine.models.monte_carlo import price_monte_carlo


class TestMonteCarloConvergence:
    """Tests for convergence of Monte Carlo to Black-Scholes."""

    def test_european_call_matches_black_scholes(self):
        """
        Test that Monte Carlo call price matches Black-Scholes within tolerance.
        """
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        bs_price = black_scholes.price(option)
        mc_price, std_error = price_monte_carlo(
            option, num_paths=50_000, antithetic=True, seed=42
        )

        # Price should be within 3 standard errors of BS price
        tolerance = 3 * std_error + 0.05  # Add small buffer
        error = abs(mc_price - bs_price)

        assert error < tolerance, (
            f"MC price {mc_price:.4f} differs from BS price {bs_price:.4f} "
            f"by {error:.4f} (tolerance: {tolerance:.4f})"
        )

    def test_european_put_matches_black_scholes(self):
        """
        Test that Monte Carlo put price matches Black-Scholes within tolerance.
        """
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        bs_price = black_scholes.price(option)
        mc_price, std_error = price_monte_carlo(
            option, num_paths=50_000, antithetic=True, seed=42
        )

        tolerance = 3 * std_error + 0.05
        error = abs(mc_price - bs_price)

        assert error < tolerance, (
            f"MC price {mc_price:.4f} differs from BS price {bs_price:.4f} "
            f"by {error:.4f} (tolerance: {tolerance:.4f})"
        )

    def test_itm_call_matches_black_scholes(self):
        """Test ITM call matches Black-Scholes."""
        option = Option(
            spot=120.0,
            strike=100.0,
            rate=0.05,
            volatility=0.25,
            time_to_maturity=0.5,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        bs_price = black_scholes.price(option)
        mc_price, std_error = price_monte_carlo(
            option, num_paths=50_000, antithetic=True, seed=123
        )

        tolerance = 3 * std_error + 0.1
        assert abs(mc_price - bs_price) < tolerance

    def test_otm_put_matches_black_scholes(self):
        """Test OTM put matches Black-Scholes."""
        option = Option(
            spot=120.0,
            strike=100.0,
            rate=0.05,
            volatility=0.25,
            time_to_maturity=0.5,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        bs_price = black_scholes.price(option)
        mc_price, std_error = price_monte_carlo(
            option, num_paths=50_000, antithetic=True, seed=456
        )

        tolerance = 3 * std_error + 0.05
        assert abs(mc_price - bs_price) < tolerance

    def test_high_volatility_matches_black_scholes(self):
        """Test high volatility option matches Black-Scholes."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.50,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        bs_price = black_scholes.price(option)
        mc_price, std_error = price_monte_carlo(
            option, num_paths=100_000, antithetic=True, seed=789
        )

        # Higher volatility means larger standard error, use more generous tolerance
        tolerance = 3 * std_error + 0.15
        assert abs(mc_price - bs_price) < tolerance


class TestStandardError:
    """Tests for standard error behavior."""

    def test_std_error_decreases_with_paths(self):
        """
        Test that standard error decreases with increasing number of paths.
        """
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        # Test with different path counts
        _, se_10k = price_monte_carlo(option, num_paths=10_000, seed=42)
        _, se_100k = price_monte_carlo(option, num_paths=100_000, seed=42)

        # SE should decrease roughly as 1/sqrt(n)
        # With 10x more paths, SE should be ~3.16x smaller
        assert se_100k < se_10k, (
            f"SE with 100k paths ({se_100k:.6f}) should be less than "
            f"SE with 10k paths ({se_10k:.6f})"
        )

        # Check approximate scaling (allow some variance)
        expected_ratio = math.sqrt(10)  # sqrt(100k/10k)
        actual_ratio = se_10k / se_100k
        assert 2.0 < actual_ratio < 5.0, (
            f"SE ratio {actual_ratio:.2f} should be close to {expected_ratio:.2f}"
        )

    def test_std_error_positive(self):
        """Test that standard error is always positive."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        _, std_error = price_monte_carlo(option, num_paths=10_000, seed=42)
        assert std_error > 0

    def test_std_error_reasonable_magnitude(self):
        """Test that standard error is reasonable relative to price."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        price, std_error = price_monte_carlo(
            option, num_paths=100_000, antithetic=True, seed=42
        )

        # SE should be small relative to price (< 1% for 100k paths)
        relative_error = std_error / price
        assert relative_error < 0.01, (
            f"Relative SE {relative_error:.4f} should be < 1% for 100k paths"
        )


class TestAntitheticVariates:
    """Tests for antithetic variates variance reduction."""

    def test_antithetic_reduces_variance(self):
        """
        Test that antithetic variates reduce standard error.
        """
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        # Same number of paths, with and without antithetic
        _, se_without = price_monte_carlo(
            option, num_paths=50_000, antithetic=False, seed=42
        )
        _, se_with = price_monte_carlo(
            option, num_paths=50_000, antithetic=True, seed=42
        )

        # Antithetic should have lower SE (or at least similar)
        # Allow small tolerance since it's stochastic
        assert se_with <= se_without * 1.1, (
            f"Antithetic SE ({se_with:.6f}) should be <= non-antithetic "
            f"SE ({se_without:.6f})"
        )

    def test_antithetic_gives_similar_price(self):
        """Test that antithetic and non-antithetic give similar prices."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        price_without, _ = price_monte_carlo(
            option, num_paths=100_000, antithetic=False, seed=42
        )
        price_with, _ = price_monte_carlo(
            option, num_paths=100_000, antithetic=True, seed=42
        )

        # Prices should be close
        assert abs(price_with - price_without) < 0.2


class TestReproducibility:
    """Tests for reproducibility with seeds."""

    def test_same_seed_same_result(self):
        """Test that same seed gives identical results."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        price1, se1 = price_monte_carlo(option, num_paths=10_000, seed=12345)
        price2, se2 = price_monte_carlo(option, num_paths=10_000, seed=12345)

        assert price1 == price2, "Same seed should give same price"
        assert se1 == se2, "Same seed should give same SE"

    def test_different_seed_different_result(self):
        """Test that different seeds give different results."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        price1, _ = price_monte_carlo(option, num_paths=10_000, seed=12345)
        price2, _ = price_monte_carlo(option, num_paths=10_000, seed=67890)

        assert price1 != price2, "Different seeds should give different prices"


class TestInputValidation:
    """Tests for input validation."""

    def test_american_option_raises_error(self):
        """Test that American options raise ValueError."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.AMERICAN
        )

        with pytest.raises(ValueError, match="only supports European options"):
            price_monte_carlo(option)

    def test_zero_paths_raises_error(self):
        """Test that zero paths raises ValueError."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        with pytest.raises(ValueError, match="Number of paths must be positive"):
            price_monte_carlo(option, num_paths=0)

    def test_negative_paths_raises_error(self):
        """Test that negative paths raises ValueError."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        with pytest.raises(ValueError, match="Number of paths must be positive"):
            price_monte_carlo(option, num_paths=-1000)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_short_maturity(self):
        """Test option with very short time to maturity."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=0.01,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        price, std_error = price_monte_carlo(
            option, num_paths=50_000, antithetic=True, seed=42
        )

        # Short maturity ATM option should have small value
        assert 0 <= price < 3
        assert std_error > 0

    def test_deep_itm_call(self):
        """Test deep in-the-money call."""
        option = Option(
            spot=150.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        price, _ = price_monte_carlo(
            option, num_paths=50_000, antithetic=True, seed=42
        )

        # Deep ITM call should be worth approximately intrinsic + time value
        intrinsic = option.spot - option.strike
        assert price >= intrinsic * 0.95  # Allow small MC error

    def test_deep_otm_put(self):
        """Test deep out-of-the-money put."""
        option = Option(
            spot=150.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        price, _ = price_monte_carlo(
            option, num_paths=50_000, antithetic=True, seed=42
        )

        # Deep OTM put should be nearly worthless
        assert price < 1.0

    def test_zero_rate(self):
        """Test with zero interest rate."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.0,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        bs_price = black_scholes.price(option)
        mc_price, std_error = price_monte_carlo(
            option, num_paths=50_000, antithetic=True, seed=42
        )

        tolerance = 3 * std_error + 0.05
        assert abs(mc_price - bs_price) < tolerance

    def test_low_path_count(self):
        """Test with very low path count still gives reasonable result."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        price, std_error = price_monte_carlo(
            option, num_paths=100, antithetic=True, seed=42
        )

        # Price should be positive and in reasonable range
        assert price > 0
        assert price < 30  # Not too far off
        # SE should be large with few paths
        assert std_error > 0.5


class TestPutCallParity:
    """Test put-call parity holds for Monte Carlo prices."""

    def test_put_call_parity(self):
        """Test that put-call parity approximately holds."""
        call = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        put = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        # Use same seed for both to reduce variance
        call_price, call_se = price_monte_carlo(
            call, num_paths=100_000, antithetic=True, seed=42
        )
        put_price, put_se = price_monte_carlo(
            put, num_paths=100_000, antithetic=True, seed=42
        )

        # C - P should equal S - K * e^(-rT)
        lhs = call_price - put_price
        rhs = call.spot - call.strike * math.exp(-call.rate * call.time_to_maturity)

        # Tolerance based on combined standard errors
        tolerance = 3 * math.sqrt(call_se**2 + put_se**2) + 0.1

        assert abs(lhs - rhs) < tolerance, (
            f"Put-call parity violated: {lhs:.4f} != {rhs:.4f} "
            f"(tolerance: {tolerance:.4f})"
        )
