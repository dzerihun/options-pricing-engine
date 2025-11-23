"""
Tests for digital (cash-or-nothing) option pricing.

Tests include:
- Comparison of Black-Scholes and Monte Carlo prices
- Monotonicity properties
- Edge cases and validation
"""

import math
import pytest

from options_pricing_engine.core.option_types import Option, OptionType, ExerciseStyle
from options_pricing_engine.models.digital import (
    price_digital_black_scholes,
    price_digital_monte_carlo,
)


class TestDigitalBSvsMC:
    """Compare Black-Scholes and Monte Carlo digital prices."""

    def test_atm_call_bs_vs_mc(self):
        """Test ATM digital call: BS matches MC."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        bs_price = price_digital_black_scholes(option, payout=100)
        mc_price, mc_se = price_digital_monte_carlo(
            option, payout=100, num_paths=100_000, seed=42
        )

        # Allow 3 standard errors
        tolerance = 3 * mc_se + 0.5
        assert abs(bs_price - mc_price) < tolerance, (
            f"BS={bs_price:.4f}, MC={mc_price:.4f}, diff={abs(bs_price-mc_price):.4f}"
        )

    def test_atm_put_bs_vs_mc(self):
        """Test ATM digital put: BS matches MC."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        bs_price = price_digital_black_scholes(option, payout=100)
        mc_price, mc_se = price_digital_monte_carlo(
            option, payout=100, num_paths=100_000, seed=42
        )

        tolerance = 3 * mc_se + 0.5
        assert abs(bs_price - mc_price) < tolerance

    def test_itm_call_bs_vs_mc(self):
        """Test ITM digital call: BS matches MC."""
        option = Option(
            spot=120.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=0.5,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        bs_price = price_digital_black_scholes(option, payout=50)
        mc_price, mc_se = price_digital_monte_carlo(
            option, payout=50, num_paths=100_000, seed=123
        )

        tolerance = 3 * mc_se + 0.5
        assert abs(bs_price - mc_price) < tolerance

    def test_otm_put_bs_vs_mc(self):
        """Test OTM digital put: BS matches MC."""
        option = Option(
            spot=120.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=0.5,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        bs_price = price_digital_black_scholes(option, payout=50)
        mc_price, mc_se = price_digital_monte_carlo(
            option, payout=50, num_paths=100_000, seed=456
        )

        tolerance = 3 * mc_se + 0.3
        assert abs(bs_price - mc_price) < tolerance


class TestDigitalMonotonicity:
    """Test monotonicity properties of digital options."""

    def test_call_price_increases_with_payout(self):
        """Test that digital call price increases with payout."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        price_1 = price_digital_black_scholes(option, payout=1)
        price_10 = price_digital_black_scholes(option, payout=10)
        price_100 = price_digital_black_scholes(option, payout=100)

        assert price_1 < price_10 < price_100

    def test_call_price_decreases_as_strike_increases(self):
        """Test that digital call price decreases as strike moves OTM."""
        prices = []
        for strike in [90, 100, 110, 120]:
            option = Option(
                spot=100.0,
                strike=strike,
                rate=0.05,
                volatility=0.20,
                time_to_maturity=1.0,
                option_type=OptionType.CALL,
                exercise_style=ExerciseStyle.EUROPEAN
            )
            prices.append(price_digital_black_scholes(option, payout=100))

        # As strike increases, digital call price decreases
        for i in range(len(prices) - 1):
            assert prices[i] > prices[i + 1], (
                f"Price at K={90+i*10} should be > price at K={100+i*10}"
            )

    def test_put_price_increases_as_strike_increases(self):
        """Test that digital put price increases as strike moves ITM."""
        prices = []
        for strike in [90, 100, 110, 120]:
            option = Option(
                spot=100.0,
                strike=strike,
                rate=0.05,
                volatility=0.20,
                time_to_maturity=1.0,
                option_type=OptionType.PUT,
                exercise_style=ExerciseStyle.EUROPEAN
            )
            prices.append(price_digital_black_scholes(option, payout=100))

        # As strike increases, digital put price increases
        for i in range(len(prices) - 1):
            assert prices[i] < prices[i + 1]

    def test_call_plus_put_equals_discounted_payout(self):
        """Test that digital call + digital put = discounted payout."""
        option_call = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        option_put = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        payout = 100
        call_price = price_digital_black_scholes(option_call, payout)
        put_price = price_digital_black_scholes(option_put, payout)

        # Call + Put should equal discounted payout
        # (one of them will always pay)
        expected = payout * math.exp(-0.05 * 1.0)

        assert abs(call_price + put_price - expected) < 1e-10


class TestDigitalBounds:
    """Test price bounds for digital options."""

    def test_call_price_bounded_by_discounted_payout(self):
        """Test that digital call price is between 0 and discounted payout."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        payout = 100
        price = price_digital_black_scholes(option, payout)
        max_price = payout * math.exp(-0.05 * 1.0)

        assert 0 < price < max_price

    def test_deep_itm_call_approaches_discounted_payout(self):
        """Test that deep ITM digital call approaches discounted payout."""
        option = Option(
            spot=200.0,  # Deep ITM
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=0.1,  # Short maturity
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        payout = 100
        price = price_digital_black_scholes(option, payout)
        max_price = payout * math.exp(-0.05 * 0.1)

        # Should be very close to discounted payout
        assert price > 0.99 * max_price

    def test_deep_otm_call_approaches_zero(self):
        """Test that deep OTM digital call approaches zero."""
        option = Option(
            spot=50.0,   # Deep OTM
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=0.1,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        payout = 100
        price = price_digital_black_scholes(option, payout)

        assert price < 0.01


class TestDigitalValidation:
    """Test input validation."""

    def test_american_raises_error(self):
        """Test that American style raises ValueError."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.AMERICAN
        )

        with pytest.raises(ValueError, match="only supports European"):
            price_digital_black_scholes(option)

    def test_negative_payout_raises_error(self):
        """Test that negative payout raises ValueError."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        with pytest.raises(ValueError, match="must be positive"):
            price_digital_black_scholes(option, payout=-10)

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

        with pytest.raises(ValueError, match="must be positive"):
            price_digital_monte_carlo(option, num_paths=0)


class TestDigitalMonteCarlo:
    """Specific tests for Monte Carlo pricing."""

    def test_reproducibility_with_seed(self):
        """Test that same seed gives same result."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        price1, _ = price_digital_monte_carlo(option, seed=12345)
        price2, _ = price_digital_monte_carlo(option, seed=12345)

        assert price1 == price2

    def test_se_decreases_with_paths(self):
        """Test that standard error decreases with more paths."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        _, se_10k = price_digital_monte_carlo(option, num_paths=10_000, seed=42)
        _, se_100k = price_digital_monte_carlo(option, num_paths=100_000, seed=42)

        assert se_100k < se_10k
