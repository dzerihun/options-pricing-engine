"""
Tests for the implied volatility solver.

Tests include:
- Round-trip tests: compute BS price -> recover IV -> check match
- Both calls and puts
- Various moneyness levels
- Edge cases and error handling
"""

import math
import pytest

from options_pricing_engine.core.option_types import Option, OptionType, ExerciseStyle
from options_pricing_engine.models import black_scholes
from options_pricing_engine.models.implied_volatility import implied_volatility, implied_volatility_newton


class TestImpliedVolatilityRoundTrip:
    """Round-trip tests: price at known vol -> recover vol."""

    def test_atm_call_round_trip(self):
        """Test IV recovery for ATM call."""
        true_vol = 0.20

        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=true_vol,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        # Compute theoretical price
        market_price = black_scholes.price(option)

        # Recover implied volatility
        iv = implied_volatility(option, market_price)

        assert abs(iv - true_vol) < 1e-6, (
            f"Recovered IV {iv:.6f} differs from true vol {true_vol:.6f}"
        )

    def test_atm_put_round_trip(self):
        """Test IV recovery for ATM put."""
        true_vol = 0.25

        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=true_vol,
            time_to_maturity=1.0,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        market_price = black_scholes.price(option)
        iv = implied_volatility(option, market_price)

        assert abs(iv - true_vol) < 1e-6

    def test_itm_call_round_trip(self):
        """Test IV recovery for ITM call."""
        true_vol = 0.30

        option = Option(
            spot=120.0,
            strike=100.0,
            rate=0.05,
            volatility=true_vol,
            time_to_maturity=0.5,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        market_price = black_scholes.price(option)
        iv = implied_volatility(option, market_price)

        assert abs(iv - true_vol) < 1e-6

    def test_otm_call_round_trip(self):
        """Test IV recovery for OTM call."""
        true_vol = 0.15

        option = Option(
            spot=80.0,
            strike=100.0,
            rate=0.05,
            volatility=true_vol,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        market_price = black_scholes.price(option)
        iv = implied_volatility(option, market_price)

        assert abs(iv - true_vol) < 1e-6

    def test_itm_put_round_trip(self):
        """Test IV recovery for ITM put."""
        true_vol = 0.35

        option = Option(
            spot=80.0,
            strike=100.0,
            rate=0.08,
            volatility=true_vol,
            time_to_maturity=0.25,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        market_price = black_scholes.price(option)
        iv = implied_volatility(option, market_price)

        assert abs(iv - true_vol) < 1e-6

    def test_otm_put_round_trip(self):
        """Test IV recovery for OTM put."""
        true_vol = 0.18

        option = Option(
            spot=120.0,
            strike=100.0,
            rate=0.05,
            volatility=true_vol,
            time_to_maturity=1.0,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        market_price = black_scholes.price(option)
        iv = implied_volatility(option, market_price)

        assert abs(iv - true_vol) < 1e-6


class TestImpliedVolatilityExtreme:
    """Tests for extreme volatility values."""

    def test_low_volatility(self):
        """Test IV recovery for very low volatility."""
        true_vol = 0.05  # 5%

        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=true_vol,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        market_price = black_scholes.price(option)
        iv = implied_volatility(option, market_price)

        assert abs(iv - true_vol) < 1e-5

    def test_high_volatility(self):
        """Test IV recovery for high volatility."""
        true_vol = 1.0  # 100%

        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=true_vol,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        market_price = black_scholes.price(option)
        iv = implied_volatility(option, market_price)

        assert abs(iv - true_vol) < 1e-5

    def test_very_high_volatility(self):
        """Test IV recovery for very high volatility."""
        true_vol = 2.0  # 200%

        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=true_vol,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        market_price = black_scholes.price(option)
        iv = implied_volatility(option, market_price)

        assert abs(iv - true_vol) < 1e-4


class TestImpliedVolatilityNewton:
    """Tests for the Newton-Raphson solver."""

    def test_newton_atm_call(self):
        """Test Newton method for ATM call."""
        true_vol = 0.20

        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=true_vol,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        market_price = black_scholes.price(option)
        iv = implied_volatility_newton(option, market_price)

        assert abs(iv - true_vol) < 1e-6

    def test_newton_atm_put(self):
        """Test Newton method for ATM put."""
        true_vol = 0.25

        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=true_vol,
            time_to_maturity=1.0,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        market_price = black_scholes.price(option)
        iv = implied_volatility_newton(option, market_price)

        assert abs(iv - true_vol) < 1e-6

    def test_newton_matches_brent(self):
        """Test that Newton and Brent give same result."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        market_price = black_scholes.price(option)

        iv_brent = implied_volatility(option, market_price)
        iv_newton = implied_volatility_newton(option, market_price)

        assert abs(iv_brent - iv_newton) < 1e-6


class TestImpliedVolatilityErrors:
    """Tests for error handling."""

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

        with pytest.raises(ValueError, match="only supports European"):
            implied_volatility(option, 10.0)

    def test_negative_price_raises_error(self):
        """Test that negative market price raises ValueError."""
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
            implied_volatility(option, -5.0)

    def test_zero_price_raises_error(self):
        """Test that zero market price raises ValueError."""
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
            implied_volatility(option, 0.0)

    def test_price_below_arbitrage_bound_call(self):
        """Test that call price below lower bound raises error."""
        option = Option(
            spot=100.0,
            strike=80.0,  # Deep ITM
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        # Lower bound is S - K*e^(-rT) ≈ 100 - 76.1 = 23.9
        with pytest.raises(ValueError, match="below the lower arbitrage bound"):
            implied_volatility(option, 10.0)

    def test_price_above_arbitrage_bound_call(self):
        """Test that call price above spot raises error."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        # Call can never be worth more than spot
        with pytest.raises(ValueError, match="exceeds the upper arbitrage bound"):
            implied_volatility(option, 150.0)

    def test_price_above_arbitrage_bound_put(self):
        """Test that put price above discounted strike raises error."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        # Put can never be worth more than K*e^(-rT) ≈ 95.1
        with pytest.raises(ValueError, match="exceeds the upper arbitrage bound"):
            implied_volatility(option, 100.0)


class TestImpliedVolatilityEdgeCases:
    """Tests for edge cases."""

    def test_short_maturity(self):
        """Test IV for short maturity option."""
        true_vol = 0.20

        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=true_vol,
            time_to_maturity=0.01,  # ~3.6 days
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        market_price = black_scholes.price(option)
        iv = implied_volatility(option, market_price)

        # Slightly larger tolerance for short maturity
        assert abs(iv - true_vol) < 1e-4

    def test_zero_rate(self):
        """Test IV with zero interest rate."""
        true_vol = 0.20

        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.0,
            volatility=true_vol,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        market_price = black_scholes.price(option)
        iv = implied_volatility(option, market_price)

        assert abs(iv - true_vol) < 1e-6

    def test_multiple_vols_same_option(self):
        """Test IV recovery for multiple volatility levels."""
        vols = [0.10, 0.20, 0.30, 0.40, 0.50]

        for true_vol in vols:
            option = Option(
                spot=100.0,
                strike=100.0,
                rate=0.05,
                volatility=true_vol,
                time_to_maturity=1.0,
                option_type=OptionType.CALL,
                exercise_style=ExerciseStyle.EUROPEAN
            )

            market_price = black_scholes.price(option)
            iv = implied_volatility(option, market_price)

            assert abs(iv - true_vol) < 1e-6, f"Failed for vol={true_vol}"
