"""
Tests for the binomial tree option pricing model.

Tests include:
- Convergence to Black-Scholes for European options
- American vs European call equality for non-dividend stocks
- Sanity checks for price behavior
"""

import math
import pytest

from options_pricing_engine.core.option_types import Option, OptionType, ExerciseStyle
from options_pricing_engine.models import black_scholes
from options_pricing_engine.models.binomial_tree import price_binomial


class TestBinomialConvergence:
    """Tests for convergence of binomial tree to Black-Scholes."""

    def test_european_call_convergence(self):
        """
        Test that European call price converges to Black-Scholes with increasing steps.
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

        # Test convergence with increasing steps
        tolerances = {25: 0.15, 50: 0.08, 100: 0.04, 200: 0.02}

        for steps, tol in tolerances.items():
            binomial_price = price_binomial(option, steps=steps)
            error = abs(binomial_price - bs_price)
            assert error < tol, (
                f"Steps={steps}: binomial={binomial_price:.4f}, "
                f"BS={bs_price:.4f}, error={error:.4f}, tol={tol}"
            )

    def test_european_put_convergence(self):
        """
        Test that European put price converges to Black-Scholes with increasing steps.
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

        tolerances = {25: 0.15, 50: 0.08, 100: 0.04, 200: 0.02}

        for steps, tol in tolerances.items():
            binomial_price = price_binomial(option, steps=steps)
            error = abs(binomial_price - bs_price)
            assert error < tol, (
                f"Steps={steps}: binomial={binomial_price:.4f}, "
                f"BS={bs_price:.4f}, error={error:.4f}, tol={tol}"
            )

    def test_itm_call_convergence(self):
        """Test convergence for in-the-money call."""
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
        binomial_price = price_binomial(option, steps=200)

        assert abs(binomial_price - bs_price) < 0.03

    def test_otm_put_convergence(self):
        """Test convergence for out-of-the-money put."""
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
        binomial_price = price_binomial(option, steps=200)

        assert abs(binomial_price - bs_price) < 0.02

    def test_high_volatility_convergence(self):
        """Test convergence with high volatility."""
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
        binomial_price = price_binomial(option, steps=200)

        # Higher volatility may need slightly larger tolerance
        assert abs(binomial_price - bs_price) < 0.05


class TestAmericanOptions:
    """Tests for American option pricing."""

    def test_american_call_equals_european_no_dividend(self):
        """
        Test that ATM American call equals European call for non-dividend stock.

        For a non-dividend-paying stock, it is never optimal to exercise an
        American call early, so its price should equal the European call price.
        """
        european_call = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        american_call = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.AMERICAN
        )

        european_price = price_binomial(european_call, steps=200)
        american_price = price_binomial(american_call, steps=200)

        # Prices should be very close (within numerical precision)
        assert abs(american_price - european_price) < 0.01, (
            f"American call ({american_price:.4f}) should equal "
            f"European call ({european_price:.4f}) for non-dividend stock"
        )

    def test_itm_american_call_equals_european(self):
        """Test that ITM American call equals European call."""
        european_call = Option(
            spot=120.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        american_call = Option(
            spot=120.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.AMERICAN
        )

        european_price = price_binomial(european_call, steps=200)
        american_price = price_binomial(american_call, steps=200)

        assert abs(american_price - european_price) < 0.01

    def test_american_put_greater_than_european(self):
        """
        Test that American put is worth at least as much as European put.

        Early exercise optionality adds value to American puts.
        """
        european_put = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        american_put = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.AMERICAN
        )

        european_price = price_binomial(european_put, steps=200)
        american_price = price_binomial(american_put, steps=200)

        # American put should be >= European put
        assert american_price >= european_price - 0.001, (
            f"American put ({american_price:.4f}) should be >= "
            f"European put ({european_price:.4f})"
        )

    def test_deep_itm_american_put_early_exercise_premium(self):
        """
        Test that deep ITM American put has early exercise premium.

        For deep ITM puts, early exercise may be optimal, so American
        put should be worth more than European put.
        """
        european_put = Option(
            spot=80.0,
            strike=100.0,
            rate=0.10,  # Higher rate makes early exercise more attractive
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        american_put = Option(
            spot=80.0,
            strike=100.0,
            rate=0.10,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.AMERICAN
        )

        european_price = price_binomial(european_put, steps=200)
        american_price = price_binomial(american_put, steps=200)

        # American put should have noticeable premium over European
        premium = american_price - european_price
        assert premium > 0.01, (
            f"Deep ITM American put should have early exercise premium, "
            f"got {premium:.4f}"
        )

    def test_american_options_at_intrinsic_value(self):
        """Test that American options are worth at least intrinsic value."""
        # Deep ITM call
        american_call = Option(
            spot=150.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.AMERICAN
        )

        call_price = price_binomial(american_call, steps=100)
        call_intrinsic = american_call.spot - american_call.strike

        assert call_price >= call_intrinsic - 0.001

        # Deep ITM put
        american_put = Option(
            spot=50.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.AMERICAN
        )

        put_price = price_binomial(american_put, steps=100)
        put_intrinsic = american_put.strike - american_put.spot

        assert put_price >= put_intrinsic - 0.001


class TestSanityChecks:
    """Sanity checks for binomial tree pricing."""

    def test_price_positive(self):
        """Test that option prices are always positive."""
        test_cases = [
            # ATM
            (100.0, 100.0, OptionType.CALL),
            (100.0, 100.0, OptionType.PUT),
            # ITM
            (120.0, 100.0, OptionType.CALL),
            (80.0, 100.0, OptionType.PUT),
            # OTM
            (80.0, 100.0, OptionType.CALL),
            (120.0, 100.0, OptionType.PUT),
        ]

        for spot, strike, opt_type in test_cases:
            option = Option(
                spot=spot,
                strike=strike,
                rate=0.05,
                volatility=0.20,
                time_to_maturity=1.0,
                option_type=opt_type,
                exercise_style=ExerciseStyle.EUROPEAN
            )

            price = price_binomial(option, steps=100)
            assert price > 0, f"Price should be positive for {opt_type.value}"

    def test_price_does_not_blow_up_with_steps(self):
        """Test that price remains stable with increasing steps."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        prices = []
        for steps in [10, 50, 100, 200, 500]:
            price = price_binomial(option, steps=steps)
            prices.append(price)
            # Price should be reasonable (not blow up)
            assert 5 < price < 20, f"Unreasonable price {price} for {steps} steps"

        # Prices should converge (later prices closer together)
        assert abs(prices[-1] - prices[-2]) < abs(prices[1] - prices[0])

    def test_price_monotonic_in_spot_call(self):
        """Test that call price increases with spot price."""
        prices = []
        for spot in [80.0, 90.0, 100.0, 110.0, 120.0]:
            option = Option(
                spot=spot,
                strike=100.0,
                rate=0.05,
                volatility=0.20,
                time_to_maturity=1.0,
                option_type=OptionType.CALL,
                exercise_style=ExerciseStyle.EUROPEAN
            )
            prices.append(price_binomial(option, steps=100))

        # Call price should increase with spot
        for i in range(len(prices) - 1):
            assert prices[i] < prices[i + 1], "Call price should increase with spot"

    def test_price_monotonic_in_spot_put(self):
        """Test that put price decreases with spot price."""
        prices = []
        for spot in [80.0, 90.0, 100.0, 110.0, 120.0]:
            option = Option(
                spot=spot,
                strike=100.0,
                rate=0.05,
                volatility=0.20,
                time_to_maturity=1.0,
                option_type=OptionType.PUT,
                exercise_style=ExerciseStyle.EUROPEAN
            )
            prices.append(price_binomial(option, steps=100))

        # Put price should decrease with spot
        for i in range(len(prices) - 1):
            assert prices[i] > prices[i + 1], "Put price should decrease with spot"

    def test_call_bounded_by_spot(self):
        """Test that call price is bounded by spot price."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        price = price_binomial(option, steps=100)
        assert price <= option.spot, "Call price should be bounded by spot"

    def test_put_bounded_by_strike(self):
        """Test that put price is bounded by discounted strike."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        price = price_binomial(option, steps=100)
        max_put_value = option.strike * math.exp(-option.rate * option.time_to_maturity)
        assert price <= max_put_value, "Put price should be bounded by discounted strike"


class TestInputValidation:
    """Tests for input validation."""

    def test_zero_steps_raises_error(self):
        """Test that zero steps raises ValueError."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        with pytest.raises(ValueError, match="Number of steps must be positive"):
            price_binomial(option, steps=0)

    def test_negative_steps_raises_error(self):
        """Test that negative steps raises ValueError."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        with pytest.raises(ValueError, match="Number of steps must be positive"):
            price_binomial(option, steps=-10)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_short_maturity(self):
        """Test option with very short time to maturity."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=0.01,  # ~3.65 days
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        price = price_binomial(option, steps=100)
        # Short maturity ATM option should have small value
        assert 0 < price < 5

    def test_low_steps_still_reasonable(self):
        """Test that even with low steps, price is reasonable."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        price = price_binomial(option, steps=5)
        bs_price = black_scholes.price(option)

        # Even with few steps, should be in reasonable range
        assert abs(price - bs_price) < 1.0

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

        price = price_binomial(option, steps=100)
        bs_price = black_scholes.price(option)

        assert abs(price - bs_price) < 0.05

    def test_high_volatility(self):
        """Test with high volatility."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=1.0,  # 100% volatility
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        price = price_binomial(option, steps=200)
        bs_price = black_scholes.price(option)

        # Higher tolerance for high volatility
        assert abs(price - bs_price) < 0.2
