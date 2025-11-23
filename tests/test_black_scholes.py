"""
Tests for the Black-Scholes option pricing model.

Tests include:
- Price verification against known values
- Put-call parity verification
- Greeks computation
- Input validation
"""

import math
import pytest

from options_pricing_engine.core.option_types import Option, OptionType, ExerciseStyle
from options_pricing_engine.models import black_scholes


class TestBlackScholesPrice:
    """Tests for the Black-Scholes pricing function."""

    def test_call_price_known_value(self):
        """Test call price against known Black-Scholes value."""
        # Standard example: S=100, K=100, r=5%, sigma=20%, T=1 year
        # Expected call price ≈ 10.4506
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        price = black_scholes.price(option)
        expected = 10.4506

        assert abs(price - expected) < 0.001, f"Expected {expected}, got {price}"

    def test_put_price_known_value(self):
        """Test put price against known Black-Scholes value."""
        # Standard example: S=100, K=100, r=5%, sigma=20%, T=1 year
        # Expected put price ≈ 5.5735
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        price = black_scholes.price(option)
        expected = 5.5735

        assert abs(price - expected) < 0.001, f"Expected {expected}, got {price}"

    def test_itm_call(self):
        """Test in-the-money call option pricing."""
        # ITM call: S=110, K=100
        option = Option(
            spot=110.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=0.5,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        price = black_scholes.price(option)
        # ITM call should be worth at least intrinsic value
        intrinsic = max(0, option.spot - option.strike)
        assert price >= intrinsic

    def test_otm_put(self):
        """Test out-of-the-money put option pricing."""
        # OTM put: S=110, K=100
        option = Option(
            spot=110.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=0.5,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        price = black_scholes.price(option)
        # OTM put should have positive time value
        assert price > 0
        # But less than strike (theoretical maximum)
        assert price < option.strike


class TestPutCallParity:
    """Tests for put-call parity relationship."""

    def test_put_call_parity_atm(self):
        """
        Test put-call parity for at-the-money option.

        Put-call parity: C - P = S - K * e^(-rT)
        """
        call = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.25,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        put = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.25,
            time_to_maturity=1.0,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        call_price = black_scholes.price(call)
        put_price = black_scholes.price(put)

        # C - P should equal S - K * e^(-rT)
        lhs = call_price - put_price
        rhs = call.spot - call.strike * math.exp(-call.rate * call.time_to_maturity)

        assert abs(lhs - rhs) < 1e-10, f"Put-call parity violated: {lhs} != {rhs}"

    def test_put_call_parity_itm(self):
        """Test put-call parity for in-the-money call."""
        call = Option(
            spot=120.0,
            strike=100.0,
            rate=0.08,
            volatility=0.30,
            time_to_maturity=0.5,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        put = Option(
            spot=120.0,
            strike=100.0,
            rate=0.08,
            volatility=0.30,
            time_to_maturity=0.5,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        call_price = black_scholes.price(call)
        put_price = black_scholes.price(put)

        lhs = call_price - put_price
        rhs = call.spot - call.strike * math.exp(-call.rate * call.time_to_maturity)

        assert abs(lhs - rhs) < 1e-10, f"Put-call parity violated: {lhs} != {rhs}"

    def test_put_call_parity_otm(self):
        """Test put-call parity for out-of-the-money call."""
        call = Option(
            spot=80.0,
            strike=100.0,
            rate=0.03,
            volatility=0.15,
            time_to_maturity=2.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        put = Option(
            spot=80.0,
            strike=100.0,
            rate=0.03,
            volatility=0.15,
            time_to_maturity=2.0,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        call_price = black_scholes.price(call)
        put_price = black_scholes.price(put)

        lhs = call_price - put_price
        rhs = call.spot - call.strike * math.exp(-call.rate * call.time_to_maturity)

        assert abs(lhs - rhs) < 1e-10, f"Put-call parity violated: {lhs} != {rhs}"


class TestGreeks:
    """Tests for the Greeks calculations."""

    @pytest.fixture
    def sample_call(self):
        """Create a sample call option for testing."""
        return Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

    @pytest.fixture
    def sample_put(self):
        """Create a sample put option for testing."""
        return Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN
        )

    def test_call_delta_range(self, sample_call):
        """Test that call delta is between 0 and 1."""
        d = black_scholes.delta(sample_call)
        assert 0 <= d <= 1, f"Call delta {d} out of range [0, 1]"

    def test_put_delta_range(self, sample_put):
        """Test that put delta is between -1 and 0."""
        d = black_scholes.delta(sample_put)
        assert -1 <= d <= 0, f"Put delta {d} out of range [-1, 0]"

    def test_delta_put_call_relationship(self, sample_call, sample_put):
        """Test that call delta - put delta = 1."""
        call_delta = black_scholes.delta(sample_call)
        put_delta = black_scholes.delta(sample_put)

        assert abs(call_delta - put_delta - 1) < 1e-10

    def test_gamma_positive(self, sample_call, sample_put):
        """Test that gamma is always positive."""
        call_gamma = black_scholes.gamma(sample_call)
        put_gamma = black_scholes.gamma(sample_put)

        assert call_gamma > 0
        assert put_gamma > 0

    def test_gamma_same_for_call_put(self, sample_call, sample_put):
        """Test that gamma is the same for call and put with same parameters."""
        call_gamma = black_scholes.gamma(sample_call)
        put_gamma = black_scholes.gamma(sample_put)

        assert abs(call_gamma - put_gamma) < 1e-10

    def test_vega_positive(self, sample_call, sample_put):
        """Test that vega is always positive."""
        call_vega = black_scholes.vega(sample_call)
        put_vega = black_scholes.vega(sample_put)

        assert call_vega > 0
        assert put_vega > 0

    def test_vega_same_for_call_put(self, sample_call, sample_put):
        """Test that vega is the same for call and put with same parameters."""
        call_vega = black_scholes.vega(sample_call)
        put_vega = black_scholes.vega(sample_put)

        assert abs(call_vega - put_vega) < 1e-10

    def test_theta_negative_for_long_positions(self, sample_call, sample_put):
        """Test that theta is typically negative (time decay)."""
        # For most options, theta is negative (lose value over time)
        call_theta = black_scholes.theta(sample_call)
        put_theta = black_scholes.theta(sample_put)

        # ATM call theta should be negative
        assert call_theta < 0
        # ATM put theta could be positive or negative depending on rates
        # Just check it's a reasonable value
        assert abs(put_theta) < 100

    def test_rho_signs(self, sample_call, sample_put):
        """Test that rho has correct sign for calls and puts."""
        call_rho = black_scholes.rho(sample_call)
        put_rho = black_scholes.rho(sample_put)

        # Call rho is positive (higher rates increase call value)
        assert call_rho > 0
        # Put rho is negative (higher rates decrease put value)
        assert put_rho < 0

    def test_delta_known_value(self):
        """Test delta against known value."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        d = black_scholes.delta(option)
        # Expected delta ≈ 0.6368
        expected = 0.6368

        assert abs(d - expected) < 0.001, f"Expected {expected}, got {d}"


class TestInputValidation:
    """Tests for input validation."""

    def test_negative_spot_raises_error(self):
        """Test that negative spot price raises ValueError."""
        with pytest.raises(ValueError, match="Spot price must be positive"):
            Option(
                spot=-100.0,
                strike=100.0,
                rate=0.05,
                volatility=0.20,
                time_to_maturity=1.0,
                option_type=OptionType.CALL,
                exercise_style=ExerciseStyle.EUROPEAN
            )

    def test_zero_strike_raises_error(self):
        """Test that zero strike price raises ValueError."""
        with pytest.raises(ValueError, match="Strike price must be positive"):
            Option(
                spot=100.0,
                strike=0.0,
                rate=0.05,
                volatility=0.20,
                time_to_maturity=1.0,
                option_type=OptionType.CALL,
                exercise_style=ExerciseStyle.EUROPEAN
            )

    def test_negative_volatility_raises_error(self):
        """Test that negative volatility raises ValueError."""
        with pytest.raises(ValueError, match="Volatility must be positive"):
            Option(
                spot=100.0,
                strike=100.0,
                rate=0.05,
                volatility=-0.20,
                time_to_maturity=1.0,
                option_type=OptionType.CALL,
                exercise_style=ExerciseStyle.EUROPEAN
            )

    def test_zero_time_to_maturity_raises_error(self):
        """Test that zero time to maturity raises ValueError."""
        with pytest.raises(ValueError, match="Time to maturity must be positive"):
            Option(
                spot=100.0,
                strike=100.0,
                rate=0.05,
                volatility=0.20,
                time_to_maturity=0.0,
                option_type=OptionType.CALL,
                exercise_style=ExerciseStyle.EUROPEAN
            )

    def test_american_option_raises_error(self):
        """Test that American options raise ValueError in Black-Scholes."""
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
            black_scholes.price(option)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_short_maturity(self):
        """Test option with very short time to maturity."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=0.001,  # Less than a day
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        price = black_scholes.price(option)
        # Should be close to intrinsic value
        assert price >= 0
        assert price < 5  # Very little time value

    def test_very_high_volatility(self):
        """Test option with very high volatility."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=2.0,  # 200% volatility
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        price = black_scholes.price(option)
        # High volatility means high option value
        assert price > 50

    def test_zero_rate(self):
        """Test option with zero interest rate."""
        call = Option(
            spot=100.0,
            strike=100.0,
            rate=0.0,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        put = Option(
            spot=100.0,
            strike=100.0,
            rate=0.0,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        # With r=0, call and put should have same price (ATM)
        call_price = black_scholes.price(call)
        put_price = black_scholes.price(put)

        assert abs(call_price - put_price) < 1e-10

    def test_deep_itm_call(self):
        """Test deep in-the-money call approaches intrinsic value."""
        option = Option(
            spot=200.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=0.1,  # Short maturity
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        price = black_scholes.price(option)
        intrinsic = option.spot - option.strike * math.exp(-option.rate * option.time_to_maturity)

        # Deep ITM call should be very close to discounted intrinsic
        assert abs(price - intrinsic) < 1

    def test_deep_otm_put(self):
        """Test deep out-of-the-money put approaches zero."""
        option = Option(
            spot=200.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=0.1,  # Short maturity
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        price = black_scholes.price(option)

        # Deep OTM put should be nearly worthless
        assert price < 0.01
