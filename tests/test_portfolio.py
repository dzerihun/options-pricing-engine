"""
Tests for portfolio management and risk analysis.

Tests include:
- Synthetic forward pricing (put-call parity)
- Portfolio Greeks aggregation
- Scenario P&L analysis
"""

import math
import pytest

from options_pricing_engine.core.option_types import Option, OptionType, ExerciseStyle
from options_pricing_engine.core.portfolio import (
    Position,
    Portfolio,
    portfolio_price,
    portfolio_greeks,
    scenario_pnl,
    create_synthetic_forward,
)
from options_pricing_engine.models import black_scholes


class TestSyntheticForward:
    """Tests for synthetic forward portfolio (long call + short put)."""

    def test_synthetic_forward_price_equals_forward(self):
        """
        Test that synthetic forward price equals S - K*exp(-rT).

        By put-call parity: C - P = S - K*exp(-rT)
        So a portfolio of +1 call -1 put should equal a forward.
        """
        spot = 100.0
        strike = 100.0
        rate = 0.05
        volatility = 0.20
        T = 1.0

        portfolio = create_synthetic_forward(spot, strike, rate, volatility, T)
        price = portfolio_price(portfolio)

        # Expected forward price
        expected = spot - strike * math.exp(-rate * T)

        assert abs(price - expected) < 1e-10, (
            f"Synthetic forward price {price:.6f} should equal {expected:.6f}"
        )

    def test_synthetic_forward_delta_equals_one(self):
        """
        Test that synthetic forward has delta ≈ 1.

        Long call delta + Short put delta = N(d1) - (N(d1) - 1) = 1
        """
        portfolio = create_synthetic_forward(100.0, 100.0, 0.05, 0.20, 1.0)
        greeks = portfolio_greeks(portfolio)

        assert abs(greeks['delta'] - 1.0) < 1e-10, (
            f"Synthetic forward delta {greeks['delta']:.6f} should be 1.0"
        )

    def test_synthetic_forward_gamma_equals_zero(self):
        """
        Test that synthetic forward has gamma ≈ 0.

        Gamma is the same for call and put, so +1 call -1 put cancels.
        """
        portfolio = create_synthetic_forward(100.0, 100.0, 0.05, 0.20, 1.0)
        greeks = portfolio_greeks(portfolio)

        assert abs(greeks['gamma']) < 1e-10, (
            f"Synthetic forward gamma {greeks['gamma']:.6f} should be 0"
        )

    def test_synthetic_forward_vega_equals_zero(self):
        """
        Test that synthetic forward has vega ≈ 0.

        Vega is the same for call and put, so +1 call -1 put cancels.
        """
        portfolio = create_synthetic_forward(100.0, 100.0, 0.05, 0.20, 1.0)
        greeks = portfolio_greeks(portfolio)

        assert abs(greeks['vega']) < 1e-10, (
            f"Synthetic forward vega {greeks['vega']:.6f} should be 0"
        )

    def test_synthetic_forward_different_strikes(self):
        """Test synthetic forward with different strike prices."""
        for strike in [80.0, 100.0, 120.0]:
            portfolio = create_synthetic_forward(100.0, strike, 0.05, 0.20, 1.0)
            price = portfolio_price(portfolio)
            expected = 100.0 - strike * math.exp(-0.05 * 1.0)

            assert abs(price - expected) < 1e-10


class TestPortfolioGreeks:
    """Tests for portfolio Greeks aggregation."""

    def test_single_position_greeks(self):
        """Test that single position Greeks match option Greeks."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        portfolio = Portfolio([Position(option, 1)])
        greeks = portfolio_greeks(portfolio)

        assert abs(greeks['delta'] - black_scholes.delta(option)) < 1e-10
        assert abs(greeks['gamma'] - black_scholes.gamma(option)) < 1e-10
        assert abs(greeks['vega'] - black_scholes.vega(option)) < 1e-10

    def test_quantity_scaling(self):
        """Test that Greeks scale with position quantity."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        portfolio_1 = Portfolio([Position(option, 1)])
        portfolio_10 = Portfolio([Position(option, 10)])

        greeks_1 = portfolio_greeks(portfolio_1)
        greeks_10 = portfolio_greeks(portfolio_10)

        assert abs(greeks_10['delta'] - 10 * greeks_1['delta']) < 1e-10
        assert abs(greeks_10['gamma'] - 10 * greeks_1['gamma']) < 1e-10
        assert abs(greeks_10['vega'] - 10 * greeks_1['vega']) < 1e-10

    def test_short_position_greeks(self):
        """Test that short positions have negative Greeks contribution."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        portfolio_long = Portfolio([Position(option, 1)])
        portfolio_short = Portfolio([Position(option, -1)])

        greeks_long = portfolio_greeks(portfolio_long)
        greeks_short = portfolio_greeks(portfolio_short)

        assert abs(greeks_short['delta'] + greeks_long['delta']) < 1e-10
        assert abs(greeks_short['gamma'] + greeks_long['gamma']) < 1e-10

    def test_multiple_positions_greeks(self):
        """Test Greeks aggregation with multiple different positions."""
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
            strike=110.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        portfolio = Portfolio([
            Position(call, 2),
            Position(put, 3),
        ])

        greeks = portfolio_greeks(portfolio)

        expected_delta = 2 * black_scholes.delta(call) + 3 * black_scholes.delta(put)
        expected_gamma = 2 * black_scholes.gamma(call) + 3 * black_scholes.gamma(put)

        assert abs(greeks['delta'] - expected_delta) < 1e-10
        assert abs(greeks['gamma'] - expected_gamma) < 1e-10


class TestScenarioPnL:
    """Tests for scenario P&L analysis."""

    def test_zero_shock_zero_pnl(self):
        """Test that zero shocks produce zero P&L."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        portfolio = Portfolio([Position(option, 1)])
        results = scenario_pnl(portfolio, [0], [0])

        assert len(results) == 1
        assert abs(results[0]['pnl']) < 1e-10, (
            f"Zero shock should give zero P&L, got {results[0]['pnl']}"
        )
        assert results[0]['spot_shock'] == 0
        assert results[0]['vol_shock'] == 0

    def test_spot_shock_direction(self):
        """Test that long call gains when spot increases."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        portfolio = Portfolio([Position(option, 1)])
        results = scenario_pnl(portfolio, [-10, 0, 10], [0])

        # Find results
        down = next(r for r in results if r['spot_shock'] == -10)
        flat = next(r for r in results if r['spot_shock'] == 0)
        up = next(r for r in results if r['spot_shock'] == 10)

        # Long call: positive delta means gains when spot up
        assert down['pnl'] < flat['pnl'] < up['pnl']

    def test_vol_shock_direction(self):
        """Test that long call gains when vol increases."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        portfolio = Portfolio([Position(option, 1)])
        results = scenario_pnl(portfolio, [0], [-0.05, 0, 0.05])

        down = next(r for r in results if r['vol_shock'] == -0.05)
        flat = next(r for r in results if r['vol_shock'] == 0)
        up = next(r for r in results if r['vol_shock'] == 0.05)

        # Long call: positive vega means gains when vol up
        assert down['pnl'] < flat['pnl'] < up['pnl']

    def test_scenario_grid_size(self):
        """Test that scenario grid has correct size."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        portfolio = Portfolio([Position(option, 1)])

        spot_shocks = [-20, -10, 0, 10, 20]
        vol_shocks = [-0.05, 0, 0.05]

        results = scenario_pnl(portfolio, spot_shocks, vol_shocks)

        expected_size = len(spot_shocks) * len(vol_shocks)
        assert len(results) == expected_size

    def test_scenario_contains_all_fields(self):
        """Test that each scenario result contains all required fields."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        portfolio = Portfolio([Position(option, 1)])
        results = scenario_pnl(portfolio, [0, 10], [0, 0.05])

        required_fields = ['spot_shock', 'vol_shock', 'new_spot', 'new_vol', 'price', 'pnl']

        for result in results:
            for field in required_fields:
                assert field in result, f"Missing field: {field}"


class TestPortfolioPrice:
    """Tests for portfolio price computation."""

    def test_single_position_price(self):
        """Test that single position price matches option price."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        portfolio = Portfolio([Position(option, 1)])
        price = portfolio_price(portfolio)

        expected = black_scholes.price(option)
        assert abs(price - expected) < 1e-10

    def test_quantity_scaling_price(self):
        """Test that price scales with quantity."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        portfolio_1 = Portfolio([Position(option, 1)])
        portfolio_5 = Portfolio([Position(option, 5)])

        price_1 = portfolio_price(portfolio_1)
        price_5 = portfolio_price(portfolio_5)

        assert abs(price_5 - 5 * price_1) < 1e-10

    def test_short_position_negative_price(self):
        """Test that short position has negative contribution to price."""
        option = Option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )

        portfolio_long = Portfolio([Position(option, 1)])
        portfolio_short = Portfolio([Position(option, -1)])

        price_long = portfolio_price(portfolio_long)
        price_short = portfolio_price(portfolio_short)

        assert abs(price_short + price_long) < 1e-10


class TestInputValidation:
    """Tests for input validation."""

    def test_empty_portfolio_raises_error(self):
        """Test that empty portfolio raises ValueError."""
        with pytest.raises(ValueError, match="at least one position"):
            Portfolio([])
