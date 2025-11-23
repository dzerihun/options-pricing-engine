# Options Pricing Engine

A clean, well-tested Python library for pricing European and American options using industry-standard quantitative methods. The library implements Black-Scholes closed-form solutions, Cox-Ross-Rubinstein binomial trees, and Monte Carlo simulation, providing a solid foundation for derivatives pricing and risk management applications.

## Pricing Methods

### Black-Scholes Model

The Black-Scholes model provides closed-form analytical solutions for European options. It's the fastest method and serves as the benchmark for other numerical approaches. The implementation includes all first-order Greeks (delta, gamma, vega, theta, rho) for risk management.

**Best for:** European options when you need exact prices and Greeks instantly.

### Binomial Tree Model

The Cox-Ross-Rubinstein (CRR) binomial tree discretizes the asset price evolution into up and down movements, then uses backward induction to compute option values. This method handles both European and American exercise styles, making it essential for pricing American puts where early exercise may be optimal.

**Best for:** American options, or when you need to visualize the price evolution and early exercise boundary.

### Monte Carlo Simulation

Monte Carlo pricing simulates thousands of possible asset price paths under geometric Brownian motion and averages the discounted payoffs. The implementation includes antithetic variates for variance reduction and returns standard errors for confidence intervals.

**Best for:** Path-dependent options, complex payoffs, or when you need confidence intervals on your estimates.

## Installation

### From Source (Development)

1. Clone the repository:
```bash
git clone https://github.com/dzerihun/options-pricing-engine.git
cd options-pricing-engine
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in editable mode with dev dependencies:
```bash
pip install -e ".[dev]"
```

4. Run tests to verify installation:
```bash
pytest tests/ -v
```

### Optional Dependencies

```bash
# For Jupyter notebooks with visualizations
pip install -e ".[notebooks]"

# For all optional dependencies
pip install -e ".[all]"
```

## Quick Start

```python
from options_pricing_engine.core.option_types import Option, OptionType, ExerciseStyle
from options_pricing_engine.models import price, delta, gamma, vega, price_binomial, price_monte_carlo

# Create a European call option
option = Option(
    spot=100.0,           # Current stock price
    strike=100.0,         # Strike price
    rate=0.05,            # Risk-free rate (5%)
    volatility=0.20,      # Volatility (20%)
    time_to_maturity=1.0, # Time to expiration (1 year)
    option_type=OptionType.CALL,
    exercise_style=ExerciseStyle.EUROPEAN
)

# Price with Black-Scholes (fastest, exact for European)
bs_price = price(option)
print(f"Black-Scholes Price: ${bs_price:.2f}")  # $10.45

# Compute Greeks
print(f"Delta: {delta(option):.4f}")  # 0.6368
print(f"Gamma: {gamma(option):.4f}")  # 0.0188
print(f"Vega:  {vega(option):.2f}")   # 37.52

# Price with binomial tree (works for American options too)
bin_price = price_binomial(option, steps=200)
print(f"Binomial Price:      ${bin_price:.2f}")  # $10.45

# Price with Monte Carlo (returns standard error)
mc_price, mc_se = price_monte_carlo(option, num_paths=100_000, seed=42)
print(f"Monte Carlo Price:   ${mc_price:.2f} ± ${mc_se:.2f}")  # $10.45 ± $0.03

# American put option (requires binomial tree)
american_put = Option(
    spot=100.0,
    strike=100.0,
    rate=0.05,
    volatility=0.20,
    time_to_maturity=1.0,
    option_type=OptionType.PUT,
    exercise_style=ExerciseStyle.AMERICAN
)

american_price = price_binomial(american_put, steps=200)
print(f"American Put Price:  ${american_price:.2f}")  # $5.65
```

## Command-Line Interface

After installation, you can price options directly from the command line:

```bash
# Price an ATM call with Black-Scholes
options-pricer --spot 100 --strike 100 --rate 0.05 --vol 0.2 --time 1.0 --type call

# Price a put with binomial tree (200 steps)
options-pricer --spot 100 --strike 95 --rate 0.05 --vol 0.25 --time 0.5 --type put --method binomial --steps 200

# Price with Monte Carlo and display Greeks
options-pricer --spot 100 --strike 105 --rate 0.05 --vol 0.3 --time 1.0 --type call --method mc --paths 100000 --greeks

# American put option
options-pricer --spot 100 --strike 100 --rate 0.05 --vol 0.2 --time 1.0 --type put --style american --method binomial
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--spot` | Current spot price (required) |
| `--strike` | Strike price (required) |
| `--rate` | Risk-free rate, e.g., 0.05 for 5% (required) |
| `--vol` | Volatility, e.g., 0.2 for 20% (required) |
| `--time` | Time to maturity in years (required) |
| `--type` | Option type: `call` or `put` (required) |
| `--style` | Exercise style: `european` or `american` (default: european) |
| `--method` | Pricing method: `bs`, `binomial`, `mc` (default: bs) |
| `--steps` | Binomial tree steps (default: 100) |
| `--paths` | Monte Carlo paths (default: 100000) |
| `--greeks` | Also display Greeks (Black-Scholes only) |

## API Reference

### Core Types

- `Option` - Dataclass holding all option parameters
- `OptionType` - Enum: `CALL`, `PUT`
- `ExerciseStyle` - Enum: `EUROPEAN`, `AMERICAN`

### Pricing Functions

| Function | Description | Supports |
|----------|-------------|----------|
| `price(option)` | Black-Scholes closed-form | European only |
| `price_binomial(option, steps=100)` | CRR binomial tree | European & American |
| `price_monte_carlo(option, num_paths=100_000, antithetic=True, seed=None)` | Monte Carlo simulation | European only |

### Greeks (Black-Scholes)

| Function | Description |
|----------|-------------|
| `delta(option)` | Price sensitivity to spot |
| `gamma(option)` | Delta sensitivity to spot |
| `vega(option)` | Price sensitivity to volatility |
| `theta(option)` | Price sensitivity to time |
| `rho(option)` | Price sensitivity to interest rate |

## Additional Features

### Implied Volatility

```python
from options_pricing_engine import implied_volatility

# Back out IV from a market price
iv = implied_volatility(option, market_price=10.50)
print(f"Implied Vol: {iv:.2%}")  # 20.12%
```

### Digital Options

```python
from options_pricing_engine import price_digital_black_scholes, price_digital_monte_carlo

# Cash-or-nothing digital call
price = price_digital_black_scholes(option, payout=100)
mc_price, se = price_digital_monte_carlo(option, payout=100, num_paths=100_000)
```

### Portfolio Risk Management

```python
from options_pricing_engine import Position, Portfolio, portfolio_greeks, scenario_pnl

positions = [Position(call_option, 10), Position(put_option, -5)]
portfolio = Portfolio(positions)

greeks = portfolio_greeks(portfolio)
scenarios = scenario_pnl(portfolio, spot_shocks=[-10, 0, 10], vol_shocks=[-0.05, 0, 0.05])
```

## Future Work

- **More exotic options** - Asian, barrier, and lookback options via Monte Carlo
- **Stochastic volatility** - Heston model implementation
- **Dividend handling** - Discrete and continuous dividend yields
- **Greeks for all models** - Finite difference Greeks for binomial and Monte Carlo
- **Performance optimization** - Numba JIT compilation for Monte Carlo paths
- **Yield curve support** - Term structure of interest rates

## License

MIT License - see [LICENSE](LICENSE) for details.
