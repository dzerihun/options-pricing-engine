"""
Command-line interface for the Options Pricing Engine.

Usage:
    options-pricer --spot 100 --strike 100 --rate 0.05 --vol 0.2 --time 1.0 --type call
    options-pricer --spot 100 --strike 100 --rate 0.05 --vol 0.2 --time 1.0 --type put --method binomial --steps 200
    options-pricer --spot 100 --strike 100 --rate 0.05 --vol 0.2 --time 1.0 --type call --method mc --paths 100000
"""

import argparse
import sys

from .core.option_types import Option, OptionType, ExerciseStyle
from .models import price, price_binomial, price_monte_carlo, delta, gamma, vega, theta, rho


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="options-pricer",
        description="Price options using Black-Scholes, binomial tree, or Monte Carlo methods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Price an ATM call with Black-Scholes
  options-pricer --spot 100 --strike 100 --rate 0.05 --vol 0.2 --time 1.0 --type call

  # Price a put with binomial tree (200 steps)
  options-pricer --spot 100 --strike 95 --rate 0.05 --vol 0.25 --time 0.5 --type put --method binomial --steps 200

  # Price with Monte Carlo and show Greeks
  options-pricer --spot 100 --strike 105 --rate 0.05 --vol 0.3 --time 1.0 --type call --method mc --greeks
        """,
    )

    # Required option parameters
    parser.add_argument("--spot", type=float, required=True, help="Current spot price")
    parser.add_argument("--strike", type=float, required=True, help="Strike price")
    parser.add_argument("--rate", type=float, required=True, help="Risk-free interest rate (e.g., 0.05 for 5%%)")
    parser.add_argument("--vol", type=float, required=True, help="Volatility (e.g., 0.2 for 20%%)")
    parser.add_argument("--time", type=float, required=True, help="Time to maturity in years")
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["call", "put"],
        help="Option type",
    )

    # Optional parameters
    parser.add_argument(
        "--style",
        type=str,
        default="european",
        choices=["european", "american"],
        help="Exercise style (default: european)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="bs",
        choices=["bs", "binomial", "mc"],
        help="Pricing method: bs (Black-Scholes), binomial, mc (Monte Carlo)",
    )

    # Method-specific parameters
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of steps for binomial tree (default: 100)",
    )
    parser.add_argument(
        "--paths",
        type=int,
        default=100000,
        help="Number of paths for Monte Carlo (default: 100000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for Monte Carlo (default: None)",
    )

    # Output options
    parser.add_argument(
        "--greeks",
        action="store_true",
        help="Also compute and display Greeks (Black-Scholes only)",
    )

    return parser


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Create the option
    option_type = OptionType.CALL if args.type == "call" else OptionType.PUT
    exercise_style = ExerciseStyle.EUROPEAN if args.style == "european" else ExerciseStyle.AMERICAN

    try:
        option = Option(
            spot=args.spot,
            strike=args.strike,
            rate=args.rate,
            volatility=args.vol,
            time_to_maturity=args.time,
            option_type=option_type,
            exercise_style=exercise_style,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Price the option
    print(f"\nOption: {args.type.upper()} @ K={args.strike}, S={args.spot}")
    print(f"Params: r={args.rate:.2%}, σ={args.vol:.2%}, T={args.time:.2f}y, {args.style}")
    print("-" * 50)

    if args.method == "bs":
        if exercise_style == ExerciseStyle.AMERICAN and option_type == OptionType.PUT:
            print("Warning: Black-Scholes doesn't account for early exercise of American puts")

        option_price = price(option)
        print(f"Black-Scholes Price: ${option_price:.4f}")

        if args.greeks:
            print("\nGreeks:")
            print(f"  Delta: {delta(option):+.4f}")
            print(f"  Gamma: {gamma(option):+.6f}")
            print(f"  Vega:  {vega(option):+.4f}")
            print(f"  Theta: {theta(option):+.4f}")
            print(f"  Rho:   {rho(option):+.4f}")

    elif args.method == "binomial":
        option_price = price_binomial(option, steps=args.steps)
        print(f"Binomial Tree Price ({args.steps} steps): ${option_price:.4f}")

    elif args.method == "mc":
        if exercise_style == ExerciseStyle.AMERICAN:
            print("Error: Monte Carlo only supports European options", file=sys.stderr)
            return 1

        mc_price, mc_se = price_monte_carlo(
            option,
            num_paths=args.paths,
            seed=args.seed,
        )
        print(f"Monte Carlo Price ({args.paths:,} paths): ${mc_price:.4f} ± ${mc_se:.4f}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
