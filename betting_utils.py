#!/usr/bin/env python3
"""Utility functions for betting calculations."""

from typing import Optional
import argparse
import sys


def kelly_criterion(prob: float, odds: float, bankroll: float) -> Optional[float]:
    """Return bet size using the Kelly Criterion.

    Parameters
    ----------
    prob : float
        Estimated probability of winning (0-1).
    odds : float
        Decimal odds from sportsbook.
    bankroll : float
        Current bankroll available.
    Returns
    -------
    float or None
        Recommended amount to wager. None if no edge.
    """
    if prob <= 0 or odds <= 1:
        return None

    edge = prob * (odds - 1) - (1 - prob)
    if edge <= 0:
        return None
    fraction = edge / (odds - 1)
    return max(0.0, bankroll * fraction)


def main(argv: list[str] | None = None) -> int:
    """Command-line interface for calculating bet size."""
    parser = argparse.ArgumentParser(description="Calculate Kelly bet size")
    parser.add_argument("probability", type=float, help="Win probability (0-1)")
    parser.add_argument("odds", type=float, help="Decimal odds")
    parser.add_argument("bankroll", type=float, help="Current bankroll")

    args = parser.parse_args(argv)

    bet = kelly_criterion(args.probability, args.odds, args.bankroll)
    if bet is None:
        print("No positive edge - do not bet")
        return 1

    print(f"Recommended wager: {bet:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

