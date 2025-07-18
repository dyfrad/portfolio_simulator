#!/usr/bin/env python3
"""
New entry point for the portfolio simulator core functionality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the core simulator
import portfolio_simulator


def main():
    """Main entry point for console script."""
    print("Portfolio Simulator Core Module Loaded Successfully")
    print("Available functions:")
    print("- portfolio_simulator.PortfolioSimulator()")
    print("- Use portfolio_simulator_ui.py for interactive interface")


if __name__ == "__main__":
    main()