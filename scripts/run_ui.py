#!/usr/bin/env python3
"""
New entry point for the portfolio simulator UI.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import and run the UI
from portfolio_simulator_ui import main

if __name__ == "__main__":
    main()