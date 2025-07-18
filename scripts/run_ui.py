#!/usr/bin/env python3
"""
Entry point for the portfolio simulator UI using new modular structure.
"""

import subprocess
import sys
import os


def main():
    """Main entry point for console script."""
    # Get the path to the portfolio_simulator_ui.py file which uses the new modular structure
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ui_script = os.path.join(script_dir, "portfolio_simulator_ui.py")
    
    # Run streamlit with the UI script
    cmd = [sys.executable, "-m", "streamlit", "run", ui_script]
    
    # Pass through any additional arguments
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    
    # Execute streamlit
    subprocess.run(cmd)


if __name__ == "__main__":
    main()