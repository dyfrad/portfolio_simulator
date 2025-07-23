# Portfolio Simulator

A personal, locally-run Streamlit dashboard for simulating and analyzing investment portfolios. Features include interactive portfolio allocation, backtesting, Monte Carlo simulations, DCA, fees/taxes, rebalancing, stress scenarios, and more. Designed for private financial planning and educational use.

---

## Features

- **Portfolio Allocation:** Adjust weights for default ETFs or add custom tickers
- **Inflation Adjustment:** Input expected annual inflation rate
- **Custom Assets:** Add tickers via text input
- **Visualizations:** Line charts for returns, drawdowns, weight drift, simulation outcomes
- **Backtesting:** Historical performance simulation, including DCA
- **DCA Support:** Monthly/quarterly contributions
- **Fees & Taxes:** TER, transaction fees, capital gains tax
- **Rebalancing:** Automatic, with drift visualization
- **Advanced Metrics:** Sharpe, Sortino, max drawdown, VaR, CVaR
- **Stress Scenarios:** Predefined (e.g., 2008 Recession, COVID Crash)
- **Portfolio Upload:** CSV for holdings or Degiro transaction history
- **Educational Tooltips:** Explanations for metrics

---

## Quick Start

1. **Python 3.8+ required** (recommend 3.12)
2. Clone the repo:
   ```bash
   git clone git@github.com:dyfrad/portfolio_simulator.git
   cd portfolio_simulator
   ```
3. Create & activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install in development mode:
   ```bash
   pip install -e .
   # Or, for legacy:
   pip install -r requirements.txt
   ```
5. (Optional) Install dev/test dependencies:
   ```bash
   pip install -e ".[dev,test]"
   ```

---

## Running the App

- **Recommended:**
  ```bash
  portfolio-ui
  portfolio-simulator
  ```
- **Legacy:**
  ```bash
  streamlit run portfolio_simulator_ui.py
  ```
- **Script:**
  ```bash
  python scripts/run_ui.py
  ```

App launches at [http://localhost:8501](http://localhost:8501).

> **Note:** Requires internet for Yahoo Finance data.

---

## Docker

1. Build image:
   ```bash
   docker build -t portfolio-simulator .
   ```
2. Run container:
   ```bash
   docker run -p 8501:8501 portfolio-simulator
   ```
3. Access at [http://localhost:8501](http://localhost:8501)

---

## Project Structure

```
portfolio_simulator/
├── src/portfolio_simulator/      # Main package
│   ├── core/                    # Business logic
│   ├── config/                  # Config management
│   └── ui/                      # UI components
├── scripts/                     # Entry point scripts
├── tests/                       # Unit & integration tests
├── pyproject.toml               # Packaging config
├── requirements.txt             # Legacy dependencies
├── Dockerfile                   # Container config
└── README.md
```

---

## Configuration

- **Constants:** `src/portfolio_simulator/config/constants.py`
- **Settings:** `src/portfolio_simulator/config/settings.py`
- **Environments:** `src/portfolio_simulator/config/environments/`
- **Environment Variables:**
  ```bash
  export ENVIRONMENT=development  # or production, testing
  export DEBUG=true
  ```

---

## Development

- **Core logic:** `src/portfolio_simulator/core/`
- **UI:** `src/portfolio_simulator/ui/components/`
- **Config:** `src/portfolio_simulator/config/`
- **Tests:** `tests/unit/`, `tests/integration/`
- **Code Quality:** `black`, `flake8`, `mypy`, `pre-commit`

---

## Testing

Run from project root:
```bash
pytest
```
- For coverage:
  ```bash
  pytest --cov=src/portfolio_simulator
  ```
- Unit or integration only:
  ```bash
  pytest tests/unit/
  pytest tests/integration/
  ```

---

## Troubleshooting

- **Data Fetch Errors:** Check ticker validity and Yahoo Finance access
- **yfinance Issues:** Update with `pip install yfinance --upgrade` ([yfinance GitHub](https://github.com/ranaroussi/yfinance))
- **Optimization Failure:** Falls back to equal weights
- **Performance:** Lower simulation count for faster runs
- **Docker:** Ensure Docker is running and ports are free
- **Import Errors:** Reinstall with `pip install -e .` and check `__init__.py` files
- **Entry Point Issues:** Reinstall, check PATH, or use legacy method
- **Test Failures:** Install test deps, run from root, check install with `pip show portfolio-simulator`

---

## Notes

- For personal/educational use only. Not financial advice.
- No external API keys required. Data from Yahoo Finance.
- Keep repo private. Do not upload sensitive data to public deployments.
- Backup your local repo regularly.

---

## Optional: Cloud Deployment

- Deploy to [Streamlit Community Cloud](https://share.streamlit.io) (private repo recommended)
- Add `requirements.txt` to repo
- Specify entry file (e.g., `portfolio_simulator_ui.py`)

---

## License

Licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). See [LICENSE](LICENSE).