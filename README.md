# Retirement Simulator

A Python toolkit for Monte Carlo retirement planning. The simulator projects portfolio balances through retirement while accounting for taxes, inflation, Social Security and pre-Medicare health costs.

## Features
- **Monte Carlo engine** – models stock and bond returns, inflation, and dynamic asset allocation.
- **Tax aware** – computes federal income tax using 2024 brackets for single, married and head of household filers.
- **Social Security** – accepts the age benefits start and adjusts payouts accordingly.
- **Health care before Medicare** – includes an optional pre‑65 premium (defaults to $650/month) that grows with inflation until age 65.
- **Mortgage and other spending** – yearly retirement needs can include mortgage and health care payments that phase out after a set number of years.
- **Mortality modeling** – uses gender‑specific probability tables to simulate death and stop withdrawals.
- **Configuration persistence** – GUI inputs are saved to and loaded from `config.json`.
- **Interactive front‑ends** – Tkinter (`monticarlo.py`) and Kivy (`kivy_app.py`) interfaces with optional plotting of successful and failed paths.

## Installation
Python 3.12 or later is recommended.

```bash
pip install numpy matplotlib kivy
```
Tkinter ships with the standard Python distribution.

## Running the simulator
### Tkinter interface
```bash
python monticarlo.py
```

### Kivy interface
```bash
python kivy_app.py
```

Both interfaces allow you to adjust general market assumptions and user‑specific parameters (age, savings, mortgage, health care, etc.), run simulations, and view success rates. Results and settings are persisted in `config.json`.

For advanced use, the `core` module exposes a `SimulationConfig` dataclass and a `simulate` function that returns the percentage of runs where funds last through all retirement years.

## Testing
Run the test suite with [pytest](https://pytest.org/):

```bash
pytest
```

## License
Released under the [GNU General Public License v3](LICENSE).

## Disclaimer
This project is for educational purposes only and does not constitute financial advice.
