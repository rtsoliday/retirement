# Retirement simulator (Python desktop)

A Tkinter front end for the Monte Carlo model in `core.py`. It projects retirement balances while accounting for taxes, inflation, Social Security, healthcare, mortgage payments, and mortality.

## Files

- `montecarlo.py`: main desktop application.
- `monticarlo.py`: legacy spelling retained as a compatibility launcher.
- `core.py`: simulation and tax calculations.
- `DeathProbsE_*_Alt2_TR2025.csv`: mortality tables used by the application.
- `config.json`: saved scenario settings.
- `tests/`: unit and regression tests.

## Run

Python 3.12 or later is recommended. Install NumPy, Numba, pandas, and Matplotlib; Tkinter must also be available with your Python installation.

From the repository root:

```bash
python Python/retirement_simulator/montecarlo.py
pytest
```

The app reads its mortality tables and saves `config.json` in this directory regardless of the current working directory. Advanced callers can import `SimulationConfig` and `simulate` from `core.py`.
