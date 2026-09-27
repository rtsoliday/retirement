# Retirement tools

This repository contains four separate programs. Each program's code, data, and tests are grouped under its own directory.

| Program | Platform | Directory |
| --- | --- | --- |
| Retirement simulator | Python desktop (Tkinter) | [`Python/retirement_simulator/`](Python/retirement_simulator/) |
| Mortgage vs. investment calculator | Python desktop (Tkinter) | [`Python/mortgage_investment/`](Python/mortgage_investment/) |
| Fidelity fund ranker | Python command line | [`Python/fidelity_fund_ranker/`](Python/fidelity_fund_ranker/) |
| Retirement Readiness Lab | Native Android | [`Android/RetirementReadinessLab/`](Android/RetirementReadinessLab/) |

See each program's README for requirements and run commands. The Python retirement simulator and Android app are separate implementations of the retirement planning concept.

## Quick start

From the repository root:

```bash
python Python/retirement_simulator/montecarlo.py
python Python/mortgage_investment/mortgage_investment.py
python Python/fidelity_fund_ranker/rank_fidelity_funds.py --help
pytest
```

Open `Android/RetirementReadinessLab/android` in Android Studio, or run its Gradle wrapper from that directory.

`Python/retirement_simulator/monticarlo.py` is a compatibility launcher for `montecarlo.py`; it is not a separate app. `core.py` is the simulator's calculation library. There is no Kivy app in this repository.

## License

Released under the [GNU General Public License v3](LICENSE).

## Disclaimer

These programs are for educational purposes only and do not constitute financial advice.
