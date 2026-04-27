# Agent Notes for retirement

This file contains a short tactical summary based on repository evidence. `../llm-wiki/scripts/refresh_wiki.py` rewrites only the machine-managed block.

<!-- BEGIN MACHINE:summary -->
## Quick start
- Repository-local guidance is sufficient: start with `AGENTS.md`, `README.md`, `docs/`, build/test/config files, and the source tree.
- A Python toolkit for Monte Carlo retirement planning. The simulator projects portfolio balances through retirement while accounting for taxes, inflation, Social Security and pre-Medicare health costs.
- Primary work areas: `tests`.

## Read first
- `README.md`: Primary project overview and workflow notes
- `pytest.ini`: Build system entry point or dependency manifest
- `core.py`: Likely operator or developer entry point
- `montecarlo.py`: Likely operator or developer entry point
- `mortgage_investment.py`: Likely operator or developer entry point

## Build and test
- Documented setup/build commands: `pip install numpy matplotlib kivy`, `make it effectively proprietary. To prevent this, the GPL assures that`.
- Documented test commands: `pytest`.
- Likely run commands or operator entry points: `python monticarlo.py`, `python kivy_app.py`.

## Operational warnings
- No operational warnings were extracted from the inspected files.

## Compatibility constraints
- No explicit compatibility constraints were extracted from the inspected files.

## Related knowledge
- Repository-local documentation should be treated as authoritative.
- If a shared `llm-wiki/` directory is present in this workspace or parent folder, consult [the matching repo page](../llm-wiki/repos/retirement.md) for additional architectural context.
- If no shared wiki is present, continue using repository-local evidence only.
- If present in this workspace, [the cross-repo map](../llm-wiki/insights/cross-repo-map.md) helps explain related repositories.
<!-- END MACHINE:summary -->

## Human notes
Add durable repo-specific instructions here.
