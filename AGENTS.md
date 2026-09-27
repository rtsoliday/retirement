# Agent Notes for retirement

This file contains a short tactical summary based on repository evidence. `../llm-wiki/scripts/refresh_wiki.py` rewrites only the machine-managed block.

<!-- BEGIN MACHINE:summary -->
## Quick start
- This repository has four separate programs: three under `Python/` and one under `Android/RetirementReadinessLab/`.
- Read the root `README.md` and the README in the relevant program directory before changing it.

## Programs and tests
- `Python/retirement_simulator/`: Tkinter app (`montecarlo.py`), calculation library (`core.py`), data, and tests. `monticarlo.py` is a compatibility launcher.
- `Python/mortgage_investment/`: standalone Tkinter calculator.
- `Python/fidelity_fund_ranker/`: standalone command-line script.
- `Android/RetirementReadinessLab/android/`: native Android Gradle project. Product docs live one level above the Gradle root.
- Run Python tests with `pytest` from the repository root; `pytest.ini` points to the retirement simulator tests.
- Run Android unit tests from the Android Gradle root with `./gradlew :app:testDebugUnitTest`.

## Related knowledge
- Repository-local documentation should be treated as authoritative.
- If a shared `llm-wiki/` directory is present in this workspace or parent folder, consult [the matching repo page](../llm-wiki/repos/retirement.md) for additional architectural context.
- If no shared wiki is present, continue using repository-local evidence only.
- If present in this workspace, [the cross-repo map](../llm-wiki/insights/cross-repo-map.md) helps explain related repositories.
<!-- END MACHINE:summary -->

## Human notes
Add durable repo-specific instructions here.
