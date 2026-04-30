# Retirement Readiness Lab

Retirement Readiness Lab is the Android-first product version of the retirement simulator concept. It is intentionally separated from the existing Python desktop application.

The app is designed as a guided retirement stress-test:

- Build a retirement scenario.
- Run a Monte Carlo readiness estimate.
- Compare retirement age, spending, Social Security, Roth conversion, healthcare, and long-term care assumptions.
- Keep financial data local by default.

## Current Status

This folder currently contains:

- A detailed implementation plan.
- Product and design documentation.
- A native Android project skeleton under `android/`.
- Initial Kotlin domain models.
- A first-pass deterministic Monte Carlo simulation engine.
- Initial Jetpack Compose screens for Dashboard, Setup, Scenarios, Lab, Assumptions, and Reports.

The Android project is scaffolded but was not built in this environment because Gradle and the Android SDK are not installed here.

## Open In Android Studio

Open this folder as an Android project:

```text
RetirementReadinessLab/android
```

Android Studio should sync Gradle dependencies from the root `settings.gradle.kts` and `build.gradle.kts` files.

## Implementation Notes

The current simulation engine is deliberately smaller than the mature Python engine. It is meant to provide a working Android product foundation and should be expanded against `docs/simulation_model.md` before release.

Before commercial release, resolve the licensing decision documented in `IMPLEMENTATION_PLAN.md`.
