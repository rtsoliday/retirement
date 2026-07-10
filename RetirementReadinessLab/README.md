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
- A native Android project under `android/`.
- Kotlin domain models and local scenario persistence.
- A monthly Monte Carlo simulation engine with deterministic seeds and calculation provenance.
- Jetpack Compose screens for Dashboard, Setup, Scenarios, Lab, Assumptions, Results, Welcome, and Reports.
- PDF/text report export paths, with JSON backup/import hidden for the first release.
- App icon, splash branding, privacy/disclaimer copy, and release-prep documentation.
- Unit tests, Compose instrumentation test compilation, and a simulator performance profile.

The app has been built and launched from Android Studio during development. Use the Gradle wrapper from `RetirementReadinessLab/android`.

## Open In Android Studio

Open this folder as an Android project:

```text
RetirementReadinessLab/android
```

Android Studio should sync Gradle dependencies from the root `settings.gradle.kts` and `build.gradle.kts` files.

## Implementation Notes

The current simulation engine is still smaller than the mature Python engine. It provides a working Android product foundation and should continue to be expanded against `docs/simulation_model.md` before release.

Before commercial release, resolve the licensing decision documented in `IMPLEMENTATION_PLAN.md`.

## Useful Commands

From `RetirementReadinessLab/android`:

```bash
./gradlew :app:testDebugUnitTest
./gradlew :app:assembleDebug
./gradlew :app:assembleDebugAndroidTest
```

Release-prep docs:

- `docs/release_checklist.md`
- `docs/closed_testing_plan.md`
- `docs/store_listing.md`
- `docs/privacy_policy_draft.md`
