# Performance Testing

## Current Coverage

The Android unit suite includes `RetirementSimulatorPerformanceTest`.

Automated profile:

- Runs the monthly simulator for 1,000 simulations.
- Verifies the result contract still has annual balance bands.
- Prints elapsed runtime to the test log.
- Uses a generous default threshold of 15 seconds to avoid flaky local builds.

Manual release profile:

- A 10,000-simulation test is included but ignored by default.
- Use it before release on representative hardware or an emulator profile that resembles a mid-range Android device.
- The goal is to confirm the simulator remains responsive when run from background dispatchers and does not change the public result contract.
- CSV comparison export is deferred for the first release.

## Commands

Standard test run:

```bash
./gradlew :app:testDebugUnitTest
```

Use a stricter local threshold for the automated profile:

```bash
./gradlew :app:testDebugUnitTest -DretirementLab.maxPerfMillis=5000
```

To run the ignored 10,000-simulation profile, remove `@Ignore` from `tenThousandSimulationManualProfileCompletesWithoutChangingResultContract` temporarily or run that method directly from Android Studio.

## Release Notes

Performance should be checked again after any change to:

- Simulation cadence.
- Tax gross-up calculations.
- Lab sweeps or optimizer loops.
- Result path collection.
- Scenario Lab sweeps or future scenario comparison batch runs.
