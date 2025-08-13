# Retirement Simulation

This project provides tools for Monte Carlo retirement planning.

The simulators now include a user-specific parameter for pre-Medicare health
insurance premiums. By default a $650 monthly cost is included and grows with
the mean inflation rate until the retiree reaches age 65.

## Testing

The test suite uses [pytest](https://pytest.org/). From the repository root, run:

```bash
pytest
```
