import numpy as np
import pytest

from core import SimulationConfig, perform_roth_conversion, tax_liability


def _make_config(enable: bool = True, rate: float = 0.12) -> SimulationConfig:
    return SimulationConfig(
        number_of_simulations=1,
        pre_retirement_mean_return=0.0,
        pre_retirement_std_dev=0.0,
        stock_mean_return=0.0,
        stock_std_dev=0.0,
        bond_mean_return=0.0,
        bond_std_dev=0.0,
        inflation_mean=0.0,
        inflation_std_dev=0.0,
        gender="male",
        current_age=60,
        retirement_age=60,
        average_yearly_need=0.0,
        current_roth=0.0,
        current_401a_and_403b=0.0,
        full_social_security_at_67=0.0,
        social_security_age_started=70,
        social_security_yearly_amount=0.0,
        mortgage_payment=0.0,
        mortgage_years_left=0,
        health_care_payment=0.0,
        percent_in_stock_after_retirement=1.0,
        bond_ratio=0.0,
        years_of_retirement=1,
        base_retirement_need=0.0,
        retirement_yearly_need=0.0,
        mortgage_years_in_retirement=0,
        mortgage_yearly_payment=0.0,
        health_care_years_in_retirement=0,
        health_care_yearly_payment=0.0,
        death_probs=np.zeros(200),
        filing_status="single",
        enable_roth_conversion=enable,
        roth_conversion_rate_cap=rate if enable else None,
    )


def test_perform_roth_conversion_fills_target_bracket():
    cfg = _make_config(rate=0.12)
    pretax_after, roth_after, taxable_after = perform_roth_conversion(
        100_000.0, 50_000.0, 20_000.0, cfg
    )

    expected_income = cfg.tax_brackets[2]
    converted = expected_income - 20_000.0
    additional_tax = tax_liability(expected_income, cfg) - tax_liability(20_000.0, cfg)

    assert taxable_after == pytest.approx(expected_income)
    assert pretax_after == pytest.approx(100_000.0 - converted)
    assert roth_after == pytest.approx(50_000.0 + converted - additional_tax)


def test_perform_roth_conversion_limits_by_balance():
    cfg = _make_config(rate=0.22)
    pretax_after, roth_after, taxable_after = perform_roth_conversion(
        8_000.0, 0.0, 0.0, cfg
    )

    additional_tax = tax_liability(8_000.0, cfg)

    assert taxable_after == pytest.approx(8_000.0)
    assert pretax_after == pytest.approx(0.0)
    assert roth_after == pytest.approx(8_000.0 - additional_tax)


def test_perform_roth_conversion_disabled():
    cfg = _make_config(enable=False)
    pretax_after, roth_after, taxable_after = perform_roth_conversion(
        50_000.0, 5_000.0, 10_000.0, cfg
    )

    assert pretax_after == 50_000.0
    assert roth_after == 5_000.0
    assert taxable_after == 10_000.0
