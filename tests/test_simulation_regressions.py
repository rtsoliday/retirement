import numpy as np

from core import MEDICARE_PART_B_BASE, MEDICARE_PART_D_BASE, SimulationConfig, simulate


def _make_config(**overrides):
    values = dict(
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
        current_age=65,
        retirement_age=65,
        average_yearly_need=0.0,
        current_roth=0.0,
        current_401a_and_403b=0.0,
        current_savings=0.0,
        savings_interest_rate=0.0,
        full_social_security_at_67=0.0,
        social_security_age_started=70,
        social_security_yearly_amount=0.0,
        mortgage_payment=0.0,
        mortgage_years_left=0,
        percent_in_stock_after_retirement=1.0,
        bond_ratio=0.0,
        years_of_retirement=1,
        base_retirement_need=0.0,
        retirement_yearly_need=0.0,
        mortgage_years_in_retirement=0,
        mortgage_yearly_payment=0.0,
        health_care_payment=0.0,
        health_care_years_in_retirement=0,
        health_care_yearly_payment=0.0,
        healthcare_inflation_mean=0.0,
        healthcare_inflation_std=0.0,
        include_medicare_premiums=True,
        include_ltc_risk=False,
        filing_status="single",
        death_probs=np.zeros(100),
    )
    values.update(overrides)
    return SimulationConfig(**values)


def test_medicare_premiums_apply_in_first_retirement_month_at_65():
    annual_medicare = MEDICARE_PART_B_BASE + MEDICARE_PART_D_BASE
    monthly_grossed_up_for_ten_percent_tax = (annual_medicare / 0.9) / 12

    cfg = _make_config(
        current_savings=monthly_grossed_up_for_ten_percent_tax * 11.5,
    )

    assert simulate(cfg) == 0.0
