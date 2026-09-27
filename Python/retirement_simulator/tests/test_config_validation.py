import numpy as np
import pytest

from core import SimulationConfig


def _make_config(death_probs):
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
        current_savings=0.0,
        savings_interest_rate=0.04,
        full_social_security_at_67=0.0,
        social_security_age_started=70,
        social_security_yearly_amount=0.0,
        mortgage_payment=0.0,
        mortgage_years_left=0,
        health_care_payment=0.0,
        percent_in_stock_after_retirement=1.0,
        bond_ratio=0.0,
        years_of_retirement=20,
        base_retirement_need=0.0,
        retirement_yearly_need=0.0,
        mortgage_years_in_retirement=0,
        mortgage_yearly_payment=0.0,
        health_care_years_in_retirement=0,
        health_care_yearly_payment=0.0,
        death_probs=death_probs,
        filing_status="single",
    )


def test_death_probs_required():
    with pytest.raises(ValueError):
        _make_config(np.array([]))


def test_death_probs_length_validation():
    # Required length is retirement_age + years_of_retirement = 80
    with pytest.raises(ValueError):
        _make_config(np.zeros(79))


def test_death_probs_length_ok():
    cfg = _make_config(np.zeros(80))
    assert cfg.death_probs.size == 80
