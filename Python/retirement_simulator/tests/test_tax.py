import pytest
from types import SimpleNamespace

from core import TAX_BRACKETS, TAX_RATES, tax_liability


@pytest.mark.parametrize(
    "status, income, expected",
    [
        # Single filer cases
        ("single", 0, 0.0),
        ("single", 11_600, 1_160.0),
        ("single", 11_601, 1_160.12),
        ("single", 47_150, 5_426.0),
        ("single", 47_151, 5_426.22),
        ("single", 100_525, 17_168.5),
        ("single", 100_526, 17_168.74),
        ("single", 191_950, 39_110.5),
        ("single", 191_951, 39_110.82),
        ("single", 243_725, 55_678.5),
        ("single", 243_726, 55_678.85),
        ("single", 609_350, 183_647.25),
        ("single", 609_351, 183_647.62),
        # Married filing jointly cases
        ("married", 0, 0.0),
        ("married", 23_200, 2_320.0),
        ("married", 23_201, 2_320.12),
        ("married", 94_300, 10_852.0),
        ("married", 94_301, 10_852.22),
        ("married", 201_050, 34_337.0),
        ("married", 201_051, 34_337.24),
        ("married", 383_900, 78_221.0),
        ("married", 383_901, 78_221.32),
        ("married", 487_450, 111_357.0),
        ("married", 487_451, 111_357.35),
        ("married", 731_200, 196_669.5),
        ("married", 731_201, 196_669.87),
        # Head of household cases
        ("head_of_household", 0, 0.0),
        ("head_of_household", 16_550, 1_655.0),
        ("head_of_household", 16_551, 1_655.12),
        ("head_of_household", 63_100, 7_241.0),
        ("head_of_household", 63_101, 7_241.22),
        ("head_of_household", 100_500, 15_469.0),
        ("head_of_household", 100_501, 15_469.24),
        ("head_of_household", 191_950, 37_417.0),
        ("head_of_household", 191_951, 37_417.32),
        ("head_of_household", 243_700, 53_977.0),
        ("head_of_household", 243_701, 53_977.35),
        ("head_of_household", 609_350, 181_954.5),
        ("head_of_household", 609_351, 181_954.87),
    ],
)
def test_tax_liability(status, income, expected):
    cfg = SimpleNamespace(
        tax_brackets=TAX_BRACKETS[status].copy(),
        tax_rates=TAX_RATES[status].copy(),
    )
    assert tax_liability(income, cfg) == pytest.approx(expected)

