import pytest

from core import taxable_social_security


@pytest.mark.parametrize(
    "filing_status, other_income, ss_amount, expected",
    [
        ("single", 0.0, 20_000.0, 0.0),
        ("single", 26_000.0, 10_000.0, 3_000.0),
        ("single", 30_000.0, 20_000.0, 9_600.0),
        ("married", 20_000.0, 30_000.0, 1_500.0),
        ("married", 60_000.0, 30_000.0, 25_500.0),
        ("head_of_household", 26_000.0, 10_000.0, 3_000.0),
    ],
)
def test_taxable_social_security(filing_status, other_income, ss_amount, expected):
    assert taxable_social_security(other_income, ss_amount, filing_status) == pytest.approx(
        expected
    )
