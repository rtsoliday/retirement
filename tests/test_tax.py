import pytest
from types import SimpleNamespace

from monticarlo import tax_liability, brackets, rates


@pytest.fixture
def cfg():
    return SimpleNamespace(tax_brackets=brackets.copy(), tax_rates=rates.copy())


@pytest.mark.parametrize(
    "income, expected",
    [
        (0, 0.0),
        (11600, 1160.0),
        (11601, 1160.12),
        (47150, 5426.0),
        (47151, 5426.22),
        (100525, 17168.5),
        (100526, 17168.74),
        (191950, 39110.5),
        (191951, 39110.82),
        (243725, 55678.5),
        (243726, 55678.85),
        (609350, 183647.25),
        (609351, 183647.62),
    ],
)
def test_tax_liability(cfg, income, expected):
    assert tax_liability(income, cfg) == pytest.approx(expected)
