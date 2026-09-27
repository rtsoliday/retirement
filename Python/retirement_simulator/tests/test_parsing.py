import pytest

from monticarlo import parse_percent, parse_dollars


@pytest.mark.parametrize(
    "text, expected",
    [
        ("10%", 0.10),
        ("0%", 0.0),
        ("100%", 1.0),
    ],
)
def test_parse_percent_valid(text, expected):
    assert parse_percent(text) == expected


@pytest.mark.parametrize("text", ["101%", "-5%", "abc%"])
def test_parse_percent_invalid(text):
    with pytest.raises(ValueError):
        parse_percent(text)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("$1,234", 1234.0),
        ("0", 0.0),
        (" 500 ", 500.0),
    ],
)
def test_parse_dollars_valid(text, expected):
    assert parse_dollars(text) == expected


@pytest.mark.parametrize("text", ["-1", "-$5", "abc"])
def test_parse_dollars_invalid(text):
    with pytest.raises(ValueError):
        parse_dollars(text)
