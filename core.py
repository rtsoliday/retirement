"""Core functionality for retirement simulations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# Monte Carlo setup
np.random.seed()


# IRS 2024 tax brackets and marginal rates by filing status
# Values sourced from IRS tables for the 2024 tax year.  Each status maps to
# the bracket thresholds that apply to taxable income for that filer type.
TAX_BRACKETS = {
    "single": [0, 11_600, 47_150, 100_525, 191_950, 243_725, 609_350],
    "married": [0, 23_200, 94_300, 201_050, 383_900, 487_450, 731_200],
    "head_of_household": [0, 16_550, 63_100, 100_500, 191_950, 243_700, 609_350],
}

TAX_RATES = {
    status: [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]
    for status in TAX_BRACKETS
}

# Provide module-level single-filer defaults for backwards compatibility
brackets = TAX_BRACKETS["single"]
rates = TAX_RATES["single"]


CONFIG_FILE = "config.json"


def parse_percent(val: str) -> float:
    """Convert a percentage string like '10%' to a float 0.10."""

    try:
        pct = float(val.strip().rstrip("%")) / 100
    except ValueError as exc:  # pragma: no cover - error path simple
        raise ValueError(f"Invalid percentage: {val!r}") from exc
    if not 0 <= pct <= 1:
        raise ValueError("Percentage must be between 0% and 100%")
    return pct


def parse_dollars(val: str) -> float:
    """Convert a currency string like '$1,234' to a float 1234.0."""

    try:
        amt = float(val.replace("$", "").replace(",", "").strip())
    except ValueError as exc:  # pragma: no cover - error path simple
        raise ValueError(f"Invalid dollar amount: {val!r}") from exc
    if amt < 0:
        raise ValueError("Dollar amount cannot be negative")
    return amt


def sample_death_year(retirement_age: int, years_of_retirement: int, death_probs) -> int:
    """Return the year index (0-indexed) in which death occurs."""

    for j in range(years_of_retirement):
        age = retirement_age + j
        if age >= len(death_probs):
            return years_of_retirement
        if np.random.random() < death_probs[age]:
            return j
    return years_of_retirement  # lived through entire horizon


@dataclass
class SimulationConfig:
    number_of_simulations: int
    pre_retirement_mean_return: float
    pre_retirement_std_dev: float
    stock_mean_return: float
    stock_std_dev: float
    bond_mean_return: float
    bond_std_dev: float
    inflation_mean: float
    inflation_std_dev: float
    gender: str
    current_age: int
    retirement_age: int
    average_yearly_need: float
    current_roth: float
    current_401a_and_403b: float
    full_social_security_at_67: float
    social_security_age_started: int
    social_security_yearly_amount: float
    mortgage_payment: float
    mortgage_years_left: int
    percent_in_stock_after_retirement: float
    bond_ratio: float
    years_of_retirement: int
    base_retirement_need: float
    retirement_yearly_need: float
    mortgage_years_in_retirement: int
    mortgage_yearly_payment: float
    health_care_payment: float
    health_care_years_in_retirement: int
    health_care_yearly_payment: float
    filing_status: str = "single"
    tax_brackets: list[float] = field(default_factory=list)
    tax_rates: list[float] = field(default_factory=list)
    death_probs: np.ndarray = field(default_factory=lambda: np.array([]))
    enable_roth_conversion: bool = False
    roth_conversion_rate_cap: Optional[float] = None
    _roth_conversion_upper_limit: Optional[float] = field(
        init=False, repr=False, default=None
    )
    _roth_conversion_rate_index: Optional[int] = field(
        init=False, repr=False, default=None
    )

    def __post_init__(self) -> None:
        if not self.tax_brackets or not self.tax_rates:
            if self.filing_status not in TAX_BRACKETS:
                raise ValueError(f"Unknown filing status: {self.filing_status}")
            self.tax_brackets = TAX_BRACKETS[self.filing_status].copy()
            self.tax_rates = TAX_RATES[self.filing_status].copy()
        self._roth_conversion_rate_index = None
        self._roth_conversion_upper_limit = None
        if self.enable_roth_conversion:
            if self.roth_conversion_rate_cap is None:
                raise ValueError(
                    "roth_conversion_rate_cap must be provided when Roth conversions are enabled"
                )
            idx = None
            for i, rate in enumerate(self.tax_rates):
                if abs(rate - self.roth_conversion_rate_cap) < 1e-9:
                    idx = i
                    break
            if idx is None:
                raise ValueError(
                    "roth_conversion_rate_cap must match one of the configured tax rates"
                )
            self._roth_conversion_rate_index = idx
            if idx + 1 < len(self.tax_brackets):
                self._roth_conversion_upper_limit = self.tax_brackets[idx + 1]
            else:
                # Top bracket has no formal cap; treat as unlimited
                self._roth_conversion_upper_limit = None


def tax_liability(income: float, cfg: SimulationConfig) -> float:
    """Compute tax owed for a given income using marginal brackets."""

    tax = 0.0
    for i in range(len(cfg.tax_rates)):
        lower = cfg.tax_brackets[i]
        upper = cfg.tax_brackets[i + 1] if i + 1 < len(cfg.tax_brackets) else income
        if income > lower:
            tax += (min(income, upper) - lower) * cfg.tax_rates[i]
        else:
            break
    return tax


def gross_from_net(net_amt: float, cfg: SimulationConfig) -> float:
    """Find G such that G - tax_liability(G) = net_amt."""

    lo, hi = net_amt, net_amt * 1.5 + 20_000
    for _ in range(40):
        mid = (lo + hi) / 2
        if mid - tax_liability(mid, cfg) < net_amt:
            lo = mid
        else:
            hi = mid
    return hi


def gross_from_net_with_ss(
    net_amt: float, social_security_yearly_amount: float, cfg: SimulationConfig
) -> float:
    """
    Find G such that
       (G + social_security_yearly_amount) - tax_liability(G + social_security_yearly_amount, cfg) == net_amt
    i.e. accounts + SS minus tax on total = net spending.
    """

    lo, hi = (
        max(0, net_amt - social_security_yearly_amount),
        (net_amt + social_security_yearly_amount) * 1.5,
    )
    for _ in range(40):
        mid = (lo + hi) / 2
        total = mid + social_security_yearly_amount
        if total - tax_liability(total, cfg) < net_amt:
            lo = mid
        else:
            hi = mid
    return hi


def _roth_conversion_headroom(
    taxable_income: float, pretax_balance: float, cfg: SimulationConfig
) -> float:
    """Return the additional taxable income that can be generated via conversions."""

    if not cfg.enable_roth_conversion or pretax_balance <= 0:
        return 0.0
    limit = cfg._roth_conversion_upper_limit
    if limit is None:
        # No explicit upper bound – convert the entire pretax balance
        return pretax_balance
    return max(0.0, min(limit - taxable_income, pretax_balance))


def perform_roth_conversion(
    pretax_balance: float, roth_balance: float, taxable_income: float, cfg: SimulationConfig
) -> Tuple[float, float, float]:
    """
    Apply a Roth conversion by filling up the configured tax bracket.

    Returns the updated pretax balance, Roth balance, and taxable income after
    any conversion. Taxes due on the conversion are withdrawn from Roth funds
    first, then pretax funds if necessary to keep balances non-negative.
    """

    headroom = _roth_conversion_headroom(taxable_income, pretax_balance, cfg)
    if headroom <= 0:
        return pretax_balance, roth_balance, taxable_income

    taxes_before = tax_liability(taxable_income, cfg)
    taxes_after = tax_liability(taxable_income + headroom, cfg)
    additional_tax = max(0.0, taxes_after - taxes_before)

    pretax_balance -= headroom
    roth_balance += headroom
    taxable_income += headroom

    if additional_tax > 0:
        if roth_balance >= additional_tax:
            roth_balance -= additional_tax
            additional_tax = 0.0
        else:
            additional_tax -= roth_balance
            roth_balance = 0.0
        if additional_tax > 0:
            pretax_balance = max(0.0, pretax_balance - additional_tax)

    return pretax_balance, roth_balance, taxable_income


def social_security_payout(full_ss_at_67: float, start_age: int) -> float:
    """Calculate yearly Social Security benefit based on start age."""

    if start_age == 67:
        return full_ss_at_67
    months_difference = (start_age - 67) * 12
    if start_age < 67:
        months_early = -months_difference
        if months_early <= 36:
            reduction = months_early * (5 / 9) / 100
        else:
            reduction = (36 * (5 / 9) + (months_early - 36) * (5 / 12)) / 100
        return full_ss_at_67 * (1 - reduction)
    else:
        months_late = min(months_difference, 36)
        increase = months_late * (2 / 3) / 100
        return full_ss_at_67 * (1 + increase)


def simulate(cfg: SimulationConfig, collect_paths: bool = False):
    """Run the Monte Carlo simulation."""

    success = 0
    success_paths = [] if collect_paths else None
    failure_paths = [] if collect_paths else None

    for _ in range(cfg.number_of_simulations):
        years_to_retirement = cfg.retirement_age - cfg.current_age
        pre_ret_returns = np.random.normal(
            cfg.pre_retirement_mean_return, cfg.pre_retirement_std_dev, years_to_retirement
        )
        growth_factor = np.prod(1 + pre_ret_returns)

        r_bal = cfg.current_roth * growth_factor
        p_bal = cfg.current_401a_and_403b * growth_factor

        base_need = cfg.base_retirement_need
        mortgage_remaining = cfg.mortgage_years_in_retirement
        health_care_remaining = cfg.health_care_years_in_retirement
        health_care_cost = (
            cfg.health_care_yearly_payment if health_care_remaining > 0 else 0
        )
        w = base_need
        if mortgage_remaining > 0:
            w += cfg.mortgage_yearly_payment
        if health_care_remaining > 0:
            w += health_care_cost
        death_year = sample_death_year(
            cfg.retirement_age, cfg.years_of_retirement, cfg.death_probs
        )
        stock_returns = np.random.normal(
            cfg.stock_mean_return, cfg.stock_std_dev, cfg.years_of_retirement
        )
        bond_returns = np.random.normal(
            cfg.bond_mean_return, cfg.bond_std_dev, cfg.years_of_retirement
        )
        infls = np.random.normal(
            cfg.inflation_mean, cfg.inflation_std_dev, cfg.years_of_retirement
        )

        path = [r_bal + p_bal] if collect_paths else None

        for year_idx, (sr, br, i) in enumerate(
            zip(stock_returns, bond_returns, infls), start=1
        ):
            if year_idx > death_year:
                success += 1
                if collect_paths:
                    success_paths.append(path)
                break

            #port_ret = cfg.percent_in_stock_after_retirement * sr + cfg.bond_ratio * br
            ratio = (r_bal + p_bal) / w
            if ratio < 30:
                stock_pct = 1.0
            elif ratio < 35:
                stock_pct = 0.9
            elif ratio < 40:
                stock_pct = 0.8
            elif ratio < 45:
                stock_pct = 0.7
            elif ratio < 50:
                stock_pct = 0.6
            else:
                stock_pct = 0.5

            port_ret = stock_pct * sr + (1 - stock_pct) * br

            r_bal *= (1 + port_ret)
            p_bal *= (1 + port_ret)

            if cfg.retirement_age + year_idx < cfg.social_security_age_started:
                gross_w = gross_from_net(w, cfg)
                taxable_income = gross_w
            else:
                gross_w = gross_from_net_with_ss(
                    w, cfg.social_security_yearly_amount, cfg
                )
                taxable_income = gross_w + cfg.social_security_yearly_amount

            if cfg.enable_roth_conversion:
                p_bal, r_bal, taxable_income = perform_roth_conversion(
                    p_bal, r_bal, taxable_income, cfg
                )
                # Spending needs already satisfied; additional taxes are paid from balances.

            if p_bal >= gross_w:
                p_bal -= gross_w
            else:
                rem_gross = gross_w - p_bal
                p_bal = 0
                rem_net = rem_gross - tax_liability(rem_gross, cfg)
                r_bal -= rem_net

            base_need *= (1 + i)
            if mortgage_remaining > 0:
                mortgage_remaining -= 1
            if health_care_remaining > 0:
                health_care_cost *= (1 + cfg.inflation_mean)
                health_care_remaining -= 1
            w = base_need
            if mortgage_remaining > 0:
                w += cfg.mortgage_yearly_payment
            if health_care_remaining > 0:
                w += health_care_cost

            if collect_paths:
                path.append(r_bal + p_bal)

            if r_bal < 0 or p_bal < 0:
                if collect_paths:
                    failure_paths.append(path)
                break
        else:
            success += 1
            if collect_paths:
                success_paths.append(path)

    if collect_paths:
        return success / cfg.number_of_simulations, success_paths, failure_paths
    return success / cfg.number_of_simulations


def load_config() -> dict:
    """Load saved configuration if available."""

    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}


def save_config(cfg: SimulationConfig) -> None:
    """Persist the provided configuration to disk."""

    data = {
        "general": {
            "number_of_simulations": cfg.number_of_simulations,
            "pre_retirement_mean_return": cfg.pre_retirement_mean_return,
            "pre_retirement_std_dev": cfg.pre_retirement_std_dev,
            "stock_mean_return": cfg.stock_mean_return,
            "stock_std_dev": cfg.stock_std_dev,
            "bond_mean_return": cfg.bond_mean_return,
            "bond_std_dev": cfg.bond_std_dev,
            "inflation_mean": cfg.inflation_mean,
            "inflation_std_dev": cfg.inflation_std_dev,
        },
        "user": {
            "gender": cfg.gender,
            "filing_status": cfg.filing_status,
            "current_age": cfg.current_age,
            "retirement_age": cfg.retirement_age,
            "average_yearly_need": cfg.average_yearly_need,
            "current_roth": cfg.current_roth,
            "current_401a_and_403b": cfg.current_401a_and_403b,
            "full_social_security_at_67": cfg.full_social_security_at_67,
            "social_security_age_started": cfg.social_security_age_started,
            "mortgage_payment": cfg.mortgage_payment,
            "mortgage_years_left": cfg.mortgage_years_left,
            "health_care_payment": cfg.health_care_payment,
            "enable_roth_conversion": cfg.enable_roth_conversion,
            "roth_conversion_rate_cap": cfg.roth_conversion_rate_cap,
        },
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=2)
