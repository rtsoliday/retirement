"""Core functionality for retirement simulations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np


# Monte Carlo setup
np.random.seed()


# IRS 2024 single-filer brackets and marginal rates
brackets = [0, 11_600, 47_150, 100_525, 191_950, 243_725, 609_350]
rates = [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]


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
    tax_brackets: list[float] = field(default_factory=lambda: brackets.copy())
    tax_rates: list[float] = field(default_factory=lambda: rates.copy())
    death_probs: np.ndarray = field(default_factory=lambda: np.array([]))


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
        w = base_need + (cfg.mortgage_yearly_payment if mortgage_remaining > 0 else 0)
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
            else:
                gross_w = gross_from_net_with_ss(
                    w, cfg.social_security_yearly_amount, cfg
                )

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
            w = base_need + (
                cfg.mortgage_yearly_payment if mortgage_remaining > 0 else 0
            )

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
            "current_age": cfg.current_age,
            "retirement_age": cfg.retirement_age,
            "average_yearly_need": cfg.average_yearly_need,
            "current_roth": cfg.current_roth,
            "current_401a_and_403b": cfg.current_401a_and_403b,
            "full_social_security_at_67": cfg.full_social_security_at_67,
            "social_security_age_started": cfg.social_security_age_started,
            "mortgage_payment": cfg.mortgage_payment,
            "mortgage_years_left": cfg.mortgage_years_left,
            "percent_in_stock_after_retirement": cfg.percent_in_stock_after_retirement,
        },
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=2)