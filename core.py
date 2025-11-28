"""Core functionality for retirement simulations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from numba import njit, prange
from numba.typed import List as NumbaList


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

# 2024 Medicare Part B monthly premiums and IRMAA surcharges
# Based on Modified Adjusted Gross Income (MAGI) from 2 years prior
MEDICARE_PART_B_BASE = 174.70 * 12  # Annual base premium for 2024
MEDICARE_PART_D_BASE = 55.50 * 12   # Average annual Part D premium for 2024

# IRMAA brackets for single filers (MAGI thresholds)
# Each tuple: (threshold, Part B monthly surcharge, Part D monthly surcharge)
IRMAA_BRACKETS_SINGLE = [
    (103_000, 0, 0),           # Base premium
    (129_000, 69.90, 12.90),   # Tier 1
    (161_000, 174.70, 33.30),  # Tier 2  
    (193_000, 279.50, 53.80),  # Tier 3
    (500_000, 384.30, 74.20),  # Tier 4
    (float('inf'), 419.30, 81.00),  # Tier 5 (highest)
]

# IRMAA brackets for married filing jointly
IRMAA_BRACKETS_MARRIED = [
    (206_000, 0, 0),
    (258_000, 69.90, 12.90),
    (322_000, 174.70, 33.30),
    (386_000, 279.50, 53.80),
    (750_000, 384.30, 74.20),
    (float('inf'), 419.30, 81.00),
]

# Long-term care statistics
# Probability of needing LTC at various ages (cumulative risk)
# Source: Various actuarial studies
LTC_PROBABILITY_BY_AGE = {
    65: 0.02,   # 2% chance of needing LTC at 65
    70: 0.05,   # 5% by age 70
    75: 0.10,   # 10% by age 75
    80: 0.20,   # 20% by age 80
    85: 0.35,   # 35% by age 85
    90: 0.50,   # 50% by age 90
}

# Average annual cost of long-term care (2024 dollars)
LTC_ANNUAL_COST = 100_000  # Approximate for nursing home care
LTC_DURATION_MEAN = 2.5    # Average years of LTC needed
LTC_DURATION_STD = 1.5     # Standard deviation

# Asset correlation defaults
# Inflation-bond correlation: historically negative (-0.3 to -0.5)
# When inflation rises unexpectedly, bond prices fall
DEFAULT_INFLATION_BOND_CORRELATION = -0.40

# Stock-bond correlation: historically low but variable (-0.2 to +0.3)
# Can be negative (flight to safety) or positive (risk-on/risk-off)
DEFAULT_STOCK_BOND_CORRELATION = 0.10

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


@njit(cache=True)
def sample_death_year(retirement_age: int, years_of_retirement: int, death_probs: np.ndarray) -> int:
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
    current_savings: float
    savings_interest_rate: float
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
    healthcare_inflation_mean: float = 0.055  # Healthcare typically inflates faster
    healthcare_inflation_std: float = 0.02
    include_medicare_premiums: bool = True
    include_ltc_risk: bool = False
    ltc_annual_cost: float = LTC_ANNUAL_COST
    inflation_bond_correlation: float = DEFAULT_INFLATION_BOND_CORRELATION
    stock_bond_correlation: float = DEFAULT_STOCK_BOND_CORRELATION
    filing_status: str = "single"
    tax_brackets: list[float] = field(default_factory=list)
    tax_rates: list[float] = field(default_factory=list)
    death_probs: np.ndarray = field(default_factory=lambda: np.array([]))
    enable_roth_conversion: bool = False
    roth_conversion_rate_cap: Optional[float] = None
    # Dynamic withdrawal strategy options
    enable_dynamic_withdrawal: bool = False
    dynamic_withdrawal_trigger: float = -0.01  # Default -1% trigger
    replenish_savings_after_drawdown: bool = True  # Whether to refill savings after market recovery
    # Monthly equivalents (derived from yearly values)
    months_of_retirement: int = field(default=0, init=False)
    months_to_retirement: int = field(default=0, init=False)
    pre_retirement_monthly_mean: float = field(default=0.0, init=False)
    pre_retirement_monthly_std: float = field(default=0.0, init=False)
    stock_monthly_mean: float = field(default=0.0, init=False)
    stock_monthly_std: float = field(default=0.0, init=False)
    bond_monthly_mean: float = field(default=0.0, init=False)
    bond_monthly_std: float = field(default=0.0, init=False)
    inflation_monthly_mean: float = field(default=0.0, init=False)
    inflation_monthly_std: float = field(default=0.0, init=False)
    healthcare_inflation_monthly_mean: float = field(default=0.0, init=False)
    healthcare_inflation_monthly_std: float = field(default=0.0, init=False)
    base_monthly_need: float = field(default=0.0, init=False)
    mortgage_monthly_payment: float = field(default=0.0, init=False)
    mortgage_months_in_retirement: int = field(default=0, init=False)
    health_care_months_in_retirement: int = field(default=0, init=False)
    health_care_monthly_payment: float = field(default=0.0, init=False)
    social_security_monthly_amount: float = field(default=0.0, init=False)
    ltc_monthly_cost: float = field(default=0.0, init=False)
    _roth_conversion_upper_limit: Optional[float] = field(
        init=False, repr=False, default=None
    )
    _roth_conversion_rate_index: Optional[int] = field(
        init=False, repr=False, default=None
    )
    _bracket_arr: Optional[np.ndarray] = field(
        init=False, repr=False, default=None
    )
    _rate_arr: Optional[np.ndarray] = field(
        init=False, repr=False, default=None
    )
    _cumulative_tax: Optional[np.ndarray] = field(
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

        # Compute monthly equivalents from yearly values
        self._compute_monthly_equivalents()

    def _compute_monthly_equivalents(self) -> None:
        """Convert yearly parameters to monthly equivalents for internal simulation."""
        import math
        
        # Time periods
        self.months_of_retirement = self.years_of_retirement * 12
        self.months_to_retirement = (self.retirement_age - self.current_age) * 12
        
        # Convert returns: monthly_mean = annual_mean / 12, monthly_std = annual_std / sqrt(12)
        sqrt_12 = math.sqrt(12)
        self.pre_retirement_monthly_mean = self.pre_retirement_mean_return / 12
        self.pre_retirement_monthly_std = self.pre_retirement_std_dev / sqrt_12
        self.stock_monthly_mean = self.stock_mean_return / 12
        self.stock_monthly_std = self.stock_std_dev / sqrt_12
        self.bond_monthly_mean = self.bond_mean_return / 12
        self.bond_monthly_std = self.bond_std_dev / sqrt_12
        self.inflation_monthly_mean = self.inflation_mean / 12
        self.inflation_monthly_std = self.inflation_std_dev / sqrt_12
        self.healthcare_inflation_monthly_mean = self.healthcare_inflation_mean / 12
        self.healthcare_inflation_monthly_std = self.healthcare_inflation_std / sqrt_12
        
        # Convert spending to monthly
        self.base_monthly_need = self.base_retirement_need / 12
        self.mortgage_monthly_payment = self.mortgage_yearly_payment / 12
        self.mortgage_months_in_retirement = self.mortgage_years_in_retirement * 12
        self.health_care_months_in_retirement = self.health_care_years_in_retirement * 12
        self.health_care_monthly_payment = self.health_care_yearly_payment / 12
        self.social_security_monthly_amount = self.social_security_yearly_amount / 12
        self.ltc_monthly_cost = self.ltc_annual_cost / 12


@njit(cache=True)
def _tax_liability_jit(
    income: float,
    bracket_arr: np.ndarray,
    rate_arr: np.ndarray,
    cumulative_tax: np.ndarray
) -> float:
    """JIT-compiled tax calculation kernel."""
    if income <= 0:
        return 0.0
    
    # Find which bracket the income falls into
    bracket_idx = np.searchsorted(bracket_arr, income, side='right') - 1
    if bracket_idx < 0:
        bracket_idx = 0
    if bracket_idx >= len(rate_arr):
        bracket_idx = len(rate_arr) - 1
    
    # Tax = cumulative tax up to this bracket + tax on income within this bracket
    return cumulative_tax[bracket_idx] + \
           (income - bracket_arr[bracket_idx]) * rate_arr[bracket_idx]


def _ensure_tax_arrays(cfg) -> tuple:
    """Initialize cached tax arrays if not already done. Returns (bracket_arr, rate_arr, cumulative_tax)."""
    # Check if using SimulationConfig or SimpleNamespace
    bracket_arr = getattr(cfg, '_bracket_arr', None)
    if bracket_arr is None:
        bracket_arr = np.array(cfg.tax_brackets, dtype=np.float64)
        rate_arr = np.array(cfg.tax_rates, dtype=np.float64)
        # Precompute cumulative tax at each bracket boundary
        cumulative_tax = np.zeros(len(cfg.tax_brackets), dtype=np.float64)
        for i in range(1, len(cfg.tax_brackets)):
            cumulative_tax[i] = cumulative_tax[i-1] + \
                (cfg.tax_brackets[i] - cfg.tax_brackets[i-1]) * cfg.tax_rates[i-1]
        # Cache if possible (SimulationConfig has these attributes)
        if hasattr(cfg, '_bracket_arr'):
            cfg._bracket_arr = bracket_arr
            cfg._rate_arr = rate_arr
            cfg._cumulative_tax = cumulative_tax
        return bracket_arr, rate_arr, cumulative_tax
    return cfg._bracket_arr, cfg._rate_arr, cfg._cumulative_tax


def tax_liability(income: float, cfg) -> float:
    """Compute tax owed for a given income using marginal brackets."""
    bracket_arr, rate_arr, cumulative_tax = _ensure_tax_arrays(cfg)
    return _tax_liability_jit(income, bracket_arr, rate_arr, cumulative_tax)


@njit(cache=True)
def _gross_from_net_jit(
    net_amt: float,
    bracket_arr: np.ndarray,
    rate_arr: np.ndarray,
    cumulative_tax: np.ndarray
) -> float:
    """JIT-compiled gross from net calculation using Newton-Raphson."""
    if net_amt <= 0:
        return 0.0
    
    # Initial guess: net_amt plus estimated tax (assume ~20% effective rate)
    gross = net_amt * 1.25
    
    # Newton-Raphson: solve f(G) = G - tax(G) - net = 0
    for _ in range(8):  # Usually converges in 3-5 iterations
        tax = _tax_liability_jit(gross, bracket_arr, rate_arr, cumulative_tax)
        net_result = gross - tax
        error = net_result - net_amt
        
        if abs(error) < 0.01:  # Converged within 1 cent
            return gross
        
        # Get marginal rate at current gross
        bracket_idx = np.searchsorted(bracket_arr, gross, side='right') - 1
        if bracket_idx < 0:
            bracket_idx = 0
        if bracket_idx >= len(rate_arr):
            bracket_idx = len(rate_arr) - 1
        marginal_rate = rate_arr[bracket_idx]
        
        # Newton step: G_new = G - f(G)/f'(G)
        derivative = 1.0 - marginal_rate
        if derivative > 0.1:  # Avoid division issues
            gross = gross - error / derivative
        else:
            gross = gross - error / 0.5  # Fallback
        
        if gross < net_amt:
            gross = net_amt  # Gross can't be less than net
    
    return gross


def gross_from_net(net_amt: float, cfg) -> float:
    """Find G such that G - tax_liability(G) = net_amt using Newton-Raphson."""
    bracket_arr, rate_arr, cumulative_tax = _ensure_tax_arrays(cfg)
    return _gross_from_net_jit(net_amt, bracket_arr, rate_arr, cumulative_tax)


@njit(cache=True)
def _gross_from_net_with_ss_jit(
    net_amt: float,
    social_security_yearly_amount: float,
    bracket_arr: np.ndarray,
    rate_arr: np.ndarray,
    cumulative_tax: np.ndarray
) -> float:
    """
    JIT-compiled version: Find G such that
       (G + social_security_yearly_amount) - tax_liability(G + social_security_yearly_amount) == net_amt
    """
    ss = social_security_yearly_amount
    
    if net_amt <= 0:
        return 0.0
    
    # Initial guess
    gross = max(0.0, net_amt - ss) * 1.25 + ss * 0.25
    
    # Newton-Raphson: solve f(G) = (G + ss) - tax(G + ss) - net = 0
    for _ in range(8):
        total = gross + ss
        tax = _tax_liability_jit(total, bracket_arr, rate_arr, cumulative_tax)
        net_result = total - tax
        error = net_result - net_amt
        
        if abs(error) < 0.01:
            return gross
        
        # Get marginal rate at current total income
        bracket_idx = np.searchsorted(bracket_arr, total, side='right') - 1
        if bracket_idx < 0:
            bracket_idx = 0
        if bracket_idx >= len(rate_arr):
            bracket_idx = len(rate_arr) - 1
        marginal_rate = rate_arr[bracket_idx]
        
        derivative = 1.0 - marginal_rate
        if derivative > 0.1:
            gross = gross - error / derivative
        else:
            gross = gross - error / 0.5
        
        if gross < 0:
            gross = 0.0
    
    return gross


def gross_from_net_with_ss(
    net_amt: float, social_security_yearly_amount: float, cfg
) -> float:
    """
    Find G such that
       (G + social_security_yearly_amount) - tax_liability(G + social_security_yearly_amount, cfg) == net_amt
    i.e. accounts + SS minus tax on total = net spending.
    Uses Newton-Raphson for fast convergence.
    """
    bracket_arr, rate_arr, cumulative_tax = _ensure_tax_arrays(cfg)
    return _gross_from_net_with_ss_jit(
        net_amt, social_security_yearly_amount,
        bracket_arr, rate_arr, cumulative_tax
    )

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


def calculate_irmaa_surcharge(
    magi: float, filing_status: str, inflation_factor: float = 1.0
) -> float:
    """
    Calculate annual IRMAA surcharge for Medicare Parts B and D.
    
    Args:
        magi: Modified Adjusted Gross Income (from 2 years prior in practice)
        filing_status: Tax filing status
        inflation_factor: Factor to adjust brackets for inflation
    
    Returns:
        Annual IRMAA surcharge amount
    """
    brackets = (
        IRMAA_BRACKETS_MARRIED if filing_status == "married" 
        else IRMAA_BRACKETS_SINGLE
    )
    
    for threshold, part_b_surcharge, part_d_surcharge in brackets:
        adjusted_threshold = threshold * inflation_factor
        if magi <= adjusted_threshold:
            return (part_b_surcharge + part_d_surcharge) * 12
    
    # If above all thresholds, use highest tier
    _, part_b_surcharge, part_d_surcharge = brackets[-1]
    return (part_b_surcharge + part_d_surcharge) * 12


def calculate_medicare_premium(
    age: int, magi: float, filing_status: str, inflation_factor: float = 1.0
) -> float:
    """
    Calculate total annual Medicare premium including IRMAA surcharges.
    
    Args:
        age: Current age (must be 65+ to have Medicare)
        magi: Modified Adjusted Gross Income
        filing_status: Tax filing status
        inflation_factor: Factor to adjust premiums for healthcare inflation
    
    Returns:
        Total annual Medicare premium (Parts B + D + IRMAA)
    """
    if age < 65:
        return 0.0
    
    base_premium = (MEDICARE_PART_B_BASE + MEDICARE_PART_D_BASE) * inflation_factor
    irmaa = calculate_irmaa_surcharge(magi, filing_status, inflation_factor)
    
    return base_premium + irmaa


# LTC probability lookup arrays for JIT (ages and probabilities)
_LTC_AGES = np.array([65, 70, 75, 80, 85, 90], dtype=np.int64)
_LTC_PROBS = np.array([0.02, 0.05, 0.10, 0.20, 0.35, 0.50], dtype=np.float64)


@njit(cache=True)
def sample_ltc_event(
    age: int, already_had_ltc: bool = False
) -> Tuple[bool, float]:
    """
    Determine if a long-term care event occurs this year.
    
    Args:
        age: Current age
        already_had_ltc: Whether person already had an LTC event
    
    Returns:
        Tuple of (ltc_needed_this_year, ltc_duration_years)
    """
    if already_had_ltc:
        return False, 0.0
    
    # Interpolate probability based on age using pre-defined arrays
    if age < _LTC_AGES[0]:
        annual_prob = 0.001  # Very low before 65
    elif age >= _LTC_AGES[-1]:
        annual_prob = 0.10  # Higher annual rate for very old
    else:
        # Find surrounding ages and interpolate
        annual_prob = 0.05  # default
        for i in range(len(_LTC_AGES) - 1):
            if _LTC_AGES[i] <= age < _LTC_AGES[i + 1]:
                # Convert cumulative to annual probability
                prob_low = _LTC_PROBS[i]
                prob_high = _LTC_PROBS[i + 1]
                years_span = _LTC_AGES[i + 1] - _LTC_AGES[i]
                # Approximate annual probability
                annual_prob = (prob_high - prob_low) / years_span
                break
    
    if np.random.random() < annual_prob:
        duration = max(0.5, np.random.normal(LTC_DURATION_MEAN, LTC_DURATION_STD))
        return True, duration
    
    return False, 0.0


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


@njit(cache=True)
def generate_correlated_returns(
    n_periods: int,
    stock_mean: float,
    stock_std: float,
    bond_mean: float,
    bond_std: float,
    inflation_mean: float,
    inflation_std: float,
    inflation_bond_corr: float,
    stock_bond_corr: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate correlated stock returns, bond returns, and inflation.
    
    Uses Cholesky decomposition to create correlated normal samples.
    Parameters should be in monthly terms for monthly simulation.
    """
    # Correlation matrix: [stocks, bonds, inflation]
    stock_inflation_corr = 0.05
    
    corr_matrix = np.array([
        [1.0, stock_bond_corr, stock_inflation_corr],
        [stock_bond_corr, 1.0, inflation_bond_corr],
        [stock_inflation_corr, inflation_bond_corr, 1.0],
    ])
    
    # Simple Cholesky decomposition (correlation matrices are usually valid)
    # Manual implementation for Numba compatibility
    L = np.zeros((3, 3))
    for i in range(3):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += L[i, k] * L[j, k]
            if i == j:
                val = corr_matrix[i, i] - s
                if val > 0:
                    L[i, j] = np.sqrt(val)
                else:
                    L[i, j] = 0.001  # Small positive value if matrix not PSD
            else:
                if L[j, j] > 1e-10:
                    L[i, j] = (corr_matrix[i, j] - s) / L[j, j]
                else:
                    L[i, j] = 0.0
    
    # Generate independent standard normal samples
    uncorrelated = np.random.standard_normal((3, n_periods))
    
    # Apply correlation structure: correlated = L @ uncorrelated
    correlated = np.zeros((3, n_periods))
    for i in range(3):
        for j in range(n_periods):
            for k in range(3):
                correlated[i, j] += L[i, k] * uncorrelated[k, j]
    
    # Scale to desired means and standard deviations
    stock_returns = stock_mean + stock_std * correlated[0]
    bond_returns = bond_mean + bond_std * correlated[1]
    inflation_rates = inflation_mean + inflation_std * correlated[2]
    
    # Ensure inflation doesn't go too negative (monthly: -0.05/12 ~ -0.004)
    min_inflation = -0.05 / 12
    for i in range(n_periods):
        if inflation_rates[i] < min_inflation:
            inflation_rates[i] = min_inflation
    
    return stock_returns, bond_returns, inflation_rates


@njit(cache=True)
def _generate_batch_correlated_returns(
    n_sims: int,
    n_periods: int,
    stock_mean: float,
    stock_std: float,
    bond_mean: float,
    bond_std: float,
    inflation_mean: float,
    inflation_std: float,
    inflation_bond_corr: float,
    stock_bond_corr: float,
    uncorrelated: np.ndarray,  # Pre-generated random numbers (n_sims, 3, n_periods)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate correlated returns for all simulations at once.
    Returns arrays of shape (n_sims, n_periods).
    Parameters should be in monthly terms for monthly simulation.
    """
    stock_inflation_corr = 0.05
    
    corr_matrix = np.array([
        [1.0, stock_bond_corr, stock_inflation_corr],
        [stock_bond_corr, 1.0, inflation_bond_corr],
        [stock_inflation_corr, inflation_bond_corr, 1.0],
    ])
    
    # Cholesky decomposition
    L = np.zeros((3, 3))
    for i in range(3):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += L[i, k] * L[j, k]
            if i == j:
                val = corr_matrix[i, i] - s
                if val > 0:
                    L[i, j] = np.sqrt(val)
                else:
                    L[i, j] = 0.001
            else:
                if L[j, j] > 1e-10:
                    L[i, j] = (corr_matrix[i, j] - s) / L[j, j]
                else:
                    L[i, j] = 0.0
    
    # Output arrays
    all_stock = np.empty((n_sims, n_periods))
    all_bond = np.empty((n_sims, n_periods))
    all_inflation = np.empty((n_sims, n_periods))
    
    # Minimum monthly inflation
    min_inflation = -0.05 / 12
    
    for sim in range(n_sims):
        # Apply correlation structure
        correlated = np.zeros((3, n_periods))
        for i in range(3):
            for j in range(n_periods):
                for k in range(3):
                    correlated[i, j] += L[i, k] * uncorrelated[sim, k, j]
        
        # Scale to desired means and standard deviations
        for j in range(n_periods):
            all_stock[sim, j] = stock_mean + stock_std * correlated[0, j]
            all_bond[sim, j] = bond_mean + bond_std * correlated[1, j]
            infl = inflation_mean + inflation_std * correlated[2, j]
            if infl < min_inflation:
                infl = min_inflation
            all_inflation[sim, j] = infl
    
    return all_stock, all_bond, all_inflation


@njit(cache=True, parallel=True)
def _simulate_parallel(
    n_sims: int,
    months_of_retirement: int,
    retirement_age: int,
    social_security_age_started: int,
    social_security_monthly_amount: float,
    current_roth: float,
    current_401a_and_403b: float,
    current_savings: float,
    savings_monthly_rate: float,
    base_monthly_need: float,
    mortgage_months_in_retirement: int,
    mortgage_monthly_payment: float,
    health_care_months_in_retirement: int,
    health_care_monthly_payment: float,
    healthcare_inflation_monthly_mean: float,
    healthcare_inflation_monthly_std: float,
    include_medicare_premiums: bool,
    include_ltc_risk: bool,
    ltc_monthly_cost: float,
    filing_status_is_married: bool,
    enable_roth_conversion: bool,
    roth_conversion_upper_limit: float,
    enable_dynamic_withdrawal: bool,
    dynamic_withdrawal_trigger: float,
    initial_cash_reserve: float,
    replenish_savings: bool,
    bracket_arr: np.ndarray,
    rate_arr: np.ndarray,
    cumulative_tax: np.ndarray,
    death_probs: np.ndarray,
    # Pre-generated random numbers (all monthly)
    growth_factors: np.ndarray,  # (n_sims,)
    all_stock_returns: np.ndarray,  # (n_sims, months_of_retirement)
    all_bond_returns: np.ndarray,  # (n_sims, months_of_retirement)
    all_inflation: np.ndarray,  # (n_sims, months_of_retirement)
    all_healthcare_inflation: np.ndarray,  # (n_sims, months_of_retirement)
    all_death_randoms: np.ndarray,  # (n_sims, years_of_retirement) - still yearly for death
    all_ltc_randoms: np.ndarray,  # (n_sims, years_of_retirement) - still yearly for LTC check
    all_ltc_duration_randoms: np.ndarray,  # (n_sims, years_of_retirement)
) -> np.ndarray:
    """
    Parallelized monthly simulation kernel. Returns array of success (1) or failure (0) for each sim.
    """
    results = np.zeros(n_sims, dtype=np.int32)
    
    # IRMAA thresholds and surcharges (monthly values) - simplified for JIT
    # Single filer thresholds
    irmaa_thresholds_single = np.array([103000.0, 129000.0, 161000.0, 193000.0, 500000.0, np.inf])
    irmaa_surcharges_single = np.array([0.0, 82.8, 208.0, 333.3, 458.5, 500.3])  # Part B + Part D monthly
    # Married thresholds
    irmaa_thresholds_married = np.array([206000.0, 258000.0, 322000.0, 386000.0, 750000.0, np.inf])
    irmaa_surcharges_married = np.array([0.0, 82.8, 208.0, 333.3, 458.5, 500.3])
    
    medicare_base_monthly = 174.70 + 55.50  # Part B + Part D base monthly
    years_of_retirement = months_of_retirement // 12
    
    for sim in prange(n_sims):
        r_bal = current_roth * growth_factors[sim]
        p_bal = current_401a_and_403b * growth_factors[sim]
        s_bal = current_savings  # Savings account (last resort)
        
        base_need = base_monthly_need
        mortgage_remaining = mortgage_months_in_retirement
        health_care_remaining = health_care_months_in_retirement
        health_care_cost = health_care_monthly_payment if health_care_remaining > 0 else 0.0
        
        healthcare_inflation_factor = 1.0
        
        # LTC tracking (in months)
        ltc_active = False
        ltc_months_remaining = 0.0
        had_ltc_event = False
        
        # Dynamic withdrawal strategy tracking
        # Running tally of returns percentage since first drawdown (RTRP)
        running_return_tally = 0.0
        # Flag to track if we're in a drawdown sequence
        in_drawdown_sequence = False
        # For dynamic withdrawal, s_bal (savings) serves as the cash reserve (C)
        # initial_cash_reserve tracks the target value to refill to
        
        # Initial monthly withdrawal need
        w = base_need
        if mortgage_remaining > 0:
            w += mortgage_monthly_payment
        if health_care_remaining > 0:
            w += health_care_cost
        
        # Sample death year inline (still yearly resolution for mortality)
        death_year = years_of_retirement
        for j in range(years_of_retirement):
            age = retirement_age + j
            if age < len(death_probs):
                if all_death_randoms[sim, j] < death_probs[age]:
                    death_year = j
                    break
        death_month = (death_year + 1) * 12
        
        # Track yearly values for tax calculations (tax is done yearly)
        year_gross_withdrawal = 0.0
        year_ss_received = 0.0
        last_yearly_taxable_income = 0.0  # For IRMAA and Roth conversion
        
        succeeded = True
        for month_idx in range(1, months_of_retirement + 1):
            if month_idx > death_month:
                break
            
            # Determine current year and month within year
            year_idx = (month_idx - 1) // 12  # 0-indexed year
            month_in_year = (month_idx - 1) % 12  # 0-11
            current_age = retirement_age + year_idx
            
            # Monthly healthcare inflation
            healthcare_infl = all_healthcare_inflation[sim, month_idx - 1]
            healthcare_inflation_factor *= (1.0 + healthcare_infl)
            
            # Monthly portfolio return based on yearly spending ratio
            sr = all_stock_returns[sim, month_idx - 1]
            br = all_bond_returns[sim, month_idx - 1]
            infl = all_inflation[sim, month_idx - 1]
            
            # Calculate yearly spending need for ratio calculation
            yearly_w = w * 12
            ratio = (r_bal + p_bal) / yearly_w if yearly_w > 0 else 100.0
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
            
            port_ret = stock_pct * sr + (1.0 - stock_pct) * br
            r_bal *= (1.0 + port_ret)
            p_bal *= (1.0 + port_ret)
            
            # Check if receiving Social Security this month
            # Person's age in months at month_idx is: retirement_age * 12 + (month_idx - 1)
            receiving_ss = (retirement_age * 12 + month_idx - 1) >= (social_security_age_started * 12)
            
            # Monthly withdrawal (pre-tax from 401a/403b)
            # We need to estimate tax monthly. For simplicity, use last year's effective rate
            # or compute based on annualized withdrawal
            monthly_net_need = w
            
            # Apply monthly interest to savings/cash reserve
            s_bal *= (1.0 + savings_monthly_rate)
            
            # Dynamic withdrawal strategy implementation
            if enable_dynamic_withdrawal:
                # Combined portfolio return percentage for this month
                rp = port_ret
                
                if rp < dynamic_withdrawal_trigger and s_bal > 0:
                    # Drawdown month - use savings (cash reserve) first
                    in_drawdown_sequence = True
                    running_return_tally += rp
                    
                    # Calculate how much we can cover with SS + savings
                    ss_this_month = social_security_monthly_amount if receiving_ss else 0.0
                    remaining_need = monthly_net_need - ss_this_month
                    
                    if remaining_need <= 0:
                        # SS covers everything
                        year_ss_received += ss_this_month
                        # No gross withdrawal needed
                    elif s_bal >= remaining_need:
                        # Savings covers the rest
                        s_bal -= remaining_need
                        year_ss_received += ss_this_month
                        # No taxable withdrawal from retirement accounts
                    else:
                        # Savings not enough, need to sell some stocks/bonds
                        remaining_after_savings = remaining_need - s_bal
                        s_bal = 0.0
                        year_ss_received += ss_this_month
                        
                        # Calculate gross needed for the remaining amount
                        # Account for SS taxation when calculating tax on withdrawal
                        annualized_remaining = remaining_after_savings * 12
                        if receiving_ss:
                            ss_annual = social_security_monthly_amount * 12
                            gross_annual = _gross_from_net_with_ss_jit(
                                annualized_remaining, ss_annual, bracket_arr, rate_arr, cumulative_tax
                            )
                        else:
                            gross_annual = _gross_from_net_jit(annualized_remaining, bracket_arr, rate_arr, cumulative_tax)
                        gross_monthly = gross_annual / 12
                        year_gross_withdrawal += gross_monthly
                        
                        # Withdraw from accounts
                        if p_bal >= gross_monthly:
                            p_bal -= gross_monthly
                        else:
                            rem_gross = gross_monthly - p_bal
                            p_bal = 0.0
                            if r_bal >= rem_gross:
                                r_bal -= rem_gross
                            else:
                                r_bal = 0.0
                                
                elif rp < dynamic_withdrawal_trigger and s_bal <= 0:
                    # Out of reserve funds month - pay expenses normally
                    in_drawdown_sequence = True
                    running_return_tally += rp
                    
                    # Normal withdrawal logic
                    annualized_need = monthly_net_need * 12
                    if receiving_ss:
                        ss_annual = social_security_monthly_amount * 12
                        gross_annual = _gross_from_net_with_ss_jit(
                            annualized_need, ss_annual, bracket_arr, rate_arr, cumulative_tax
                        )
                        gross_monthly = gross_annual / 12
                        year_ss_received += social_security_monthly_amount
                    else:
                        gross_annual = _gross_from_net_jit(annualized_need, bracket_arr, rate_arr, cumulative_tax)
                        gross_monthly = gross_annual / 12
                    
                    year_gross_withdrawal += gross_monthly
                    
                    if p_bal >= gross_monthly:
                        p_bal -= gross_monthly
                    else:
                        rem_gross = gross_monthly - p_bal
                        p_bal = 0.0
                        if r_bal >= rem_gross:
                            r_bal -= rem_gross
                        else:
                            rem_gross -= r_bal
                            r_bal = 0.0
                            s_bal -= rem_gross
                            
                elif in_drawdown_sequence and running_return_tally < 0:
                    # Market recovering but not back to initial drawdown level
                    running_return_tally += rp
                    
                    # Normal withdrawal logic
                    annualized_need = monthly_net_need * 12
                    if receiving_ss:
                        ss_annual = social_security_monthly_amount * 12
                        gross_annual = _gross_from_net_with_ss_jit(
                            annualized_need, ss_annual, bracket_arr, rate_arr, cumulative_tax
                        )
                        gross_monthly = gross_annual / 12
                        year_ss_received += social_security_monthly_amount
                    else:
                        gross_annual = _gross_from_net_jit(annualized_need, bracket_arr, rate_arr, cumulative_tax)
                        gross_monthly = gross_annual / 12
                    
                    year_gross_withdrawal += gross_monthly
                    
                    if p_bal >= gross_monthly:
                        p_bal -= gross_monthly
                    else:
                        rem_gross = gross_monthly - p_bal
                        p_bal = 0.0
                        if r_bal >= rem_gross:
                            r_bal -= rem_gross
                        else:
                            rem_gross -= r_bal
                            r_bal = 0.0
                            s_bal -= rem_gross
                            
                elif in_drawdown_sequence and s_bal < initial_cash_reserve and replenish_savings:
                    # Market recovered and refilling savings (cash reserve)
                    # Pay expenses normally AND refill savings
                    annualized_need = monthly_net_need * 12
                    if receiving_ss:
                        ss_annual = social_security_monthly_amount * 12
                        gross_annual = _gross_from_net_with_ss_jit(
                            annualized_need, ss_annual, bracket_arr, rate_arr, cumulative_tax
                        )
                        gross_monthly = gross_annual / 12
                        year_ss_received += social_security_monthly_amount
                    else:
                        gross_annual = _gross_from_net_jit(annualized_need, bracket_arr, rate_arr, cumulative_tax)
                        gross_monthly = gross_annual / 12
                    
                    year_gross_withdrawal += gross_monthly
                    
                    # Calculate amount to refill savings (same ratio as normal expenses)
                    refill_amount = min(initial_cash_reserve - s_bal, monthly_net_need * 0.5)
                    total_gross_needed = gross_monthly + refill_amount
                    
                    if p_bal >= total_gross_needed:
                        p_bal -= total_gross_needed
                        s_bal += refill_amount
                    else:
                        rem_gross = total_gross_needed - p_bal
                        p_bal = 0.0
                        if r_bal >= rem_gross:
                            r_bal -= rem_gross
                            s_bal += refill_amount
                        else:
                            # Not enough in retirement accounts to fully refill
                            # Just refill what we can after covering expenses
                            available = r_bal
                            r_bal = 0.0
                            # Calculate how much is left for refill after covering the expense portion
                            # rem_gross includes both expense shortfall and refill amount
                            # expense_shortfall = rem_gross - refill_amount (what p_bal couldn't cover)
                            # If available > expense_shortfall, the remainder can go to refill
                            actual_refill = max(0.0, available - (rem_gross - refill_amount))
                            s_bal += actual_refill
                    
                    # Check if fully refilled
                    if s_bal >= initial_cash_reserve:
                        s_bal = initial_cash_reserve
                        running_return_tally = 0.0
                        in_drawdown_sequence = False
                else:
                    # Not in drawdown or savings already fully funded (or no replenishment) - normal withdrawal
                    running_return_tally = 0.0
                    in_drawdown_sequence = False
                    
                    annualized_need = monthly_net_need * 12
                    if receiving_ss:
                        ss_annual = social_security_monthly_amount * 12
                        gross_annual = _gross_from_net_with_ss_jit(
                            annualized_need, ss_annual, bracket_arr, rate_arr, cumulative_tax
                        )
                        gross_monthly = gross_annual / 12
                        year_ss_received += social_security_monthly_amount
                    else:
                        gross_annual = _gross_from_net_jit(annualized_need, bracket_arr, rate_arr, cumulative_tax)
                        gross_monthly = gross_annual / 12
                    
                    year_gross_withdrawal += gross_monthly
                    
                    if p_bal >= gross_monthly:
                        p_bal -= gross_monthly
                    else:
                        rem_gross = gross_monthly - p_bal
                        p_bal = 0.0
                        if r_bal >= rem_gross:
                            r_bal -= rem_gross
                        else:
                            rem_gross -= r_bal
                            r_bal = 0.0
                            s_bal -= rem_gross
            else:
                # Original withdrawal logic (dynamic withdrawal disabled)
                # Estimate gross monthly withdrawal
                # Annualize the need, compute tax, then divide by 12
                annualized_need = monthly_net_need * 12
                if receiving_ss:
                    ss_annual = social_security_monthly_amount * 12
                    gross_annual = _gross_from_net_with_ss_jit(
                        annualized_need, ss_annual, bracket_arr, rate_arr, cumulative_tax
                    )
                    gross_monthly = gross_annual / 12
                    year_ss_received += social_security_monthly_amount
                else:
                    gross_annual = _gross_from_net_jit(annualized_need, bracket_arr, rate_arr, cumulative_tax)
                    gross_monthly = gross_annual / 12
                
                year_gross_withdrawal += gross_monthly
                
                # Withdraw from accounts: pre-tax first, then Roth, then savings (last resort)
                if p_bal >= gross_monthly:
                    p_bal -= gross_monthly
                else:
                    rem_gross = gross_monthly - p_bal
                    p_bal = 0.0
                    # Remaining comes from Roth (tax-free)
                    if r_bal >= rem_gross:
                        r_bal -= rem_gross
                    else:
                        rem_gross -= r_bal
                        r_bal = 0.0
                        # Last resort: savings account
                        s_bal -= rem_gross
            
            # Update base need with monthly inflation
            base_need *= (1.0 + infl)
            if mortgage_remaining > 0:
                mortgage_remaining -= 1
            
            if health_care_remaining > 0:
                health_care_cost *= (1.0 + healthcare_infl)
                health_care_remaining -= 1
            
            # Medicare costs (age 65+) - monthly
            medicare_cost = 0.0
            if include_medicare_premiums and current_age >= 65:
                base_premium = medicare_base_monthly * healthcare_inflation_factor
                # IRMAA surcharge based on last year's income
                surcharge = 0.0
                if filing_status_is_married:
                    for ti in range(len(irmaa_thresholds_married)):
                        if last_yearly_taxable_income <= irmaa_thresholds_married[ti] * healthcare_inflation_factor:
                            surcharge = irmaa_surcharges_married[ti] * healthcare_inflation_factor
                            break
                else:
                    for ti in range(len(irmaa_thresholds_single)):
                        if last_yearly_taxable_income <= irmaa_thresholds_single[ti] * healthcare_inflation_factor:
                            surcharge = irmaa_surcharges_single[ti] * healthcare_inflation_factor
                            break
                medicare_cost = base_premium + surcharge
            
            # LTC costs (monthly)
            ltc_cost = 0.0
            entering_ltc_this_month = False
            if include_ltc_risk:
                if ltc_active and ltc_months_remaining > 0:
                    ltc_cost = ltc_monthly_cost * healthcare_inflation_factor
                    ltc_months_remaining -= 1.0
                    if ltc_months_remaining <= 0:
                        ltc_active = False
                elif not had_ltc_event and month_in_year == 0:
                    # Only check for LTC at start of each year
                    if current_age < 65:
                        annual_prob = 0.001
                    elif current_age >= 90:
                        annual_prob = 0.10
                    else:
                        annual_prob = 0.05
                        if 65 <= current_age < 70:
                            annual_prob = (0.05 - 0.02) / 5
                        elif 70 <= current_age < 75:
                            annual_prob = (0.10 - 0.05) / 5
                        elif 75 <= current_age < 80:
                            annual_prob = (0.20 - 0.10) / 5
                        elif 80 <= current_age < 85:
                            annual_prob = (0.35 - 0.20) / 5
                        elif 85 <= current_age < 90:
                            annual_prob = (0.50 - 0.35) / 5
                    
                    if all_ltc_randoms[sim, year_idx] < annual_prob:
                        had_ltc_event = True
                        ltc_active = True
                        ltc_years = max(0.5, 2.5 + 1.5 * all_ltc_duration_randoms[sim, year_idx])
                        ltc_months_remaining = ltc_years * 12
                        ltc_cost = ltc_monthly_cost * healthcare_inflation_factor
                        entering_ltc_this_month = True
            
            if entering_ltc_this_month and mortgage_remaining > 0:
                mortgage_remaining = 0
            
            # Calculate next month's spending need
            if ltc_active:
                w = ltc_cost + medicare_cost
            else:
                w = base_need
                if mortgage_remaining > 0:
                    w += mortgage_monthly_payment
                if health_care_remaining > 0:
                    w += health_care_cost
                w += medicare_cost
            
            # At end of year, do Roth conversion and update yearly income tracking
            if month_in_year == 11:
                taxable_income = year_gross_withdrawal + year_ss_received
                
                # Roth conversion at end of year
                if enable_roth_conversion and p_bal > 0:
                    if roth_conversion_upper_limit > 0:
                        headroom = max(0.0, min(roth_conversion_upper_limit - taxable_income, p_bal))
                    else:
                        headroom = p_bal
                    if headroom > 0:
                        taxes_before = _tax_liability_jit(taxable_income, bracket_arr, rate_arr, cumulative_tax)
                        taxes_after = _tax_liability_jit(taxable_income + headroom, bracket_arr, rate_arr, cumulative_tax)
                        additional_tax = max(0.0, taxes_after - taxes_before)
                        p_bal -= headroom
                        r_bal += headroom
                        taxable_income += headroom
                        if additional_tax > 0:
                            if r_bal >= additional_tax:
                                r_bal -= additional_tax
                            else:
                                additional_tax -= r_bal
                                r_bal = 0.0
                                p_bal = max(0.0, p_bal - additional_tax)
                
                last_yearly_taxable_income = taxable_income
                year_gross_withdrawal = 0.0
                year_ss_received = 0.0
            
            if r_bal < 0 or p_bal < 0 or s_bal < 0:
                succeeded = False
                break
        
        if succeeded:
            results[sim] = 1
    
    return results


def simulate(cfg: SimulationConfig, collect_paths: bool = False):
    """Run the Monte Carlo simulation with monthly resolution."""
    
    n_sims = cfg.number_of_simulations
    months_to_retirement = cfg.months_to_retirement
    months_of_retirement = cfg.months_of_retirement
    years_of_retirement = cfg.years_of_retirement
    
    # Pre-generate all random numbers
    # Pre-retirement returns (monthly)
    if months_to_retirement > 0:
        pre_ret_randoms = np.random.normal(
            cfg.pre_retirement_monthly_mean, 
            cfg.pre_retirement_monthly_std, 
            (n_sims, months_to_retirement)
        )
        growth_factors = np.prod(1 + pre_ret_randoms, axis=1)
    else:
        growth_factors = np.ones(n_sims)
    
    # Correlated returns random numbers (monthly)
    uncorrelated = np.random.standard_normal((n_sims, 3, months_of_retirement))
    
    all_stock, all_bond, all_inflation = _generate_batch_correlated_returns(
        n_sims=n_sims,
        n_periods=months_of_retirement,
        stock_mean=cfg.stock_monthly_mean,
        stock_std=cfg.stock_monthly_std,
        bond_mean=cfg.bond_monthly_mean,
        bond_std=cfg.bond_monthly_std,
        inflation_mean=cfg.inflation_monthly_mean,
        inflation_std=cfg.inflation_monthly_std,
        inflation_bond_corr=cfg.inflation_bond_correlation,
        stock_bond_corr=cfg.stock_bond_correlation,
        uncorrelated=uncorrelated,
    )
    
    # Healthcare inflation (monthly)
    all_healthcare_inflation = np.random.normal(
        cfg.healthcare_inflation_monthly_mean, 
        cfg.healthcare_inflation_monthly_std, 
        (n_sims, months_of_retirement)
    )
    
    # Death probability randoms (yearly - mortality tables are yearly)
    all_death_randoms = np.random.random((n_sims, years_of_retirement))
    
    # LTC randoms (yearly for probability check)
    all_ltc_randoms = np.random.random((n_sims, years_of_retirement))
    all_ltc_duration_randoms = np.random.standard_normal((n_sims, years_of_retirement))
    
    # Ensure tax arrays are initialized
    bracket_arr, rate_arr, cumulative_tax = _ensure_tax_arrays(cfg)
    
    # Get Roth conversion limit
    roth_conversion_upper_limit = 0.0
    if cfg.enable_roth_conversion and cfg._roth_conversion_upper_limit is not None:
        roth_conversion_upper_limit = cfg._roth_conversion_upper_limit
    elif cfg.enable_roth_conversion:
        roth_conversion_upper_limit = -1.0  # Signal for unlimited
    
    # Run parallel simulation
    results = _simulate_parallel(
        n_sims=n_sims,
        months_of_retirement=months_of_retirement,
        retirement_age=cfg.retirement_age,
        social_security_age_started=cfg.social_security_age_started,
        social_security_monthly_amount=cfg.social_security_monthly_amount,
        current_roth=cfg.current_roth,
        current_401a_and_403b=cfg.current_401a_and_403b,
        current_savings=cfg.current_savings,
        savings_monthly_rate=cfg.savings_interest_rate / 12,
        base_monthly_need=cfg.base_monthly_need,
        mortgage_months_in_retirement=cfg.mortgage_months_in_retirement,
        mortgage_monthly_payment=cfg.mortgage_monthly_payment,
        health_care_months_in_retirement=cfg.health_care_months_in_retirement,
        health_care_monthly_payment=cfg.health_care_monthly_payment,
        healthcare_inflation_monthly_mean=cfg.healthcare_inflation_monthly_mean,
        healthcare_inflation_monthly_std=cfg.healthcare_inflation_monthly_std,
        include_medicare_premiums=cfg.include_medicare_premiums,
        include_ltc_risk=cfg.include_ltc_risk,
        ltc_monthly_cost=cfg.ltc_monthly_cost,
        filing_status_is_married=(cfg.filing_status == "married"),
        enable_roth_conversion=cfg.enable_roth_conversion,
        roth_conversion_upper_limit=roth_conversion_upper_limit,
        enable_dynamic_withdrawal=cfg.enable_dynamic_withdrawal,
        dynamic_withdrawal_trigger=cfg.dynamic_withdrawal_trigger,
        initial_cash_reserve=cfg.current_savings if cfg.enable_dynamic_withdrawal else 0.0,
        replenish_savings=cfg.replenish_savings_after_drawdown,
        bracket_arr=bracket_arr,
        rate_arr=rate_arr,
        cumulative_tax=cumulative_tax,
        death_probs=cfg.death_probs,
        growth_factors=growth_factors,
        all_stock_returns=all_stock,
        all_bond_returns=all_bond,
        all_inflation=all_inflation,
        all_healthcare_inflation=all_healthcare_inflation,
        all_death_randoms=all_death_randoms,
        all_ltc_randoms=all_ltc_randoms,
        all_ltc_duration_randoms=all_ltc_duration_randoms,
    )
    
    success_count = np.sum(results)
    
    # If paths are requested, run a subset sequentially to collect them
    if collect_paths:
        success_paths, failure_paths = _collect_paths_sequential(cfg, min(n_sims, 2000))
        return success_count / n_sims, success_paths, failure_paths
    
    return success_count / n_sims


def _collect_paths_sequential(cfg: SimulationConfig, n_sims: int):
    """Collect simulation paths for visualization (runs sequentially with monthly resolution).
    
    Returns yearly balance snapshots for plotting even though simulation runs monthly.
    """
    success_paths = []
    failure_paths = []
    
    months_to_retirement = cfg.months_to_retirement
    months_of_retirement = cfg.months_of_retirement
    years_of_retirement = cfg.years_of_retirement
    bracket_arr, rate_arr, cumulative_tax = _ensure_tax_arrays(cfg)
    
    for _ in range(n_sims):
        # Pre-retirement growth (monthly)
        if months_to_retirement > 0:
            pre_ret_returns = np.random.normal(
                cfg.pre_retirement_monthly_mean, 
                cfg.pre_retirement_monthly_std, 
                months_to_retirement
            )
            growth_factor = np.prod(1 + pre_ret_returns)
        else:
            growth_factor = 1.0

        r_bal = cfg.current_roth * growth_factor
        p_bal = cfg.current_401a_and_403b * growth_factor
        s_bal = cfg.current_savings  # Savings account (last resort)
        savings_monthly_rate = cfg.savings_interest_rate / 12

        base_need = cfg.base_monthly_need
        mortgage_remaining = cfg.mortgage_months_in_retirement
        health_care_remaining = cfg.health_care_months_in_retirement
        health_care_cost = cfg.health_care_monthly_payment if health_care_remaining > 0 else 0.0
        
        healthcare_inflation_factor = 1.0
        ltc_active = False
        ltc_months_remaining = 0.0
        had_ltc_event = False
        
        # Dynamic withdrawal strategy tracking
        # initial_cash_reserve tracks the target value to refill to
        # s_bal serves as the cash reserve (C)
        initial_cash_reserve = cfg.current_savings if cfg.enable_dynamic_withdrawal else 0.0
        running_return_tally = 0.0
        in_drawdown_sequence = False
        
        w = base_need
        if mortgage_remaining > 0:
            w += cfg.mortgage_monthly_payment
        if health_care_remaining > 0:
            w += health_care_cost
            
        death_year = sample_death_year(
            cfg.retirement_age, years_of_retirement, cfg.death_probs
        )
        death_month = (death_year + 1) * 12
        
        # Generate monthly returns
        stock_returns, bond_returns, infls = generate_correlated_returns(
            n_periods=months_of_retirement,
            stock_mean=cfg.stock_monthly_mean,
            stock_std=cfg.stock_monthly_std,
            bond_mean=cfg.bond_monthly_mean,
            bond_std=cfg.bond_monthly_std,
            inflation_mean=cfg.inflation_monthly_mean,
            inflation_std=cfg.inflation_monthly_std,
            inflation_bond_corr=cfg.inflation_bond_correlation,
            stock_bond_corr=cfg.stock_bond_correlation,
        )

        # Path stores yearly snapshots (at end of each year)
        path = [r_bal + p_bal + s_bal]
        
        # Track yearly values for tax calculations
        year_gross_withdrawal = 0.0
        year_ss_received = 0.0
        last_yearly_taxable_income = 0.0
        
        failed = False
        for month_idx in range(1, months_of_retirement + 1):
            if month_idx > death_month:
                # Died - simulation ends as success
                break
            
            year_idx = (month_idx - 1) // 12
            month_in_year = (month_idx - 1) % 12
            current_age = cfg.retirement_age + year_idx
            
            # Monthly healthcare inflation
            healthcare_infl = np.random.normal(
                cfg.healthcare_inflation_monthly_mean, 
                cfg.healthcare_inflation_monthly_std
            )
            healthcare_inflation_factor *= (1.0 + healthcare_infl)
            
            sr = stock_returns[month_idx - 1]
            br = bond_returns[month_idx - 1]
            infl = infls[month_idx - 1]
            
            # Portfolio allocation based on yearly spending ratio
            yearly_w = w * 12
            ratio = (r_bal + p_bal) / yearly_w if yearly_w > 0 else 100.0
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

            port_ret = stock_pct * sr + (1.0 - stock_pct) * br
            r_bal *= (1.0 + port_ret)
            p_bal *= (1.0 + port_ret)

            # Check Social Security eligibility
            # Person's age in months at month_idx is: retirement_age * 12 + (month_idx - 1)
            receiving_ss = (cfg.retirement_age * 12 + month_idx - 1) >= (cfg.social_security_age_started * 12)
            
            # Monthly withdrawal calculation
            monthly_net_need = w

            # Apply monthly interest to savings
            s_bal *= (1.0 + savings_monthly_rate)
            
            # Dynamic withdrawal strategy implementation
            if cfg.enable_dynamic_withdrawal:
                rp = port_ret
                
                if rp < cfg.dynamic_withdrawal_trigger and s_bal > 0:
                    # Drawdown month - use savings (cash reserve) first
                    in_drawdown_sequence = True
                    running_return_tally += rp
                    
                    ss_this_month = cfg.social_security_monthly_amount if receiving_ss else 0.0
                    remaining_need = monthly_net_need - ss_this_month
                    
                    if remaining_need <= 0:
                        year_ss_received += ss_this_month
                    elif s_bal >= remaining_need:
                        s_bal -= remaining_need
                        year_ss_received += ss_this_month
                    else:
                        remaining_after_savings = remaining_need - s_bal
                        s_bal = 0.0
                        year_ss_received += ss_this_month
                        
                        # Account for SS taxation when calculating tax on withdrawal
                        annualized_remaining = remaining_after_savings * 12
                        if receiving_ss:
                            ss_annual = cfg.social_security_monthly_amount * 12
                            gross_annual = _gross_from_net_with_ss_jit(
                                annualized_remaining, ss_annual, bracket_arr, rate_arr, cumulative_tax
                            )
                        else:
                            gross_annual = _gross_from_net_jit(annualized_remaining, bracket_arr, rate_arr, cumulative_tax)
                        gross_monthly = gross_annual / 12
                        year_gross_withdrawal += gross_monthly
                        
                        if p_bal >= gross_monthly:
                            p_bal -= gross_monthly
                        else:
                            rem_gross = gross_monthly - p_bal
                            p_bal = 0.0
                            if r_bal >= rem_gross:
                                r_bal -= rem_gross
                            else:
                                r_bal = 0.0
                                
                elif rp < cfg.dynamic_withdrawal_trigger and s_bal <= 0:
                    in_drawdown_sequence = True
                    running_return_tally += rp
                    
                    annualized_need = monthly_net_need * 12
                    if receiving_ss:
                        ss_annual = cfg.social_security_monthly_amount * 12
                        gross_annual = _gross_from_net_with_ss_jit(
                            annualized_need, ss_annual, bracket_arr, rate_arr, cumulative_tax
                        )
                        gross_monthly = gross_annual / 12
                        year_ss_received += cfg.social_security_monthly_amount
                    else:
                        gross_annual = _gross_from_net_jit(annualized_need, bracket_arr, rate_arr, cumulative_tax)
                        gross_monthly = gross_annual / 12
                    
                    year_gross_withdrawal += gross_monthly
                    
                    if p_bal >= gross_monthly:
                        p_bal -= gross_monthly
                    else:
                        rem_gross = gross_monthly - p_bal
                        p_bal = 0.0
                        if r_bal >= rem_gross:
                            r_bal -= rem_gross
                        else:
                            rem_gross -= r_bal
                            r_bal = 0.0
                            s_bal -= rem_gross
                            
                elif in_drawdown_sequence and running_return_tally < 0:
                    running_return_tally += rp
                    
                    annualized_need = monthly_net_need * 12
                    if receiving_ss:
                        ss_annual = cfg.social_security_monthly_amount * 12
                        gross_annual = _gross_from_net_with_ss_jit(
                            annualized_need, ss_annual, bracket_arr, rate_arr, cumulative_tax
                        )
                        gross_monthly = gross_annual / 12
                        year_ss_received += cfg.social_security_monthly_amount
                    else:
                        gross_annual = _gross_from_net_jit(annualized_need, bracket_arr, rate_arr, cumulative_tax)
                        gross_monthly = gross_annual / 12
                    
                    year_gross_withdrawal += gross_monthly
                    
                    if p_bal >= gross_monthly:
                        p_bal -= gross_monthly
                    else:
                        rem_gross = gross_monthly - p_bal
                        p_bal = 0.0
                        if r_bal >= rem_gross:
                            r_bal -= rem_gross
                        else:
                            rem_gross -= r_bal
                            r_bal = 0.0
                            s_bal -= rem_gross
                            
                elif in_drawdown_sequence and s_bal < initial_cash_reserve and cfg.replenish_savings_after_drawdown:
                    # Market recovered and refilling savings (cash reserve)
                    annualized_need = monthly_net_need * 12
                    if receiving_ss:
                        ss_annual = cfg.social_security_monthly_amount * 12
                        gross_annual = _gross_from_net_with_ss_jit(
                            annualized_need, ss_annual, bracket_arr, rate_arr, cumulative_tax
                        )
                        gross_monthly = gross_annual / 12
                        year_ss_received += cfg.social_security_monthly_amount
                    else:
                        gross_annual = _gross_from_net_jit(annualized_need, bracket_arr, rate_arr, cumulative_tax)
                        gross_monthly = gross_annual / 12
                    
                    year_gross_withdrawal += gross_monthly
                    
                    refill_amount = min(initial_cash_reserve - s_bal, monthly_net_need * 0.5)
                    total_gross_needed = gross_monthly + refill_amount
                    
                    if p_bal >= total_gross_needed:
                        p_bal -= total_gross_needed
                        s_bal += refill_amount
                    else:
                        rem_gross = total_gross_needed - p_bal
                        p_bal = 0.0
                        if r_bal >= rem_gross:
                            r_bal -= rem_gross
                            s_bal += refill_amount
                        else:
                            # Not enough in retirement accounts to fully refill
                            # Just refill what we can after covering expenses
                            available = r_bal
                            r_bal = 0.0
                            # Calculate how much is left for refill after covering the expense portion
                            # rem_gross includes both expense shortfall and refill amount
                            # expense_shortfall = rem_gross - refill_amount (what p_bal couldn't cover)
                            # If available > expense_shortfall, the remainder can go to refill
                            actual_refill = max(0.0, available - (rem_gross - refill_amount))
                            s_bal += actual_refill
                    
                    if s_bal >= initial_cash_reserve:
                        s_bal = initial_cash_reserve
                        running_return_tally = 0.0
                        in_drawdown_sequence = False
                else:
                    # Not in drawdown or savings already fully funded (or no replenishment)
                    running_return_tally = 0.0
                    in_drawdown_sequence = False
                    
                    annualized_need = monthly_net_need * 12
                    if receiving_ss:
                        ss_annual = cfg.social_security_monthly_amount * 12
                        gross_annual = _gross_from_net_with_ss_jit(
                            annualized_need, ss_annual, bracket_arr, rate_arr, cumulative_tax
                        )
                        gross_monthly = gross_annual / 12
                        year_ss_received += cfg.social_security_monthly_amount
                    else:
                        gross_annual = _gross_from_net_jit(annualized_need, bracket_arr, rate_arr, cumulative_tax)
                        gross_monthly = gross_annual / 12
                    
                    year_gross_withdrawal += gross_monthly
                    
                    if p_bal >= gross_monthly:
                        p_bal -= gross_monthly
                    else:
                        rem_gross = gross_monthly - p_bal
                        p_bal = 0.0
                        if r_bal >= rem_gross:
                            r_bal -= rem_gross
                        else:
                            rem_gross -= r_bal
                            r_bal = 0.0
                            s_bal -= rem_gross
            else:
                # Original withdrawal logic
                annualized_need = monthly_net_need * 12
                if receiving_ss:
                    ss_annual = cfg.social_security_monthly_amount * 12
                    gross_annual = _gross_from_net_with_ss_jit(
                        annualized_need, ss_annual, bracket_arr, rate_arr, cumulative_tax
                    )
                    gross_monthly = gross_annual / 12
                    year_ss_received += cfg.social_security_monthly_amount
                else:
                    gross_annual = _gross_from_net_jit(annualized_need, bracket_arr, rate_arr, cumulative_tax)
                    gross_monthly = gross_annual / 12
                
                year_gross_withdrawal += gross_monthly
                
                if p_bal >= gross_monthly:
                    p_bal -= gross_monthly
                else:
                    rem_gross = gross_monthly - p_bal
                    p_bal = 0.0
                    if r_bal >= rem_gross:
                        r_bal -= rem_gross
                    else:
                        rem_gross -= r_bal
                        r_bal = 0.0
                        s_bal -= rem_gross

            # Update base need with monthly inflation
            base_need *= (1.0 + infl)
            if mortgage_remaining > 0:
                mortgage_remaining -= 1
            
            if health_care_remaining > 0:
                health_care_cost *= (1.0 + healthcare_infl)
                health_care_remaining -= 1
            
            # Medicare costs (monthly)
            medicare_cost = 0.0
            if cfg.include_medicare_premiums and current_age >= 65:
                medicare_cost = calculate_medicare_premium(
                    current_age, 
                    last_yearly_taxable_income, 
                    cfg.filing_status,
                    healthcare_inflation_factor
                ) / 12  # Monthly portion
            
            # LTC costs (monthly)
            ltc_cost = 0.0
            entering_ltc_this_month = False
            if cfg.include_ltc_risk:
                if ltc_active and ltc_months_remaining > 0:
                    ltc_cost = cfg.ltc_monthly_cost * healthcare_inflation_factor
                    ltc_months_remaining -= 1.0
                    if ltc_months_remaining <= 0:
                        ltc_active = False
                elif not had_ltc_event and month_in_year == 0:
                    ltc_needed, ltc_duration = sample_ltc_event(current_age, had_ltc_event)
                    if ltc_needed:
                        had_ltc_event = True
                        ltc_active = True
                        ltc_months_remaining = ltc_duration * 12
                        ltc_cost = cfg.ltc_monthly_cost * healthcare_inflation_factor
                        entering_ltc_this_month = True
            
            if entering_ltc_this_month and mortgage_remaining > 0:
                mortgage_remaining = 0
            
            # Calculate next month's spending need
            if ltc_active:
                w = ltc_cost + medicare_cost
            else:
                w = base_need
                if mortgage_remaining > 0:
                    w += cfg.mortgage_monthly_payment
                if health_care_remaining > 0:
                    w += health_care_cost
                w += medicare_cost
            
            # End of year processing
            if month_in_year == 11:
                taxable_income = year_gross_withdrawal + year_ss_received
                
                # Roth conversion at end of year
                if cfg.enable_roth_conversion:
                    p_bal, r_bal, taxable_income = perform_roth_conversion(
                        p_bal, r_bal, taxable_income, cfg
                    )
                
                last_yearly_taxable_income = taxable_income
                year_gross_withdrawal = 0.0
                year_ss_received = 0.0
                
                # Record yearly snapshot for path (include savings in total)
                path.append(r_bal + p_bal + s_bal)

            if r_bal < 0 or p_bal < 0 or s_bal < 0:
                # Record final snapshot and mark as failure
                if month_in_year != 11:  # If we haven't already added end-of-year
                    path.append(r_bal + p_bal + s_bal)
                failure_paths.append(path)
                failed = True
                break
        
        if not failed:
            success_paths.append(path)

    return success_paths, failure_paths


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
            "inflation_bond_correlation": cfg.inflation_bond_correlation,
            "stock_bond_correlation": cfg.stock_bond_correlation,
        },
        "user": {
            "gender": cfg.gender,
            "filing_status": cfg.filing_status,
            "current_age": cfg.current_age,
            "retirement_age": cfg.retirement_age,
            "average_yearly_need": cfg.average_yearly_need,
            "current_roth": cfg.current_roth,
            "current_401a_and_403b": cfg.current_401a_and_403b,
            "current_savings": cfg.current_savings,
            "savings_interest_rate": cfg.savings_interest_rate,
            "full_social_security_at_67": cfg.full_social_security_at_67,
            "social_security_age_started": cfg.social_security_age_started,
            "mortgage_payment": cfg.mortgage_payment,
            "mortgage_years_left": cfg.mortgage_years_left,
            "health_care_payment": cfg.health_care_payment,
            "healthcare_inflation_mean": cfg.healthcare_inflation_mean,
            "healthcare_inflation_std": cfg.healthcare_inflation_std,
            "include_medicare_premiums": cfg.include_medicare_premiums,
            "include_ltc_risk": cfg.include_ltc_risk,
            "ltc_annual_cost": cfg.ltc_annual_cost,
            "enable_roth_conversion": cfg.enable_roth_conversion,
            "roth_conversion_rate_cap": cfg.roth_conversion_rate_cap,
            "enable_dynamic_withdrawal": cfg.enable_dynamic_withdrawal,
            "dynamic_withdrawal_trigger": cfg.dynamic_withdrawal_trigger,
        },
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=2)
