import numpy as np
from bisect import bisect_right
import pandas as pd

# Monte Carlo setup
np.random.seed()

def sample_death_year(retire_age, n_years):
    """Return the year index (0-indexed) in which death occurs."""
    for j in range(n_years):
        age = retire_age + j
        if age >= len(death_probs):
            return n_years
        if np.random.random() < death_probs[age]:
            return j
    return n_years  # lived through entire horizon

# IRS 2024 single-filer brackets and marginal rates
brackets = [0, 11_600, 47_150, 100_525, 191_950, 243_725, 609_350]
rates    = [0.10, 0.12, 0.22,   0.24,    0.32,    0.35,    0.37]

def tax_liability(income):
    tax = 0.0
    for i in range(len(rates)):
        lower = brackets[i]
        upper = brackets[i+1] if i+1 < len(brackets) else income
        if income > lower:
            tax += (min(income, upper) - lower) * rates[i]
        else:
            break
    return tax

def gross_from_net(net_amt):
    """Find G such that G - tax_liability(G) = net_amt."""
    lo, hi = net_amt, net_amt * 1.5 + 20_000
    for _ in range(40):
        mid = (lo + hi) / 2
        if mid - tax_liability(mid) < net_amt:
            lo = mid
        else:
            hi = mid
    return hi

def gross_from_net_with_ss(net_amt, ss_gross):
    """
    Find G such that
       (G + ss_gross) - tax_liability(G + ss_gross) == net_amt
    i.e. accounts + SS minus tax on total = net spending.
    """
    lo, hi = max(0, net_amt - ss_gross), (net_amt + ss_gross) * 1.5
    for _ in range(40):
        mid = (lo + hi) / 2
        total = mid + ss_gross
        if total - tax_liability(total) < net_amt:
            lo = mid
        else:
            hi = mid
    return hi

def simulate(yearly_net_withdrawal):
    success = 0
    for _ in range(n_sim):
        r_bal, p_bal = roth_start, pretax_start
        w = yearly_net_withdrawal
        death_year = sample_death_year(retire_age, n_years)
        stock_returns = np.random.normal(stock_mean_return, stock_std_dev, n_years)
        bond_returns  = np.random.normal(bond_mean_return,  bond_std_dev,  n_years)
        infls         = np.random.normal(infl_mean,           infl_std,     n_years)

        for year_idx, (sr, br, i) in enumerate(zip(stock_returns, bond_returns, infls), start=1):
            if year_idx > death_year:
                success += 1
                break  # retiree is assumed dead; stop withdrawals

            # portfolio return = weighted average of stock & bond returns
            port_ret = stock_ratio * sr + bond_ratio * br

            # grow balances by portfolio return
            r_bal *= (1 + port_ret)
            p_bal *= (1 + port_ret)

            # pick the right gross calculation
            if retire_age + year_idx < 62:
                gross_w = gross_from_net(w)
            else:
                gross_w = gross_from_net_with_ss(w, ss_gross)

            # withdraw from pre-tax first
            if p_bal >= gross_w:
                p_bal -= gross_w
            else:
                rem_gross = gross_w - p_bal
                p_bal = 0
                rem_net = rem_gross - tax_liability(rem_gross)
                r_bal -= rem_net

            # inflation-adjust next year's net need
            w *= (1 + i)

            # if either balance goes negative → ruin
            if r_bal < 0 or p_bal < 0:
                break
        else:
            success += 1

    return success / n_sim

# General Parameters
n_sim, n_years = 2_000, 60 # 60 years to cover up to age 118
stock_mean_return, stock_std_dev = 0.1046, 0.208 # based on 1926-2023 S&P 500 returns
bond_mean_return,  bond_std_dev  = 0.03,   0.053 # based on 1926-2023 10-year Treasury returns
infl_mean, infl_std   = 0.033, 0.04 # based on 1926-2023 CPI returns

# User-specific Parameters
gender = 'male'
current_age = 50 # current age
retire_age  = 58 # starting retirement age
average_yearly_need = 77_334 # average yearly spending need today
roth_current, pretax_current = 100_000, 800_000 # current balances in Roth and pre-tax (401k or 403b) accounts
ss_gross = 33_456 # gross Social Security income starting at age 62
stock_ratio = 0.99
bond_ratio  = 1 - stock_ratio

# Pre-computations. Assume all funds are invested in stocks until retirement.
retirement_yearly_need = average_yearly_need * (1 + infl_mean) ** (retire_age - current_age)
roth_start = roth_current * (1 + stock_mean_return) ** (retire_age - current_age)
pretax_start = pretax_current * (1 + stock_mean_return) ** (retire_age - current_age)

# Load mortality data
if (gender == 'male'):
    df = pd.read_csv('DeathProbsE_M_Alt2_TR2025.csv')
else:
    df = pd.read_csv('DeathProbsE_F_Alt2_TR2025.csv')
death_row = df[df['Year'] == 2025].iloc[0]  # row for 2025
death_probs = death_row.drop('Year').astype(float).values  # probabilities for ages 0–119

# Run simulation
rate = simulate(retirement_yearly_need) * 100

# Display results
print(f"\nRetirement age: {retire_age}")
print(f"{stock_ratio * 100:,.0f}/{bond_ratio * 100:,.0f} allocation success rate with ${retirement_yearly_need:,} net withdrawal: {rate:.1f}%")
print(f"Gross needed in year 1 to net ${retirement_yearly_need:,}: ${gross_from_net(retirement_yearly_need):,.0f}")
print(f"Gross needed in year {62 - retire_age} to net ${retirement_yearly_need:,} (with SS): ${gross_from_net_with_ss(retirement_yearly_need, ss_gross):,.0f}")

