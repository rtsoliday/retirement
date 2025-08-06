import numpy as np
from bisect import bisect_right
import pandas as pd
import tkinter as tk
from tkinter import ttk

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

# Default parameter values
DEFAULT_GENERAL = {
    "n_sim": 2_000,
    "n_years": 60,
    "stock_mean_return": 0.1046,
    "stock_std_dev": 0.208,
    "bond_mean_return": 0.03,
    "bond_std_dev": 0.053,
    "infl_mean": 0.033,
    "infl_std": 0.04,
}

DEFAULT_USER = {
    "gender": "male",
    "current_age": 50,
    "retire_age": 58,
    "average_yearly_need": 77_334,
    "roth_current": 100_000,
    "pretax_current": 800_000,
    "ss_gross": 33_456,
    "stock_ratio": 0.99,
}


def run_sim():
    global n_sim, n_years, stock_mean_return, stock_std_dev, bond_mean_return
    global bond_std_dev, infl_mean, infl_std, gender, current_age, retire_age
    global average_yearly_need, roth_current, pretax_current, ss_gross
    global stock_ratio, bond_ratio, retirement_yearly_need, roth_start
    global pretax_start, death_probs

    n_sim = int(gen_entries["n_sim"].get())
    n_years = int(gen_entries["n_years"].get())
    stock_mean_return = float(gen_entries["stock_mean_return"].get())
    stock_std_dev = float(gen_entries["stock_std_dev"].get())
    bond_mean_return = float(gen_entries["bond_mean_return"].get())
    bond_std_dev = float(gen_entries["bond_std_dev"].get())
    infl_mean = float(gen_entries["infl_mean"].get())
    infl_std = float(gen_entries["infl_std"].get())

    gender = user_entries["gender"].get().strip().lower()
    current_age = int(user_entries["current_age"].get())
    retire_age = int(user_entries["retire_age"].get())
    average_yearly_need = float(user_entries["average_yearly_need"].get())
    roth_current = float(user_entries["roth_current"].get())
    pretax_current = float(user_entries["pretax_current"].get())
    ss_gross = float(user_entries["ss_gross"].get())
    stock_ratio = float(user_entries["stock_ratio"].get())
    bond_ratio = 1 - stock_ratio

    retirement_yearly_need = average_yearly_need * (1 + infl_mean) ** (
        retire_age - current_age
    )
    roth_start = roth_current * (1 + stock_mean_return) ** (
        retire_age - current_age
    )
    pretax_start = pretax_current * (1 + stock_mean_return) ** (
        retire_age - current_age
    )

    precomp_ret_need_var.set(
        f"Year 1 net need: ${retirement_yearly_need:,.0f}"
    )
    precomp_roth_start_var.set(
        f"Roth at retirement: ${roth_start:,.0f}"
    )
    precomp_pretax_start_var.set(
        f"Pre-tax at retirement: ${pretax_start:,.0f}"
    )

    file = (
        "DeathProbsE_M_Alt2_TR2025.csv"
        if gender.startswith("m")
        else "DeathProbsE_F_Alt2_TR2025.csv"
    )
    df = pd.read_csv(file)
    death_row = df[df["Year"] == 2025].iloc[0]
    death_probs = death_row.drop("Year").astype(float).values

    rate = simulate(retirement_yearly_need) * 100

    results = [
        f"Success rate: {rate:.1f}%",
        f"Gross needed in year 1: ${gross_from_net(retirement_yearly_need):,.0f}",
        f"Gross needed in year {62 - retire_age} (with SS): ${gross_from_net_with_ss(retirement_yearly_need, ss_gross):,.0f}",
    ]
    results_var.set("\n".join(results))


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Retirement Simulator")
    root.geometry("360x640")

    gen_entries = {}
    user_entries = {}

    general_frame = ttk.LabelFrame(root, text="General Parameters")
    general_frame.pack(fill="x", padx=10, pady=5)
    for key, default in DEFAULT_GENERAL.items():
        row = ttk.Frame(general_frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=key.replace("_", " ").title()).pack(side="left")
        ent = ttk.Entry(row)
        ent.insert(0, str(default))
        ent.pack(side="right", fill="x", expand=True)
        gen_entries[key] = ent

    user_frame = ttk.LabelFrame(root, text="User-specific Parameters")
    user_frame.pack(fill="x", padx=10, pady=5)
    for key, default in DEFAULT_USER.items():
        row = ttk.Frame(user_frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=key.replace("_", " ").title()).pack(side="left")
        ent = ttk.Entry(row)
        ent.insert(0, str(default))
        ent.pack(side="right", fill="x", expand=True)
        user_entries[key] = ent

    precomp_frame = ttk.LabelFrame(root, text="Pre-computations")
    precomp_frame.pack(fill="x", padx=10, pady=5)
    precomp_ret_need_var = tk.StringVar()
    precomp_roth_start_var = tk.StringVar()
    precomp_pretax_start_var = tk.StringVar()
    ttk.Label(precomp_frame, textvariable=precomp_ret_need_var).pack(anchor="w")
    ttk.Label(precomp_frame, textvariable=precomp_roth_start_var).pack(anchor="w")
    ttk.Label(precomp_frame, textvariable=precomp_pretax_start_var).pack(anchor="w")

    run_frame = ttk.Frame(root)
    run_frame.pack(fill="x", padx=10, pady=5)
    ttk.Button(run_frame, text="Run Simulation", command=run_sim).pack()

    results_frame = ttk.LabelFrame(root, text="Results")
    results_frame.pack(fill="both", expand=True, padx=10, pady=5)
    results_var = tk.StringVar()
    ttk.Label(results_frame, textvariable=results_var, wraplength=320).pack(anchor="w")

    root.mainloop()

