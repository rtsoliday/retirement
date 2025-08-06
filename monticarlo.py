import numpy as np
from bisect import bisect_right
import pandas as pd
import tkinter as tk
from tkinter import ttk

# Monte Carlo setup
np.random.seed()


def parse_percent(val: str) -> float:
    """Convert a percentage string like '10%' to a float 0.10."""
    return float(val.strip().rstrip("%")) / 100

def sample_death_year(retirement_age, years_of_retirement):
    """Return the year index (0-indexed) in which death occurs."""
    for j in range(years_of_retirement):
        age = retirement_age + j
        if age >= len(death_probs):
            return years_of_retirement
        if np.random.random() < death_probs[age]:
            return j
    return years_of_retirement  # lived through entire horizon

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

def gross_from_net_with_ss(net_amt, social_security_at_62):
    """
    Find G such that
       (G + social_security_at_62) - tax_liability(G + social_security_at_62) == net_amt
    i.e. accounts + SS minus tax on total = net spending.
    """
    lo, hi = max(0, net_amt - social_security_at_62), (net_amt + social_security_at_62) * 1.5
    for _ in range(40):
        mid = (lo + hi) / 2
        total = mid + social_security_at_62
        if total - tax_liability(total) < net_amt:
            lo = mid
        else:
            hi = mid
    return hi

def simulate(yearly_net_withdrawal):
    success = 0
    for _ in range(number_of_simulations):
        r_bal, p_bal = roth_start, pretax_start
        w = yearly_net_withdrawal
        death_year = sample_death_year(retirement_age, years_of_retirement)
        stock_returns = np.random.normal(stock_mean_return, stock_std_dev, years_of_retirement)
        bond_returns  = np.random.normal(bond_mean_return, bond_std_dev, years_of_retirement)
        infls         = np.random.normal(inflation_mean, inflation_std_dev, years_of_retirement)

        for year_idx, (sr, br, i) in enumerate(zip(stock_returns, bond_returns, infls), start=1):
            if year_idx > death_year:
                success += 1
                break  # retiree is assumed dead; stop withdrawals

            # portfolio return = weighted average of stock & bond returns
            port_ret = percent_in_stock_after_retirement * sr + bond_ratio * br

            # grow balances by portfolio return
            r_bal *= (1 + port_ret)
            p_bal *= (1 + port_ret)

            # pick the right gross calculation
            if retirement_age + year_idx < 62:
                gross_w = gross_from_net(w)
            else:
                gross_w = gross_from_net_with_ss(w, social_security_at_62)

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

    return success / number_of_simulations

# Default parameter values
DEFAULT_GENERAL = {
    "number_of_simulations": 2_000,
    "stock_mean_return": 0.1046,
    "stock_std_dev": 0.208,
    "bond_mean_return": 0.03,
    "bond_std_dev": 0.053,
    "inflation_mean": 0.033,
    "inflation_std_dev": 0.04,
}

PERCENT_FIELDS = {
    "stock_mean_return",
    "stock_std_dev",
    "bond_mean_return",
    "bond_std_dev",
    "inflation_mean",
    "inflation_std_dev",
    "percent_in_stock_after_retirement",
}

DEFAULT_USER = {
    "gender": "male",
    "current_age": 50,
    "retirement_age": 58,
    "average_yearly_need": 75_000,
    "current_roth": 100_000,
    "current_401a_and_403b": 800_000,
    "social_security_at_62": 30_000,
    "percent_in_stock_after_retirement": 0.7,
}


def run_sim():
    global number_of_simulations, years_of_retirement, stock_mean_return, stock_std_dev, bond_mean_return
    global bond_std_dev, inflation_mean, inflation_std_dev, gender, current_age, retirement_age
    global average_yearly_need, current_roth, current_401a_and_403b, social_security_at_62
    global percent_in_stock_after_retirement, bond_ratio, retirement_yearly_need, roth_start
    global pretax_start, death_probs

    number_of_simulations = int(gen_entries["number_of_simulations"].get())
    stock_mean_return = parse_percent(gen_entries["stock_mean_return"].get())
    stock_std_dev = parse_percent(gen_entries["stock_std_dev"].get())
    bond_mean_return = parse_percent(gen_entries["bond_mean_return"].get())
    bond_std_dev = parse_percent(gen_entries["bond_std_dev"].get())
    inflation_mean = parse_percent(gen_entries["inflation_mean"].get())
    inflation_std_dev = parse_percent(gen_entries["inflation_std_dev"].get())

    gender = user_entries["gender"].get().strip().lower()
    current_age = int(user_entries["current_age"].get())
    retirement_age = int(user_entries["retirement_age"].get())
    average_yearly_need = float(user_entries["average_yearly_need"].get())
    current_roth = float(user_entries["current_roth"].get())
    current_401a_and_403b = float(user_entries["current_401a_and_403b"].get())
    social_security_at_62 = float(user_entries["social_security_at_62"].get())
    percent_in_stock_after_retirement = parse_percent(user_entries["percent_in_stock_after_retirement"].get())
    bond_ratio = 1 - percent_in_stock_after_retirement

    years_of_retirement = 119 - retirement_age

    retirement_yearly_need = average_yearly_need * (1 + inflation_mean) ** (
        retirement_age - current_age
    )
    roth_start = current_roth * (1 + stock_mean_return) ** (
        retirement_age - current_age
    )
    pretax_start = current_401a_and_403b * (1 + stock_mean_return) ** (
        retirement_age - current_age
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
        f"Gross needed in year {62 - retirement_age} (with SS): ${gross_from_net_with_ss(retirement_yearly_need, social_security_at_62):,.0f}",
    ]
    results_var.set("\n".join(results))


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Retirement Simulator")
    root.geometry("360x640")

    gen_entries = {}
    user_entries = {}

    label_width = max(
        len(k.replace("_", " ").title())
        for k in list(DEFAULT_GENERAL) + list(DEFAULT_USER)
    )

    general_frame = ttk.LabelFrame(root, text="General Parameters")
    general_frame.pack(fill="x", padx=10, pady=5)
    for key, default in DEFAULT_GENERAL.items():
        row = ttk.Frame(general_frame)
        row.pack(fill="x", pady=2)
        ttk.Label(
            row,
            text=key.replace("_", " ").title(),
            width=label_width,
            anchor="w",
        ).pack(side="left")
        ent = ttk.Entry(row)
        if key in PERCENT_FIELDS:
            ent.insert(0, f"{default * 100:.2f}%")
        else:
            ent.insert(0, str(default))
        ent.pack(side="left", fill="x", expand=True)
        gen_entries[key] = ent

    user_frame = ttk.LabelFrame(root, text="User-specific Parameters")
    user_frame.pack(fill="x", padx=10, pady=5)
    for key, default in DEFAULT_USER.items():
        row = ttk.Frame(user_frame)
        row.pack(fill="x", pady=2)
        ttk.Label(
            row,
            text=key.replace("_", " ").title(),
            width=label_width,
            anchor="w",
        ).pack(side="left")
        if key == "gender":
            gender_var = tk.StringVar(value=default)
            rb_frame = ttk.Frame(row)
            rb_frame.pack(side="left", fill="x", expand=True)
            ttk.Radiobutton(rb_frame, text="M  ", variable=gender_var, value="male").pack(side="left")
            ttk.Radiobutton(rb_frame, text="F", variable=gender_var, value="female").pack(side="left")
            user_entries[key] = gender_var
        else:
            ent = ttk.Entry(row)
            if key in PERCENT_FIELDS:
                ent.insert(0, f"{default * 100:.2f}%")
            else:
                ent.insert(0, str(default))
            ent.pack(side="left", fill="x", expand=True)
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

