import numpy as np
from bisect import bisect_right
import pandas as pd
import tkinter as tk
from tkinter import ttk
import json
import os

# Monte Carlo setup
np.random.seed()


def parse_percent(val: str) -> float:
    """Convert a percentage string like '10%' to a float 0.10."""
    return float(val.strip().rstrip("%")) / 100


def parse_dollars(val: str) -> float:
    """Convert a currency string like '$1,234' to a float 1234.0."""
    return float(val.replace("$", "").replace(",", "").strip())

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

def simulate(base_yearly_need):
    global retirement_age, current_age
    global current_roth, current_401a_and_403b
    global pre_retirement_mean_return, pre_retirement_std_dev
    global roth_start, pretax_start
    global mortgage_years_in_retirement, mortgage_yearly_payment

    success = 0
    for _ in range(number_of_simulations):
        # Grow pre-retirement balances using both mean return and volatility.
        years_to_retirement = retirement_age - current_age
        pre_ret_returns = np.random.normal(
            pre_retirement_mean_return, pre_retirement_std_dev, years_to_retirement
        )
        growth_factor = np.prod(1 + pre_ret_returns)

        roth_start = current_roth * growth_factor
        pretax_start = current_401a_and_403b * growth_factor



        r_bal, p_bal = roth_start, pretax_start
        base_need = base_yearly_need
        mortgage_remaining = mortgage_years_in_retirement
        w = base_need + (mortgage_yearly_payment if mortgage_remaining > 0 else 0)
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

            # inflation-adjust base need and recompute net need
            base_need *= (1 + i)
            if mortgage_remaining > 0:
                mortgage_remaining -= 1
            w = base_need + (mortgage_yearly_payment if mortgage_remaining > 0 else 0)

            # if either balance goes negative → ruin
            if r_bal < 0 or p_bal < 0:
                break
        else:
            success += 1

    return success / number_of_simulations

# Default parameter values
DEFAULT_GENERAL = {
    "number_of_simulations": 2_000,
    "pre_retirement_mean_return": 0.1472,
    "pre_retirement_std_dev": 0.292,
    "stock_mean_return": 0.1046,
    "stock_std_dev": 0.208,
    "bond_mean_return": 0.03,
    "bond_std_dev": 0.053,
    "inflation_mean": 0.033,
    "inflation_std_dev": 0.04,
}

PERCENT_FIELDS = {
    "pre_retirement_mean_return",
    "pre_retirement_std_dev",
    "stock_mean_return",
    "stock_std_dev",
    "bond_mean_return",
    "bond_std_dev",
    "inflation_mean",
    "inflation_std_dev",
    "percent_in_stock_after_retirement",
}

DOLLAR_FIELDS = {
    "average_yearly_need",
    "current_roth",
    "current_401a_and_403b",
    "social_security_at_62",
    "mortgage_payment",
}

DEFAULT_USER = {
    "gender": "male",
    "current_age": 50,
    "retirement_age": 58,
    "average_yearly_need": 75_000,
    "current_roth": 100_000,
    "current_401a_and_403b": 800_000,
    "social_security_at_62": 30_000,
    "mortgage_payment": 0,
    "mortgage_years_left": 0,
    "percent_in_stock_after_retirement": 0.7,
}


CONFIG_FILE = "config.json"


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}


def save_config():
    data = {
        "general": {
            "number_of_simulations": number_of_simulations,
            "pre_retirement_mean_return": pre_retirement_mean_return,
            "pre_retirement_std_dev": pre_retirement_std_dev,
            "stock_mean_return": stock_mean_return,
            "stock_std_dev": stock_std_dev,
            "bond_mean_return": bond_mean_return,
            "bond_std_dev": bond_std_dev,
            "inflation_mean": inflation_mean,
            "inflation_std_dev": inflation_std_dev,
        },
        "user": {
            "gender": gender,
            "current_age": current_age,
            "retirement_age": retirement_age,
            "average_yearly_need": average_yearly_need,
            "current_roth": current_roth,
            "current_401a_and_403b": current_401a_and_403b,
            "social_security_at_62": social_security_at_62,
            "mortgage_payment": mortgage_payment,
            "mortgage_years_left": mortgage_years_left,
            "percent_in_stock_after_retirement": percent_in_stock_after_retirement,
        },
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _load_inputs(percent_override: float | None = None):
    """Load GUI inputs into globals and perform common pre-computations."""
    global number_of_simulations, years_of_retirement, stock_mean_return, stock_std_dev
    global pre_retirement_mean_return, pre_retirement_std_dev
    global bond_mean_return, bond_std_dev, inflation_mean, inflation_std_dev
    global gender, current_age, retirement_age, average_yearly_need, current_roth
    global current_401a_and_403b, social_security_at_62, retirement_yearly_need
    global mortgage_payment, mortgage_years_left, mortgage_yearly_payment
    global mortgage_years_in_retirement, base_retirement_need
    global roth_start, pretax_start, death_probs

    number_of_simulations = int(gen_entries["number_of_simulations"].get())
    pre_retirement_mean_return = parse_percent(gen_entries["pre_retirement_mean_return"].get())
    pre_retirement_std_dev = parse_percent(gen_entries["pre_retirement_std_dev"].get())
    stock_mean_return = parse_percent(gen_entries["stock_mean_return"].get())
    stock_std_dev = parse_percent(gen_entries["stock_std_dev"].get())
    bond_mean_return = parse_percent(gen_entries["bond_mean_return"].get())
    bond_std_dev = parse_percent(gen_entries["bond_std_dev"].get())
    inflation_mean = parse_percent(gen_entries["inflation_mean"].get())
    inflation_std_dev = parse_percent(gen_entries["inflation_std_dev"].get())

    gender = user_entries["gender"].get().strip().lower()
    current_age = int(user_entries["current_age"].get())
    retirement_age = int(user_entries["retirement_age"].get())
    average_yearly_need = parse_dollars(user_entries["average_yearly_need"].get())
    current_roth = parse_dollars(user_entries["current_roth"].get())
    current_401a_and_403b = parse_dollars(user_entries["current_401a_and_403b"].get())
    social_security_at_62 = parse_dollars(user_entries["social_security_at_62"].get())
    mortgage_payment = parse_dollars(user_entries["mortgage_payment"].get())
    mortgage_years_left = int(user_entries["mortgage_years_left"].get())
    mortgage_yearly_payment = mortgage_payment * 12

    years_of_retirement = 119 - retirement_age

    years_to_retirement = retirement_age - current_age
    base_retirement_need = average_yearly_need * (1 + inflation_mean) ** years_to_retirement
    mortgage_years_in_retirement = max(0, mortgage_years_left - years_to_retirement)
    retirement_yearly_need = base_retirement_need + (
        mortgage_yearly_payment if mortgage_years_in_retirement > 0 else 0
    )

    precomp_ret_need_var.set(
        f"Year 1 net need: ${retirement_yearly_need:,.0f}"
    )
    #precomp_roth_start_var.set(
    #    f"Roth at retirement: ${roth_start:,.0f}"
    #)
    #precomp_pretax_start_var.set(
    #    f"Pre-tax at retirement: ${pretax_start:,.0f}"
    #)

    results_var.set("Working...")
    root.update_idletasks()

    file = (
        "DeathProbsE_M_Alt2_TR2025.csv"
        if gender.startswith("m")
        else "DeathProbsE_F_Alt2_TR2025.csv"
    )
    df = pd.read_csv(file)
    death_row = df[df["Year"] == 2025].iloc[0]
    death_probs = death_row.drop("Year").astype(float).values

    if percent_override is not None:
        global percent_in_stock_after_retirement, bond_ratio
        percent_in_stock_after_retirement = percent_override
        bond_ratio = 1 - percent_override


def run_sim():
    """Run a single simulation using the current GUI inputs."""
    _load_inputs(parse_percent(user_entries["percent_in_stock_after_retirement"].get()))

    rate = simulate(base_retirement_need) * 100

    if retirement_age < 62:
        years_until_ss = 62 - retirement_age
        need_at_62 = base_retirement_need * (1 + inflation_mean) ** years_until_ss
        if mortgage_years_in_retirement > years_until_ss:
            need_at_62 += mortgage_yearly_payment
        results = [
            f"Success rate: {rate:.1f}%",
            f"Gross needed in year 1: ${gross_from_net(retirement_yearly_need):,.0f}",
            f"Gross needed in year {years_until_ss} (with SS): ${gross_from_net_with_ss(need_at_62, social_security_at_62):,.0f}",
        ]
    else:
        results = [
            f"Success rate: {rate:.1f}%",
            f"Gross needed in year 1 (with SS): ${gross_from_net_with_ss(retirement_yearly_need, social_security_at_62):,.0f}",
        ]
    results_var.set("\n".join(results))
    save_config()


def optimize_percent():
    """Search for the stock percentage that maximizes success rate."""
    _load_inputs(1.0)

    best_percent = 1.0
    best_rate = simulate(base_retirement_need) * 100
    prev_rate = best_rate
    percent = 0.9

    while percent >= 0:
        percent_in_stock_after_retirement = percent
        bond_ratio = 1 - percent
        rate = simulate(base_retirement_need) * 100
        if rate < prev_rate:
            break
        if rate >= best_rate:
            best_rate = rate
            best_percent = percent
        prev_rate = rate
        percent -= 0.10

    percent = best_percent - 0.05
    prev_rate = best_rate
    while percent >= 0:
        percent_in_stock_after_retirement = percent
        bond_ratio = 1 - percent
        rate = simulate(base_retirement_need) * 100
        if rate < prev_rate:
            break
        if rate >= best_rate:
            best_rate = rate
            best_percent = percent
        prev_rate = rate
        percent -= 0.05

    user_entries["percent_in_stock_after_retirement"].delete(0, tk.END)
    user_entries["percent_in_stock_after_retirement"].insert(0, f"{best_percent*100:.2f}%")

    if retirement_age < 62:
        years_until_ss = 62 - retirement_age
        need_at_62 = base_retirement_need * (1 + inflation_mean) ** years_until_ss
        if mortgage_years_in_retirement > years_until_ss:
            need_at_62 += mortgage_yearly_payment
        results = [
            f"Best percent in stock: {best_percent*100:.1f}%",
            f"Success rate: {best_rate:.1f}%",
            f"Gross needed in year 1: ${gross_from_net(retirement_yearly_need):,.0f}",
            f"Gross needed in year {years_until_ss} (with SS): ${gross_from_net_with_ss(need_at_62, social_security_at_62):,.0f}",
        ]
    else:
        results = [
            f"Best percent in stock: {best_percent*100:.1f}%",
            f"Success rate: {best_rate:.1f}%",
            f"Gross needed in year 1 (with SS): ${gross_from_net_with_ss(retirement_yearly_need, social_security_at_62):,.0f}",
        ]
    results_var.set("\n".join(results))


def load_defaults():
    for key, default in DEFAULT_GENERAL.items():
        ent = gen_entries[key]
        ent.delete(0, tk.END)
        if key in PERCENT_FIELDS:
            ent.insert(0, f"{default * 100:.2f}%")
        else:
            ent.insert(0, str(default))
    for key, default in DEFAULT_USER.items():
        if key == "gender":
            user_entries[key].set(default)
        else:
            ent = user_entries[key]
            ent.delete(0, tk.END)
            if key in PERCENT_FIELDS:
                ent.insert(0, f"{default * 100:.2f}%")
            elif key in DOLLAR_FIELDS:
                ent.insert(0, f"${default:,.0f}")
            else:
                ent.insert(0, str(default))


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Retirement Simulator")
    root.geometry("360x840")

    gen_entries = {}
    user_entries = {}

    config = load_config()
    gen_cfg = config.get("general", {})
    user_cfg = config.get("user", {})

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
        val = gen_cfg.get(key, default)
        if key in PERCENT_FIELDS:
            ent.insert(0, f"{val * 100:.2f}%")
        else:
            ent.insert(0, str(val))
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
        val = user_cfg.get(key, default)
        if key == "gender":
            gender_var = tk.StringVar(value=val)
            rb_frame = ttk.Frame(row)
            rb_frame.pack(side="left", fill="x", expand=True)
            ttk.Radiobutton(rb_frame, text="M  ", variable=gender_var, value="male").pack(side="left")
            ttk.Radiobutton(rb_frame, text="F", variable=gender_var, value="female").pack(side="left")
            user_entries[key] = gender_var
        else:
            ent = ttk.Entry(row)
            if key in PERCENT_FIELDS:
                ent.insert(0, f"{val * 100:.2f}%")
            elif key in DOLLAR_FIELDS:
                ent.insert(0, f"${val:,.0f}")
            else:
                ent.insert(0, str(val))
            ent.pack(side="left", fill="x", expand=True)
            user_entries[key] = ent

    precomp_frame = ttk.LabelFrame(root, text="Pre-computations")
    precomp_frame.pack(fill="x", padx=10, pady=5)
    precomp_ret_need_var = tk.StringVar()
    #precomp_roth_start_var = tk.StringVar()
    #precomp_pretax_start_var = tk.StringVar()
    ttk.Label(precomp_frame, textvariable=precomp_ret_need_var).pack(anchor="w")
    #ttk.Label(precomp_frame, textvariable=precomp_roth_start_var).pack(anchor="w")
    #ttk.Label(precomp_frame, textvariable=precomp_pretax_start_var).pack(anchor="w")

    run_frame = ttk.Frame(root)
    run_frame.pack(fill="x", padx=10, pady=5)
    ttk.Button(run_frame, text="Run Simulations", command=run_sim).pack()
    ttk.Button(run_frame, text="Optimize % In Stocks", command=optimize_percent).pack()
    ttk.Button(run_frame, text="Load Defaults", command=load_defaults).pack()

    results_frame = ttk.LabelFrame(root, text="Results")
    results_frame.pack(fill="both", expand=True, padx=10, pady=5)
    results_var = tk.StringVar()
    ttk.Label(results_frame, textvariable=results_var, wraplength=320).pack(anchor="w")

    root.mainloop()

