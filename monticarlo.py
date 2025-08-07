import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

from core import (
    SimulationConfig,
    parse_percent,
    parse_dollars,
    tax_liability,
    gross_from_net,
    gross_from_net_with_ss,
    social_security_payout,
    simulate,
    load_config,
    save_config,
    brackets,
    rates,
)


class ToolTip:
    """Simple hover tooltip for a widget."""

    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event=None):
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 10
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("tahoma", "8", "normal"),
        ).pack(ipadx=1)

    def _hide(self, _event=None):
        tw = self.tipwindow
        self.tipwindow = None
        if tw is not None:
            tw.destroy()

def plot_paths(success_paths, failure_paths):
    """Plot retirement fund paths for successful and failed simulations."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.transforms as transforms

    failed_max_by_year = {}
    fail_x, fail_y = [], []
    if failure_paths:
        for path in failure_paths:
            for year_idx, val in enumerate(path):
                if val > 0:
                    fail_x.append(year_idx)
                    fail_y.append(val)
                    failed_max_by_year[year_idx] = max(
                        failed_max_by_year.get(year_idx, float("-inf")), val
                    )

    success_x, success_y, colors = [], [], []
    success_min_by_year = {}
    if success_paths:
        for path in success_paths:
            for year_idx, val in enumerate(path):
                if val > 0:
                    success_x.append(year_idx)
                    success_y.append(val)
                    success_min_by_year[year_idx] = min(
                        success_min_by_year.get(year_idx, float("inf")), val
                    )
                    if val > failed_max_by_year.get(year_idx, float("-inf")):
                        colors.append("lime")
                    else:
                        colors.append("green")

    fail_red_x, fail_red_y, fail_maroon_x, fail_maroon_y = [], [], [], []
    for x, y in zip(fail_x, fail_y):
        if y < success_min_by_year.get(x, float("inf")):
            fail_red_x.append(x)
            fail_red_y.append(y)
        else:
            fail_maroon_x.append(x)
            fail_maroon_y.append(y)
    # compute mean funds by year across all paths
    mean_by_year = {}
    for path in (success_paths or []) + (failure_paths or []):
        for year_idx, val in enumerate(path):
            if val > 0:
                mean_by_year.setdefault(year_idx, []).append(val)
    mean_x = sorted(mean_by_year.keys())
    mean_y = [np.mean(mean_by_year[i]) for i in mean_x]

    if success_x or fail_red_x or fail_maroon_x:
        fig, ax = plt.subplots(figsize=(8, 4))
        trans_success = transforms.ScaledTranslation(2 / fig.dpi, 0, fig.dpi_scale_trans)
        trans_failure = transforms.ScaledTranslation(5 / fig.dpi, 0, fig.dpi_scale_trans)

        legend_handles = []
        legend_labels = []
        legend_items = []

        if success_x:
            success_scatter = ax.scatter(
                success_x,
                success_y,
                s=1,
                c=colors,
                linewidths=1,
                transform=ax.transData + trans_success,
                label="Success",
            )
            legend_handles.append(success_scatter)
            legend_labels.append("Success")
            legend_items.append(success_scatter)

        fail_scatter_maroon = fail_scatter_red = fail_scatter_handle = None
        if fail_maroon_x:
            fail_scatter_maroon = ax.scatter(
                fail_maroon_x,
                fail_maroon_y,
                s=1,
                c="maroon",
                linewidths=1,
                transform=ax.transData + trans_failure,
                label="Failure",
            )
            fail_scatter_handle = fail_scatter_maroon
        if fail_red_x:
            fail_scatter_red = ax.scatter(
                fail_red_x,
                fail_red_y,
                s=1,
                c="red",
                linewidths=1,
                transform=ax.transData + trans_failure,
                label="Failure" if fail_scatter_handle is None else "_nolegend_",
            )
            if fail_scatter_handle is None:
                fail_scatter_handle = fail_scatter_red
        if fail_scatter_handle:
            legend_handles.append(fail_scatter_handle)
            legend_labels.append("Failure")
            legend_items.append([fail_scatter_maroon, fail_scatter_red])

        mean_line = None
        if mean_x:
            mean_line = ax.plot(
                mean_x, mean_y, color="black", linewidth=1, label="Mean"
            )[0]
            legend_handles.append(mean_line)
            legend_labels.append("Mean")
            legend_items.append(mean_line)

        all_x = success_x + fail_red_x + fail_maroon_x
        all_y = success_y + fail_red_y + fail_maroon_y
        ax.set_yscale("log")
        ax.set_xlabel("Years in retirement")
        ax.set_ylabel("Total funds ($)")
        ax.set_title("Simulation paths")
        if all_x:
            ax.set_xlim(0, max(all_x))
        if all_y:
            ax.set_ylim(min(all_y), max(all_y))

        if legend_handles:
            legend = ax.legend(legend_handles, legend_labels)
            lookup = {}
            for leg_handle, orig in zip(legend.legend_handles, legend_items):
                leg_handle.set_picker(True)
                lookup[leg_handle] = orig

            def on_pick(event):
                leg_handle = event.artist
                orig = lookup.get(leg_handle)
                if orig is None:
                    return
                if isinstance(orig, list):
                    visible = not orig[0].get_visible()
                    for art in orig:
                        if art is not None:
                            art.set_visible(visible)
                else:
                    visible = not orig.get_visible()
                    orig.set_visible(visible)
                leg_handle.set_alpha(1.0 if visible else 0.2)
                fig.canvas.draw()

            fig.canvas.mpl_connect("pick_event", on_pick)

        plt.show()

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
    "full_social_security_at_67",
    "mortgage_payment",
}

DEFAULT_USER = {
    "gender": "male",
    "current_age": 50,
    "retirement_age": 58,
    "average_yearly_need": 75_000,
    "current_roth": 100_000,
    "current_401a_and_403b": 800_000,
    "full_social_security_at_67": 30_000,
    "social_security_age_started": 62,
    "mortgage_payment": 0,
    "mortgage_years_left": 0,
    "percent_in_stock_after_retirement": 0.7,
}

LABEL_OVERRIDES = {
    "full_social_security_at_67": "Full Social Security at 67",
    "social_security_age_started": "Social Security Age Started",
}

ENTRY_HELP = {
    "number_of_simulations": "How many Monte Carlo runs to perform.",
    "pre_retirement_mean_return": "Expected annual return before retirement (percentage).",
    "pre_retirement_std_dev": "Volatility of pre-retirement returns (percentage).",
    "stock_mean_return": "Average yearly stock market return (percentage).",
    "stock_std_dev": "Volatility of stock returns (percentage).",
    "bond_mean_return": "Average yearly bond return (percentage).",
    "bond_std_dev": "Volatility of bond returns (percentage).",
    "inflation_mean": "Expected average annual inflation rate (percentage).",
    "inflation_std_dev": "Volatility of annual inflation (percentage).",
    "gender": "Select the retiree's gender for mortality assumptions.",
    "current_age": "Current age of the retiree.",
    "retirement_age": "Age at which retirement begins.",
    "average_yearly_need": "Estimated yearly spending in today's dollars.",
    "current_roth": "Current balance in Roth accounts.",
    "current_401a_and_403b": "Current balance in 401a/403b accounts.",
    "full_social_security_at_67": "Annual Social Security benefit if started at age 67.",
    "social_security_age_started": "Age when Social Security benefits start.",
    "mortgage_payment": "Yearly mortgage payment in retirement.",
    "mortgage_years_left": "Number of years remaining on the mortgage currently.",
    "percent_in_stock_after_retirement": "Fraction of portfolio in stocks during retirement.",
}

def _load_inputs(percent_override: float | None = None) -> SimulationConfig:
    """Parse GUI inputs and return a SimulationConfig."""
    import pandas as pd
    number_of_simulations = int(gen_entries["number_of_simulations"].get())
    if number_of_simulations <= 0:
        raise ValueError("Number of simulations must be positive")
    pre_retirement_mean_return = parse_percent(
        gen_entries["pre_retirement_mean_return"].get()
    )
    pre_retirement_std_dev = parse_percent(
        gen_entries["pre_retirement_std_dev"].get()
    )
    stock_mean_return = parse_percent(gen_entries["stock_mean_return"].get())
    stock_std_dev = parse_percent(gen_entries["stock_std_dev"].get())
    bond_mean_return = parse_percent(gen_entries["bond_mean_return"].get())
    bond_std_dev = parse_percent(gen_entries["bond_std_dev"].get())
    inflation_mean = parse_percent(gen_entries["inflation_mean"].get())
    inflation_std_dev = parse_percent(gen_entries["inflation_std_dev"].get())

    gender = user_entries["gender"].get().strip().lower()
    current_age = int(user_entries["current_age"].get())
    retirement_age = int(user_entries["retirement_age"].get())
    if current_age < 0 or retirement_age < 0:
        raise ValueError("Ages must be non-negative")
    if retirement_age <= current_age:
        raise ValueError("Retirement age must be greater than current age")
    if retirement_age >= 119:
        raise ValueError("Retirement age must be less than 119")
    average_yearly_need = parse_dollars(user_entries["average_yearly_need"].get())
    current_roth = parse_dollars(user_entries["current_roth"].get())
    current_401a_and_403b = parse_dollars(user_entries["current_401a_and_403b"].get())
    full_social_security_at_67 = parse_dollars(
        user_entries["full_social_security_at_67"].get()
    )
    social_security_age_started = int(
        user_entries["social_security_age_started"].get()
    )
    if social_security_age_started < 0:
        raise ValueError("Social Security age must be non-negative")
    social_security_yearly_amount = social_security_payout(
        full_social_security_at_67, social_security_age_started
    )
    mortgage_payment = parse_dollars(user_entries["mortgage_payment"].get())
    mortgage_years_left = int(user_entries["mortgage_years_left"].get())
    if mortgage_years_left < 0:
        raise ValueError("Mortgage years left cannot be negative")
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

    if percent_override is None:
        percent_in_stock_after_retirement = parse_percent(
            user_entries["percent_in_stock_after_retirement"].get()
        )
    else:
        if not 0 <= percent_override <= 1:
            raise ValueError(
                "Percent in stock after retirement must be between 0 and 100"
            )
        percent_in_stock_after_retirement = percent_override
    bond_ratio = 1 - percent_in_stock_after_retirement

    cfg = SimulationConfig(
        number_of_simulations=number_of_simulations,
        pre_retirement_mean_return=pre_retirement_mean_return,
        pre_retirement_std_dev=pre_retirement_std_dev,
        stock_mean_return=stock_mean_return,
        stock_std_dev=stock_std_dev,
        bond_mean_return=bond_mean_return,
        bond_std_dev=bond_std_dev,
        inflation_mean=inflation_mean,
        inflation_std_dev=inflation_std_dev,
        gender=gender,
        current_age=current_age,
        retirement_age=retirement_age,
        average_yearly_need=average_yearly_need,
        current_roth=current_roth,
        current_401a_and_403b=current_401a_and_403b,
        full_social_security_at_67=full_social_security_at_67,
        social_security_age_started=social_security_age_started,
        social_security_yearly_amount=social_security_yearly_amount,
        mortgage_payment=mortgage_payment,
        mortgage_years_left=mortgage_years_left,
        percent_in_stock_after_retirement=percent_in_stock_after_retirement,
        bond_ratio=bond_ratio,
        years_of_retirement=years_of_retirement,
        base_retirement_need=base_retirement_need,
        retirement_yearly_need=retirement_yearly_need,
        mortgage_years_in_retirement=mortgage_years_in_retirement,
        mortgage_yearly_payment=mortgage_yearly_payment,
        death_probs=death_probs,
    )
    return cfg


def run_sim():
    """Run a single simulation using the current GUI inputs."""
    try:
        cfg = _load_inputs(
            parse_percent(user_entries["percent_in_stock_after_retirement"].get())
        )
    except ValueError as exc:
        messagebox.showerror("Input error", str(exc))
        return

    rate, success_paths, failure_paths = simulate(cfg, collect_paths=True)
    rate *= 100

    start_success = [p[0] for p in success_paths]
    start_failure = [p[0] for p in failure_paths]
    maroon_thresh = green_thresh = None
    if cfg.retirement_age > cfg.current_age and start_success and start_failure:
        success_min0 = min(start_success)
        failure_max0 = max(start_failure)
        maroon_candidates = [f for f in start_failure if f >= success_min0]
        green_candidates = [s for s in start_success if s <= failure_max0]
        if maroon_candidates:
            maroon_thresh = min(maroon_candidates)
        if green_candidates:
            green_thresh = max(green_candidates)

    if cfg.retirement_age < cfg.social_security_age_started:
        years_until_ss = cfg.social_security_age_started - cfg.retirement_age
        need_at_ss = cfg.base_retirement_need * (1 + cfg.inflation_mean) ** years_until_ss
        if cfg.mortgage_years_in_retirement > years_until_ss:
            need_at_ss += cfg.mortgage_yearly_payment
        results = [
            f"Success rate: {rate:.1f}%",
            f"Gross needed in year 1: ${gross_from_net(cfg.retirement_yearly_need, cfg):,.0f}",
            f"Gross needed in year {years_until_ss} (with SS): ${gross_from_net_with_ss(need_at_ss, cfg.social_security_yearly_amount, cfg):,.0f}",
        ]
    else:
        results = [
            f"Success rate: {rate:.1f}%",
            f"Gross needed in year 1 (with SS): ${gross_from_net_with_ss(cfg.retirement_yearly_need, cfg.social_security_yearly_amount, cfg):,.0f}",
        ]
    if maroon_thresh is not None:
        results.append(
            f"Warning: do not retire if total funds are below ${maroon_thresh:,.0f} in year 0 of retirement."
        )
    if green_thresh is not None:
        results.append(
            f"It is safe to retire if total funds are above ${green_thresh:,.0f} in year 0 of retirement."
        )
    results_var.set("\n".join(results))
    save_config(cfg)
    plot_paths(success_paths, failure_paths)


def optimize_percent():
    """Search for the stock percentage that maximizes success rate."""
    try:
        cfg = _load_inputs(1.0)
    except ValueError as exc:
        messagebox.showerror("Input error", str(exc))
        return

    best_percent = 1.0
    cfg.percent_in_stock_after_retirement = 1.0
    cfg.bond_ratio = 0.0
    best_rate = simulate(cfg) * 100
    prev_rate = best_rate
    percent = 0.9

    while percent >= 0:
        cfg.percent_in_stock_after_retirement = percent
        cfg.bond_ratio = 1 - percent
        rate = simulate(cfg) * 100
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
        cfg.percent_in_stock_after_retirement = percent
        cfg.bond_ratio = 1 - percent
        rate = simulate(cfg) * 100
        if rate < prev_rate:
            break
        if rate >= best_rate:
            best_rate = rate
            best_percent = percent
        prev_rate = rate
        percent -= 0.05

    user_entries["percent_in_stock_after_retirement"].delete(0, tk.END)
    user_entries["percent_in_stock_after_retirement"].insert(0, f"{best_percent*100:.2f}%")

    if cfg.retirement_age < cfg.social_security_age_started:
        years_until_ss = cfg.social_security_age_started - cfg.retirement_age
        need_at_ss = cfg.base_retirement_need * (1 + cfg.inflation_mean) ** years_until_ss
        if cfg.mortgage_years_in_retirement > years_until_ss:
            need_at_ss += cfg.mortgage_yearly_payment
        results = [
            f"Best percent in stock: {best_percent*100:.1f}%",
            f"Success rate: {best_rate:.1f}%",
            f"Gross needed in year 1: ${gross_from_net(cfg.retirement_yearly_need, cfg):,.0f}",
            f"Gross needed in year {years_until_ss} (with SS): ${gross_from_net_with_ss(need_at_ss, cfg.social_security_yearly_amount, cfg):,.0f}",
        ]
    else:
        results = [
            f"Best percent in stock: {best_percent*100:.1f}%",
            f"Success rate: {best_rate:.1f}%",
            f"Gross needed in year 1 (with SS): ${gross_from_net_with_ss(cfg.retirement_yearly_need, cfg.social_security_yearly_amount, cfg):,.0f}",
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
        len(LABEL_OVERRIDES.get(k, k.replace("_", " ").title()))
        for k in list(DEFAULT_GENERAL) + list(DEFAULT_USER)
    )

    general_frame = ttk.LabelFrame(root, text="General Parameters")
    general_frame.pack(fill="x", padx=10, pady=5)
    for key, default in DEFAULT_GENERAL.items():
        row = ttk.Frame(general_frame)
        row.pack(fill="x", pady=2)
        ttk.Label(
            row,
            text=LABEL_OVERRIDES.get(key, key.replace("_", " ").title()),
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
        ToolTip(ent, ENTRY_HELP.get(key, ""))

    user_frame = ttk.LabelFrame(root, text="User-specific Parameters")
    user_frame.pack(fill="x", padx=10, pady=5)
    for key, default in DEFAULT_USER.items():
        row = ttk.Frame(user_frame)
        row.pack(fill="x", pady=2)
        ttk.Label(
            row,
            text=LABEL_OVERRIDES.get(key, key.replace("_", " ").title()),
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
            ToolTip(rb_frame, ENTRY_HELP.get(key, ""))
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
            ToolTip(ent, ENTRY_HELP.get(key, ""))

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

