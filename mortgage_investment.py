"""
Mortgage vs Investment Calculator

This script calculates the minimum investment balance needed to cover 
all mortgage payments, given that the investment rate of return exceeds 
the mortgage interest rate.

The idea is: if your investment grows faster than your mortgage costs,
you need less cash invested than the outstanding mortgage balance to
fully pay it off on schedule.

Now includes capital gains tax calculations on investment withdrawals.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional


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


def parse_percent(val: str) -> float:
    """Convert a percentage string like '14.72%' or '14.72' to a decimal."""
    val = val.strip().rstrip('%')
    return float(val) / 100


def parse_dollars(val: str) -> float:
    """Convert a dollar string like '$143,098.56' to a float."""
    val = val.strip().lstrip('$').replace(',', '')
    return float(val)


from typing import Optional


def calculate_months_remaining(principal: float, annual_rate: float, monthly_payment: float) -> Optional[int]:
    """
    Calculate how many months remain on a mortgage.
    
    Uses the formula: n = -log(1 - (r * P) / M) / log(1 + r)
    where:
        n = number of months
        r = monthly interest rate
        P = principal (outstanding balance)
        M = monthly payment
    
    Returns None if payment doesn't cover interest.
    """
    import math
    
    if annual_rate <= 0:
        # No interest - simple division
        return int(math.ceil(principal / monthly_payment))
    
    monthly_rate = annual_rate / 12
    
    # Check if payment can cover interest
    if monthly_payment <= principal * monthly_rate:
        return None  # Payment doesn't cover interest
    
    n = -math.log(1 - (monthly_rate * principal) / monthly_payment) / math.log(1 + monthly_rate)
    return int(math.ceil(n))


def simulate_investment_mortgage(
    starting_investment: float,
    investment_annual_rate: float,
    mortgage_outstanding: float,
    mortgage_annual_rate: float,
    monthly_payment: float,
    capital_gains_rate: float = 0.15,
) -> tuple[bool, list[dict]]:
    """
    Simulate month-by-month investment growth vs mortgage payments.
    
    Includes capital gains tax on withdrawals. We track cost basis and gains
    separately. When withdrawing, we assume proportional sale (average cost basis).
    
    Args:
        starting_investment: Initial investment balance (this is 100% cost basis)
        investment_annual_rate: Annual rate of return on investments
        mortgage_outstanding: Starting mortgage balance
        mortgage_annual_rate: Annual mortgage interest rate
        monthly_payment: Monthly mortgage payment
        capital_gains_rate: Tax rate on capital gains (default 15%)
    
    Returns:
        (success, history) where success is True if investment covers all payments,
        and history is a list of monthly snapshots.
    """
    investment_monthly_rate = investment_annual_rate / 12
    mortgage_monthly_rate = mortgage_annual_rate / 12
    
    investment_balance = starting_investment
    cost_basis = starting_investment  # Track cost basis for tax calculations
    mortgage_balance = mortgage_outstanding
    
    history = []
    month = 0
    total_taxes_paid = 0.0
    
    while mortgage_balance > 0.01:  # Small tolerance for floating point
        month += 1
        
        # Investment grows (gains increase, cost basis stays same until withdrawal)
        investment_growth = investment_balance * investment_monthly_rate
        investment_balance += investment_growth
        
        # Mortgage accrues interest
        mortgage_interest = mortgage_balance * mortgage_monthly_rate
        
        # Determine actual payment (may be less than scheduled if mortgage is almost paid)
        actual_payment = min(monthly_payment, mortgage_balance + mortgage_interest)
        
        # Apply payment to mortgage
        mortgage_balance = mortgage_balance + mortgage_interest - actual_payment
        mortgage_balance = max(0, mortgage_balance)  # Can't go negative
        
        # Calculate capital gains tax on withdrawal
        # Gain ratio = (balance - cost_basis) / balance
        if investment_balance > 0:
            gain_ratio = max(0, (investment_balance - cost_basis) / investment_balance)
        else:
            gain_ratio = 0
        
        # When we withdraw, we're selling a proportional mix of basis and gains
        gains_in_withdrawal = actual_payment * gain_ratio
        capital_gains_tax = gains_in_withdrawal * capital_gains_rate
        total_taxes_paid += capital_gains_tax
        
        # Total withdrawal needed = payment + tax on the withdrawal
        # But wait - we also need to pay tax on the gains from the tax portion!
        # This creates a recursive relationship. Let's solve it:
        # Let W = total withdrawal needed
        # W = payment + (W * gain_ratio * tax_rate)
        # W = payment / (1 - gain_ratio * tax_rate)
        tax_adjusted_multiplier = 1 / (1 - gain_ratio * capital_gains_rate)
        total_withdrawal = actual_payment * tax_adjusted_multiplier
        actual_tax = (total_withdrawal - actual_payment)
        total_taxes_paid = total_taxes_paid - capital_gains_tax + actual_tax  # Correct the tax
        
        # Update cost basis proportionally to withdrawal
        if investment_balance > 0:
            basis_ratio = cost_basis / investment_balance
            cost_basis -= total_withdrawal * basis_ratio
            cost_basis = max(0, cost_basis)
        
        # Withdraw from investment to make payment + tax
        investment_balance -= total_withdrawal
        
        history.append({
            'month': month,
            'investment_balance': investment_balance,
            'mortgage_balance': mortgage_balance,
            'payment': actual_payment,
            'investment_growth': investment_growth,
            'mortgage_interest': mortgage_interest,
            'capital_gains_tax': actual_tax,
            'total_withdrawal': total_withdrawal,
            'cost_basis': cost_basis,
            'gain_ratio': gain_ratio,
        })
        
        # Check if investment ran out
        if investment_balance < -0.01:
            return False, history
        
        # Safety check for infinite loops
        if month > 600:  # 50 years
            return False, history
    
    return True, history


def find_minimum_investment(
    investment_annual_rate: float,
    mortgage_outstanding: float,
    mortgage_annual_rate: float,
    monthly_payment: float,
    capital_gains_rate: float = 0.15,
    tolerance: float = 0.01,
) -> tuple[float, list[dict]]:
    """
    Binary search to find the minimum investment needed to cover all mortgage payments.
    
    Returns:
        (minimum_investment, history) for the solution
    """
    # Lower bound: theoretically could be 0 if investment rate is infinitely high
    # Upper bound: the full mortgage outstanding (if rates were equal)
    low = 0.0
    high = mortgage_outstanding
    
    # First check if upper bound works
    success, history = simulate_investment_mortgage(
        high, investment_annual_rate, mortgage_outstanding, mortgage_annual_rate, 
        monthly_payment, capital_gains_rate
    )
    if not success:
        # Even the full mortgage amount isn't enough - investment rate too low
        # Increase upper bound
        high = mortgage_outstanding * 2
        success, history = simulate_investment_mortgage(
            high, investment_annual_rate, mortgage_outstanding, mortgage_annual_rate, 
            monthly_payment, capital_gains_rate
        )
        if not success:
            raise ValueError(
                "Investment rate is too low to cover mortgage payments. "
                "Consider a higher investment return or lower mortgage payment."
            )
    
    best_history = history
    
    # Binary search
    while high - low > tolerance:
        mid = (low + high) / 2
        success, history = simulate_investment_mortgage(
            mid, investment_annual_rate, mortgage_outstanding, mortgage_annual_rate, 
            monthly_payment, capital_gains_rate
        )
        if success:
            high = mid
            best_history = history
        else:
            low = mid
    
    return high, best_history


def plot_results(history: list[dict], starting_investment: float):
    """Plot the investment and mortgage balances over time."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    
    months = [h['month'] for h in history]
    investment = [h['investment_balance'] for h in history]
    mortgage = [h['mortgage_balance'] for h in history]
    
    # Add starting point
    months = [0] + months
    investment = [starting_investment] + investment
    mortgage = [history[0]['mortgage_balance'] + history[0]['payment'] - history[0]['mortgage_interest']] + mortgage
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(months, investment, label='Investment Balance', color='green', linewidth=2)
    ax.plot(months, mortgage, label='Mortgage Balance', color='red', linewidth=2)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Balance ($)')
    ax.set_title('Investment vs Mortgage Balance Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as currency
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.show()


# Default parameter values
DEFAULT_PARAMS = {
    "investment_rate": 0.1472,
    "mortgage_outstanding": 143098.56,
    "mortgage_rate": 0.0225,
    "monthly_payment": 1294.45,
    "capital_gains_rate": 0.15,
}

LABEL_OVERRIDES = {
    "investment_rate": "Investment Annual Rate of Return",
    "mortgage_outstanding": "Mortgage Outstanding Balance",
    "mortgage_rate": "Mortgage Annual Interest Rate",
    "monthly_payment": "Mortgage Monthly Payment",
    "capital_gains_rate": "Capital Gains Tax Rate",
}

ENTRY_HELP = {
    "investment_rate": "Expected annual return on investments (e.g., 14.72% for S&P 500 historical average with dividends reinvested).",
    "mortgage_outstanding": "Current outstanding balance on the mortgage.",
    "mortgage_rate": "Annual interest rate on the mortgage.",
    "monthly_payment": "Monthly mortgage payment amount (principal + interest).",
    "capital_gains_rate": "Long-term capital gains tax rate (15% for most taxpayers, 20% for high earners, 0% for low earners).",
}


def run_calculation():
    """Run the calculation using the current GUI inputs."""
    try:
        investment_rate = parse_percent(entries["investment_rate"].get())
        mortgage_outstanding = parse_dollars(entries["mortgage_outstanding"].get())
        mortgage_rate = parse_percent(entries["mortgage_rate"].get())
        monthly_payment = parse_dollars(entries["monthly_payment"].get())
        capital_gains_rate = parse_percent(entries["capital_gains_rate"].get())
        
        if investment_rate <= 0:
            raise ValueError("Investment rate must be positive")
        if mortgage_outstanding <= 0:
            raise ValueError("Mortgage outstanding must be positive")
        if mortgage_rate < 0:
            raise ValueError("Mortgage rate cannot be negative")
        if monthly_payment <= 0:
            raise ValueError("Monthly payment must be positive")
        if capital_gains_rate < 0 or capital_gains_rate > 1:
            raise ValueError("Capital gains rate must be between 0% and 100%")
            
    except ValueError as exc:
        messagebox.showerror("Input error", str(exc))
        return
    
    results_var.set("Calculating...")
    root.update_idletasks()
    
    try:
        # Calculate months remaining on mortgage
        months_remaining = calculate_months_remaining(mortgage_outstanding, mortgage_rate, monthly_payment)
        
        if months_remaining is None:
            raise ValueError("Monthly payment does not cover interest. Increase payment.")
        
        # Find minimum investment needed
        min_investment, history = find_minimum_investment(
            investment_rate, mortgage_outstanding, mortgage_rate, monthly_payment, capital_gains_rate
        )
        
        # Calculate savings
        savings = mortgage_outstanding - min_investment
        savings_percent = (savings / mortgage_outstanding) * 100
        
        # Calculate totals
        total_payments = sum(h['payment'] for h in history)
        total_interest_paid = sum(h['mortgage_interest'] for h in history)
        total_investment_growth = sum(h['investment_growth'] for h in history)
        total_taxes_paid = sum(h['capital_gains_tax'] for h in history)
        total_withdrawals = sum(h['total_withdrawal'] for h in history)
        
        # Final investment balance
        final_investment = history[-1]['investment_balance'] if history else 0
        
        results = [
            f"Months to pay off mortgage: {months_remaining}",
            f"Years to pay off: {months_remaining / 12:.1f}",
            "",
            f"Minimum investment needed: ${min_investment:,.2f}",
            f"Mortgage outstanding: ${mortgage_outstanding:,.2f}",
            f"Savings vs paying off now: ${savings:,.2f} ({savings_percent:.1f}%)",
            "",
            f"Total mortgage payments: ${total_payments:,.2f}",
            f"Total mortgage interest paid: ${total_interest_paid:,.2f}",
            f"Total capital gains taxes paid: ${total_taxes_paid:,.2f}",
            f"Total withdrawn from investments: ${total_withdrawals:,.2f}",
            "",
            f"Total investment growth: ${total_investment_growth:,.2f}",
            f"Final investment balance: ${final_investment:,.2f}",
        ]
        results_var.set("\n".join(results))
        
        # Plot the results
        plot_results(history, min_investment)
        
    except ValueError as exc:
        messagebox.showerror("Calculation error", str(exc))
        results_var.set("")


def explain_calculations():
    """Show a detailed explanation of the calculation methodology."""
    explanation = """
Calculation Methodology:

This calculator determines the minimum investment balance needed
to cover all mortgage payments until the mortgage is paid off,
including capital gains taxes on investment withdrawals.

The key insight is: if your investment grows faster than your
mortgage costs (even after taxes), you need less cash invested
than the outstanding mortgage balance to fully pay it off.

Monthly Process:
1. Investment balance grows by (balance × monthly_rate)
2. Mortgage accrues interest
3. Calculate withdrawal needed for mortgage payment + taxes
4. Capital gains tax applies to the gains portion of withdrawal
5. We use average cost basis method for tax calculations

Capital Gains Tax Model:
- Your initial investment is 100% cost basis (no gains yet)
- Each month, gains accumulate from investment growth
- When withdrawing, we sell proportionally (basis + gains)
- Tax = (withdrawal × gain_ratio) × tax_rate
- We solve for total withdrawal needed to net the payment

Example: If 40% of your investment is gains, and you need $1000:
- You must sell more than $1000 to cover the $1000 + taxes
- Formula: withdrawal = payment / (1 - gain_ratio × tax_rate)

Note: This assumes constant rates (no volatility) and uses
long-term capital gains rates. Short-term gains (< 1 year)
are taxed at ordinary income rates which are higher.
    """
    messagebox.showinfo("Calculation Details", explanation.strip())


def load_defaults():
    """Reset all entries to default values."""
    for key, default in DEFAULT_PARAMS.items():
        ent = entries[key]
        ent.delete(0, tk.END)
        if key in {"investment_rate", "mortgage_rate", "capital_gains_rate"}:
            ent.insert(0, f"{default * 100:.2f}%")
        elif key in {"mortgage_outstanding", "monthly_payment"}:
            ent.insert(0, f"${default:,.2f}")
        else:
            ent.insert(0, str(default))


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Mortgage vs Investment Calculator")
    root.geometry("500x520")
    
    entries = {}
    
    label_width = max(
        len(LABEL_OVERRIDES.get(k, k.replace("_", " ").title()))
        for k in DEFAULT_PARAMS
    )
    
    # Parameters frame
    params_frame = ttk.LabelFrame(root, text="Parameters")
    params_frame.pack(fill="x", padx=10, pady=5)
    
    for key, default in DEFAULT_PARAMS.items():
        row = ttk.Frame(params_frame)
        row.pack(fill="x", pady=2)
        ttk.Label(
            row,
            text=LABEL_OVERRIDES.get(key, key.replace("_", " ").title()),
            width=label_width,
            anchor="w",
        ).pack(side="left")
        ent = ttk.Entry(row)
        if key in {"investment_rate", "mortgage_rate", "capital_gains_rate"}:
            ent.insert(0, f"{default * 100:.2f}%")
        elif key in {"mortgage_outstanding", "monthly_payment"}:
            ent.insert(0, f"${default:,.2f}")
        else:
            ent.insert(0, str(default))
        ent.pack(side="left", fill="x", expand=True)
        entries[key] = ent
        ToolTip(ent, ENTRY_HELP.get(key, ""))
    
    # Buttons frame
    button_frame = ttk.Frame(root)
    button_frame.pack(fill="x", padx=10, pady=5)
    ttk.Button(button_frame, text="Calculate", command=run_calculation).pack()
    ttk.Button(button_frame, text="Explain Calculations", command=explain_calculations).pack()
    ttk.Button(button_frame, text="Load Defaults", command=load_defaults).pack()
    
    # Results frame
    results_frame = ttk.LabelFrame(root, text="Results")
    results_frame.pack(fill="both", expand=True, padx=10, pady=5)
    results_var = tk.StringVar()
    ttk.Label(results_frame, textvariable=results_var, wraplength=450, justify="left").pack(anchor="w", padx=5, pady=5)
    
    root.mainloop()
