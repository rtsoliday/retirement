from __future__ import annotations

import csv
from typing import Optional

from kivy.app import App
from kivy.factory import Factory
from kivy.properties import StringProperty
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.metrics import dp

import numpy as np

from core import (
    SimulationConfig,
    parse_percent,
    parse_dollars,
    social_security_payout,
    gross_from_net,
    gross_from_net_with_ss,
    simulate,
    save_config,
    load_config,
)


# Default parameter values mirror those in the Tkinter app
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
    "stock_reduction_alpha": 1.0,
    "min_percent_in_stock": 0.2,
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
    "min_percent_in_stock",
}

DOLLAR_FIELDS = {
    "average_yearly_need",
    "current_roth",
    "current_401a_and_403b",
    "full_social_security_at_67",
    "mortgage_payment",
}


class RetirementApp(App):
    """Kivy front-end for the retirement simulator."""

    result_text = StringProperty("")
    precomp_text = StringProperty("")

    def build(self):
        return Factory.RetirementRoot()

    def on_start(self):
        """Populate fields with defaults and any saved configuration."""
        self.load_defaults()
        self._apply_saved_config()

    def load_defaults(self) -> None:
        """Reset all input fields to built-in defaults."""
        ids = self.root.ids
        for key, default in DEFAULT_GENERAL.items():
            if key in PERCENT_FIELDS:
                ids[key].text = f"{default * 100:.2f}%"
            else:
                ids[key].text = str(default)
        for key, default in DEFAULT_USER.items():
            if key == "gender":
                ids[key].text = default
            elif key in PERCENT_FIELDS:
                ids[key].text = f"{default * 100:.2f}%"
            elif key in DOLLAR_FIELDS:
                ids[key].text = f"${default:,.0f}"
            else:
                ids[key].text = str(default)

    def _apply_saved_config(self) -> None:
        """Load configuration from disk and update the UI."""
        cfg = load_config()
        ids = self.root.ids
        gen_cfg = cfg.get("general", {})
        user_cfg = cfg.get("user", {})
        for key, val in gen_cfg.items():
            if key not in ids:
                continue
            if key in PERCENT_FIELDS:
                ids[key].text = f"{val * 100:.2f}%"
            else:
                ids[key].text = str(val)
        for key, val in user_cfg.items():
            if key not in ids:
                continue
            if key == "gender":
                ids[key].text = val
            elif key in PERCENT_FIELDS:
                ids[key].text = f"{val * 100:.2f}%"
            elif key in DOLLAR_FIELDS:
                ids[key].text = f"${val:,.0f}"
            else:
                ids[key].text = str(val)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _error(self, msg: str) -> None:
        Popup(title="Error", content=Label(text=msg), size_hint=(0.75, 0.5)).open()

    def _load_inputs(self, percent_override: Optional[float] = None) -> SimulationConfig:
        """Gather UI inputs and build a SimulationConfig."""

        ids = self.root.ids
        number_of_simulations = int(ids.number_of_simulations.text)
        pre_retirement_mean_return = parse_percent(ids.pre_retirement_mean_return.text)
        pre_retirement_std_dev = parse_percent(ids.pre_retirement_std_dev.text)
        stock_mean_return = parse_percent(ids.stock_mean_return.text)
        stock_std_dev = parse_percent(ids.stock_std_dev.text)
        bond_mean_return = parse_percent(ids.bond_mean_return.text)
        bond_std_dev = parse_percent(ids.bond_std_dev.text)
        inflation_mean = parse_percent(ids.inflation_mean.text)
        inflation_std_dev = parse_percent(ids.inflation_std_dev.text)

        gender = ids.gender.text.strip().lower()
        current_age = int(ids.current_age.text)
        retirement_age = int(ids.retirement_age.text)
        if retirement_age <= current_age:
            raise ValueError("Retirement age must be greater than current age")
        average_yearly_need = parse_dollars(ids.average_yearly_need.text)
        current_roth = parse_dollars(ids.current_roth.text)
        current_401a_and_403b = parse_dollars(ids.current_401a_and_403b.text)
        full_social_security_at_67 = parse_dollars(ids.full_social_security_at_67.text)
        social_security_age_started = int(ids.social_security_age_started.text)
        social_security_yearly_amount = social_security_payout(
            full_social_security_at_67, social_security_age_started
        )
        mortgage_payment = parse_dollars(ids.mortgage_payment.text)
        mortgage_years_left = int(ids.mortgage_years_left.text)
        mortgage_yearly_payment = mortgage_payment * 12

        stock_reduction_alpha = float(ids.stock_reduction_alpha.text)
        min_percent_in_stock = parse_percent(ids.min_percent_in_stock.text)

        years_of_retirement = 119 - retirement_age
        years_to_retirement = retirement_age - current_age
        base_retirement_need = average_yearly_need * (1 + inflation_mean) ** years_to_retirement
        mortgage_years_in_retirement = max(0, mortgage_years_left - years_to_retirement)
        retirement_yearly_need = base_retirement_need + (
            mortgage_yearly_payment if mortgage_years_in_retirement > 0 else 0
        )

        self.precomp_text = f"Year 1 net need: ${retirement_yearly_need:,.0f}"

        file = (
            "DeathProbsE_M_Alt2_TR2025.csv"
            if gender.startswith("m")
            else "DeathProbsE_F_Alt2_TR2025.csv"
        )
        death_probs = None
        with open(file) as f:
            reader = csv.reader(f)
            next(reader)  # header
            for row in reader:
                if row and row[0] == "2025":
                    death_probs = np.array([float(x) for x in row[1:]], dtype=float)
                    break
        if death_probs is None:
            raise ValueError("Death probabilities not found")

        if percent_override is None:
            percent_in_stock_after_retirement = parse_percent(
                ids.percent_in_stock_after_retirement.text
            )
        else:
            if not 0 <= percent_override <= 1:
                raise ValueError("Percent in stock must be between 0 and 100")
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
            stock_reduction_alpha=stock_reduction_alpha,
            min_percent_in_stock=min_percent_in_stock,
            years_of_retirement=years_of_retirement,
            base_retirement_need=base_retirement_need,
            retirement_yearly_need=retirement_yearly_need,
            mortgage_years_in_retirement=mortgage_years_in_retirement,
            mortgage_yearly_payment=mortgage_yearly_payment,
            death_probs=death_probs,
        )
        return cfg

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------
    def run_simulation(self) -> None:
        try:
            cfg = self._load_inputs()
        except Exception as exc:  # pragma: no cover - UI only
            self._error(str(exc))
            return

        rate, success_paths, failure_paths, pct_paths = simulate(cfg, collect_paths=True)
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

        self.result_text = "\n".join(results)
        self._last_success_paths = success_paths
        self._last_failure_paths = failure_paths
        self._last_pct_paths = pct_paths
        btn = self.root.ids.get("plot_button")
        if btn is not None:
            btn.opacity = 1
            btn.disabled = False
            btn.height = dp(40)
        save_config(cfg)

    def show_plot(self) -> None:
        """Display a plot of the latest simulation paths."""
        if not (
            getattr(self, "_last_success_paths", None)
            or getattr(self, "_last_failure_paths", None)
        ):
            self._error("Run simulations first")
            return
        from monticarlo import plot_paths, plot_percent_in_stock

        plot_paths(self._last_success_paths, self._last_failure_paths)
        checkbox = self.root.ids.get("pct_checkbox")
        if checkbox and checkbox.active:
            plot_percent_in_stock(self._last_pct_paths)

    def optimize_percent(self) -> None:
        try:
            cfg = self._load_inputs(1.0)
        except Exception as exc:  # pragma: no cover - UI only
            self._error(str(exc))
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

        self.root.ids.percent_in_stock_after_retirement.text = f"{best_percent*100:.2f}%"

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
        self.result_text = "\n".join(results)


if __name__ == "__main__":  # pragma: no cover - manual run only
    RetirementApp().run()
