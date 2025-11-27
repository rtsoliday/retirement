from __future__ import annotations

import csv

from kivy.app import App
from kivy.factory import Factory
from kivy.properties import StringProperty
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
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

DEFAULT_HEALTHCARE_INFLATION_MEAN = 0.055
DEFAULT_HEALTHCARE_INFLATION_STD = 0.02


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
    "filing_status": "single",
    "current_age": 50,
    "retirement_age": 58,
    "average_yearly_need": 75_000,
    "current_roth": 100_000,
    "current_401a_and_403b": 800_000,
    "current_savings": 50_000,
    "savings_interest_rate": 0.04,
    "full_social_security_at_67": 30_000,
    "social_security_age_started": 62,
    "health_care_payment": 650,
    "mortgage_payment": 0,
    "mortgage_years_left": 0,
    "enable_roth_conversion": False,
    "roth_conversion_rate_cap": 0.22,
    "enable_dynamic_withdrawal": False,
    "dynamic_withdrawal_trigger": -0.01,
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
    "savings_interest_rate",
}

DOLLAR_FIELDS = {
    "average_yearly_need",
    "current_roth",
    "current_401a_and_403b",
    "current_savings",
    "full_social_security_at_67",
    "health_care_payment",
    "mortgage_payment",
}

BOOL_FIELDS = {"enable_roth_conversion", "enable_dynamic_withdrawal"}


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
            if key in {"gender", "filing_status"}:
                ids[key].text = default
            elif key in BOOL_FIELDS:
                ids[key].active = bool(default)
            elif key == "roth_conversion_rate_cap":
                ids[key].text = f"{default * 100:.0f}%"
            elif key == "dynamic_withdrawal_trigger":
                ids[key].text = f"{default * 100:.0f}%"
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
            if key in {"gender", "filing_status"}:
                ids[key].text = val
            elif key in BOOL_FIELDS:
                ids[key].active = bool(val)
            elif key == "roth_conversion_rate_cap":
                if val is None:
                    continue
                ids[key].text = f"{val * 100:.0f}%"
            elif key == "dynamic_withdrawal_trigger":
                if val is None:
                    continue
                ids[key].text = f"{val * 100:.0f}%"
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

    def _load_inputs(self) -> SimulationConfig:
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
        filing_status = ids.filing_status.text.strip().lower()
        current_age = int(ids.current_age.text)
        retirement_age = int(ids.retirement_age.text)
        if retirement_age < current_age:
            raise ValueError("Retirement age must be greater than or equal to current age")
        if retirement_age >= 119:
            raise ValueError("Retirement age must be less than 119")
        average_yearly_need = parse_dollars(ids.average_yearly_need.text)
        current_roth = parse_dollars(ids.current_roth.text)
        current_401a_and_403b = parse_dollars(ids.current_401a_and_403b.text)
        current_savings = parse_dollars(ids.current_savings.text)
        savings_interest_rate = parse_percent(ids.savings_interest_rate.text)
        full_social_security_at_67 = parse_dollars(ids.full_social_security_at_67.text)
        social_security_age_started = int(ids.social_security_age_started.text)
        if social_security_age_started < 62:
            raise ValueError("Social Security cannot be claimed before age 62")
        if social_security_age_started > 70:
            raise ValueError("Social Security claiming is capped at age 70")
        social_security_yearly_amount = social_security_payout(
            full_social_security_at_67, social_security_age_started
        )
        health_care_payment = parse_dollars(ids.health_care_payment.text)
        mortgage_payment = parse_dollars(ids.mortgage_payment.text)
        mortgage_years_left = int(ids.mortgage_years_left.text)
        years_to_retirement = retirement_age - current_age
        healthcare_inflation_mean = DEFAULT_HEALTHCARE_INFLATION_MEAN
        health_care_yearly_payment = health_care_payment * 12 * (
            1 + healthcare_inflation_mean
        ) ** years_to_retirement
        mortgage_yearly_payment = mortgage_payment * 12

        years_of_retirement = 119 - retirement_age
        base_retirement_need = average_yearly_need * (1 + inflation_mean) ** years_to_retirement
        mortgage_years_in_retirement = max(0, mortgage_years_left - years_to_retirement)
        health_care_years_in_retirement = max(0, 65 - retirement_age)
        retirement_yearly_need = base_retirement_need
        if mortgage_years_in_retirement > 0:
            retirement_yearly_need += mortgage_yearly_payment
        if health_care_years_in_retirement > 0:
            retirement_yearly_need += health_care_yearly_payment

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

        percent_in_stock_after_retirement = 1.0
        bond_ratio = 0.0

        enable_roth_conversion = bool(ids.enable_roth_conversion.active)
        rate_text = ids.roth_conversion_rate_cap.text.strip()
        roth_conversion_rate_cap = parse_percent(rate_text) if rate_text else None

        enable_dynamic_withdrawal = bool(ids.enable_dynamic_withdrawal.active)
        trigger_text = ids.dynamic_withdrawal_trigger.text.strip()
        # Parse trigger as a negative percentage (e.g., "-1%" becomes -0.01)
        if trigger_text:
            trigger_val = float(trigger_text.strip().rstrip("%")) / 100
            dynamic_withdrawal_trigger = trigger_val
        else:
            dynamic_withdrawal_trigger = -0.01

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
            current_savings=current_savings,
            savings_interest_rate=savings_interest_rate,
            full_social_security_at_67=full_social_security_at_67,
            social_security_age_started=social_security_age_started,
            social_security_yearly_amount=social_security_yearly_amount,
            mortgage_payment=mortgage_payment,
            mortgage_years_left=mortgage_years_left,
            health_care_payment=health_care_payment,
            percent_in_stock_after_retirement=percent_in_stock_after_retirement,
            bond_ratio=bond_ratio,
            years_of_retirement=years_of_retirement,
            base_retirement_need=base_retirement_need,
            retirement_yearly_need=retirement_yearly_need,
            mortgage_years_in_retirement=mortgage_years_in_retirement,
            mortgage_yearly_payment=mortgage_yearly_payment,
            health_care_years_in_retirement=health_care_years_in_retirement,
            health_care_yearly_payment=health_care_yearly_payment,
            healthcare_inflation_mean=healthcare_inflation_mean,
            healthcare_inflation_std=DEFAULT_HEALTHCARE_INFLATION_STD,
            death_probs=death_probs,
            filing_status=filing_status,
            enable_roth_conversion=enable_roth_conversion,
            roth_conversion_rate_cap=roth_conversion_rate_cap,
            enable_dynamic_withdrawal=enable_dynamic_withdrawal,
            dynamic_withdrawal_trigger=dynamic_withdrawal_trigger,
        )
        return cfg

    def _build_explanation(self, cfg: SimulationConfig) -> str:
        """Return a detailed explanation of inputs and calculations."""
        years_to_retirement = cfg.retirement_age - cfg.current_age
        file = (
            "DeathProbsE_M_Alt2_TR2025.csv"
            if cfg.gender.startswith("m")
            else "DeathProbsE_F_Alt2_TR2025.csv"
        )
        explanation = [
            "Input values:",
            f"  Number of simulations: {cfg.number_of_simulations}",
            (
                "  Pre-retirement mean return: "
                f"{cfg.pre_retirement_mean_return * 100:.2f}% ("
                f"σ {cfg.pre_retirement_std_dev * 100:.2f}%)"
            ),
            (
                "  Stock return: "
                f"{cfg.stock_mean_return * 100:.2f}% ("
                f"σ {cfg.stock_std_dev * 100:.2f}%)"
            ),
            (
                "  Bond return: "
                f"{cfg.bond_mean_return * 100:.2f}% ("
                f"σ {cfg.bond_std_dev * 100:.2f}%)"
            ),
            (
                "  Inflation: "
                f"{cfg.inflation_mean * 100:.2f}% ("
                f"σ {cfg.inflation_std_dev * 100:.2f}%)"
            ),
            (
                "  Gender: "
                f"{cfg.gender}, Filing status: {cfg.filing_status}"
            ),
            (
                "  Current age: "
                f"{cfg.current_age}, Retirement age: {cfg.retirement_age}"
            ),
            f"  Average yearly need: ${cfg.average_yearly_need:,.0f}",
            f"  Roth balance: ${cfg.current_roth:,.0f}",
            f"  401a/403b balance: ${cfg.current_401a_and_403b:,.0f}",
            f"  Social Security at 67: ${cfg.full_social_security_at_67:,.0f}",
            (
                "  Social Security starting age: "
                f"{cfg.social_security_age_started} "
                f"(annual benefit ${cfg.social_security_yearly_amount:,.0f})"
            ),
            (
                "  Monthly health care payment: "
                f"${cfg.health_care_payment:,.0f}"
            ),
            (
                "  Monthly mortgage payment: "
                f"${cfg.mortgage_payment:,.0f} with {cfg.mortgage_years_left} years left"
            ),
            "",
            "Derived values:",
            f"  Years until retirement: {years_to_retirement}",
            f"  Years simulated in retirement: {cfg.years_of_retirement}",
            (
                "  Base retirement need after inflation: "
                f"${cfg.base_retirement_need:,.0f}"
            ),
            f"  Mortgage years in retirement: {cfg.mortgage_years_in_retirement}",
            f"  Health care years in retirement: {cfg.health_care_years_in_retirement}",
            f"  Mortgage yearly payment: ${cfg.mortgage_yearly_payment:,.0f}",
            (
                "  Health care yearly payment at retirement: "
                f"${cfg.health_care_yearly_payment:,.0f}"
            ),
            f"  Year 1 net retirement need: ${cfg.retirement_yearly_need:,.0f}",
            (
                "  Roth conversion strategy: "
                + (
                    f"Fill up to the {cfg.roth_conversion_rate_cap * 100:.0f}% bracket"
                    if cfg.enable_roth_conversion and cfg.roth_conversion_rate_cap is not None
                    else "No conversions"
                )
            ),
            (
                "  Dynamic withdrawal strategy: "
                + (
                    f"Enabled (trigger: {cfg.dynamic_withdrawal_trigger * 100:.0f}%)"
                    if cfg.enable_dynamic_withdrawal
                    else "Disabled"
                )
            ),
            "",
            "Process:",
            f"  Loads mortality probabilities from {file}.",
            (
                "  For each simulation, yearly returns and inflation are "
                "sampled from normal distributions using the provided means "
                "and standard deviations."
            ),
            (
                "  Balances grow, spending is withdrawn, taxes applied, and "
                "Social Security added when eligible."
            ),
            (
                "  The simulation runs until age 119 or funds deplete; the "
                "success rate is the percentage of runs where money lasts "
                "through all retirement years."
            ),
        ]
        return "\n".join(explanation)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------
    def run_simulation(self) -> None:
        try:
            cfg = self._load_inputs()
        except Exception as exc:  # pragma: no cover - UI only
            self._error(str(exc))
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
            if cfg.health_care_years_in_retirement > years_until_ss:
                need_at_ss += cfg.health_care_yearly_payment * (
                    1 + cfg.healthcare_inflation_mean
                ) ** years_until_ss
            results = [
                f"Success rate: {rate:.1f}%",
                f"Gross needed in year 1: ${gross_from_net(cfg.retirement_yearly_need, cfg):,.0f}",
                f"Gross needed in year {years_until_ss + 1} (with SS): ${gross_from_net_with_ss(need_at_ss, cfg.social_security_yearly_amount, cfg):,.0f}",
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
        btn = self.root.ids.get("plot_button")
        if btn is not None:
            btn.opacity = 1
            btn.disabled = False
            btn.height = dp(40)
        save_config(cfg)

    def explain_calculations(self) -> None:
        try:
            cfg = self._load_inputs()
        except Exception as exc:  # pragma: no cover - UI only
            self._error(str(exc))
            return
        explanation = self._build_explanation(cfg)
        content = ScrollView(size_hint=(1, 1))
        lbl = Label(
            text=explanation,
            size_hint_y=None,
            text_size=(dp(400), None),
            halign="left",
            valign="top",
        )
        lbl.bind(texture_size=lambda inst, val: setattr(inst, "height", val[1]))
        content.add_widget(lbl)
        Popup(title="Simulation Details", content=content, size_hint=(0.9, 0.9)).open()

    def show_plot(self) -> None:
        """Display a plot of the latest simulation paths."""
        if not (
            getattr(self, "_last_success_paths", None)
            or getattr(self, "_last_failure_paths", None)
        ):
            self._error("Run simulations first")
            return
        from monticarlo import plot_paths

        plot_paths(self._last_success_paths, self._last_failure_paths)


if __name__ == "__main__":  # pragma: no cover - manual run only
    RetirementApp().run()
