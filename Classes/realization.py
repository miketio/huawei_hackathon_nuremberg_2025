import copy
import pandas as pd
from typing import Dict, List
from Classes.batteryOptimizer import BatteryOptimizer


# --- Financials ---
FINANCIALS = {
    'de': {'wacc': 0.083, 'inflation': 0.02},
    'at': {'wacc': 0.083, 'inflation': 0.033},
    'ch': {'wacc': 0.083, 'inflation': 0.001},
    'hu': {'wacc': 0.12, 'inflation': 0.029},
    'cz': {'wacc': 0.15, 'inflation': 0.046}
}
# --- Configuration Sweep ---
C_RATES = [0.25, 0.33, 0.50]
DAILY_CYCLES = [1.0, 1.5, 2.0]

battery_configs = {
    "luna_battery_init": {
        "E_nom_MWh": 4.5,           # 4.5 MWh
        "C_rate": 0.5,                # 0.5 C
        "P_max_MW": 0.5 * 4.5,       # = 2250 kW
        "eta_charge": 0.9,
        "eta_discharge": 0.9,
        "soc_min": 0.1,
        "soc_max": 0.9,
        "initial_soc_frac": 0.5,
        "max_daily_cycles": 2.0,       # From spec
    },
}

class InvestmentAnalyzer:
    """
    Evaluates investment metrics over a multi-year horizon (default 10 years).
    Notes on units:
      - capex_kEUR_per_MWh: CAPEX expressed in kEUR per MWh (e.g., 200 means 200 kEUR/MWh).
      - revenue_per_year: total nominal annual revenue for the entire battery (EUR/year).
      - E_nom_MWh: battery energy capacity in MWh.
    """
    def __init__(self, financials: Dict[str, Dict], capex_kEUR_per_MWh: float = 200.0, years: int = 10):
        self.financials = financials
        # explicit name to make units obvious: kEUR per MWh
        self.capex_kEUR_per_MWh = capex_kEUR_per_MWh
        self.years = years
        # project years (for labeling), e.g. 2024..(2024+years-1)
        self.project_years = list(range(2024, 2024 + years))

    def analyze_investment(self, revenue_per_year: float, battery_config: Dict, country: str) -> Dict:
        E_mwh = float(battery_config["E_nom_MWh"])
        capex_per_mwh_kEUR = self.capex_kEUR_per_MWh

        fin = self.financials[country]
        wacc = fin["wacc"]
        inflation = fin["inflation"]

        # Compute nominal revenues for each year (EUR).
        # Assumption: the first year's nominal revenue equals revenue_per_year (t=0).
        revenues_eur = [revenue_per_year * ((1 + inflation) ** t) for t in range(self.years)]

        # Discount nominal revenues to present value using WACC.
        # We assume each year's payment arrives at the end of the year, so discount by (t+1).
        pv_revenues_eur = sum(r / ((1 + wacc) ** (t + 1)) for t, r in enumerate(revenues_eur))

        # Convert PV to kEUR and then to a per-MWh basis
        pv_revenues_kEUR = pv_revenues_eur / 1000.0
        pv_profit_per_mwh_kEUR = pv_revenues_kEUR / E_mwh

        # Yearly profit per MWh in kEUR/MWh (nominal, first year)
        yearly_profit_per_mwh_kEUR = (revenue_per_year / 1000.0) / E_mwh

        # Levelized ROI (%) = discounted PV profit per MWh divided by CAPEX per MWh
        roi_levelized = (pv_profit_per_mwh_kEUR / capex_per_mwh_kEUR) * 100.0

        return {
            "Country": country.upper(),
            "WACC": wacc,
            "Inflation Rate": inflation,
            "Discount Rate": wacc,
            "Initial Investment[kEUR/MWh]": round(capex_per_mwh_kEUR, 2),
            "Yearly profits[kEUR/MWh]": round(yearly_profit_per_mwh_kEUR, 2),
            "Levelized ROI[%]": round(roi_levelized, 2)
        }

    def generate_yearly_df(self, revenue_per_year: float, battery_config: Dict, country: str) -> pd.DataFrame:
        """
        Generate a year-by-year table (one row per project year) and append a final
        row with Levelized ROI. All displayed profit values are in kEUR/MWh.
        """
        E_mwh = float(battery_config["E_nom_MWh"])

        fin = self.financials[country]
        wacc = fin["wacc"]
        inflation = fin["inflation"]

        rows = []
        total_pv_kEUR_per_MWh = 0.0

        for t, year in enumerate(self.project_years):
            # Nominal yearly profit per MWh in kEUR/MWh (apply inflation to nominal revenue)
            yearly_nominal_kEUR_per_MWh = (revenue_per_year * ((1 + inflation) ** t)) / 1000.0 / E_mwh
            rows.append({
                "Year": year,
                # show CAPEX per MWh only in the first row (kEUR/MWh)
                "Initial Investment[kEUR/MWh]": self.capex_kEUR_per_MWh if year == self.project_years[0] else 0.0,
                "Yearly profits[kEUR/MWh]": round(yearly_nominal_kEUR_per_MWh, 2)
            })

            # Discount the yearly profit (kEUR/MWh) to present value and accumulate
            pv = yearly_nominal_kEUR_per_MWh / ((1 + wacc) ** (t + 1))
            total_pv_kEUR_per_MWh += pv

        # Compute levelized ROI as PV (kEUR/MWh) divided by CAPEX (kEUR/MWh)
        levelized_roi_pct = total_pv_kEUR_per_MWh / self.capex_kEUR_per_MWh * 100.0

        df = pd.DataFrame(rows)
        df.loc[len(df)] = ["Levelized ROI", "", round(levelized_roi_pct, 2)]
        return df


class ConfigurationOptimizer:
    """
    Runs optimization across multiple C-rates and daily cycles.
    """
    def __init__(self, data_loader, investment_analyzer):
        self.data_loader = data_loader
        self.investment_analyzer = investment_analyzer
        # kept for potential historical tracking, but run() returns a local results list
        self.results = []

    def run(self, country: str, c_rates: List[float], daily_cycles_list: List[int], battery_configs: Dict) -> pd.DataFrame:
        battery_key = "luna_battery_init"
        if battery_key not in battery_configs:
            raise KeyError(f"Battery config '{battery_key}' not found. Available: {list(battery_configs.keys())}")

        # load all required data once per run
        data = self.data_loader.load_all_data()
        results = []  # local container to avoid accumulating results across multiple run() calls

        for c_rate in c_rates:
            for cycles in daily_cycles_list:
                # deep copy to avoid mutating the original battery_configs dict
                config = copy.deepcopy(battery_configs[battery_key])
                config["C_rate"] = c_rate
                config["max_daily_cycles"] = cycles
                # compute maximum power (MW) from energy (MWh) and C-rate
                config["P_max_MW"] = config["E_nom_MWh"] * config["C_rate"]

                optimizer = BatteryOptimizer(country, config, data)
                result = optimizer.optimize()

                metrics = self.investment_analyzer.analyze_investment(
                    result.revenue, config, country
                )
                metrics["C-rate"] = c_rate
                metrics["number of cycles"] = cycles

                results.append(metrics)
                # optionally keep a global history
                self.results.append(metrics)

        # return a DataFrame containing one row per configuration evaluated in this run
        return pd.DataFrame(results)
