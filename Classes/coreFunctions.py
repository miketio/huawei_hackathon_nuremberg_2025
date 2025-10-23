import pandas as pd
import numpy as np
from Classes.batteryOptimizer import BatteryOptimizer

# --- Core functions ---
def run_configuration_optimization(config_optimizer, country_list, c_rates, daily_cycles, battery_configs):
    config_records = {}
    for country in country_list:
        try:
            df_country = config_optimizer.run(
                country=country,
                c_rates=c_rates,
                daily_cycles_list=daily_cycles,
                battery_configs=battery_configs,
            )
            df_country["Country"] = country  # ✅ keep country info for global best selection
            config_records[country] = df_country[[
                "C-rate", "number of cycles", "Yearly profits[kEUR/MWh]", "Levelized ROI[%]", "Country"
            ]]
            print(f"✅ Configuration for {country} completed: {len(df_country)} rows")
        except Exception as e:
            print(f"❌ Failed for {country}: {e}")
    return config_records



def mock_run_configuration_optimization(config_optimizer=None, country_list=None, c_rates=None, daily_cycles=None, battery_configs=None):
    """
    Mock version of run_configuration_optimization for testing.
    Generates synthetic data instead of calling config_optimizer.run().
    """
    if country_list is None:
        country_list = ["de", "at", "ch"]

    config_records = {}
    rng = np.random.default_rng(42)  # reproducible randomness

    for country in country_list:
        try:
            # Fake 5 configurations per country
            df_country = pd.DataFrame({
                "C-rate": rng.choice(c_rates or [0.5, 1, 2], size=5),
                "number of cycles": rng.choice(daily_cycles or [100, 200, 300], size=5),
                "Yearly profits[kEUR/MWh]": rng.normal(loc=50, scale=10, size=5).round(2),
                "Levelized ROI[%]": rng.uniform(5, 20, size=5).round(2),
            })
            df_country["Country"] = country

            config_records[country] = df_country[[
                "C-rate", "number of cycles", "Yearly profits[kEUR/MWh]", "Levelized ROI[%]", "Country"
            ]]

            print(f"✅ (MOCK) Configuration for {country} generated: {len(df_country)} rows")

        except Exception as e:
            print(f"❌ (MOCK) Failed for {country}: {e}")

    return config_records


def run_investment_analysis(config_records, country_list, analyzer, base_config):
    """
    Run investment analysis for each country and return results
    as a dictionary of DataFrames (header + yearly data combined).
    """
    print("📈 Running Investment Analysis (year-by-year)...")
    investment_results = {}

    for country in country_list:
        if country not in config_records or config_records[country].empty:
            continue

        # Pick best config for this country (highest ROI)
        country_data = config_records[country]
        best_row = country_data.loc[country_data["Levelized ROI[%]"].idxmax()]
        revenue_per_year = best_row["Yearly profits[kEUR/MWh]"] * 1000 * base_config["E_nom_MWh"]

        # Generate yearly breakdown + metrics
        yearly_df = analyzer.generate_yearly_df(revenue_per_year, base_config, country)
        metrics = analyzer.analyze_investment(revenue_per_year, base_config, country)

        # Build header rows
        header_df = pd.DataFrame({
            "Parameter": ["WACC", "Inflation Rate", "Discount rate", "Yearly profits (2024)"],
            "Value": [
                metrics["WACC"],
                metrics["Inflation Rate"],
                metrics["Discount Rate"],
                metrics["Yearly profits[kEUR/MWh]"],
            ],
        })

        # Store both parts in dict for saving
        investment_results[country] = {
            "header": header_df,
            "yearly": yearly_df,
        }

        print(f"✅ Investment analysis completed for {country}")

    return investment_results


def run_operation_optimization(config_records, country_list, base_config_key, battery_configs, loader):
    print("⚡ Running Operation Optimization for the best configuration globally...")
    all_configs_df = pd.concat(config_records.values(), ignore_index=True)
    best_overall = all_configs_df.loc[all_configs_df["Levelized ROI[%]"].idxmax()]

    country_best = best_overall["Country"].lower()
    if country_best not in country_list:
        print(f"❌ Best country '{country_best}' not in the original list. Defaulting to 'de'.")
        country_best = "de"

    conf = battery_configs[base_config_key].copy()
    conf["C_rate"] = best_overall["C-rate"]
    conf["max_daily_cycles"] = best_overall["number of cycles"]
    conf["P_max_MW"] = conf["E_nom_MWh"] * conf["C_rate"]

    optimizer = BatteryOptimizer(country=country_best, battery_config=conf, data=loader.load_all_data())
    try:
        result = optimizer.optimize()
        operation_df = result.operation_df
    except Exception as e:
        print(f"❌ Operation optimization failed for {country_best.upper()}: {e}")
        operation_df = pd.DataFrame(columns=[
            "Timestamp", "Stored energy[MWh]", "SoC[-]", "Charge[MWh]", "Discharge [MWh]",
            "Day-ahead buy[MWh]", "Day-ahead sell[MWh]",
            "FCR Capacity[MW]", "aFRR Capacity POS[MW]", "aFRR Capacity NEG[MW]"
        ])

    return operation_df, country_best, conf