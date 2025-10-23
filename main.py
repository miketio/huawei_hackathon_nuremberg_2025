from Classes.dataLoader import DataLoader
from Classes.dataSaver import DataSaver
from Classes.testFunctions import test_operation_validation, test_battery_optimizer, test_investment_and_configuration
from Classes.realization import FINANCIALS, C_RATES, DAILY_CYCLES, battery_configs, InvestmentAnalyzer, ConfigurationOptimizer
from Classes.coreFunctions import run_configuration_optimization, run_investment_analysis, run_operation_optimization, mock_run_configuration_optimization
import os
import argparse


# --- Main entry point ---
def main(test_mode=False):
    print("🚀 TechArena 2025 - Phase I: Starting main execution...")
    if test_mode:   
        test_battery_optimizer()
        test_investment_and_configuration()
    input_dir, output_dir = "input", "output"
    os.makedirs(output_dir, exist_ok=True)

    loader = DataLoader(input_dir=input_dir)
    analyzer = InvestmentAnalyzer(FINANCIALS, capex_kEUR_per_MWh=200.0, years=10)
    config_optimizer = ConfigurationOptimizer(loader, analyzer)
    saver = DataSaver(output_dir)

    country_list = ["de", "at", "ch", "hu", "cz"]
    base_config_key = "luna_battery_init"

    config_records = run_configuration_optimization(config_optimizer, country_list, C_RATES, DAILY_CYCLES, battery_configs)
    saver.save_configurations(config_records)

    investment_results = run_investment_analysis(config_records, country_list, analyzer, battery_configs[base_config_key])
    saver.save_investments(investment_results)

    operation_df, country_best, conf = run_operation_optimization(config_records, country_list, base_config_key, battery_configs, loader)
    saver.save_operations(operation_df, country_best, conf)
    
    if test_mode:
        test_operation_validation(start="2024-06-01", end="2024-06-09")

    print("🎉 All outputs generated successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    args = parser.parse_args()

    main(test_mode=args.test)
    