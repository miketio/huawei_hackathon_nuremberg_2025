import os
import pandas as pd
from typing import Dict, Any
from Classes.dataLoader import DataLoader
from Classes.batteryOptimizer import BatteryOptimizer
from Classes.realization import InvestmentAnalyzer, ConfigurationOptimizer, FINANCIALS, C_RATES, DAILY_CYCLES, battery_configs

class OperationValidationCaller:
    """
    Load operation CSV, construct an OptimizationResult wrapper and call
    BatteryOptimizer.validate_solution using an existing optimizer instance
    or by constructing a BatteryOptimizer if battery_config/country/loader are provided.
    """

    def __init__(self, op_path: str):
        self.op_path = op_path
        if not os.path.exists(self.op_path):
            raise FileNotFoundError(f"Operation file not found: {self.op_path}")

    def _load_operation_df(self) -> pd.DataFrame:
        """Load the operation data from CSV or Excel into a DataFrame with a Timestamp column."""
        ext = os.path.splitext(self.op_path)[1].lower()

        if ext in [".xlsx", ".xls"]:
            # Always pull from "Operation" sheet
            df = pd.read_excel(self.op_path, sheet_name="Operation")
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        # Ensure Timestamp column exists and is parsed
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        else:
            try:
                df.index = pd.to_datetime(df.index, errors="coerce")
                df = df.reset_index().rename(columns={"index": "Timestamp"})
            except Exception:
                raise ValueError("Could not parse Timestamp column or index.")

        return df
    
    def _load_best_config(self) -> Dict[str, Any]:
        """If Excel file, load the Best_Config sheet into a dict."""
        ext = os.path.splitext(self.op_path)[1].lower()
        if ext not in [".xlsx", ".xls"]:
            raise ValueError("Best_Config is only available in Excel files.")

        df = pd.read_excel(self.op_path, sheet_name="Best_Config")
        row = df.iloc[0].to_dict()
        return {
            "country": row.get("Best Country", "").lower(),
            "C_rate": row.get("C-rate"),
            "max_daily_cycles": row.get("Cycles"),
            "E_nom_MWh": row.get("Nominal Energy [MWh]"),
            "P_max_MW": row.get("Nominal Power [MW]"),
        }

    def run_from_best_config(self, data_loader, start: str = "2024-06-01", end: str = "2024-06-09", eps: float = 1e-3) -> Dict[str, Any]:
        """
        Create a BatteryOptimizer from Best_Config sheet and validate the Operation sheet.
        """
        best_conf = self._load_best_config()
        battery_config = {
            "C_rate": best_conf["C_rate"],
            "max_daily_cycles": best_conf["max_daily_cycles"],
            "E_nom_MWh": best_conf["E_nom_MWh"],
            "P_max_MW": best_conf["P_max_MW"],
        }

        optimizer = BatteryOptimizer(
            country=best_conf["country"],
            battery_config=battery_config,
            data=data_loader.load_all_data(),
        )
        operation_df = self._load_operation_df()

        return optimizer.validate_solution(operation_df, start=start, end=end, eps=eps)
    
def test_operation_validation(start="2024-06-01", end="2024-06-09"):
    print("🚀 Testing OperationValidationCaller...")

    # Directly from Excel file (both Operation + Best_Config)
    caller = OperationValidationCaller("output/TechArena_Phase1_Operation.xlsx")
    loader = DataLoader()

    violations = caller.run_from_best_config(loader, start=start, end=end)

    print("Validation results:")
    print(violations)

def test_battery_optimizer(
    country: str = "de",
    battery_key: str = "luna_battery_init",
    input_dir: str = "input",
    output_dir: str = "output"
):
    """
    Test function to validate BatteryOptimizer functionality.
    """
    print("🚀 Starting BatteryOptimizer test...\n")
    print(f"🔧 Parameters:")
    print(f"   Country: {country.lower()}")
    print(f"   Battery Config: {battery_key}")
    print(f"   Input Dir: {input_dir}")
    print(f"   Output Dir: {output_dir}\n")

    # -------------------------------
    # 1. Check input file
    # -------------------------------
    # input_file = Path(input_dir) / "TechArena2025_ElectricityPriceData_v2.xlsx"
    # if not input_file.exists():
    #     raise FileNotFoundError(f"Input file not found: {input_file.resolve()}\n"
    #                             "Make sure it's placed in the 'input' folder.")

    # -------------------------------
    # 2. Load data
    # -------------------------------
    print("📂 Loading data...")
    loader = DataLoader()
    data_frames = loader.load_all_data()
    print("✅ Data loaded successfully!")
    print("   Available sheets: da, fcr, afrr")
    print(f"   DA data shape: {data_frames['da'].shape}")
    print(f"   FCR data shape: {data_frames['fcr'].shape}")
    print(f"   aFRR data shape: {data_frames['afrr'].shape}\n")

    # -------------------------------
    # 3. Validate battery config
    # -------------------------------
    if battery_key not in battery_configs:
        available = list(battery_configs.keys())
        raise KeyError(f"Battery config '{battery_key}' not found.\nAvailable configs: {available}")
    print(f"🔋 Using battery config: {battery_key}")
    print(f"   Capacity: {battery_configs[battery_key]['E_nom_MWh']} MWh")
    print(f"   Power: {battery_configs[battery_key]['P_max_MW']} MW")
    print(f"   C-rate: {battery_configs[battery_key]['P_max_MW'] / battery_configs[battery_key]['E_nom_MWh']:.2f}C\n")
    print(f"   Charge Efficiency: {battery_configs[battery_key].get('eta_charge', 1.0)}")
    print(f"   Discharge Efficiency: {battery_configs[battery_key].get('eta_discharge', 1.0)}")

    # -------------------------------
    # 4. Run optimization
    # -------------------------------
    print("⚡ Creating BatteryOptimizer instance...")
    optimizer = BatteryOptimizer(country=country, battery_config=battery_configs[battery_key], data=data_frames)

    print("🧮 Running optimization (this may take a moment for 1-year data)...")
    try:
        result = optimizer.optimize()
        print("✅ Optimization completed successfully!\n")
    except Exception as e:
        raise RuntimeError(f"Optimization failed: {e}")

    # -------------------------------
    # 5. Display results
    # -------------------------------
    print("📊 Optimization Results:")
    print(f"   Energy Revenue: {result.energy_revenue:,.2f} EUR")
    print(f"   Capacity Revenue: {result.capacity_revenue:,.2f} EUR")
    print(f"   Energy Charged: {result.charge_energy:.2f} MWh")
    print(f"   Energy Discharged: {result.discharge_energy:.2f} MWh")
    print(f"   Avg Efficiency: {(result.discharge_energy / result.charge_energy * 100):.1f}%\n")

    print("📈 First 10 rows of operation DataFrame:")
    print(result.operation_df.head(10))
    print("\n🔎 Operation DataFrame columns:")
    for col in result.operation_df.columns:
        print(f"   - {col}")

    # -------------------------------
    # 7. Final validation
    # -------------------------------
    required_columns = [
        "Timestamp",
        "Stored energy[MWh]",
        "SoC[-]",
        "Charge[MWh]",
        "Discharge [MWh]",
        "Day-ahead buy[MWh]",
        "Day-ahead sell[MWh]",
        "FCR Capacity[MW]",
        "aFRR Capacity POS[MW]",
        "aFRR Capacity NEG[MW]"
    ]

    missing = [col for col in required_columns if col not in result.operation_df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
    else:
        print("✅ All required columns are present!")

    print("\n🎉 BatteryOptimizer test PASSED!")

    print("\n🔍 Validating solution constraints over 2024-06-01 to 2024-06-08...")
    violations = optimizer.validate_solution(result.operation_df, start="2024-06-01", end="2024-06-08", eps=0)

def test_investment_and_configuration():
    print("🚀 Testing InvestmentAnalyzer and ConfigurationOptimizer...\n")

    # --- Setup ---
    loader = DataLoader(input_dir="input")
    analyzer = InvestmentAnalyzer(FINANCIALS, capex_kEUR_per_MWh=200.0, years=10)
    config_optimizer = ConfigurationOptimizer(loader, analyzer)

    # --- Test 1: InvestmentAnalyzer ---
    print("🧪 Test 1: InvestmentAnalyzer.analyze_investment()")
    mock_revenue = 80_000  # EUR/year per MWh
    mock_config = {"E_nom_MWh": 1.0}
    metrics = analyzer.analyze_investment(mock_revenue, mock_config, "de")
    print("Result:", {k: f"{v:.2f}" if isinstance(v, (int, float)) else v for k, v in metrics.items()})

    print("\n🧪 Test 2: InvestmentAnalyzer.generate_yearly_df()")
    df_yearly = analyzer.generate_yearly_df(mock_revenue, mock_config, "de")
    print(df_yearly.tail(3))

    # --- Test 3: ConfigurationOptimizer ---
    print("\n🧪 Test 3: ConfigurationOptimizer.run()")


    # try:
    config_df = config_optimizer.run(country="de", c_rates=C_RATES, daily_cycles_list=DAILY_CYCLES, battery_configs=battery_configs)
    print("✅ Configuration results:")
    print(config_df[["C-rate", "number of cycles", "Levelized ROI[%]", "Yearly profits[kEUR/MWh]"]])
    # except Exception as e:
    #     print(f"❌ ConfigurationOptimizer failed: {e}")

    # # --- Save test outputs ---
    # output_dir = Path("output")
    # output_dir.mkdir(exist_ok=True)

    # config_df.to_csv(output_dir / "TechArena_Phase1_Configuration_TEST.csv", index=False)
    # df_yearly.to_csv(output_dir / "TechArena_Phase1_Investment_TEST.csv", index=False)
    # print(f"\n💾 Test outputs saved to {output_dir}/")

    print("\n🎉 All tests completed!")
