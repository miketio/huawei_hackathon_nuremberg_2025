import pandas as pd
from pathlib import Path

class DataSaver:
    """Utility class for saving configuration, investment, and operation results."""

    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def save_configurations(self, config_records: dict, filename="TechArena_Phase1_Configuration.xlsx"):
        path = self.output_dir / filename
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            for country, df in config_records.items():
                df.to_excel(writer, sheet_name=country, index=False)
        print(f"✅ Configuration Excel saved to {path}")

    def save_investments(self, investment_results: dict, filename="TechArena_Phase1_Investment.xlsx"):
        """
        Save investment results (header + yearly data per country) into Excel.
        """
        path = self.output_dir / filename
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for country, parts in investment_results.items():
                header_df = parts["header"]
                yearly_df = parts["yearly"]

                # Write header at the top
                header_df.to_excel(writer, sheet_name=country, index=False, startrow=0)

                # Write yearly table starting at row 6
                yearly_df.to_excel(writer, sheet_name=country, index=False, startrow=6, header=True)

                # Add Levelized ROI row
                last_row_idx = 6 + len(yearly_df)
                worksheet = writer.sheets[country]
                worksheet.cell(row=last_row_idx + 1, column=1, value="Levelized ROI")
                worksheet.cell(row=last_row_idx + 1, column=3, value=yearly_df.iloc[-1]["Yearly profits[kEUR/MWh]"])

        print(f"✅ Investment Excel saved to {path}")

    def save_operations(self, operation_df, country_best: str, conf: dict, filename="TechArena_Phase1_Operation"):
        """
        Save operation results to both CSV and Excel formats,
        and include a one-row summary of the best country/config.
        """
        # --- Summary row ---
        summary_df = pd.DataFrame([{
            "Best Country": country_best.upper(),
            "C-rate": conf.get("C_rate"),
            "Cycles": conf.get("max_daily_cycles"),
            "Nominal Energy [MWh]": conf.get("E_nom_MWh"),
            "Nominal Power [MW]": conf.get("P_max_MW"),
        }])

        # --- Save Excel (separate sheets) ---
        xlsx_path = self.output_dir / f"{filename}.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Best_Config", index=False)
            operation_df.to_excel(writer, sheet_name="Operation", index=False)
        print(f"✅ Operation results (with summary sheet) saved to {xlsx_path}")

