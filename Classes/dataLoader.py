import os
import pandas as pd
from typing import List, Dict

class DataLoader:
    """
    Handles loading and cleaning of input Excel data.
    """
    def __init__(self, input_dir: str = "input"):
        self.input_dir = input_dir
        self.file_path = os.path.join(input_dir, "TechArena2025_data.xlsx")

    def detect_header_rows(self, sheet_name: str, max_rows: int = 10) -> List[int]:
        df = pd.read_excel(self.file_path, sheet_name=sheet_name, header=None, nrows=max_rows)
        has_timestep = df.apply(lambda r: r.astype(str).str.contains('timestep', case=False).any(), axis=1)
        ts_row_idx = has_timestep.idxmax() if has_timestep.any() else 0

        country_row = ts_row_idx - 1
        countries = {'DE', 'AT', 'CH', 'HU', 'CZ'}
        if country_row >= 0:
            row_vals = df.iloc[country_row].astype(str).str.strip().str.upper()
            if row_vals.isin(countries).sum() >= 2:
                return [country_row, ts_row_idx]
        return [ts_row_idx]

    def load_sheet(self, sheet_name: str) -> pd.DataFrame:
        header = self.detect_header_rows(sheet_name)
        df = pd.read_excel(self.file_path, sheet_name=sheet_name, header=header)

        if isinstance(df.columns, pd.MultiIndex):
            new_cols = []
            for cols in df.columns:
                valid_parts = [str(c).strip() for c in cols if pd.notna(c) and str(c).strip() and not str(c).startswith('Unnamed')]
                new_cols.append('_'.join(valid_parts) if valid_parts else 'Unknown')
            df.columns = new_cols

        timestep_col = next((col for col in df.columns if 'timestep' in str(col).lower()), df.columns[0])
        df = df.rename(columns={timestep_col: 'Timestep'})
        df['Timestep'] = pd.to_datetime(df['Timestep'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Timestep']).set_index('Timestep').sort_index()

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').round(2)
        df = df.dropna(axis=1, how='all')

        df.columns = (df.columns
                      .str.replace(r'\s+', '_', regex=True)
                      .str.replace(r'_+', '_', regex=True)
                      .str.strip('_')
                      .str.lower())
        return df

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        da = self.load_sheet("Day-ahead prices")
        fcr = self.load_sheet("FCR prices")
        afrr = self.load_sheet("aFRR capacity prices")

        # Normalize DE_LU -> DE
        da.columns = da.columns.str.replace(r'^de_lu$', 'de', case=False, regex=True)

        return {"da": da, "fcr": fcr, "afrr": afrr}
