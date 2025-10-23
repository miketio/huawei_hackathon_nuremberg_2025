import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from typing import Dict, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class OptimizationResult:
    operation_df: pd.DataFrame
    energy_revenue: float
    capacity_revenue: float
    revenue: float
    charge_energy: float
    discharge_energy: float

    
class BatteryOptimizer:
    """
    Solves the BESS optimal operation problem for a single country and battery configuration.
    Maximizes revenue from:
    - Day-ahead arbitrage (15-min)
    - FCR capacity (4-hour blocks)
    - aFRR capacity (4-hour blocks, pos/neg)

    Generates operation output in required format for TechArena.
    """

    def __init__(self, country: str, battery_config: Dict, data: Dict[str, pd.DataFrame]):
        """
        Args:
            country: lowercase country code (e.g., 'de')
            battery_config: dict with keys E_nom_MWh, P_max_MW, eta_charge, etc.
            data: dict with keys "da", "fcr", "afrr" (DataFrames)
        """
        self.country = country.lower()
        self.conf = battery_config
        self.data = data

        # Extract battery parameters
        self.E_nom = float(self.conf['E_nom_MWh'])
        self.P_max = float(self.conf['P_max_MW'])
        self.eta_c = float(self.conf.get('eta_charge', 1.0))
        self.eta_d = float(self.conf.get('eta_discharge', 1.0))
        self.soc_min = float(self.conf.get('soc_min_frac', 0.0))
        self.soc_max = float(self.conf.get('soc_max_frac', 1.0))
        self.soc_0 = float(self.conf.get('soc0_frac', 0.5))
        self.max_daily_cycles = self.conf.get('max_daily_cycles')
        # Time settings
        self.dt_hours = 0.25  # 15 minutes

        # Validate and preprocess input data
        self.da_prices, self.fcr_prices, self.afrr_prices = self._load_and_align_markets()

        # Derived time vectors
        self.T_index = self.da_prices.index
        self.n = len(self.T_index)

        self.fcr_index = self.fcr_prices.index
        self.Bf = len(self.fcr_index)
        self.afrr_index = self.afrr_prices.index
        self.Ba = len(self.afrr_index)

        # Variable indexing (in optimization vector x)
        self.i_ch = 0
        self.i_dis = self.i_ch + self.n
        self.i_soc = self.i_dis + self.n
        self.i_fcr = self.i_soc + self.n
        self.i_pos = self.i_fcr + self.Bf
        self.i_neg = self.i_pos + self.Ba
        self.m = self.i_neg + self.Ba  # total variables

        # Mapping: timestep → FCR/aFRR block index
        self.fcr_block_idx = self._map_to_blocks(self.T_index, self.fcr_index)
        self.afrr_block_idx = self._map_to_blocks(self.T_index, self.afrr_index)

    def _map_to_blocks(self, timesteps: pd.DatetimeIndex, block_index: pd.DatetimeIndex) -> np.ndarray:
        """Map each 15-min timestep to its corresponding 4-hour block."""
        indexer = block_index.get_indexer(timesteps, method='ffill')
        indexer[indexer == -1] = 0  # fallback to first block
        return indexer.astype(int)

    def _load_and_align_markets(self) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """Load and align DA, FCR, aFRR prices for the target country."""
        da_raw = self.data["da"]
        fcr_raw = self.data["fcr"]
        afrr_raw = self.data["afrr"]

        # Normalize DE_LU → de
        da_cols = da_raw.columns.str.replace(r'^de_lu$', 'de', case=False, regex=True)
        da_raw.columns = da_cols

        # --- Day-ahead ---
        if self.country not in da_raw.columns:
            raise ValueError(f"Country '{self.country}' not in DA data. Available: {list(da_raw.columns)}")
        da_prices = da_raw[self.country].copy().astype(float)

        # --- FCR ---
        short = self.country.lower()
        if short not in fcr_raw.columns:
            raise ValueError(f"Country '{short}' not in FCR data. Available: {list(fcr_raw.columns)}")
        fcr_prices = fcr_raw[short].copy().astype(float)

        # --- aFRR ---
        pos_col = f"{short}_pos"
        neg_col = f"{short}_neg"
        afrr_pos = afrr_raw[pos_col] if pos_col in afrr_raw.columns else pd.Series(0.0, index=afrr_raw.index)
        afrr_neg = afrr_raw[neg_col] if neg_col in afrr_raw.columns else pd.Series(0.0, index=afrr_raw.index)

        afrr_prices = pd.DataFrame({"Pos": afrr_pos, "Neg": afrr_neg}, index=afrr_raw.index).astype(float)

        # Ensure datetime index and fill missing
        for series in [da_prices, fcr_prices, afrr_prices]:
            series.index = pd.to_datetime(series.index, errors='coerce')
        da_prices = da_prices.dropna()
        fcr_prices = fcr_prices.dropna()
        afrr_prices = afrr_prices.dropna()

        return da_prices.fillna(0), fcr_prices.fillna(0), afrr_prices.fillna(0)

    def optimize(self) -> OptimizationResult:
        """Solve the LP and return structured results."""
        print(f"Optimizing for {self.country} with {self.conf.get('max_daily_cycles')} cycles and {self.conf.get('C_rate')} rate...")
        eps = 1e-2  # small epsilon for numerical stability
        # Build objective vector
        c = self._build_objective()

        # Equality constraints: SOC dynamics
        A_eq, b_eq = self._build_equality_constraints()

        # Inequality constraints: power, energy, aFRR guarantees, step limits
        A_ub, b_ub = self._build_inequality_constraints(eps=eps)

        # Add daily cycle limits if specified
        if self.max_daily_cycles is not None:
            A_ub, b_ub = self._add_daily_cycle_constraints(A_ub, b_ub, eps=eps)

        # Variable bounds
        bounds = self._build_bounds()

        # Solve
        res = linprog(
            c=c,
            A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            bounds=bounds,
            method='highs',
            options={'presolve': True}
        )

        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")

        # Extract solution
        x = res.x
        p_ch = x[self.i_ch:self.i_ch + self.n]
        p_dis = x[self.i_dis:self.i_dis + self.n]
        soc = x[self.i_soc:self.i_soc + self.n]
        fcr_vals = x[self.i_fcr:self.i_fcr + self.Bf]
        pos_vals = x[self.i_pos:self.i_pos + self.Ba]
        neg_vals = x[self.i_neg:self.i_neg + self.Ba]

        # Compute total revenue
        energy_rev, cap_rev = self._calculate_total_revenue(p_ch, p_dis, fcr_vals, pos_vals, neg_vals)

        # Create operation DataFrame in required format
        operation_df = self._build_operation_dataframe(p_ch, p_dis, soc, fcr_vals, pos_vals, neg_vals)

        # Energy stats
        charge_energy = (p_ch * self.dt_hours / self.eta_c).sum()
        discharge_energy = (p_dis * self.dt_hours * self.eta_d).sum()

        return OptimizationResult(
            operation_df=operation_df,
            energy_revenue=energy_rev,
            capacity_revenue=cap_rev,
            revenue=energy_rev + cap_rev,
            charge_energy=charge_energy,
            discharge_energy=discharge_energy
        )

    def _build_objective(self) -> np.ndarray:
        """Minimize negative revenue (i.e., maximize revenue)."""
        c = np.zeros(self.m)

        # DA arbitrage: cost of charging, revenue from discharging
        c[self.i_ch:self.i_ch + self.n] = self.da_prices.values * self.dt_hours  / self.eta_c # +cost → minimize
        c[self.i_dis:self.i_dis + self.n] = -self.da_prices.values * self.dt_hours  * self.eta_d # -revenue → minimize

        # FCR capacity revenue (per MW block)
        for i, ts in enumerate(self.fcr_index):
            price_MW = float(self.fcr_prices.loc[ts])
            c[self.i_fcr + i] = -price_MW * 4.0 * min(self.eta_c, self.eta_d)  # 4-hour blocks

        # aFRR capacity revenue (pos/neg)
        for i, ts in enumerate(self.afrr_index):
            pos_price = float(self.afrr_prices.loc[ts, "Pos"])
            neg_price = float(self.afrr_prices.loc[ts, "Neg"])
            c[self.i_pos + i] = -pos_price * 4.0 * self.eta_d
            c[self.i_neg + i] = -neg_price * 4.0 * self.eta_c

        return c



    def _build_equality_constraints(self) -> Tuple[sparse.spmatrix, np.ndarray]:
        """
        SOC dynamics: soc[t] = soc[t-1] + (p_ch[t] * dt) / E_nom - (p_dis[t] * dt) / (E_nom)
        Translated into: A_eq @ x = b_eq
        """
        rows_eq = self.n
        A_eq = sparse.lil_matrix((rows_eq, self.m), dtype=float)
        b_eq = np.zeros(rows_eq, dtype=float)

        # Coefficients for charge and discharge (normalized by E_nom)
        coef_pch = (self.dt_hours) / self.E_nom    # positive contribution to SOC
        coef_pdis = (self.dt_hours) / (self.E_nom)  # negative contribution

        for t in range(self.n):
            # soc[t] coefficient
            A_eq[t, self.i_soc + t] = 1.0
            # p_ch[t] coefficient
            A_eq[t, self.i_ch + t] = -coef_pch  # because p_ch increases SOC → subtracted in equation form
            # p_dis[t] coefficient
            A_eq[t, self.i_dis + t] = coef_pdis  # because p_dis decreases SOC

            if t == 0:
                # Initial condition: soc[0] = soc_0
                b_eq[t] = self.soc_0
            else:
                # soc[t-1] coefficient
                A_eq[t, self.i_soc + (t - 1)] = -1.0
                b_eq[t] = 0.0

        return A_eq.tocsr(), b_eq


    def _build_inequality_constraints(self, eps: float =1e-2) -> Tuple[sparse.spmatrix, np.ndarray]:
        """
        Builds inequality constraints matrix and vector with detailed formulas:
        
        (1) Power limit: total instantaneous power ≤ P_max
            p_ch[t] + p_dis[t] + p_fcr[bidx] + p_pos[aidx] + p_neg[aidx] ≤ P_max

        (2) SOC upper bound: battery SOC after charging and delivering NEG/FCR ≤ SOC_max * E_nom
            SOC[t]*E_nom + p_ch[t]*dt + p_neg[aidx]*tau_rem + p_fcr[bidx]*tau_rem ≤ SOC_max*E_nom

        (3) SOC lower bound: battery SOC after discharging and delivering POS/FCR ≥ SOC_min * E_nom
            SOC[t]*E_nom - p_dis[t]*dt - p_pos[aidx]*tau_rem - p_fcr[bidx]*tau_rem ≥ SOC_min*E_nom
            ⇒ -SOC[t]*E_nom + p_dis[t]*dt + p_pos[aidx]*tau_rem + p_fcr[bidx]*tau_rem ≤ -SOC_min*E_nom
        """
        rows_ub = 3 * self.n
        A_ub = sparse.lil_matrix((rows_ub, self.m), dtype=float)
        b_ub = np.zeros(rows_ub, dtype=float)

        block_hours = 4.0  # Duration of FCR and aFRR blocks [hours]

        for t in range(self.n):
            row_base = 3 * t

            # Current FCR/aFRR block indices
            bidx = self.fcr_block_idx[t]
            aidx = self.afrr_block_idx[t]

            # Remaining time in current 4-hour block
            block_start = self.afrr_index[aidx]
            elapsed = (self.T_index[t] - block_start).total_seconds() / 3600.0
            tau_rem = max(0.0, block_hours - elapsed)

            # ---- (1) Power constraint ----
            # Formula: p_ch[t] + p_dis[t] + p_fcr[bidx] + p_pos[aidx] + p_neg[aidx] ≤ P_max
            A_ub[row_base, self.i_ch + t] = 1.0
            A_ub[row_base, self.i_dis + t] = 1.0
            A_ub[row_base, self.i_fcr + bidx] = 1.0
            A_ub[row_base, self.i_pos + aidx] = 1.0
            A_ub[row_base, self.i_neg + aidx] = 1.0
            b_ub[row_base] = self.P_max - eps

            # ---- (2) SOC upper bound ----
            # Formula: SOC[t]*E_nom + p_ch[t]*dt + p_neg[aidx]*tau_rem + p_fcr[bidx]*tau_rem ≤ SOC_max*E_nom
            A_ub[row_base + 1, self.i_soc + t] = self.E_nom
            A_ub[row_base + 1, self.i_ch + t] = self.dt_hours
            A_ub[row_base + 1, self.i_neg + aidx] = tau_rem
            A_ub[row_base + 1, self.i_fcr + bidx] = tau_rem
            b_ub[row_base + 1] = self.E_nom * self.soc_max - eps

            # ---- (3) SOC lower bound ----
            # Formula: -SOC[t]*E_nom + p_dis[t]*dt + p_pos[aidx]*tau_rem + p_fcr[bidx]*tau_rem ≤ -SOC_min*E_nom
            A_ub[row_base + 2, self.i_soc + t] = -self.E_nom
            A_ub[row_base + 2, self.i_dis + t] = self.dt_hours
            A_ub[row_base + 2, self.i_pos + aidx] = tau_rem
            A_ub[row_base + 2, self.i_fcr + bidx] = tau_rem
            b_ub[row_base + 2] = -self.E_nom * self.soc_min - eps

        return A_ub.tocsr(), b_ub

    def _add_daily_cycle_constraints(self, A_ub: sparse.spmatrix, b_ub: np.ndarray, eps: float = 1e-2) -> Tuple[sparse.spmatrix, np.ndarray]:
        """
        Adds constraints on maximum daily charge/discharge energy to limit number of cycles.
        """
        if self.max_daily_cycles is None:
            return A_ub, b_ub

        # Get unique days
        days = pd.to_datetime(self.T_index).normalize().unique()
        extra_rows = 2 * len(days)  # one for charge, one for discharge per day
        A_extra = sparse.lil_matrix((extra_rows, self.m), dtype=float)
        b_extra = np.zeros(extra_rows, dtype=float)

        for j, day in enumerate(days):
            mask = (pd.to_datetime(self.T_index).normalize() == day)
            idxs = np.where(mask)[0]

            # === Charge energy limit ===
            for t in idxs:
                A_extra[2 * j, self.i_ch + t] = self.dt_hours
            b_extra[2 * j] = self.E_nom * self.max_daily_cycles - eps

            # === Discharge energy limit ===
            for t in idxs:
                A_extra[2 * j + 1, self.i_dis + t] = self.dt_hours
            b_extra[2 * j + 1] = self.E_nom * self.max_daily_cycles - eps

        # Stack new constraints vertically
        A_ub = sparse.vstack([A_ub, A_extra], format="csr")
        b_ub = np.concatenate([b_ub, b_extra])

        return A_ub, b_ub
    
    def _build_bounds(self) -> list:
        """Variable bounds: (min, max) for each variable in x."""
        bounds = []

        # p_ch: [0, P_max]
        bounds += [(0.0, self.P_max)] * self.n
        # p_dis: [0, P_max]
        bounds += [(0.0, self.P_max)] * self.n
        # soc: [soc_min, soc_max]
        bounds += [(self.soc_min, self.soc_max)] * self.n
        # FCR: [0, P_max]
        bounds += [(0.0, self.P_max)] * self.Bf
        # aFRR pos: [0, P_max]
        bounds += [(0.0, self.P_max)] * self.Ba
        # aFRR neg: [0, P_max]
        bounds += [(0.0, self.P_max)] * self.Ba

        return bounds

    def _calculate_total_revenue(self, p_ch: np.ndarray, p_dis: np.ndarray,
                                 fcr_vals: np.ndarray, pos_vals: np.ndarray, neg_vals: np.ndarray) -> float:
        """Compute total annual revenue from all sources."""
        cap_rev = 0.0

        # market buy energy (MWh) at t = p_ch / eta_c * dt
        # market sell energy (MWh) at t = p_dis * eta_d * dt
        da = self.da_prices.values
        buy_energy = (p_ch / self.eta_c) * self.dt_hours   # MWh bought from grid
        sell_energy = (p_dis * self.eta_d) * self.dt_hours # MWh sold to grid
        energy_rev = ( - da * buy_energy + da * sell_energy ).sum()

        # FCR revenue
        cap_rev += np.sum(fcr_vals * self.fcr_prices.values * 4.0 * min(self.eta_c, self.eta_d))

        # aFRR revenue using positional indexing
        cap_rev += np.sum(pos_vals * self.afrr_prices["Pos"].values * 4.0 * self.eta_d)
        cap_rev += np.sum(neg_vals * self.afrr_prices["Neg"].values * 4.0 * self.eta_c)

        return energy_rev, cap_rev

    def _build_operation_dataframe(self, p_ch: np.ndarray, p_dis: np.ndarray, soc: np.ndarray,
                                   fcr_vals: np.ndarray, pos_vals: np.ndarray, neg_vals: np.ndarray) -> pd.DataFrame:
        """Build output DataFrame in exact required format."""
        # Map block-level reserves to 15-min level
        fcr_15min = fcr_vals[self.fcr_block_idx]
        pos_15min = pos_vals[self.afrr_block_idx]
        neg_15min = neg_vals[self.afrr_block_idx]

        df = pd.DataFrame({
            "Timestamp": self.T_index,
            "Stored energy[MWh]": soc * self.E_nom,
            "SoC[-]": soc,
            "Charge[MWh]": p_ch * self.dt_hours,
            "Discharge [MWh]": p_dis * self.dt_hours,
            "Day-ahead buy[MWh]": p_ch * self.dt_hours / self.eta_c,
            "Day-ahead sell[MWh]": p_dis * self.dt_hours * self.eta_d,
            "FCR Capacity[MW]": fcr_15min,
            "aFRR Capacity POS[MW]": pos_15min,
            "aFRR Capacity NEG[MW]": neg_15min
        })

        # Ensure correct data types and round
        df["Stored energy[MWh]"] = df["Stored energy[MWh]"].round(3)
        df["SoC[-]"] = df["SoC[-]"].round(4)
        df["Charge[MWh]"] = df["Charge[MWh]"].round(3)
        df["Discharge [MWh]"] = df["Discharge [MWh]"].round(3)
        df["Day-ahead buy[MWh]"] = df["Day-ahead buy[MWh]"].round(3)
        df["Day-ahead sell[MWh]"] = df["Day-ahead sell[MWh]"].round(3)
        df["FCR Capacity[MW]"] = df["FCR Capacity[MW]"].round(3)
        df["aFRR Capacity POS[MW]"] = df["aFRR Capacity POS[MW]"].round(3)
        df["aFRR Capacity NEG[MW]"] = df["aFRR Capacity NEG[MW]"].round(3)

        return df

    def validate_solution(self, operation_df: pd.DataFrame,
                    start: str = "2024-06-01", end: str = "2024-06-09",
                    eps: float = 1e-3) -> dict:
        """
        Validate the optimization solution using only the operation_df contained in the result.
        Performs the same constraint checks and plots as before.

        Args:
            result: OptimizationResult returned from .optimize()
            start: plotting/validation window start (inclusive)
            end: plotting/validation window end (exclusive)
            eps: numerical tolerance for constraint violations

        Returns:
            dict with counts of violations for each check
        """
        # Use a copy so we don't mutate the original DataFrame
        op_df = operation_df.copy()

        # Ensure Timestamp is datetime index
        if "Timestamp" in op_df.columns:
            op_df["Timestamp"] = pd.to_datetime(op_df["Timestamp"], errors="coerce")
            op_df = op_df.set_index("Timestamp")
        else:
            # assume index is already datetime like before
            op_df.index = pd.to_datetime(op_df.index, errors="coerce")

        # Sort index to be safe
        op_df = op_df.sort_index()

        # Fill missing reserve columns with zeros if not present
        for col in ["FCR Capacity[MW]", "aFRR Capacity POS[MW]", "aFRR Capacity NEG[MW]",
                    "Charge[MWh]", "Discharge [MWh]", "SoC[-]"]:
            if col not in op_df.columns:
                op_df[col] = 0.0

        # local variables
        T_index = op_df.index
        dt_hours = self.dt_hours
        E_nom = self.E_nom
        P_max = self.P_max
        x_min = self.soc_min
        x_max = self.soc_max
        max_daily_cycles = self.max_daily_cycles

        # Slice for plotting/validation
        mask = (T_index >= pd.to_datetime(start)) & (T_index < pd.to_datetime(end))
        if mask.sum() == 0:
            # if no points in requested window, widen to available data
            mask = slice(None)
        T_plot = T_index[mask]
        op_plot = op_df.loc[mask]

        # Derive reserves at each timestep from operation_df
        fcr_vals_15min = op_df["FCR Capacity[MW]"].fillna(0.0).values
        pos_vals_15min = op_df["aFRR Capacity POS[MW]"].fillna(0.0).values
        neg_vals_15min = op_df["aFRR Capacity NEG[MW]"].fillna(0.0).values
        reserves = fcr_vals_15min + pos_vals_15min + neg_vals_15min

        # Compute time remaining in the current 4-hour block by flooring timestamps to 4H
        # and using difference between timestamp and block start.
        # This matches the previous 4-hour block assumption.
        block_start = T_index.floor('4H')
        elapsed_hours = (T_index - block_start).total_seconds() / 3600.0
        tau_rem_arr = np.maximum(0.0, 4.0 - elapsed_hours)  # shape = len(T_index)

        # === Constraint checks (all derived from operation_df) ===
        violations = {}

        # 1) Power: p_ch + p_dis + reserves <= P_max
        # Convert stored energy per timestep (MWh) to power (MW): power = energy / dt_hours
        p_ch = (op_df["Charge[MWh]"].fillna(0.0) / dt_hours).values
        p_dis = (op_df["Discharge [MWh]"].fillna(0.0) / dt_hours).values
        p_total = p_ch + p_dis + reserves
        violations["power"] = int((p_total > P_max + eps).sum())

        # 2) Energy: instantaneous arbitrage energy + potential reserve energy over remaining block <= E_nom
        E_arb = (p_ch + p_dis) * dt_hours                     # instantaneous arbitrage energy [MWh]
        E_res = reserves * tau_rem_arr                        # potential reserve energy [MWh]
        E_total = E_arb + E_res
        violations["energy"] = int((E_total > E_nom + eps).sum())

        # 3) SOC bounds (use SoC[-] column which is fraction)
        soc = op_df["SoC[-]"].fillna(method="ffill").fillna(self.soc_0).values
        violations["soc_low"] = int((soc < x_min - eps).sum())
        violations["soc_high"] = int((soc > x_max + eps).sum())

        # 4) Daily cycle limits (sum of charge/discharge energy per day)
        if max_daily_cycles is not None:
            # ensure Charge and Discharge are energy per timestep (MWh)
            daily_charge = op_df["Charge[MWh]"].fillna(0.0).resample("1D").sum()
            daily_discharge = op_df["Discharge [MWh]"].fillna(0.0).resample("1D").sum()
            daily_limit = E_nom * max_daily_cycles
            violations["daily_ch"] = int((daily_charge > daily_limit + eps).sum())
            violations["daily_dis"] = int((daily_discharge > daily_limit + eps).sum())
        else:
            violations["daily_ch"] = 0
            violations["daily_dis"] = 0

        # Print short report
        print("✅ Constraint Validation Report")
        print("=" * 40)
        for k, v in violations.items():
            status = "❌ FAILED" if v > 0 else "✅ OK"
            print(f"{k:15} : {v:3} violations {status}")



        fig, axs = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

        if len(T_plot) > 0:
            # SOC plot
            axs[0].plot(T_plot, op_plot["SoC[-]"], label="SOC", linewidth=1)
            axs[0].axhline(x_min, color="red", linestyle="--", label="Min SOC")
            axs[0].axhline(x_max, color="green", linestyle="--", label="Max SOC")
            axs[0].set_ylabel("SOC [-]")
            axs[0].legend()
            axs[0].grid(True, alpha=0.3)

            # Power plot
            p_ch_plot = (op_plot["Charge[MWh]"] / dt_hours).fillna(0.0)
            p_dis_plot = (op_plot["Discharge [MWh]"] / dt_hours).fillna(0.0)
            reserves_plot = (op_plot["FCR Capacity[MW]"].fillna(0.0)
                            + op_plot["aFRR Capacity POS[MW]"].fillna(0.0)
                            + op_plot["aFRR Capacity NEG[MW]"].fillna(0.0))
            p_total_plot = p_ch_plot + p_dis_plot + reserves_plot

            axs[1].plot(T_plot, p_ch_plot, label="Charge Power", alpha=0.8)
            axs[1].plot(T_plot, p_dis_plot, label="Discharge Power", alpha=0.8)
            axs[1].plot(T_plot, reserves_plot, label="Reserve Power", alpha=0.7)
            axs[1].plot(T_plot, p_total_plot, label="Total Power", linewidth=2)
            axs[1].axhline(P_max, color="red", linestyle="--", label="P_max")
            axs[1].set_ylabel("Power [MW]")
            axs[1].legend()
            axs[1].grid(True, alpha=0.3)

            # Energy usage plot (arbitrage + reserve potential)
            # Align tau_rem_arr for plotting window:
            tau_rem_plot = tau_rem_arr[mask] if isinstance(mask, (np.ndarray, slice)) else tau_rem_arr
            E_arb_plot = (p_ch_plot + p_dis_plot) * dt_hours
            E_res_plot = reserves_plot.values * np.array(tau_rem_plot)
            E_total_plot = E_arb_plot.values + E_res_plot

            axs[2].plot(T_plot, E_total_plot, label="Total Potential Energy Use", linewidth=2)
            axs[2].plot(T_plot, E_arb_plot, label="Arbitrage Energy", alpha=0.7)
            axs[2].plot(T_plot, E_res_plot, label="Reserve Energy", alpha=0.7)
            axs[2].axhline(E_nom, color="red", linestyle="--", label="E_nom")
            axs[2].set_ylabel("Energy [MWh]")
            axs[2].legend()
            axs[2].grid(True, alpha=0.3)

            # Daily cycles bar plot
            if max_daily_cycles is not None:
                daily_charge = op_df["Charge[MWh]"].fillna(0.0).resample("1D").sum()
                daily_discharge = op_df["Discharge [MWh]"].fillna(0.0).resample("1D").sum()
                limit = E_nom * max_daily_cycles
                days = daily_charge.index

                axs[3].bar(days, daily_charge, width=0.8, label="Daily Charge Energy", alpha=0.8)
                axs[3].bar(days, daily_discharge, width=0.5, label="Daily Discharge Energy", alpha=0.8)
                axs[3].axhline(limit, color="red", linestyle="--", label=f"Max ({limit:.2f} MWh)")
                # set x-limits to the plotting window if available
                try:
                    axs[3].set_xlim([T_plot[0], T_plot[-1]])
                except Exception:
                    pass
                axs[3].set_ylabel("Energy [MWh]")
                axs[3].legend()
                axs[3].grid(True, alpha=0.3)

        plt.xlabel("Time")
        plt.tight_layout()
        plt.show()

        return violations
