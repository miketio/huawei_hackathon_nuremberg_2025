# ⚡ TechArena 2025 — Technical README

> This project solves the **optimal operation of a Battery Energy Storage System (BESS)** on European electricity markets (Day-Ahead, FCR, aFRR).
> The problem is formulated as a **Linear Program (LP)**: maximize revenues from arbitrage and reserve markets subject to physical and contractual constraints.

---

# 1. Project Purpose

* Load and preprocess time-series data: **Day-Ahead (15 min)**, **FCR (4 h blocks)**, **aFRR (4 h blocks, pos/neg)**.
* Formulate and solve an LP optimization problem for battery scheduling.
* Generate detailed operation profiles (15-min granularity) and compute revenues.
* Sweep configurations (C-rate × daily cycles) to evaluate **investment metrics** such as yearly profits and **Levelized ROI**.
* Save results into structured Excel outputs.
* Provide validation tools to check compliance with physical and contractual constraints.

---

# 2. How to Run

Run the full pipeline:

```bash
python main.py
```

Run in **test mode**:

```bash
python main.py --test
```

Run a single optimizer manually:

```python
from Classes.dataLoader import DataLoader
from Classes.batteryOptimizer import BatteryOptimizer

loader = DataLoader("input")
data = loader.load_all_data()

config = {"E_nom_MWh": 4.5, "P_max_MW": 2.25, "eta_charge": 0.9, "eta_discharge": 0.9, "soc_min_frac":0.1, "soc_max_frac":0.9}
optimizer = BatteryOptimizer("de", config, data)
result = optimizer.optimize()
print(result.revenue)
```

---

# 3. Project Structure

```
.
├── Classes/
│   ├── batteryOptimizer.py      # Core LP optimizer (objective, constraints, bounds)
│   ├── dataLoader.py            # Load & clean input Excel data
│   ├── dataSaver.py             # Save results to Excel
│   ├── realization.py           # InvestmentAnalyzer + ConfigurationOptimizer
│   ├── coreFunctions.py         # High-level orchestration functions
│   ├── testFunctions.py         # Validation and unit tests
├── main.py                      # Entry point (pipeline execution)
├── input/                       # Input Excel (default: TechArena2025_data.xlsx)
├── output/                      # Generated Excel results (Configuration, Investment, Operation)
```

---

# 4. Data Format

### Input Excel (default: `input/TechArena2025_data.xlsx`)

* **`Day-ahead prices`** — 15-min indexed, columns = countries (`de`, `at`, `ch`, `hu`, `cz`).
* **`FCR prices`** — 4-h indexed, columns = countries.
* **`aFRR capacity prices`** — 4-h indexed, columns = `{country}_pos`, `{country}_neg`.

### Output Excel (in `output/`)

* `TechArena_Phase1_Configuration.xlsx` — optimal configurations per country.
* `TechArena_Phase1_Investment.xlsx` — investment metrics + year-by-year breakdown.
* `TechArena_Phase1_Operation.xlsx` — detailed operation (15-min resolution).

**Operation file columns:**

```
Timestamp
Stored energy[MWh]
SoC[-]
Charge[MWh]
Discharge [MWh]
Day-ahead buy[MWh]
Day-ahead sell[MWh]
FCR Capacity[MW]
aFRR Capacity POS[MW]
aFRR Capacity NEG[MW]
```

---

# 5. Mathematical Model (LP Formulation)

The optimizer builds and solves a **linear system of equations and inequalities**.
Decision vector $x$ includes:

* $p^{ch}_t$: charging power \[MW],
* $p^{dis}_t$: discharging power \[MW],
* $x_t$: state of charge (fraction),
* $r^{fcr}_b$: FCR capacity \[MW],
* $r^{pos}_b, r^{neg}_b$: aFRR up/down capacity \[MW].

---

## 5.1 Objective Function

Maximize revenues (implemented as minimization of negative revenue in code):

$$
\max_x \; 
\sum_{t=1}^n \Delta t \left( \pi^{DA}_t \cdot (\eta_d p^{dis}_t - \tfrac{1}{\eta_c}p^{ch}_t) \right)
+ \sum_{b=1}^{B_f} \pi^{FCR}_b \, r^{fcr}_b \cdot 4 \cdot \min(\eta_c,\eta_d)
+ \sum_{b=1}^{B_a} \left(\pi^{pos}_b \, r^{pos}_b \cdot 4 \eta_d + \pi^{neg}_b \, r^{neg}_b \cdot 4 \eta_c \right)
$$

where:

* $\pi^{DA}_t$: day-ahead price \[€/MWh],
* $\pi^{FCR}_b$, $\pi^{pos}_b$, $\pi^{neg}_b$: reserve prices \[€/MW].

Notes:

* Implementation builds vector `c` so that `linprog` minimizes `c^T x`. Charging terms are positive cost (including 1/eta\_c factor), discharging terms are negative (including eta\_d factor), and reserve revenues are negative with multiplicative efficiency factors.

---

## 5.2 SoC Dynamics (Equality Constraints)

For every timestep $t$:

$$
 x_t = x_{t-1} + \frac{\Delta t}{E_{nom}} \left( p^{ch}_t - p^{dis}_t \right), \quad x_0 = x^{init}
$$

This forms $A_{eq} x = b_{eq}$. These equations ensure energy conservation and link the power variables to SoC.

---

## 5.3 Power Limit (Inequality)

$$
 p^{ch}_t + p^{dis}_t + r^{fcr}_{b(t)} + r^{pos}_{a(t)} + r^{neg}_{a(t)} \;\le\; P_{max}
$$

Total instantaneous power (arbitrage + reserved capacities) cannot exceed rated battery power.

---

## 5.4 SoC Upper Bound with Reserves

$$
 E_{nom}\, x_t + \Delta t\, p^{ch}_t + \tau_{rem}(t)\,(r^{neg}_{a(t)} + r^{fcr}_{b(t)}) \;\le\; E_{nom}\, soc_{max}
$$

Interpreted as: after executing charging and potentially absorbing all NEG and FCR (for the remaining block), SoC must remain ≤ `soc_max`.

---

## 5.5 SoC Lower Bound with Reserves

$$
 - E_{nom}\, x_t + \Delta t\, p^{dis}_t + \tau_{rem}(t)\,(r^{pos}_{a(t)} + r^{fcr}_{b(t)}) \;\le\; -E_{nom}\, soc_{min}
$$

Interpreted as: after executing discharging and delivering all POS and FCR (for the remaining block), SoC must remain ≥ `soc_min` (rewritten as ≤ inequality for LP form).

---

## 5.6 Daily Cycle Limits

For each day:

$$
 \sum_{t \in day} \Delta t \, p^{ch}_t \;\le\; E_{nom}\cdot N_{cycles}, 
 \quad 
 \sum_{t \in day} \Delta t \, p^{dis}_t \;\le\; E_{nom}\cdot N_{cycles}
$$

Limits total daily throughput to a maximum number of cycles (`max_daily_cycles`) to control usage intensity and approximate degradation constraints.

---

## 5.7 Variable Bounds

* $0 \le p^{ch}_t \le P_{max}$
* $0 \le p^{dis}_t \le P_{max}$
* $soc_{min} \le x_t \le soc_{max}$
* $0 \le r^{fcr}_b, r^{pos}_b, r^{neg}_b \le P_{max}$

---

# 6. Economic Model

The optimization results (revenues per configuration and country) are post-processed into
**investment metrics**. This section describes how yearly profits, present values, and
Levelized ROI are computed.

---

## 6.1 Yearly Profits

The optimizer yields a **total annual revenue** for the battery system:

$$
R^{tot}_{year} \; [\text{EUR/year}]
$$

To normalize per unit of energy capacity, we divide by the installed energy $E_{nom}$:

$$
\pi_{year}^{(0)} \;=\; \frac{R^{tot}_{year}}{1000 \cdot E_{nom}}
\quad [\text{kEUR/MWh}]
$$

where:
* $E_{nom}$ = nominal battery capacity \[MWh],
* $1000$ = scaling factor EUR → kEUR.

This value corresponds to the **first-year nominal profit per MWh**.

---

## 6.2 Nominal Cash Flows with Inflation

Future yearly profits are escalated with the country-specific inflation rate $\iota$:

$$
\pi_{year}^{(t)} \;=\; \pi_{year}^{(0)} \cdot (1+\iota)^t
$$

for $t = 0,1,2,\dots,T-1$, where $T$ is the project lifetime in years (typically 10).

---

## 6.3 Discounting with WACC

To evaluate investment profitability, nominal cash flows are **discounted** back to present
value using the Weighted Average Cost of Capital (WACC):

$$
PV \;=\; \sum_{t=0}^{T-1} \frac{\pi_{year}^{(t)}}{(1+WACC)^{t+1}}
$$

* WACC reflects the weighted required return of both equity and debt holders.
* In this project, the **discount rate** is taken equal to the WACC, since we
  evaluate free cash flows to the firm (FCFF) in nominal terms.

---

## 6.4 Levelized ROI

The initial investment per unit energy capacity is the CAPEX:

$$
CAPEX \;=\; c_{capex} \quad [\text{kEUR/MWh}]
$$

where $c_{capex}$ is a given cost assumption (e.g. 200 kEUR/MWh).

The Levelized ROI (%) is then computed as the ratio of discounted profits to the initial
investment:

$$
ROI_{lev} \;=\; \frac{PV}{CAPEX} \cdot 100 \;\; [\%]
$$

---

## 6.5 Summary of Reported Indicators

For each country, the investment output file reports:

* **WACC** (nominal, from financial assumptions),
* **Inflation rate** (nominal, from financial assumptions),
* **Discount rate** (= WACC, since flows are FCFF in nominal terms),
* **Initial Investment [kEUR/MWh]**,
* **Yearly profits [kEUR/MWh]** (first year),
* **Levelized ROI [%]** (over project horizon).


# 7. Implementation Notes

* Objective, constraints, and bounds are constructed in `Classes/batteryOptimizer.py`:

  * `_build_objective()` builds vector `c`.
  * `_build_equality_constraints()` builds `A_eq` and `b_eq`.
  * `_build_inequality_constraints()` builds `A_ub` and `b_ub`.
  * `_add_daily_cycle_constraints()` appends daily throughput constraints.
  * `_build_bounds()` returns per-variable bounds.
* Solver: `scipy.optimize.linprog(method="highs")`.
* Efficiencies (`eta_c`, `eta_d`) are applied both in objective and when computing market buy/sell amounts and reserve revenue multipliers.
* `tau_rem(t)` is computed as remaining hours in the current 4-hour block (used to estimate potential energy associated with block-level reserves).
* `validate_solution` reproduces and checks constraints on the `operation_df` and produces diagnostic plots (SoC, power, energy budgets, daily cycles).
* The LP structure is modular and ready to extend: adding degradation cost terms, ramp-rate constraints, or stochastic scenarios is done by adding variables/rows in `c`, `A_eq`/`b_eq`, and `A_ub`/`b_ub`.

---

*End of README.*
