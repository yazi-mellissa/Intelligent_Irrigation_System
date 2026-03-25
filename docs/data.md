# Data provenance & pipeline (from the report + `Colabs/Data.ipynb`)

This project combines:

1) **Observed/reanalysis climate data** (NASA POWER), and  
2) **Simulated agronomic variables** (DSSAT CROPGRO-Tomato via DSSATTools)  
to produce two datasets used by the ML models.

## 1) Raw climate data: NASA POWER

**File**: `Data/POWER_Point_Daily_20000101_20241212_033d36N_006d88E_LST.csv`

**How it is loaded in the original notebook**:

- The CSV contains a metadata header → loaded with `skiprows=25`.
- Columns used include (examples): `T2M_MAX`, `T2M_MIN`, `T2M`, `RH2M`, `WS2M`, `ALLSKY_SFC_SW_DWN`, `PRECTOTCORR`, …
- Location in the notebook/report: ~**33.3614°N, 6.8753°E** (El Oued area, Algeria).
- NASA POWER uses **`-999`** to encode missing/uncomputable values (not always present in this file).

## 2) Reference evapotranspiration (ETo)

The report requires computing **ETo** (reference evapotranspiration) using the FAO-56 Penman–Monteith equation.

In `Colabs/Data.ipynb`, this is implemented through the Python package `eto`:

- A date index is created from `YEAR` + `DOY`.
- A column mapping is applied (e.g. `T2M_MAX → T_max`, `T2M_MIN → T_min`, `RH2M → RH`, `ALLSKY_SFC_SW_DWN → R_s`).
- The notebook uses:
  - `z_msl = 75` (altitude in meters),
  - `lat = 33.3614`, `lon = 6.8753`,
  - `TZ_lon = 6` (timezone longitude, UTC+1 in the notebook comments).

The resulting daily series is saved into a new column: **`ETo`**.

## 3) Agronomic simulation with DSSAT (CROPGRO-Tomato)

The report uses **DSSAT** to simulate soil/crop dynamics and irrigation variables.

In `Colabs/Data.ipynb`, this is driven from Python with **DSSATTools**:

- **Crop**: tomato cultivar `TM0001`
- **Soil**: `SoilProfile(default_class='LS')` (Loamy Sand)
- **Management** (per year):
  - `sim_start = April 14`
  - `planting_date = April 15`
  - `emergence_date = April 20`
  - `irrigation = "A"` (automatic)
  - `harvest = "M"` (maturing)
  - plus planting details like row spacing, planting depth, etc.
- Years simulated in the notebook: **2000 → 2023** (the code uses `range(begin, end)` with `begin=2000`, `end=2024`).

DSSAT outputs are read from:

- `dssat.output["SoilWat"]` (soil water)
- `dssat.output["PlantGro"]` (plant growth)

## 4) Generated datasets: Data1 and Data2

The notebook merges per-day DSSAT outputs with the climate/ETo dataframe (`df_eto`) and concatenates all simulated years.

### Data1 (`Data/output_data1.csv`)

**Purpose in the report**: train LSTM1 (next-day soil water prediction).

**Columns**:

- Climate: `T2M_MAX`, `T2M_MIN`, `WS2M`, `RH2M`, `ALLSKY_SFC_SW_DWN`, `ETo`
- Soil water: `SW1D`, `SW2D`, `SW3D`, `SWTD`
- Irrigation: `IRRC`

### Data2 (`Data/output_data2.csv`)

**Purpose in the report**: train LSTM2 (end-of-season yield/biomass prediction; proxy via `CWAD`).

**Columns** (as present in the repo file):

- Soil/crop: `SWTD`, `SWXD`, `LAID`, `HIAD`, `WSPD`, `WSGD`, `NSTD`, `CWAD`
- Climate: `T2M`, `PREC`, `T2M_MAX`, `T2M_MIN`, `WS2M`, `RH2M`, `ALLSKY_SFC_SW_DWN`, `ETo`
- Irrigation: `IRRC`

**Data quality note**: `output_data2.csv` contains `HIAD` twice (second copy appears as `HIAD.1`) and both columns are identical; in later code we treat that as a duplicate and drop `HIAD.1`.

## Quick inspection script

To print shapes / columns / basic checks for the three CSVs:

```bash
python scripts/data/inspect_datasets.py
```

