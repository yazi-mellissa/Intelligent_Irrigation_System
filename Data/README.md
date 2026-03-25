# Data folder

This repository includes three CSV files:

## `Data/POWER_Point_Daily_20000101_20241212_033d36N_006d88E_LST.csv`

- **Source**: NASA POWER (daily point data).
- **Coverage**: 2000-01-01 → 2024-12-12 (daily).
- **Location**: ~33.3614°N, 6.8753°E (El Oued area, Algeria).
- **Notes**:
  - The original file includes a metadata header; in the original notebook it is loaded with `skiprows=25`.
  - Missing/uncomputable values may be encoded as `-999` (NASA POWER convention).

## `Data/output_data1.csv` (Data1)

- **How it was created**: DSSAT (CROPGRO-Tomato) simulation outputs merged with climate + computed ETo.
- **What it contains**: daily climate variables + simulated soil water variables (incl. `SWTD`) and irrigation (`IRRC`).
- **Main use in the report**: train LSTM1 to predict next-day soil water (`SWTD`).

## `Data/output_data2.csv` (Data2)

- **How it was created**: DSSAT (CROPGRO-Tomato) simulation outputs merged with climate + computed ETo.
- **What it contains**: daily climate + soil water + crop growth variables (incl. `CWAD`) and irrigation (`IRRC`).
- **Main use in the report**: train LSTM2 to predict end-of-season yield/biomass (proxy via `CWAD`).

For the full step-by-step pipeline used to generate Data1/Data2 (ETo computation + DSSATTools simulation), see `docs/data.md`.

