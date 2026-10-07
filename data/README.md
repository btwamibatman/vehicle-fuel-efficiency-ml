# Data

## Source (verified)

| Item | Value |
|---|---|
| Dataset | **Fuel Economy Data** (`vehicles.csv`) - U.S. Environmental Protection Agency (EPA) / U.S. Department of Energy (DOE), published on fueleconomy.gov |
| Page | https://www.fueleconomy.gov/feg/ws/index.shtml (data dictionary and download) |
| Download | https://www.fueleconomy.gov/feg/epadata/vehicles.csv.zip |
| Coverage | Model years 1984-2027 in the file (the 2027 rows are model years already announced); this project uses **2015 and later** |
| Raw size | 50,407 rows x 84 columns (checked on 2026-10-07) |
| Target | `mpg` = raw column `comb08`: EPA **combined** fuel economy in **US miles per gallon** (about 55% city / 45% highway). Higher = more efficient |

**Licence / usage terms:** this is U.S. federal government data. **Aktore: open the fueleconomy.gov page above, copy the exact usage/licence statement into this file and cite the source in the slides.** I did not confirm the licence wording automatically, so no claim is made here.

## Get the raw data

```bash
python -m vehicle_efficiency.data download          # idempotent; --force to re-download
```

`data/raw/vehicles.csv` is **never edited** (git-ignored, re-downloadable). If the download is blocked, download the zip from the link above and extract `vehicles.csv` into `data/raw/`. Because EPA updates the file regularly, the row count can change over time; record the download date when you report results.

## Scope filter (done in code: `data.prepare`)

Applied before splitting; it learns nothing from the data.

| Rule | Why |
|---|---|
| model year >= 2015 (`MIN_YEAR` in `data.py`) | the requirement "recent vehicles only" |
| drop EV, fuel-cell, plug-in hybrid, CNG/bi-fuel rows (`atvType`) and electricity/hydrogen/natural-gas fuel types | their efficiency is reported in **MPGe**, which is not comparable to MPG |
| kept: gasoline, diesel, flex-fuel and ordinary hybrids | efficiency is real MPG |

Result at time of writing: **13,975 rows**, model years 2015-2027, MPG range 9-59, mean 23.2, **no missing values** in the used columns. 416 rows are exact duplicates of another row on all project columns (kept together in one split).

## Columns used

| Project column | Raw column | Type | Meaning |
|---|---|---|---|
| `mpg` | `comb08` | target | combined MPG (US) |
| `model_year` | `year` | integer | 2015-2027 |
| `cylinders` | `cylinders` | integer | engine cylinders |
| `displacement` | `displ` | continuous | engine displacement in **litres** |
| `drive` | `drive` | categorical | e.g. Front-Wheel Drive, All-Wheel Drive |
| `vehicle_class` | `VClass` | categorical | EPA size class (22 values) |
| `fuel_type` | `fuelType1` | categorical | Regular/Premium/Midgrade Gasoline, Diesel |
| `transmission` | derived from `trany` | categorical | Automatic / Manual / Other |
| `hybrid` | derived from `atvType` | 0/1 | ordinary hybrid |
| `turbo`, `supercharged` | `tCharger`, `sCharger` | 0/1 | forced induction |
| `make`, `model` | `make`, `model` | text | **not features**: kept for grouping and error analysis only |

## Requested features not available

- **Weight** and **horsepower** are **not** in this dataset. This conflicts with the original project brief, which listed them as example features. Cylinders, displacement and model year are available. The team should confirm with the course requirements that this is acceptable.
- `data/README.md` limitations below apply to any conclusions.

## Limitations (Aktore to extend)

- No weight, horsepower or aerodynamic data - a major driver of real-world MPG is missing, so error floors may be high.
- MPG values are **EPA laboratory test ratings**, not measured real-world consumption.
- The file contains many near-identical rows (same model with small variants, same model in consecutive years). A random split can put such near-duplicates in train and validation, which can make scores optimistic. Only *exact* duplicates are grouped.
- EV, plug-in hybrid and fuel-cell vehicles are excluded, so conclusions say nothing about them.
- EPA revises the file; rerunning later may give slightly different data.

## Split design

Random 70/15/15 train/validation/test over records, seed 42. Intended scenario: predict combined MPG for a new, recent vehicle configuration from its specification. Exact duplicate records (identical on all project columns) are grouped (`GroupShuffleSplit`) so they cannot cross splits. A stricter alternative worth discussing in the defense: group by `make`+`model` so whole models are unseen at test time. The assignment (raw `row_id` -> split) is written to `data/processed/split_assignments.csv` when you run the baseline.
