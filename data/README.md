# Data

## Source and access

The source is the U.S. EPA / DOE FuelEconomy.gov **Vehicle** download, `vehicles.csv`:

- Data page and dictionary: https://www.fueleconomy.gov/feg/ws/index.shtml
- Download used by the project: https://www.fueleconomy.gov/feg/epadata/vehicles.csv.zip
- Acquisition: `python -m vehicle_efficiency.data download` downloads the ZIP and extracts the unchanged file to `data/raw/vehicles.csv`. It is intentionally Git-ignored because EPA updates it.
- Accessed for this EDA: 2026-10-08. The service describes the file as covering model years 1984 through the current model year and supplies CSV/XML downloads.

### Licence / usage terms

EPA's [Standard Open Data License](https://edg.epa.gov/EPA_Data_License.htm) says that, unless otherwise specified, EPA-produced data are in the public domain and not subject to domestic copyright protection under 17 U.S.C. §105. Use is therefore permitted for this course project with attribution to EPA/DOE FuelEconomy.gov. The same statement disclaims warranty for accuracy or fitness; users should consider the metadata and limitations. This repository retains the source URL and download procedure rather than redistributing the raw file.

## Project subset

The raw file downloaded on 2026-10-08 has **50,407 rows x 84 columns**. `vehicle_efficiency.data.prepare` retains model year >= 2015 and removes EV, fuel-cell, plug-in-hybrid, CNG/bi-fuel, electricity, hydrogen, and natural-gas entries, because their efficiency can be reported as MPGe rather than comparable gasoline MPG. The resulting modelling frame has **13,975 rows x 14 columns**. The deterministic seed-42 split is approximately 70/15/15; the EDA training partition has **9,816 rows**.

| Project column | Raw column | Type | Meaning / unit |
|---|---|---|---|
| `mpg` | `comb08` | integer, target | EPA combined fuel economy, US miles per gallon (MPG); higher is more efficient |
| `model_year` | `year` | integer | vehicle model year |
| `cylinders` | `cylinders` | numeric | number of engine cylinders |
| `displacement` | `displ` | numeric | engine displacement, litres |
| `drive` | `drive` | categorical | driven axle / drive type |
| `vehicle_class` | `VClass` | categorical | EPA vehicle size class |
| `fuel_type` | `fuelType1` | categorical | primary fuel type |
| `transmission` | derived from `trany` | categorical | Automatic, Manual, or Other |
| `hybrid`, `turbo`, `supercharged` | derived from `atvType`, `tCharger`, `sCharger` | binary | technology flags |
| `make`, `model` | `make`, `model` | text | retained for grouping/error analysis, not modelling features |

EPA defines `comb08` as combined MPG for the primary fuel and `displ` as displacement in litres in its [data dictionary](https://www.fueleconomy.gov/feg/ws/index.shtml). The target is an EPA laboratory rating, not a driver-measured consumption value.

## Data-quality findings for preprocessing

These findings are from `load_split("train")` on the 2026-10-08 download (seed 42), and are reproduced by `notebooks/01_eda.ipynb`.

- All 14 project columns have the expected numeric/string type; no used-column values are missing in the training split.
- There are no exact duplicate rows once `row_id` is included. There are **325** duplicate vehicle records after omitting `row_id`; exact duplicate project records are grouped before splitting, so they do not cross partitions.
- No non-positive MPG, displacement, or cylinder values were found. The observed training ranges are MPG 9--59, displacement 0.9--8.4 L, cylinders 3--16, and model year 2015--2027.
- Rare but plausible values need no automatic deletion: 12 rows have 16 cylinders, 20 have 5 cylinders, 17 have MPG >=55, and 14 have displacement >=8 L. Preserve them; use robust diagnostics/model validation rather than treating them as errors.
- `cylinders` is numeric in the loaded frame but semantically discrete. Khamza should retain the existing documented treatment and keep all imputation/scaling inside the training-fitted pipeline. No cleaning change is requested by this EDA.

## Limitations

- This dataset does **not** provide curb weight or horsepower, so the brief's weight- and horsepower-versus-efficiency plots cannot be made from the selected source. The EDA uses displacement and model year instead.
- Ratings are standardised EPA estimates rather than real-world fuel use; driving behaviour, weather, load, and maintenance are absent.
- The scope excludes EVs, fuel-cell vehicles, plug-in hybrids, and CNG/bi-fuel vehicles. Conclusions must not be extended to them.
- Near-identical variants and repeated models may make a record-level random split optimistic. Exact duplicates are grouped, but a future robustness check could group by make/model.
- EPA may revise records, so row counts and results can change after a later download.
