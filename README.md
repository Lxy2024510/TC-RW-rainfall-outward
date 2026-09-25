# TC-RW-rainfall-outward

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22943232.svg)](https://doi.org/10.5281/zenodo.22943232)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code supporting the manuscript **“Rapid weakening of tropical cyclones shifts rainfall outward.”**

This repository contains the calculation, statistical-analysis, and plotting code used to investigate changes in the radial distribution of tropical-cyclone (TC) heavy rainfall during rapid weakening.

## 1. Download the data

The data and archived results are distributed separately through Zenodo:

> **Zenodo data record:** [https://doi.org/10.5281/zenodo.22943232](https://doi.org/10.5281/zenodo.22943232)

Download the following two archives from the Zenodo record:

- `Data.rar` — input data required by the released calculation and plotting scripts;
- `Results.rar` — archived figures, tables, and quality-control outputs.

Place both archives in the root directory of this repository and extract them without changing their internal directory structure:

```bash
cd /path/to/TC-RW-rainfall-outward
7z x Data.rar
7z x Results.rar
```

Alternatively, with `unrar`:

```bash
unrar x Data.rar
unrar x Results.rar
```

After extraction, `Data/` and `Results/` must be at the same directory level as `Cal_code/` and `Plot_code/`.

## 2. Repository structure

```text
TC-RW-rainfall-outward/
├── Cal_code/                 # Data processing and statistical-analysis scripts
├── Plot_code/                # Main figures, Extended Data figures, and tables
├── Data/                     # Downloaded from Zenodo; not stored in Git
│   ├── Raw/
│   ├── Intermediate/
│   └── Processed/
├── Results/                  # Downloaded from Zenodo or regenerated locally
├── environment.yml           # Portable Conda environment specification
├── package_versions.txt      # Reference package-version inventory
├── LICENSE                   # MIT License
└── README.md
```

`Data/`, `Results/`, `Data.rar`, and `Results.rar` are intentionally excluded from Git because they are distributed through Zenodo.

## 3. Software environment

The archived workflow was run with:

- Python 3.9.21
- pip 25.0

The recommended way to recreate the software environment is:

```bash
conda env create -f environment.yml
conda activate xin
```

The main Python packages include NumPy, pandas, SciPy, xarray, netCDF4, h5netcdf, Matplotlib, GeoPandas, Shapely, pyproj, Cartopy, and tqdm. See `environment.yml` and `package_versions.txt` for the environment specification and package inventory.

## 4. Reproducing the figures and tables

For most users, the recommended procedure is to download `Data.rar` from Zenodo and run only the scripts under `Plot_code/`. The computationally expensive raw-data extraction steps do not need to be repeated.

Run all commands from the repository root:

```bash
cd /path/to/TC-RW-rainfall-outward
```

Examples:

```bash
python Plot_code/Main_figures/MAIN_FIG1AB.py
python Plot_code/Main_figures/MAIN_FIG1C.py
python Plot_code/Extended_tables/EXTENDED_DATA_TABLE1.py
```

Generated figures and tables are written below `Results/`. The plotting scripts read from `Data/` and do not modify the archived input files.

## 5. Reproducing the calculation workflow

The complete calculation workflow is provided under `Cal_code/`. Scripts beginning with `RUN1`, `RUN2`, and so forth should generally be run in ascending numerical order within each subdirectory.

The main processing sequence is:

```text
IBTrACS preparation
        ↓
Rainfall extraction and 24-h window construction
        ↓
Environmental and dynamical-variable processing
        ↓
Bootstrap and other statistical analyses
        ↓
Population-exposure analysis
        ↓
Figure and table generation
```

The principal calculation modules include:

- `Cal_code/IBTrACS/` — best-track preparation, coastline distance, and TC motion;
- `Cal_code/MSWEP/` — primary rainfall analysis;
- `Cal_code/IMERG/` and `Cal_code/CMORPH/` — rainfall-product sensitivity analyses;
- `Cal_code/ERA5/` — environmental variables and atmospheric diagnostics, including SST, RH, VWS, vertical motion, convergence, and radial mass flux;
- `Cal_code/Exposure/` — population-exposure analysis.

Some raw-data extraction steps require substantial disk space, memory, and processing time. The processed data supplied through Zenodo are sufficient for reproducing the published figures and tables.

## 6. Project-root detection

The scripts use paths relative to the repository root and normally detect that root from the script location. If the repository is moved or called from another working directory, the root can be specified explicitly:

```bash
export TC_RW_PROJECT_ROOT=/path/to/TC-RW-rainfall-outward
```

For the directory layout used in the archived workflow, for example:

```bash
export TC_RW_PROJECT_ROOT=/mnt/e/data/TC-RW-V1
```

## 7. Main datasets

The analyses use the following principal datasets:

- International Best Track Archive for Climate Stewardship (IBTrACS);
- Multi-Source Weighted-Ensemble Precipitation (MSWEP);
- Integrated Multi-satellitE Retrievals for GPM (IMERG);
- Climate Prediction Center morphing technique precipitation (CMORPH);
- ERA5 reanalysis;
- GSHHG coastline data;
- LandScan population data.

Users who rerun the workflow from the original source data are responsible for obtaining those datasets from their official providers and complying with their respective licences and terms of use. The Zenodo archive contains the data released for reproducing the analyses, figures, and tables in this repository.

## 8. Reproducibility notes

- The analysis period is 1982–2024 unless a script explicitly specifies a different period.
- Twenty-four-hour changes are calculated as the endpoint value minus the starting-point value unless otherwise stated.
- Rapid-weakening and rapid-intensification samples are selected using the intensity-change definitions implemented in the corresponding scripts.
- Bootstrap procedures use 5,000 repetitions unless otherwise stated.
- Fixed random seeds are used where specified by the scripts.
- Quality-control information is written below `Results/Quality_control/`.
- Large intermediate products and temporary caches are not tracked by Git.

## 9. Citation

If you use the code, data, or results, please cite both the manuscript and the Zenodo record. The complete manuscript citation will be added after publication.

> **Data and results archive:** [https://doi.org/10.5281/zenodo.22943232](https://doi.org/10.5281/zenodo.22943232)

The DOI link should be used instead of a direct file URL because it provides a persistent reference to the archived record.

## 10. License

The original code in this repository is released under the [MIT License](LICENSE). Third-party datasets remain subject to the licences and terms of their respective providers.

## 11. Contact

Questions and reproducibility issues can be submitted through the GitHub issue tracker. Corresponding-author information will be added with the final manuscript citation.
