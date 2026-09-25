├── Plot_code/
├── Data/
│   ├── Raw/
│   ├── Intermediate/
│   └── Processed/
└── Results/
```

The scripts use paths relative to the repository root. The root is normally detected automatically from the script location. It can also be specified explicitly with the `TC_RW_PROJECT_ROOT` environment variable:

```bash
export TC_RW_PROJECT_ROOT=/path/to/TC-RW-rainfall-outward
```

### Two reproduction options

1. **Reproduce figures and tables only.** Download the processed data from Zenodo and run the scripts in `Plot_code/`. This is the recommended route for reproducing the results presented in the manuscript.
2. **Reproduce the complete processing workflow.** Obtain the required source datasets from their original providers, place them under the corresponding `Data/Raw/` directories, and run the numbered scripts in `Cal_code/`. Some extraction steps are computationally and storage intensive.

## Software environment

The workflow was developed in Python. Python 3.9 or later is recommended. The principal dependencies include:

- NumPy
- pandas
- SciPy
- xarray
- netCDF4
- h5netcdf
- Matplotlib
- GeoPandas
- Shapely
- pyproj
- tqdm

A suitable environment can be created with Conda, for example:

```bash
conda create -n tc-rw python=3.9 numpy pandas scipy xarray netcdf4 h5netcdf \
  matplotlib geopandas shapely pyproj tqdm
conda activate tc-rw
```

Package versions used for the archived analyses should be recorded in an accompanying environment file before the final repository release.

## Running the code

Run commands from the repository root. For example:

```bash
cd /path/to/TC-RW-rainfall-outward
python Cal_code/MSWEP/RUN1_EXTRACT_MSWEP_500KM.py
```

Continue with the numbered scripts in the same analysis directory. Separate workflows are provided for the rainfall products, IBTrACS, ERA5 environmental variables, atmospheric diagnostics, population exposure, and sensitivity analyses.

To reproduce a figure or table directly from the processed Zenodo data, run the corresponding script under `Plot_code/`. For example:

```bash
python Plot_code/Main_figures/MAIN_FIG1AB.py
python Plot_code/Extended_tables/EXTENDED_DATA_TABLE1.py
```

Generated products are written below `Results/`. Plotting scripts do not modify the archived input data.

## Main datasets

The workflow uses the following principal data products:

- International Best Track Archive for Climate Stewardship (IBTrACS)
- Multi-Source Weighted-Ensemble Precipitation (MSWEP)
- Integrated Multi-satellitE Retrievals for GPM (IMERG)
- Climate Prediction Center morphing technique precipitation (CMORPH)
- ERA5 reanalysis
- GSHHG coastline data
- LandScan population data

Users who rerun the complete workflow are responsible for obtaining these source datasets from their official providers and complying with their respective licences and terms of use. Processed inputs needed for figure and table reproduction are supplied through the Zenodo record where redistribution is permitted.

## Reproducibility notes

- Twenty-four-hour changes are generally calculated as the endpoint value minus the starting-point value.
- Rapid-weakening and rapid-intensification samples are identified using the intensity-change definitions implemented in the analysis scripts.
- Random procedures use fixed seeds where specified in the corresponding scripts.
- Quality-control records are written below `Results/Quality_control/`.
- Intermediate products can be large and are therefore not tracked in the Git repository.

## Citation

If you use this code or the archived data, please cite both the manuscript and the Zenodo record:

*Rapid weakening of tropical cyclones shifts rainfall outward*. Manuscript citation to be updated after publication.

> Data and code-supporting archive: [https://doi.org/10.5281/zenodo.22943232](https://doi.org/10.5281/zenodo.22943232)

The DOI above should be used in preference to a direct Zenodo file URL because it provides a persistent link to the archived record.

## Licence

Add the selected software licence as a `LICENSE` file before public release. The licence for this repository applies only to the original code and documentation; third-party datasets retain their own licences and terms of use.

## Contact

For questions about the code or data, please open a GitHub issue or contact the corresponding author listed in the manuscript.
