# Kentucky Drought Regionalization

This repository contains Python code for regionalizing meteorological drought patterns across Kentucky using 130 years (1895-2025) of monthly Standardized Precipitation Index (SPI) data from NOAA's nClimGrid dataset. Principal Component Analysis (PCA) is applied to reduce the dimensionality of the gridded SPI data, followed by Ward's hierarchical clustering to identify six spatially coherent drought regions. Regional trend analysis (Mann-Kendall tests), seasonal climatologies, wavelet analysis, and drought event characterization are then performed to quantify how drought behavior varies across the state.

## Data

The raw nClimGrid NetCDF files and shapefiles are not included due to file size. SPI data can be downloaded from [NOAA NCEI](https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/).

## Requirements

- Python 3.8+
- numpy, pandas, matplotlib, scipy, scikit-learn
- geopandas, shapely, h5py
- pymannkendall, pywt

## Author

Kamal Biswakarma
