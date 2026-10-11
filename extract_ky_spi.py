import h5py
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import polygonize, unary_union
from datetime import date, timedelta
import pandas as pd

print("Loading Kentucky boundary...")
ky = gpd.read_file('C:/2_Drought/Shapefiles/Ky_State_Boundary.shp')
print(f"  CRS: {ky.crs}")

# Reproject to WGS84 (lon/lat) to match the SPI grid
ky_wgs84 = ky.to_crs(epsg=4326)
lines = ky_wgs84.union_all()
# Shapefile contains boundary lines, not polygons — polygonize to get filled polygon
ky_geom = unary_union(list(polygonize(lines)))
print(f"  Geometry type: {ky_geom.geom_type}")
print(f"  KY bounds: {ky_geom.bounds}")

print("\nOpening SPI NetCDF...")
path = 'C:/2_Drought/nclimgrid-spi-gamma-01.nc'
with h5py.File(path, 'r') as f:
    lats = f['lat'][:]   # shape (596,)
    lons = f['lon'][:]   # shape (1385,)
    times = f['time'][:] # shape (1574,)

    # KY bounding box to subset grid before point-in-polygon
    minx, miny, maxx, maxy = ky_geom.bounds
    lat_mask = (lats >= miny - 0.1) & (lats <= maxy + 0.1)
    lon_mask = (lons >= minx - 0.1) & (lons <= maxx + 0.1)

    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]

    sub_lats = lats[lat_idx]
    sub_lons = lons[lon_idx]

    print(f"  Bounding box subset: {len(lat_idx)} lats x {len(lon_idx)} lons")

    # Build grid points and find those inside KY polygon
    print("  Finding grid points inside Kentucky...")
    lon_grid, lat_grid = np.meshgrid(sub_lons, sub_lats)
    pts = [Point(lo, la) for lo, la in zip(lon_grid.ravel(), lat_grid.ravel())]

    inside = np.array([ky_geom.contains(p) for p in pts])
    inside = inside.reshape(len(lat_idx), len(lon_idx))

    n_pts = inside.sum()
    print(f"  Grid points inside KY: {n_pts}")

    # Extract SPI for those points across all time steps
    print("  Extracting SPI data (this may take a moment for 1574 months)...")
    # Read full spatial subset, then mask
    # spi shape: (time, lat, lon)
    lat_start, lat_end = lat_idx[0], lat_idx[-1] + 1
    lon_start, lon_end = lon_idx[0], lon_idx[-1] + 1

    spi_sub = f['spi_01'][:, lat_start:lat_end, lon_start:lon_end]  # (1574, nlat, nlon)
    fill_val = -999.9

    # Replace fill with NaN
    spi_sub = spi_sub.astype(np.float32)
    spi_sub[spi_sub <= fill_val + 1] = np.nan

    # Apply KY mask: set non-KY points to NaN
    spi_sub[:, ~inside] = np.nan

# Convert time
base = date(1800, 1, 1)
dates = [base + timedelta(days=int(d)) for d in times]

# Compute spatial mean SPI for KY at each time step
ky_spi_mean = np.nanmean(spi_sub.reshape(len(times), -1), axis=1)
ky_spi_std  = np.nanstd(spi_sub.reshape(len(times), -1), axis=1)
ky_spi_min  = np.nanmin(spi_sub.reshape(len(times), -1), axis=1)
ky_spi_max  = np.nanmax(spi_sub.reshape(len(times), -1), axis=1)

df = pd.DataFrame({
    'date': dates,
    'spi_01_mean': ky_spi_mean,
    'spi_01_std':  ky_spi_std,
    'spi_01_min':  ky_spi_min,
    'spi_01_max':  ky_spi_max,
})
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

out_csv = 'C:/2_Drought/KY_SPI01_monthly.csv'
df.to_csv(out_csv)
print(f"\nSaved mean SPI timeseries to: {out_csv}")
print(df.head(10))
print(f"\nShape: {df.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print(f"Mean SPI stats:\n{df['spi_01_mean'].describe()}")

# Also save the full gridded KY SPI as numpy arrays for further analysis
print("\nSaving gridded arrays...")
np.save('C:/2_Drought/KY_SPI01_grid.npy', spi_sub)
np.save('C:/2_Drought/KY_SPI01_lats.npy', sub_lats)
np.save('C:/2_Drought/KY_SPI01_lons.npy', sub_lons)
np.save('C:/2_Drought/KY_SPI01_mask.npy', inside)

print("Saved gridded arrays:")
print(f"  KY_SPI01_grid.npy  shape: {spi_sub.shape}  (time x lat x lon)")
print(f"  KY_SPI01_lats.npy  shape: {sub_lats.shape}")
print(f"  KY_SPI01_lons.npy  shape: {sub_lons.shape}")
print(f"  KY_SPI01_mask.npy  shape: {inside.shape}  (KY boolean mask)")
print("\nDone.")
