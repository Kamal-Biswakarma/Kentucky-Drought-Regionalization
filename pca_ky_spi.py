import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import geopandas as gpd
from shapely.ops import polygonize, unary_union
from datetime import date, timedelta
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load extracted KY data ────────────────────────────────────────────────
print("Loading KY SPI data...")
spi_grid = np.load('C:/2_Drought/KY_SPI01_grid.npy')   # (1574, 68, 187)
lats     = np.load('C:/2_Drought/KY_SPI01_lats.npy')
lons     = np.load('C:/2_Drought/KY_SPI01_lons.npy')
mask     = np.load('C:/2_Drought/KY_SPI01_mask.npy')   # (68, 187) bool

n_times, n_lat, n_lon = spi_grid.shape
n_pts = mask.sum()
print(f"  Grid shape: {spi_grid.shape}  |  KY cells: {n_pts}")

# Build date index
import h5py
with h5py.File('C:/2_Drought/nclimgrid-spi-gamma-01.nc', 'r') as f:
    time_vals = f['time'][:]
base = date(1800, 1, 1)
dates = pd.to_datetime([base + timedelta(days=int(d)) for d in time_vals])

# ── 2. Build 2D matrix: (time × KY_cells) ───────────────────────────────────
print("Building data matrix...")
X = spi_grid.reshape(n_times, n_lat * n_lon)[:, mask.ravel()]  # (1574, 6158)
print(f"  Matrix shape: {X.shape}")

# Check for any remaining NaN columns (should be none inside KY)
nan_cols = np.isnan(X).any(axis=0)
print(f"  Columns with any NaN: {nan_cols.sum()}")
if nan_cols.sum() > 0:
    X = X[:, ~nan_cols]
    print(f"  After dropping NaN cols: {X.shape}")

# ── 3. Standardize (zero-mean each grid cell, unit variance) ─────────────────
# SPI is ~N(0,1) already but spatial mean removal reveals pattern differences
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # (time, cells): standardize across time for each cell

# ── 4. Run PCA ───────────────────────────────────────────────────────────────
print("Running PCA...")
n_components = 30
pca = PCA(n_components=n_components, random_state=42)
scores = pca.fit_transform(X_scaled)  # (time, n_components) — PC time series
loadings = pca.components_            # (n_components, cells) — spatial patterns

var_exp   = pca.explained_variance_ratio_ * 100
cum_var   = np.cumsum(var_exp)
print(f"  Variance explained by first 10 PCs: {cum_var[9]:.1f}%")
print(f"  Variance explained by first 15 PCs: {cum_var[14]:.1f}%")
print(f"  Variance explained by first 20 PCs: {cum_var[19]:.1f}%")

# ── 5. Load KY boundary for maps ─────────────────────────────────────────────
ky_shp  = gpd.read_file('C:/2_Drought/Shapefiles/Ky_State_Boundary.shp').to_crs(epsg=4326)
ky_poly = unary_union(list(polygonize(ky_shp.union_all())))
ky_gdf  = gpd.GeoDataFrame(geometry=[ky_poly], crs='EPSG:4326')

# Helper: place loading values back onto the 2D KY grid
def loadings_to_grid(loading_vec):
    grid = np.full(mask.shape, np.nan)
    # account for NaN columns dropped
    valid_idx = np.where(mask.ravel())[0]
    if nan_cols.sum() > 0:
        valid_idx = valid_idx[~nan_cols]
    grid.ravel()[valid_idx] = loading_vec
    return grid

lon_grid, lat_grid = np.meshgrid(lons, lats)

# ── 6. FIGURE 1: Scree plot ───────────────────────────────────────────────────
print("Plotting scree plot...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('PCA on Kentucky SPI-01 (1895–2026)\nScree Analysis', fontsize=14, fontweight='bold')

ax = axes[0]
ax.bar(range(1, n_components+1), var_exp, color='steelblue', edgecolor='white', linewidth=0.5)
ax.axvline(x=10.5, color='red', linestyle='--', linewidth=1.2, label='K=10 (prev. result)')
ax.set_xlabel('Principal Component', fontsize=12)
ax.set_ylabel('Variance Explained (%)', fontsize=12)
ax.set_title('Individual Variance Explained', fontsize=12)
ax.set_xticks(range(1, n_components+1, 2))
ax.legend()
for i, v in enumerate(var_exp[:15]):
    ax.text(i+1, v+0.1, f'{v:.1f}', ha='center', va='bottom', fontsize=7)

ax = axes[1]
ax.plot(range(1, n_components+1), cum_var, 'o-', color='steelblue', linewidth=2, markersize=5)
ax.axhline(y=70, color='green',  linestyle='--', linewidth=1, label='70% threshold')
ax.axhline(y=80, color='orange', linestyle='--', linewidth=1, label='80% threshold')
ax.axhline(y=90, color='red',    linestyle='--', linewidth=1, label='90% threshold')
ax.axvline(x=10.5, color='gray', linestyle='--', linewidth=1.2, label='K=10')
ax.set_xlabel('Number of Components', fontsize=12)
ax.set_ylabel('Cumulative Variance Explained (%)', fontsize=12)
ax.set_title('Cumulative Variance Explained', fontsize=12)
ax.set_xticks(range(1, n_components+1, 2))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Annotate key thresholds
for thresh, col in [(70,'green'),(80,'orange'),(90,'red')]:
    idx = np.where(cum_var >= thresh)[0]
    if len(idx):
        n = idx[0] + 1
        ax.annotate(f'PC{n}', xy=(n, thresh), xytext=(n+1, thresh-3),
                    fontsize=8, color=col,
                    arrowprops=dict(arrowstyle='->', color=col, lw=1))

plt.tight_layout()
plt.savefig('C:/2_Drought/Fig_PCA_Scree.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: Fig_PCA_Scree.png")

# ── 7. FIGURE 2: Spatial loadings maps PC1–PC8 ───────────────────────────────
print("Plotting PC loadings maps...")
n_maps = 8
ncols  = 4
nrows  = 2
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 8))
fig.suptitle('Kentucky SPI-01 PCA — Spatial Loadings (PC1–PC8)', fontsize=14, fontweight='bold')

cmap = plt.cm.RdBu_r
for pc_idx in range(n_maps):
    ax = axes[pc_idx // ncols, pc_idx % ncols]
    grid = loadings_to_grid(loadings[pc_idx])
    vmax = np.nanpercentile(np.abs(grid), 98)
    im = ax.pcolormesh(lon_grid, lat_grid, grid,
                       cmap=cmap, vmin=-vmax, vmax=vmax, shading='auto')
    ky_gdf.boundary.plot(ax=ax, color='black', linewidth=0.8)
    ax.set_title(f'PC{pc_idx+1}  ({var_exp[pc_idx]:.1f}%)', fontsize=11, fontweight='bold')
    ax.set_xlim(lons.min()-0.1, lons.max()+0.1)
    ax.set_ylim(lats.min()-0.1, lats.max()+0.1)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

plt.tight_layout()
plt.savefig('C:/2_Drought/Fig_PCA_Loadings_PC1-8.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: Fig_PCA_Loadings_PC1-8.png")

# ── 8. FIGURE 3: PC time series (scores) for PC1–PC6 ─────────────────────────
print("Plotting PC scores timeseries...")
fig, axes = plt.subplots(6, 1, figsize=(14, 14), sharex=True)
fig.suptitle('Kentucky SPI-01 PCA — PC Scores (1895–2026)', fontsize=14, fontweight='bold')

for i, ax in enumerate(axes):
    s = scores[:, i]
    ax.fill_between(dates, s, 0, where=(s >= 0), color='steelblue', alpha=0.7, linewidth=0)
    ax.fill_between(dates, s, 0, where=(s < 0),  color='firebrick',  alpha=0.7, linewidth=0)
    ax.axhline(0, color='black', linewidth=0.6)
    # Running mean
    rm = pd.Series(s, index=dates).rolling('3650D').mean()
    ax.plot(dates, rm, color='black', linewidth=1.2, label='10-yr smooth')
    ax.set_ylabel(f'PC{i+1}\n({var_exp[i]:.1f}%)', fontsize=9)
    ax.set_ylim(-max(abs(s.max()), abs(s.min()))*1.1, max(abs(s.max()), abs(s.min()))*1.1)
    ax.grid(True, alpha=0.2, axis='y')

axes[-1].set_xlabel('Year', fontsize=11)
plt.tight_layout()
plt.savefig('C:/2_Drought/Fig_PCA_Scores_PC1-6.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: Fig_PCA_Scores_PC1-6.png")

# ── 9. Print summary table ────────────────────────────────────────────────────
print("\n=== PCA SUMMARY ===")
print(f"{'PC':<5} {'Var%':<8} {'Cum%':<8}")
print("-" * 21)
for i in range(20):
    print(f"PC{i+1:<3} {var_exp[i]:<8.2f} {cum_var[i]:<8.2f}")

# Save loadings/scores for clustering
np.save('C:/2_Drought/KY_PCA_scores.npy',   scores)    # (1574, 30)
np.save('C:/2_Drought/KY_PCA_loadings.npy', loadings)  # (30, n_cells)
np.save('C:/2_Drought/KY_PCA_varexp.npy',   var_exp)

print("\nSaved: KY_PCA_scores.npy, KY_PCA_loadings.npy, KY_PCA_varexp.npy")
print("\nDone. Review scree plot to decide number of PCs before clustering.")
