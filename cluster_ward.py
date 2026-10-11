import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
import geopandas as gpd
from shapely.ops import polygonize, unary_union
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load PCA results ───────────────────────────────────────────────────────
print("Loading PCA results...")
loadings  = np.load('C:/2_Drought/KY_PCA_loadings.npy')   # (30, n_cells)
var_exp   = np.load('C:/2_Drought/KY_PCA_varexp.npy')      # (30,)
mask      = np.load('C:/2_Drought/KY_SPI01_mask.npy')      # (68, 187)
lats      = np.load('C:/2_Drought/KY_SPI01_lats.npy')
lons      = np.load('C:/2_Drought/KY_SPI01_lons.npy')

# Account for the 28 NaN columns dropped during PCA
spi_grid  = np.load('C:/2_Drought/KY_SPI01_grid.npy')
n_times, n_lat, n_lon = spi_grid.shape
X_raw = spi_grid.reshape(n_times, n_lat * n_lon)[:, mask.ravel()]
nan_cols = np.isnan(X_raw).any(axis=0)   # (6158,) — 28 True
valid_mask_flat = mask.ravel().copy()
# valid_mask_flat has 6158 True positions; 28 of those are NaN cols
# loadings shape is (30, 6130) — matching the 6130 non-NaN cells

# ── 2. Compute correlation loadings (Leathers method) ─────────────────────────
# corr_loading[j, i] = components_[j,i] * sqrt(eigenvalue_j)
# eigenvalue_j = explained_variance_ratio_j * total_variance
# For standardized data: total_variance = n_cells = 6130
n_cells = loadings.shape[1]
eigenvalues = var_exp[:loadings.shape[0]] / 100 * n_cells
corr_loadings = loadings * np.sqrt(eigenvalues[:, None])   # (30, 6130)

print(f"  Correlation loadings shape: {corr_loadings.shape}")
print(f"  PC1 loading range: {corr_loadings[0].min():.3f} to {corr_loadings[0].max():.3f}")

# ── 3. Select number of PCs based on scree (elbow at PC3-4, retain PC1-5) ─────
N_PCS = 5
L = corr_loadings[:N_PCS, :].T   # (6130, 5) — each row = one grid cell's loadings
print(f"\nUsing PC1–PC{N_PCS} for clustering  (cumulative var: {var_exp[:N_PCS].sum():.1f}%)")
print(f"  Loading matrix for clustering: {L.shape}")

# ── 4. Standardize loadings so each PC contributes equally ────────────────────
from sklearn.preprocessing import StandardScaler
L_scaled = StandardScaler().fit_transform(L)

# ── 5. Ward's linkage ─────────────────────────────────────────────────────────
print("\nComputing Ward's linkage (this may take ~1 min for 6130 cells)...")
Z = linkage(L_scaled, method='ward', metric='euclidean')
print("  Done.")

# Merge distances (last 30 merges — helpful for finding optimal K)
merge_dist = Z[:, 2]
last30 = merge_dist[-30:]
acceleration = np.diff(last30, 2)   # second derivative
print(f"\n  Last 10 merge distances: {last30[-10:].round(3)}")

# ── 6. FIGURE 1: Dendrogram + acceleration ────────────────────────────────────
print("\nPlotting dendrogram...")
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle("Ward's Linkage Dendrogram — Kentucky SPI-01 (PC1–5)", fontsize=14, fontweight='bold')

# Truncated dendrogram (last 50 merges)
ax = axes[0]
dendrogram(Z, ax=ax, truncate_mode='lastp', p=50, leaf_rotation=90,
           color_threshold=0, above_threshold_color='steelblue')
ax.set_title('Truncated Dendrogram (last 50 merges)', fontsize=12)
ax.set_xlabel('Cluster (size)', fontsize=10)
ax.set_ylabel('Ward Linkage Distance', fontsize=10)

# Draw horizontal lines at candidate K cuts
cuts = {6: 'red', 8: 'darkorange', 10: 'green', 12: 'purple', 14: 'brown'}
n_total = len(Z) + 1
for k, col in cuts.items():
    # distance at which we'd have k clusters = merge_dist[-(k-1)-1]
    cut_d = merge_dist[-(k - 1)]
    ax.axhline(cut_d, color=col, linestyle='--', linewidth=1.2, label=f'K={k}  (d={cut_d:.2f})')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

# Acceleration plot (second derivative of merge distances)
ax2 = axes[1]
x_vals = np.arange(len(merge_dist)-2, 0, -1) + 2   # K values corresponding to acceleration
acc_plot = np.diff(merge_dist, 2)
ax2.bar(range(len(acc_plot[-28:])), acc_plot[-28:][::-1],
        color=['firebrick' if v > 0 else 'steelblue' for v in acc_plot[-28:][::-1]])
k_labels = list(range(3, 31))
ax2.set_xticks(range(len(k_labels)))
ax2.set_xticklabels([f'K={k}' for k in k_labels], rotation=45, fontsize=8)
ax2.set_title('Acceleration of Merge Distances (peaks suggest optimal K)', fontsize=11)
ax2.set_ylabel('2nd Derivative of Linkage Distance', fontsize=10)
ax2.axhline(0, color='black', linewidth=0.8)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('C:/2_Drought/Fig_Ward_Dendrogram.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: Fig_Ward_Dendrogram.png")

# ── 7. Map multiple cluster solutions ─────────────────────────────────────────
print("\nGenerating cluster maps for K = 6, 8, 10, 12, 14...")

# Load KY boundary
ky_shp  = gpd.read_file('C:/2_Drought/Shapefiles/Ky_State_Boundary.shp').to_crs(epsg=4326)
ky_poly = unary_union(list(polygonize(ky_shp.union_all())))
ky_gdf  = gpd.GeoDataFrame(geometry=[ky_poly], crs='EPSG:4326')

lon_grid, lat_grid = np.meshgrid(lons, lats)

# Map PCA-valid cells back to the 68×187 grid
# valid_mask: mask (6158 True in 68×187) minus the 28 NaN columns
ky_flat_idx = np.where(mask.ravel())[0]           # 6158 indices in flat 68×187
valid_flat_idx = ky_flat_idx[~nan_cols]            # 6130 indices (no NaN cols)

def clusters_to_grid(labels):
    grid = np.full(mask.shape, np.nan)
    grid.ravel()[valid_flat_idx] = labels
    return grid

k_list   = [6, 8, 10, 12, 14]
all_labels = {}
for k in k_list:
    all_labels[k] = fcluster(Z, k, criterion='maxclust')

# Use a qualitative colormap with enough colors
cmap14 = plt.cm.get_cmap('tab20', 14)

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle("Ward's Linkage Cluster Solutions — Kentucky SPI-01 (PC1–5)",
             fontsize=14, fontweight='bold')
axes = axes.ravel()

for idx, k in enumerate(k_list):
    ax = axes[idx]
    labels = all_labels[k]
    grid   = clusters_to_grid(labels)
    cmap_k = plt.cm.get_cmap('tab20', k)
    im = ax.pcolormesh(lon_grid, lat_grid, grid,
                       cmap=cmap_k, vmin=0.5, vmax=k + 0.5, shading='auto')
    ky_gdf.boundary.plot(ax=ax, color='black', linewidth=1.0)
    ax.set_title(f'K = {k} regions', fontsize=12, fontweight='bold')
    ax.set_xlim(lons.min()-0.1, lons.max()+0.1)
    ax.set_ylim(lats.min()-0.1, lats.max()+0.1)
    ax.set_xticks([]); ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_ticks(range(1, k+1))

axes[-1].set_visible(False)
plt.tight_layout()
plt.savefig('C:/2_Drought/Fig_Ward_ClusterMaps.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: Fig_Ward_ClusterMaps.png")

# ── 8. Within-cluster variance (WCSS equivalent) for each K ──────────────────
print("\nComputing within-cluster variance for K = 3 to 15...")
wcv = []
k_range = range(3, 16)
for k in k_range:
    labels = fcluster(Z, k, criterion='maxclust')
    wc = 0.0
    for c in range(1, k+1):
        pts = L_scaled[labels == c]
        if len(pts) > 1:
            wc += np.sum((pts - pts.mean(axis=0))**2)
    wcv.append(wc)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(list(k_range), wcv, 'o-', color='steelblue', linewidth=2, markersize=7)
for k, col in [(6,'red'),(8,'darkorange'),(10,'green'),(12,'purple')]:
    ax.axvline(k, color=col, linestyle='--', linewidth=1, alpha=0.7, label=f'K={k}')
ax.set_xlabel('Number of Clusters (K)', fontsize=12)
ax.set_ylabel('Within-Cluster Variance', fontsize=12)
ax.set_title("Within-Cluster Variance vs K — Ward's Linkage", fontsize=13, fontweight='bold')
ax.set_xticks(list(k_range))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('C:/2_Drought/Fig_Ward_WCV.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: Fig_Ward_WCV.png")

# ── 9. Sensitivity test: vary N_PCS (PC1-4, PC1-5, PC1-6) ────────────────────
print("\nSensitivity test: comparing K=10 solution for PC1-4, PC1-5, PC1-6...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Sensitivity Test — K=10 Clusters for Different PC Retentions",
             fontsize=13, fontweight='bold')

for idx, n_pcs in enumerate([4, 5, 6]):
    L_test = corr_loadings[:n_pcs, :].T
    L_test_sc = StandardScaler().fit_transform(L_test)
    Z_test = linkage(L_test_sc, method='ward', metric='euclidean')
    lbl = fcluster(Z_test, 10, criterion='maxclust')
    grid = clusters_to_grid(lbl)
    cmap_k = plt.cm.get_cmap('tab20', 10)
    ax = axes[idx]
    im = ax.pcolormesh(lon_grid, lat_grid, grid,
                       cmap=cmap_k, vmin=0.5, vmax=10.5, shading='auto')
    ky_gdf.boundary.plot(ax=ax, color='black', linewidth=1.0)
    ax.set_title(f'PC1–{n_pcs}  ({var_exp[:n_pcs].sum():.1f}% var)', fontsize=12, fontweight='bold')
    ax.set_xlim(lons.min()-0.1, lons.max()+0.1)
    ax.set_ylim(lats.min()-0.1, lats.max()+0.1)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

plt.tight_layout()
plt.savefig('C:/2_Drought/Fig_Ward_Sensitivity_PCs.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: Fig_Ward_Sensitivity_PCs.png")

# ── 10. Print summary ─────────────────────────────────────────────────────────
print("\n=== WITHIN-CLUSTER VARIANCE SUMMARY ===")
print(f"{'K':<5} {'WCV':<12} {'% drop from K-1'}")
print("-"*35)
for i, (k, w) in enumerate(zip(k_range, wcv)):
    drop = f"{(wcv[i-1]-w)/wcv[i-1]*100:.1f}%" if i > 0 else "—"
    print(f"{k:<5} {w:<12.1f} {drop}")

# Save K=10 labels for further analysis
labels_k10 = fcluster(Z, 10, criterion='maxclust')
np.save('C:/2_Drought/KY_Ward_K10_labels.npy', labels_k10)
np.save('C:/2_Drought/KY_Ward_linkage.npy', Z)
print("\nSaved: KY_Ward_K10_labels.npy, KY_Ward_linkage.npy")
print("\nDone. Review figures to choose optimal K.")
