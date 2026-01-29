"""
Plot FIM results from rooney_biegler_fim_results.json
Creates a 3D surface plot with log10(det) vs hour1 and hour2
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Read the results file from the same directory as this script
with open(script_dir / 'rooney_biegler_fim_verification.json', 'r') as f:
    data = json.load(f)

# Convert results to DataFrame
results_df = pd.DataFrame(data['results'])

# Filter valid results (where log10_det is not None)
valid_df = results_df[results_df['log10_det'].notna()].copy()

print(f"Loaded {len(results_df)} results")
print(f"Valid results: {len(valid_df)}")

# Find the best design
best_idx = valid_df['log10_det'].idxmax()
best = valid_df.loc[best_idx]
print(f"\nBest design:")
print(f"  Hour1: {best['hour1']:.2f}")
print(f"  Hour2: {best['hour2']:.2f}")
print(f"  log10(det): {best['log10_det']:.4f}")

# Create the 3D plot
fig = plt.figure(figsize=(14, 10))

# 3D surface plot
ax1 = fig.add_subplot(121, projection='3d')

# Get unique hour values and create grid
hour1_unique = np.unique(results_df['hour1'])
hour2_unique = np.unique(results_df['hour2'])
n_hours = len(hour1_unique)

print(f"\nGrid size: {n_hours} x {n_hours}")

# Create meshgrid for 3D surface
H1, H2 = np.meshgrid(hour1_unique, hour2_unique)

# Create Z values grid (use all results, fill invalid with NaN)
Z = np.full(H1.shape, np.nan)
for idx, row in results_df.iterrows():
    i = np.where(hour2_unique == row['hour2'])[0][0]
    j = np.where(hour1_unique == row['hour1'])[0][0]
    Z[i, j] = row['log10_det'] if row['log10_det'] is not None else np.nan

# Surface plot (matplotlib handles NaN gracefully)
surf = ax1.plot_surface(H1, H2, Z, cmap='viridis', alpha=0.8, edgecolor='none')

# Mark the best point
ax1.scatter(
    [best['hour1']],
    [best['hour2']],
    [best['log10_det']],
    color='red',
    s=250,
    marker='*',
    edgecolors='black',
    linewidths=2,
    alpha=0.9,
    label=f'GridBest: ({best["hour1"]:.2f}, {best["hour2"]:.2f})',
)

# Mark the optimal point from optimization
ax1.scatter(
    [1.9322],
    [10],
    [6.2256],
    color='cyan',
    s=200,
    marker='o',
    edgecolors='blue',
    linewidths=2,
    alpha=0.95,
    label=f'Optimal: (1.9322, 10.00)',
)
ax1.set_xlabel('Hour 1', fontsize=12, labelpad=10)
ax1.set_ylabel('Hour 2', fontsize=12, labelpad=10)
ax1.set_zlabel('log10(det(FIM))', fontsize=12, labelpad=10)
ax1.set_title('FIM Determinant Surface', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')

# Add colorbar
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5, label='log10(det(FIM))')

# Set viewing angle
ax1.view_init(elev=25, azim=45)

# 2D contour plot
ax2 = fig.add_subplot(122)

# Use the same grid from 3D plot
# Contour plot
contour = ax2.contourf(H1, H2, Z, levels=20, cmap='viridis')
ax2.contour(H1, H2, Z, levels=20, colors='black', alpha=0.3, linewidths=0.5)

# Mark the best point
ax2.scatter(
    [best['hour1']],
    [best['hour2']],
    color='red',
    s=350,
    marker='*',
    edgecolors='black',
    linewidths=2,
    alpha=0.9,
    label=f'GridBest: ({best["hour1"]:.2f}, {best["hour2"]:.2f})',
    zorder=5,
)

# Mark the optimal point from optimization
ax2.scatter(
    [1.9322],
    [10],
    color='cyan',
    s=300,
    marker='o',
    edgecolors='blue',
    linewidths=2,
    alpha=0.95,
    label=f'Optimal: (1.9322, 10.00)',
    zorder=6,
)

ax2.set_xlabel('Hour 1', fontsize=12)
ax2.set_ylabel('Hour 2', fontsize=12)
ax2.set_title('FIM Determinant Contour Map', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# Add colorbar
fig.colorbar(contour, ax=ax2, label='log10(det(FIM))')

plt.tight_layout()
output_path = script_dir / 'rooney_biegler_fim_plot_concatenated.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved as '{output_path}'")
plt.show()
