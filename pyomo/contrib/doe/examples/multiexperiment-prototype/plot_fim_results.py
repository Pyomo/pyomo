"""
Plot FIM results from rooney_biegler_fim_2exp_verification.json
Creates 2D contour plots for D-optimality, A-optimality, and Pseudo-A-optimality
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys

# Add the parent directory to the path to import from pyomo.contrib.doe
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from pyomo.contrib.doe.utils import compute_FIM_metrics

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Read the results file from the same directory as this script
with open(script_dir / 'rooney_biegler_fim_2exp_verification.json', 'r') as f:
    data = json.load(f)

# Convert results to DataFrame
results_df = pd.DataFrame(data['results'])

print(f"Loaded {len(results_df)} results")
print(f"Computing FIM metrics for all results...\n")

# Compute metrics for each result
d_optimality = []
a_optimality = []
pseudo_a_optimality = []

for idx, row in results_df.iterrows():
    if row['log10_det'] is not None:
        FIM = np.array(row['FIM'])
        try:
            (
                det_FIM,
                trace_cov,
                trace_FIM,
                E_vals,
                E_vecs,
                D_opt,
                A_opt,
                pseudo_A_opt,
                E_opt,
                ME_opt,
            ) = compute_FIM_metrics(FIM)

            d_optimality.append(D_opt)
            a_optimality.append(A_opt)
            pseudo_a_optimality.append(pseudo_A_opt)
        except:
            d_optimality.append(np.nan)
            a_optimality.append(np.nan)
            pseudo_a_optimality.append(np.nan)
    else:
        d_optimality.append(np.nan)
        a_optimality.append(np.nan)
        pseudo_a_optimality.append(np.nan)

# Add metrics to dataframe
results_df['D_optimality'] = d_optimality
results_df['A_optimality'] = a_optimality
results_df['pseudo_A_optimality'] = pseudo_a_optimality

# Find optimal values for each metric
# D-optimality: maximize
valid_d = results_df[results_df['D_optimality'].notna()].copy()
best_d_idx = valid_d['D_optimality'].idxmax()
best_d = valid_d.loc[best_d_idx]

# A-optimality: minimize (trace of covariance)
valid_a = results_df[results_df['A_optimality'].notna()].copy()
best_a_idx = valid_a['A_optimality'].idxmin()
best_a = valid_a.loc[best_a_idx]

# Pseudo-A-optimality: maximize (trace of FIM)
valid_pa = results_df[results_df['pseudo_A_optimality'].notna()].copy()
best_pa_idx = valid_pa['pseudo_A_optimality'].idxmax()
best_pa = valid_pa.loc[best_pa_idx]

print(f"Valid results: {len(valid_d)}")
print(f"\nBest D-optimality design:")
print(f"  Hour1: {best_d['hour1']:.4f}, Hour2: {best_d['hour2']:.4f}")
print(f"  log10(det): {best_d['D_optimality']:.4f}")

print(f"\nBest A-optimality design (minimize trace of covariance):")
print(f"  Hour1: {best_a['hour1']:.4f}, Hour2: {best_a['hour2']:.4f}")
print(f"  log10(trace(cov)): {best_a['A_optimality']:.4f}")

print(f"\nBest Pseudo-A-optimality design (maximize trace of FIM):")
print(f"  Hour1: {best_pa['hour1']:.4f}, Hour2: {best_pa['hour2']:.4f}")
print(f"  log10(trace(FIM)): {best_pa['pseudo_A_optimality']:.4f}")


# Create the plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Get unique hour values and create grid
hour1_unique = np.unique(results_df['hour1'])
hour2_unique = np.unique(results_df['hour2'])

print(f"\nGrid size: {len(hour1_unique)} x {len(hour2_unique)}")

# Create meshgrid
H1, H2 = np.meshgrid(hour1_unique, hour2_unique)


# Helper function to create grid for metric
def create_metric_grid(df, metric_name):
    Z = np.full(H1.shape, np.nan)
    for idx, row in df.iterrows():
        i = np.where(hour2_unique == row['hour2'])[0][0]
        j = np.where(hour1_unique == row['hour1'])[0][0]
        Z[i, j] = row[metric_name] if not np.isnan(row[metric_name]) else np.nan
    return Z


# 1. D-optimality plot
ax1 = axes[0]
Z_d = create_metric_grid(results_df, 'D_optimality')
contour1 = ax1.contourf(H1, H2, Z_d, levels=20, cmap='viridis')
ax1.contour(H1, H2, Z_d, levels=20, colors='black', alpha=0.3, linewidths=0.5)
ax1.scatter(
    [best_d['hour1']],
    [best_d['hour2']],
    color='red',
    s=350,
    marker='*',
    edgecolors='white',
    linewidths=2,
    alpha=0.95,
    label=f'Optimal: ({best_d["hour1"]:.2f}, {best_d["hour2"]:.2f})',
    zorder=5,
)
# Plot optimal point from optimize_experiments()
ax1.scatter(
    1.93,
    10.0,
    color='cyan',
    s=200,
    marker='X',
    edgecolors='black',
    linewidths=1.5,
    alpha=0.9,
    label='Optimum from optimization',
    zorder=5,
)
ax1.set_xlabel('Hour 1', fontsize=12)
ax1.set_ylabel('Hour 2', fontsize=12)
ax1.set_title(
    'D-Optimality (Maximize)\nlog10(det(FIM))', fontsize=13, fontweight='bold'
)
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)
fig.colorbar(contour1, ax=ax1, label='log10(det(FIM))')

# 2. A-optimality plot
ax2 = axes[1]
Z_a = create_metric_grid(results_df, 'A_optimality')
contour2 = ax2.contourf(H1, H2, Z_a, levels=20, cmap='viridis')
ax2.contour(H1, H2, Z_a, levels=20, colors='black', alpha=0.3, linewidths=0.5)
ax2.scatter(
    [best_a['hour1']],
    [best_a['hour2']],
    color='red',
    s=350,
    marker='*',
    edgecolors='white',
    linewidths=2,
    alpha=0.95,
    label=f'Optimal: ({best_a["hour1"]:.2f}, {best_a["hour2"]:.2f})',
    zorder=5,
)
ax2.scatter(
    0.9728,
    10.0,
    color='cyan',
    s=200,
    marker='X',
    edgecolors='black',
    linewidths=1.5,
    alpha=0.9,
    label='Optimum from optimization',
    zorder=5,
)
ax2.set_xlabel('Hour 1', fontsize=12)
ax2.set_ylabel('Hour 2', fontsize=12)
ax2.set_title(
    'A-Optimality (Minimize)\nlog10(trace(FIM⁻¹))', fontsize=13, fontweight='bold'
)
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)
fig.colorbar(contour2, ax=ax2, label='log10(trace(cov))')

# 3. Pseudo-A-optimality plot
ax3 = axes[2]
Z_pa = create_metric_grid(results_df, 'pseudo_A_optimality')
contour3 = ax3.contourf(H1, H2, Z_pa, levels=20, cmap='viridis')
ax3.contour(H1, H2, Z_pa, levels=20, colors='black', alpha=0.3, linewidths=0.5)
ax3.scatter(
    [best_pa['hour1']],
    [best_pa['hour2']],
    color='red',
    s=350,
    marker='*',
    edgecolors='white',
    linewidths=2,
    alpha=0.95,
    label=f'Optimal: ({best_pa["hour1"]:.2f}, {best_pa["hour2"]:.2f})',
    zorder=5,
)
ax3.scatter(
    2.0037,
    2.0039,
    color='cyan',
    s=200,
    marker='X',
    edgecolors='black',
    linewidths=1.5,
    alpha=0.9,
    label='Optimum from optimization',
    zorder=5,
)


ax3.set_xlabel('Hour 1', fontsize=12)
ax3.set_ylabel('Hour 2', fontsize=12)
ax3.set_title(
    'Pseudo-A-Optimality (Maximize)\nlog10(trace(FIM))', fontsize=13, fontweight='bold'
)
ax3.legend(loc='best', fontsize=9)
ax3.grid(True, alpha=0.3)
fig.colorbar(contour3, ax=ax3, label='log10(trace(FIM))')

plt.tight_layout()
output_path = script_dir / 'rooney_biegler_fim_metrics_plot.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved as '{output_path}'")
plt.show()
