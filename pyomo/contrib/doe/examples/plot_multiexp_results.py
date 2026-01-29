"""
Plot multi-experiment optimization results for different objective functions.
Compares determinant, trace, and pseudo_trace objectives.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Read the combined results file
with open(script_dir / 'rooney_biegler_multiexp_all_objectives.json', 'r') as f:
    all_results = json.load(f)

# Define objective options and colors
objective_options = ['determinant', 'trace', 'pseudo_trace']
colors = {'determinant': '#1f77b4', 'trace': "#df8536", 'pseudo_trace': '#2ca02c'}
markers = {'determinant': 'o', 'trace': 's', 'pseudo_trace': '^'}

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 10))

# 1. Design variable comparison
ax1 = fig.add_subplot(2, 3, 1)
for obj_option in objective_options:
    results = all_results[obj_option]
    scenario = results['scenarios'][0]  # First scenario

    hour_values = []
    exp_indices = []
    for exp in scenario['experiments']:
        hour_values.append(exp['design_variables']['hour'])
        exp_indices.append(exp['experiment_idx'] + 1)

    ax1.scatter(
        exp_indices,
        hour_values,
        label=obj_option,
        color=colors[obj_option],
        marker=markers[obj_option],
        s=150,
        alpha=0.7,
        edgecolors='black',
        linewidths=1.5,
    )

ax1.set_xlabel('Experiment Index', fontsize=11)
ax1.set_ylabel('Hour (Design Variable)', fontsize=11)
ax1.set_title('Design Variables by Objective', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(exp_indices)

# 2. Log10 D-optimality comparison
ax2 = fig.add_subplot(2, 3, 2)
d_opt_values = []
obj_labels = []
for obj_option in objective_options:
    results = all_results[obj_option]
    scenario = results['scenarios'][0]
    d_opt_values.append(scenario['log10_D_opt'])
    obj_labels.append(obj_option)

bars = ax2.bar(
    range(len(objective_options)),
    d_opt_values,
    color=[colors[obj] for obj in objective_options],
    alpha=0.7,
    edgecolor='black',
    linewidth=1.5,
)
ax2.set_xlabel('Objective Function', fontsize=11)
ax2.set_ylabel('log10(det(FIM))', fontsize=11)
ax2.set_title('D-optimality (Determinant)', fontsize=12, fontweight='bold')
ax2.set_xticks(range(len(objective_options)))
ax2.set_xticklabels(obj_labels, rotation=15, ha='right')
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, d_opt_values)):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        f'{val:.4f}',
        ha='center',
        va='bottom',
        fontsize=9,
        fontweight='bold',
    )

# 3. Log10 A-optimality comparison
ax3 = fig.add_subplot(2, 3, 3)
a_opt_values = []
for obj_option in objective_options:
    results = all_results[obj_option]
    scenario = results['scenarios'][0]
    a_opt_values.append(scenario['log10_A_opt'])

bars = ax3.bar(
    range(len(objective_options)),
    a_opt_values,
    color=[colors[obj] for obj in objective_options],
    alpha=0.7,
    edgecolor='black',
    linewidth=1.5,
)
ax3.set_xlabel('Objective Function', fontsize=11)
ax3.set_ylabel('log10(trace(FIM⁻¹))', fontsize=11)
ax3.set_title('A-optimality (Trace)', fontsize=12, fontweight='bold')
ax3.set_xticks(range(len(objective_options)))
ax3.set_xticklabels(obj_labels, rotation=15, ha='right')
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, a_opt_values)):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        f'{val:.4f}',
        ha='center',
        va='bottom',
        fontsize=9,
        fontweight='bold',
    )

# 4. Log10 E-optimality comparison
ax4 = fig.add_subplot(2, 3, 4)
e_opt_values = []
for obj_option in objective_options:
    results = all_results[obj_option]
    scenario = results['scenarios'][0]
    e_opt_values.append(scenario['log10_E_opt'])

bars = ax4.bar(
    range(len(objective_options)),
    e_opt_values,
    color=[colors[obj] for obj in objective_options],
    alpha=0.7,
    edgecolor='black',
    linewidth=1.5,
)
ax4.set_xlabel('Objective Function', fontsize=11)
ax4.set_ylabel('log10(max eigenvalue(FIM⁻¹))', fontsize=11)
ax4.set_title('E-optimality (Min Eigenvalue)', fontsize=12, fontweight='bold')
ax4.set_xticks(range(len(objective_options)))
ax4.set_xticklabels(obj_labels, rotation=15, ha='right')
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, e_opt_values)):
    ax4.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        f'{val:.4f}',
        ha='center',
        va='bottom',
        fontsize=9,
        fontweight='bold',
    )

# 5. FIM Condition Number comparison
ax5 = fig.add_subplot(2, 3, 5)
cond_values = []
for obj_option in objective_options:
    results = all_results[obj_option]
    scenario = results['scenarios'][0]
    cond_values.append(scenario['FIM_condition_number'])

bars = ax5.bar(
    range(len(objective_options)),
    cond_values,
    color=[colors[obj] for obj in objective_options],
    alpha=0.7,
    edgecolor='black',
    linewidth=1.5,
)
ax5.set_xlabel('Objective Function', fontsize=11)
ax5.set_ylabel('Condition Number', fontsize=11)
ax5.set_title('FIM Condition Number', fontsize=12, fontweight='bold')
ax5.set_xticks(range(len(objective_options)))
ax5.set_xticklabels(obj_labels, rotation=15, ha='right')
ax5.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, cond_values)):
    ax5.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f'{val:.2f}',
        ha='center',
        va='bottom',
        fontsize=9,
        fontweight='bold',
    )

# 6. Computational Time comparison
ax6 = fig.add_subplot(2, 3, 6)
solve_times = []
for obj_option in objective_options:
    results = all_results[obj_option]
    solve_times.append(results['timing']['solve_time'])

bars = ax6.bar(
    range(len(objective_options)),
    solve_times,
    color=[colors[obj] for obj in objective_options],
    alpha=0.7,
    edgecolor='black',
    linewidth=1.5,
)
ax6.set_xlabel('Objective Function', fontsize=11)
ax6.set_ylabel('Solve Time (seconds)', fontsize=11)
ax6.set_title('Computational Time', fontsize=12, fontweight='bold')
ax6.set_xticks(range(len(objective_options)))
ax6.set_xticklabels(obj_labels, rotation=15, ha='right')
ax6.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, solve_times)):
    ax6.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f'{val:.3f}s',
        ha='center',
        va='bottom',
        fontsize=9,
        fontweight='bold',
    )

plt.tight_layout()
output_path = script_dir / 'rooney_biegler_multiexp_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved as '{output_path}'")

# Print summary
print("\n" + "=" * 70)
print("Multi-Experiment Optimization Results Summary")
print("=" * 70)
for obj_option in objective_options:
    results = all_results[obj_option]
    scenario = results['scenarios'][0]
    print(f"\n{obj_option.upper()}:")
    print(f"  Design variables:")
    for exp in scenario['experiments']:
        print(
            f"    Exp {exp['experiment_idx']+1}: hour = {exp['design_variables']['hour']:.4f}"
        )
    print(f"  log10(D-opt): {scenario['log10_D_opt']:.4f}")
    print(f"  log10(A-opt): {scenario['log10_A_opt']:.4f}")
    print(f"  log10(E-opt): {scenario['log10_E_opt']:.4f}")
    print(f"  Condition #:  {scenario['FIM_condition_number']:.2f}")
    print(f"  Solve time:   {results['timing']['solve_time']:.3f}s")

print("\n" + "=" * 70)

plt.show()
