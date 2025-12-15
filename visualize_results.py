"""
Generate convergence plots for visual presentation.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('pricing_results.json', 'r') as f:
    results = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Binomial convergence
ax1 = axes[0]
steps = results['binomial']['steps']
abs_errors = results['binomial']['abs_errors']

ax1.plot(steps, abs_errors, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Steps', fontsize=12)
ax1.set_ylabel('Absolute Error ($)', fontsize=12)
ax1.set_title('Binomial Tree Convergence', fontsize=14)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# Add annotation for best result
best_steps = results['binomial']['best']['steps']
best_error = results['binomial']['best']['error']
ax1.annotate(f'{best_steps} steps\n${best_error:.6f} error',
             xy=(best_steps, best_error),
             xytext=(best_steps*1.5, best_error*2),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10, color='red')

# Monte Carlo convergence
ax2 = axes[1]
paths = results['monte_carlo']['paths']
ses = results['monte_carlo']['standard_errors']

ax2.plot(paths, ses, 'go-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Paths', fontsize=12)
ax2.set_ylabel('Standard Error ($)', fontsize=12)
ax2.set_title('Monte Carlo Standard Error', fontsize=14)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

# Add O(1/sqrt(n)) reference line
ref_paths = np.array([10000, 500000])
ref_se = ses[0] * np.sqrt(paths[0] / ref_paths)
ax2.plot(ref_paths, ref_se, 'r--', alpha=0.5, label='O(1/√n) scaling')
ax2.legend()

plt.tight_layout()
plt.savefig('convergence_analysis.png', dpi=300, bbox_inches='tight')
print("Saved convergence_analysis.png")
plt.close()
