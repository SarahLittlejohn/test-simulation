import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from generate_gaussian_data import generate_gaussian_matrix_variable_impact

# Parameters
d = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
baseline = 7
initial_amplitude = 3
length_impact = 200
d_impact = 5.5 
min_switching_rates = []
min_times = []
all_fitted_values = []

gaussian_matrix = generate_gaussian_matrix_variable_impact(baseline, initial_amplitude, d, length_impact, impact=d_impact)

def reverse_bell_curve(x, a, b, c, d):
    """Reverse Gaussian (bell curve) model."""
    return -a * np.exp(-((x - b)**2) / (2 * c**2)) + d

def fit_switching_rate(x, y):
    """Fit the data to the reverse bell curve model."""
    popt, _ = curve_fit(reverse_bell_curve, x, y, p0=[1, np.mean(x), 50, 7])
    return reverse_bell_curve(x, *popt), popt

global_min_rate1 = float('inf')
global_min_rate2 = float('inf')
global_min_d1 = None
global_min_d2 = None

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

baseline_row = gaussian_matrix[0]
x_baseline = np.arange(len(baseline_row))
valid_baseline_indices = ~np.isnan(baseline_row)
x_valid_baseline = x_baseline[valid_baseline_indices]
y_valid_baseline = baseline_row[valid_baseline_indices]

axs[0].plot(x_valid_baseline, y_valid_baseline, color='grey', linewidth=2, label='Baseline (Row 1)')

adjusted_d = d[:-1]  

for i, (row, dist) in enumerate(zip(gaussian_matrix[1:], adjusted_d)):
    valid_indices = ~np.isnan(row)
    x_valid = np.where(valid_indices)[0]
    y_valid = row[valid_indices]

    if len(x_valid) == 0:
        continue

    fitted_values, _ = fit_switching_rate(x_valid, y_valid)
    all_fitted_values.append((x_valid, fitted_values))

    min_rate = np.min(fitted_values)
    min_index = np.argmin(fitted_values)
    min_time = x_valid[min_index]

    if min_rate < global_min_rate1:
        global_min_rate2 = global_min_rate1
        global_min_d2 = global_min_d1
        global_min_rate1 = min_rate
        global_min_d1 = dist
    elif min_rate < global_min_rate2:
        global_min_rate2 = min_rate
        global_min_d2 = dist

    axs[0].plot(x_valid, y_valid, alpha=0.7, label=f'd = {dist}')
    axs[0].plot(x_valid, fitted_values, '--', linewidth=1.5, label=f'Fit d = {dist}')

axs[0].set_title('Switching Rates with Fitted Curves')
axs[0].set_xlabel('Time Steps')
axs[0].set_ylabel('Switching Rate')
axs[0].legend(title="Distance (d)", loc='upper right', bbox_to_anchor=(1.2, 1))

if global_min_d1 is not None and global_min_d2 is not None:
    d1, d2 = global_min_d1, global_min_d2
    print(d1)
    print(d2)
    m1, m2 = global_min_rate1, global_min_rate2
    print(m2 - m1)
    print(initial_amplitude - m1)
    weighting = (initial_amplitude - m2) / (m2 - m1)
    print(weighting)
    d_impact_extracted = d1 + weighting
    print(d_impact_extracted)
else:
    d_impact_extracted = None

axs[1].set_aspect('equal', adjustable='box')
for dist in adjusted_d:
    circle = plt.Circle((dist, 0), 0.2, color='C0', fill=True)
    axs[1].add_patch(circle)

if d_impact_extracted is not None:
    axs[1].scatter(d_impact_extracted, 0, color='red', marker='x', s=100, linewidths=2, label=f'Extracted d_impact = {d_impact_extracted:.3f}')

axs[1].set_title('Qubit layout (1D)')
axs[1].set_xlabel('Distance (d)')
axs[1].set_xlim(-1, max(d) + 1)
axs[1].set_ylim(-1, 1)
axs[1].legend(loc='upper right')

plt.tight_layout()
plt.show()

print(f"Extracted d_impact: {d_impact_extracted:.3f}")
print(f"Minimum Switching Rates: m1 = {global_min_rate1:.4f}, m2 = {global_min_rate2:.4f}")
