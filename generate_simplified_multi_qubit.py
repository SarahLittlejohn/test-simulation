import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from generate_gaussian_data import generate_gaussian_matrix

# Defining parameters
d = [0, 0.5, 1, 1.5 , 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
baseline = 7
initial_amplitude = 3
length_impact = 200

# Generate the Gaussian matrix
gaussian_matrix = generate_gaussian_matrix(baseline, initial_amplitude, d, length_impact)

# Define reverse bell curve function
def reverse_bell_curve(x, a, b, c, d):
    return -a * np.exp(-((x - b)**2) / (2 * c**2)) + d

# Fit the reverse bell curve to the valid data
def fit_switching_rate(x, y):
    popt, _ = curve_fit(reverse_bell_curve, x, y, p0=[1, np.mean(x), 50, 7])
    return reverse_bell_curve(x, *popt), popt

# Plot all rows and their corresponding fits
plt.figure(figsize=(12, 8))

min_switching_rates = []

for i, row in enumerate(gaussian_matrix):
    valid_indices = ~np.isnan(row)  # Mask for valid (non-NaN) data
    x_valid = np.where(valid_indices)[0]  # Indices of valid values
    y_valid = row[valid_indices]  # Non-NaN values

    # Skip rows with no valid data
    if len(x_valid) == 0:
        continue

    # Fit the curve
    fitted_values, _ = fit_switching_rate(x_valid, y_valid)

    plt.plot(row, label=f"Row {i+1} (Data)", linewidth=1)
    plt.plot(x_valid, fitted_values, linestyle='--', linewidth=1.5, label=f"Row {i+1} (Fit)")

    if i > 0:
        min_switching_rates.append(np.min(fitted_values))

print(min_switching_rates)

# Customize the plot
plt.title("Gaussian Dips and Fitted Reverse Bell Curves")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend(ncol=2, fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()
