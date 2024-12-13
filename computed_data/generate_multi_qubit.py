import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from computed_data.generate_parity_series import generate_parity_series_dynamic
from computed_data.model_switching_rate_from_gen_parity import segment_and_compute_switching_rates
from perfect_data.generate_gaussian_data import generate_gaussian_matrix

# Defining parameters
d = [0, 0.5, 1, 1.5 , 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
baseline = 7
initial_amplitude = 3
baseline_noise_std = 0.31
segment_length = 1000
length_impact=200

# Generate the Gaussian matrix
gaussian_matrix = generate_gaussian_matrix(baseline, initial_amplitude, d, length_impact)

# Initialize results
all_switching_rates = []
all_fitted_switching_rates = []

# Define the reverse bell curve fit
def reverse_bell_curve(x, a, b, c, d):
    return -a * np.exp(-((x - b)**2) / (2 * c**2)) + d

def fit_switching_rate(computed_switching_rates, bell_curve_mid_point):
    x = np.linspace(0, len(computed_switching_rates), len(computed_switching_rates))
    popt, _ = curve_fit(reverse_bell_curve, x, computed_switching_rates, p0=[10, bell_curve_mid_point, 100, 7])
    return reverse_bell_curve(x, *popt)

# Loop through the rows of the Gaussian matrix (skip baseline)
for i, total_series in enumerate(gaussian_matrix[1:]):  # Skip the baseline row
    # Generate parity data and compute switching rates
    parity_series = generate_parity_series_dynamic(total_series, len(total_series))
    switching_rates = segment_and_compute_switching_rates(parity_series, segment_length)
    
    # Fit the switching rates to the reverse bell curve
    bell_curve_mid_point = len(total_series) // 2
    fitted_switching_rates = fit_switching_rate(switching_rates, bell_curve_mid_point)
    
    # Store results
    all_switching_rates.append(switching_rates)
    all_fitted_switching_rates.append(fitted_switching_rates)
    print(f"Row {i + 1} (Distance d = {d[i]}) complete")

# Extract min switching rates
min_switching_rates = [np.min(rates) for rates in all_fitted_switching_rates]

# Exponential decay fit
def exp_decay(distance, lambda_):
    return 7 - 4 * np.exp(-lambda_ * distance)

d = np.array(d)
params, _ = curve_fit(exp_decay, d, min_switching_rates)
lambda_estimate = params[0]

# Collect min t-values
min_t_values = [np.argmin(rates) for rates in all_switching_rates]
print(min_t_values[1])
for val in min_t_values:
    print(val)

# Linear model fit
def linear_model(distance, sigma):
    return min_t_values[0] + distance * sigma

params, _ = curve_fit(linear_model, d, min_t_values)
sigma_estimate = params[0]

# Plotting results
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Subplot 1: Computed Switching Rates
for i, rates in enumerate(all_switching_rates):
    axs[0, 0].plot(rates, label=f'd = {d[i]}', alpha=0.7)
axs[0, 0].set_title('Computed Switching Rates')
axs[0, 0].set_xlabel('Time Steps')
axs[0, 0].set_ylabel('Switching Rate')

# Subplot 2: Fitted Switching Rates
for i, rates in enumerate(all_fitted_switching_rates):
    axs[0, 1].plot(rates, linestyle='--', alpha=0.7, label=f'd = {d[i]}')
axs[0, 1].set_title('Fitted Switching Rates')
axs[0, 1].set_xlabel('Time Steps')
axs[0, 1].set_ylabel('Switching Rate')
axs[0, 1].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small', title="Distance d")


# Subplot 3: Min Switching Rates
axs[1, 0].scatter(d, min_switching_rates, color='red', label='Data')
axs[1, 0].plot(d, exp_decay(d, lambda_estimate), label=f'Fitted Exp Decay (λ={lambda_estimate:.4f})')
axs[1, 0].set_title('Exponential Decay Fit')
axs[1, 0].set_xlabel('Distance (d)')
axs[1, 0].set_ylabel('Min Switching Rate')
axs[1, 0].legend()

# Subplot 4: Linear Fit of Time Steps
axs[1, 1].scatter(d, min_t_values, color='red', label='Data')
axs[1, 1].plot(d, linear_model(d, sigma_estimate), label=f'Linear Fit (σ={sigma_estimate:.4f})')
axs[1, 1].set_title('Linear Fit of Time Steps')
axs[1, 1].set_xlabel('Distance (d)')
axs[1, 1].set_ylabel('Time Step')

# Final layout
plt.tight_layout()
plt.legend()
plt.show()
