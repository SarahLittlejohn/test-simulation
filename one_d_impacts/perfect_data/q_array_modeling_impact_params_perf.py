import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tools.generate_gaussian_data import generate_gaussian_matrix

# This file is a showing how to find the parameters that define how the impact propagates accross the lines of qubits

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
    popt, pcov = curve_fit(reverse_bell_curve, x, y, p0=[1, np.mean(x), 50, 7])
    return reverse_bell_curve(x, *popt), popt, pcov

min_switching_rates = []
min_times = []
all_fitted_values = []
all_error_values = []

# fit each row
for i, row in enumerate(gaussian_matrix):
    valid_indices = ~np.isnan(row)
    x_valid = np.where(valid_indices)[0]
    y_valid = row[valid_indices]

    # Skip rows with no valid data
    if len(x_valid) == 0:
        continue

    # Fit the curve
    fitted_values, opt_params, covariance_matrix = fit_switching_rate(x_valid, y_valid)
    all_fitted_values.append((x_valid, fitted_values))

    if i > 0:
        print(f"Row: {i}")
        print(f"Error: {np.sqrt(np.diag(covariance_matrix))}")
        print(f"Matrix: {covariance_matrix}")
        all_error_values.append(np.sqrt(np.diag(covariance_matrix)))
        min_rate = np.min(fitted_values)
        min_index = np.argmin(fitted_values)
        min_time = x_valid[min_index] 

        min_switching_rates.append(min_rate)
        min_times.append(min_time)

# for i, (rate, time) in enumerate(zip(min_switching_rates, min_times), start=1):
#     print(f"Row {i}: Min Switching Rate = {rate:.4f}, Time = {time}")

# Fit the exponetial decay
def exp_decay(distance, lambda_):
    return 7 - 4 * np.exp(-lambda_ * distance)

def linear_model(distance, sigma):
    return min_times[0] + distance * sigma

d = np.array(d)

params_A, cov_A = curve_fit(exp_decay, d, min_switching_rates)
lambda_estimate = params_A[0]
lambda_errors = np.sqrt(np.diag(cov_A))[0]

params_t, cov_t = curve_fit(linear_model, d, min_times)
sigma_estimate = params_t[0]
sigma_errors = np.sqrt(np.diag(cov_t))[0]

min_switching_rate_errors = [errors[0] for errors in all_error_values]
time_step_errors = [errors[1] for errors in all_error_values]
min_switching_rate_errors = np.array(min_switching_rate_errors)
print("switching rate errors::")
print(min_switching_rate_errors)
time_step_errors = np.array(time_step_errors)
min_switching_rates = np.array(min_switching_rates)

print("switching rate errors")
print(min_switching_rate_errors)

print(f'Lambda estimate: {lambda_estimate} +/- {lambda_errors}')
print(f'Sigma estimate: {sigma_estimate} +/- {sigma_errors}')

# Plot
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

for i, (row, (x_valid, fitted_values)) in enumerate(zip(gaussian_matrix[1:], all_fitted_values)):
    axs[0, 0].plot(row, label=f'd = {d[i]}', alpha=0.5)
    axs[0, 0].plot(x_valid, fitted_values, linestyle='--', linewidth=1.5, label=f'Fit d = {d[i]}')
axs[0, 0].set_title('Computed Switching Rates with Fitted Curves')
axs[0, 0].set_xlabel('Time Steps')
axs[0, 0].set_ylabel('Switching Rate')

axs[0, 1].set_aspect('equal', adjustable='box')
for i, distance in enumerate(d):
    circle = plt.Circle((distance, 0), 0.1, color='C0', fill=True)
    axs[0, 1].add_patch(circle)
impact_handle = axs[0, 1].scatter(0, 0, color='red', marker='x', s=100, linewidths=2, label='Impact at d=0')
axs[0, 1].set_xlim(-1, max(d) + 1)
axs[0, 1].set_ylim(-1, 1)
axs[0, 1].set_title("Qubit layout (1D)")
axs[0, 1].set_xlabel("Distance (d)")
axs[0, 1].set_yticks([])
axs[0, 1].set_xticks(np.arange(0, max(d) + 1, step=1))
handles, labels = axs[0, 1].get_legend_handles_labels()
# handles.append(impact_handle)
labels.append("Impact at d=0")
axs[0, 1].legend(handles=handles, labels=labels, loc='upper left')

axs[1, 0].errorbar(d, min_switching_rates, yerr=min_switching_rate_errors, fmt='o', color='red', label='Data with Error Bars')
axs[1, 0].plot(d, exp_decay(d, lambda_estimate), label=f'Fitted Exp Decay (λ={lambda_estimate:.4f} +/- {lambda_errors:.4f})')
axs[1, 0].set_title('Exponential Decay Fit')
axs[1, 0].set_xlabel('Distance (d)')
axs[1, 0].set_ylabel('Min Switching Rate')
axs[1, 0].legend()

axs[1, 1].errorbar(d, min_times, yerr=time_step_errors, fmt='o', color='red', label='Data with Error Bars')
axs[1, 1].plot(d, linear_model(d, sigma_estimate), label=f'Linear Fit (σ={sigma_estimate:.4f} +/- {sigma_errors:.4f})')
axs[1, 1].set_title('Linear Fit of Time Steps')
axs[1, 1].set_xlabel('Distance (d)')
axs[1, 1].set_ylabel('Time Step')
axs[1, 1].legend()

plt.tight_layout()
plt.show()

print(min_switching_rates)
print(all_error_values[0])