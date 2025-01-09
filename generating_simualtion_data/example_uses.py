from generating_simualtion_data.generate_parity_series import generate_parity_series
from generating_simualtion_data.find_switching_rate import find_static_switching_rate_clean_series, find_dynamic_switching_rates_noisy_series, find_static_switching_rate_noisy_series
import numpy as np
from generating_simualtion_data.generate_parity_series import generate_parity_series_dynamic
from generating_simualtion_data.find_switching_rate import find_dynamic_switching_rates_noisy_series
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# # Example of finding a clean static switching rate
# series = generate_parity_series(30000, 3)
# find_static_switching_rate_clean_series(series)

# # Example of generating dynamic switchign rates and fitting a bell curve 
# def generate_switching_rate():
#     baseline = 7
#     baseline_noise_std = 0.31
#     spike_noise_std = 0.31
#     length_baseline = 50
#     length_spike = 200
#     length_return = 200
#     length_post_spike_baseline = 550

#     # Creating the actual switching rate data
#     baseline_series = np.random.normal(loc=baseline, scale=baseline_noise_std, size=length_baseline)
#     spike_series = np.linspace(baseline, baseline - 4, length_spike)  # Linear increase
#     spike_series += np.random.normal(scale=spike_noise_std, size=length_spike)  # Add noise
#     return_series = np.linspace(baseline - 4, baseline, length_return)
#     return_series += np.random.normal(scale=baseline_noise_std, size=length_return)  # Add noise
#     post_spike_baseline_series = np.random.normal(loc=baseline, scale=baseline_noise_std, size=length_post_spike_baseline)
#     total_series = np.concatenate([baseline_series, spike_series, return_series, post_spike_baseline_series])

#     # Creating the parity series out of that
#     parity_series = generate_parity_series_dynamic(total_series, 1000)
#     segment_length = 1000

#     # Computing the switching rates from the parity data
#     return find_dynamic_switching_rates_noisy_series(parity_series, segment_length)

# x = np.linspace(0, 1000, 1000)
# switching_rates = generate_switching_rate()

# # Defining and fitting the reverse bell curve
# def reverse_bell_curve(x, a, b, c, d):
#     return -a * np.exp(-((x - b)**2) / (2 * c**2)) + d
# popt, pcov = curve_fit(reverse_bell_curve, x, switching_rates, p0=[10, 200, 100, 7])
# a, b, c, d = popt
# fitted_curve = reverse_bell_curve(x, a, b, c, d)

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(x, switching_rates, label="Computed Switching Rate", alpha=0.7)
# plt.plot(x, fitted_curve, label="Fitted Reverse Bell Curve", color='red', linewidth=2)
# plt.xlabel("Switching Rate")
# plt.ylabel("Time Step")
# plt.title("Inputted Switching Rate and Computed Switching Rates with Fit")
# plt.legend()
# plt.grid()
# plt.show()