import matplotlib.pyplot as plt
from generate_switching_series import generate_e2e_series_with_noise
from model_noisy_switching_series import model_switching_rate_noisy_series_gaussian
import numpy as np

# Generate a series with noise
series = generate_e2e_series_with_noise(30000, 5.4, 0.01)

# Example usage
# model_switching_rate_noisy_series_gaussian(series)

def segmented_standard_deviation(noisy_series, n_segments=10):
    segment_length = len(noisy_series) // n_segments
    switching_rates = []

    for i in range(n_segments):
        # Define the segment
        start = i * segment_length
        end = start + segment_length
        segment = noisy_series[start:end]

        # Calculate the switching rate for the segment
        rate = model_switching_rate_noisy_series_gaussian(segment)

        # Only add the rate if it's valid
        if rate is not None:
            switching_rates.append(rate)

    # Check if we have any valid rates
    if len(switching_rates) == 0:
        print("No valid switching rates were computed. Please check the input data and model parameters.")
        return None

    # Compute mean and standard deviation of switching rates
    mean_rate = np.mean(switching_rates)
    std_dev = np.std(switching_rates)
    error = 2 * std_dev
    rate = model_switching_rate_noisy_series_gaussian(noisy_series)

    print(f"Switching rate total: {mean_rate:.2f} +/- {error:.2f}")

# Example usage
segmented_standard_deviation(series, n_segments=10)


# model_switching_rate_noisy_series_gaussian(series)