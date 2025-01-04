import numpy as np
import matplotlib.pyplot as plt
from computed_data.generate_parity_series import generate_parity_series_dynamic
from computed_data.model_switching_rate_from_gen_parity import segment_and_compute_switching_rates, model_switching_rate_noisy_series

def generate_switching_rate():
    baseline = 7
    baseline_noise_std = 0.31
    spike_noise_std = 0.31
    length_baseline = 50
    length_spike = 200
    length_return = 200
    length_post_spike_baseline = 550

    baseline_series = np.random.normal(loc=baseline, scale=baseline_noise_std, size=length_baseline)
    spike_series = np.linspace(baseline, baseline - 4, length_spike)  # Linear increase
    spike_series += np.random.normal(scale=spike_noise_std, size=length_spike)  # Add noise
    return_series = np.linspace(baseline - 4, baseline, length_return)
    return_series += np.random.normal(scale=baseline_noise_std, size=length_return)  # Add noise
    post_spike_baseline_series = np.random.normal(loc=baseline, scale=baseline_noise_std, size=length_post_spike_baseline)
    total_series = np.concatenate([baseline_series, spike_series, return_series, post_spike_baseline_series])


    parity_series = generate_parity_series_dynamic(total_series, 1000)
    segment_length = 1000

    return segment_and_compute_switching_rates(parity_series, segment_length)

    # plt.figure(figsize=(12, 6))

    # plt.plot(total_series, marker='o', linestyle='-', color='b', label='Inputted Switching Rate')
    # plt.plot(switching_rates, marker='o', linestyle='--', color='r', label='Computed Switching Rate')

    # plt.xlabel('Switching Rate')
    # plt.ylabel('Time Step')
    # plt.title('Inputted Switching Rate and Computed Switching Rates')
    # plt.grid(True)
    # plt.legend()

    # plt.show()

    # difference = total_series - switching_rates

    # plt.figure(figsize=(12, 6))
    # plt.plot(difference, marker='o', linestyle='-', color='g', label='Difference')
    # plt.xlabel('Time Step')
    # plt.ylabel('Difference')
    # plt.title('Difference between inputted and computed and switching rates')
    # plt.grid(True)
    # plt.legend()
    # plt.show()