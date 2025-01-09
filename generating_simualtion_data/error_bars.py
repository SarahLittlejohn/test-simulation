from generating_simualtion_data.generate_parity_series import generate_parity_series
from generating_simualtion_data.find_switching_rate import find_static_switching_rate_noisy_series
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

switching_rate_sample = [7.11, 7.12, 7.13, 7.14, 7.15]
series_length_sample = [500, 1000, 1500, 2000, 2500, 3000]
sample_size = 50
standard_deviations_matrix = []

def find_standard_dev():
    for rate in switching_rate_sample:
        row_std_devs = []
        for length_sample in series_length_sample:
            series = generate_parity_series(length_sample, rate)
            
            computed_switching_rates = [
                find_static_switching_rate_noisy_series(series) for _ in range(sample_size)
            ]
            
            row_std_devs.append(np.std(computed_switching_rates))
        
        standard_deviations_matrix.append(row_std_devs)

    standard_deviations_matrix = np.array(standard_deviations_matrix)

    for i, row in enumerate(standard_deviations_matrix):
        plt.plot(series_length_sample, row, label=f"Rate {switching_rate_sample[i]}")

    plt.xlabel("Series Length")
    plt.ylabel("Standard Deviation")
    plt.title("Standard Deviation vs Series Length for Different Switching Rates")
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_and_plot_switching_rates(switching_rates, length_sample, sample_size):
    means = []
    std_devs = []

    for rate in switching_rates:
        series = generate_parity_series(length_sample, rate)
        
        static_rates = [
            find_static_switching_rate_noisy_series(series) for _ in range(sample_size)
        ]

        means.append(np.mean(static_rates))
        std_devs.append(np.std(static_rates))

    results_table = pd.DataFrame({
        'Actual Switching Rate': switching_rates,
        'Mean Switching Rate': means,
        'Standard Deviation': std_devs
    })
    print(results_table)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(switching_rates, means, yerr=std_devs, fmt='o', ecolor='red', capsize=5, label='Mean with Error Bars')
    ax.set_title('Mean Switching Rates with Standard Deviation')
    ax.set_xlabel('Actual Switching Rates')
    ax.set_ylabel('Mean Static Switching Rate')
    ax.legend()
    plt.show()


def calculate_switching_rate_and_errors(parity_series, sample_size):
    static_rates = [
        find_static_switching_rate_noisy_series(parity_series) for _ in range(sample_size)
    ]
    return np.mean(static_rates), np.std(static_rates)