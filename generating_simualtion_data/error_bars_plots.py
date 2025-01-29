from generating_simualtion_data.generate_parity_series import generate_parity_series, generate_e2e_parity_series_with_noise
from generating_simualtion_data.find_switching_rate import find_static_switching_rate_clean_series, find_static_switching_rate_noisy_series
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

switching_rate_sample = [3, 4, 5, 6, 7, 8, 9]
series_length_sample = 3000
error_bars = []

def plot_error_bars_for_switching_rates_clean():
    rates = []
    switching_rates = []
    switching_rate_stds = []

    for rate in switching_rate_sample:
        series = generate_parity_series(series_length_sample, rate)
        switching_rate, switching_rate_std = find_static_switching_rate_clean_series(series)
        rates.append(rate)
        switching_rates.append(switching_rate)
        switching_rate_stds.append(switching_rate_std)

    plt.figure(figsize=(8, 6))
    plt.errorbar(rates, switching_rates, yerr=switching_rate_stds, fmt='o', capsize=5, label='Switching Rate')
    plt.title('Generated Rate vs Input Rate (with Error Bars)')
    plt.xlabel('Input Rate')
    plt.ylabel('Generated Rate')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

def plot_error_bars_for_switching_rates_noisy():
    rates = []
    switching_rates = []
    switching_rate_stds = []

    for rate in switching_rate_sample:
        series = generate_e2e_parity_series_with_noise(series_length_sample, rate, 0.005)
        switching_rate, switching_rate_std = find_static_switching_rate_noisy_series(series)
        rates.append(rate)
        switching_rates.append(switching_rate)
        switching_rate_stds.append(switching_rate_std)

    plt.figure(figsize=(8, 6))
    plt.errorbar(rates, switching_rates, yerr=switching_rate_stds, fmt='o', capsize=5, label='Switching Rate')
    plt.title('Generated Rate vs Input Rate (with Error Bars) - HMM')
    plt.xlabel('Input Rate')
    plt.ylabel('Generated Rate')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()