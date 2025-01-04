import matplotlib.pyplot as plt
from computed_data.generate_parity_series import generate_e2e_parity_series_with_noise
from computed_data.model_noisy_parity_series import model_switching_rate_noisy_series
import numpy as np
import pandas as pd
from pandas.plotting import table

series = generate_e2e_parity_series_with_noise(10000, 5, 0.005)

def bootstrap_model_switching_rate(series, n_bootstrap=1000, segment_length=None):
    if segment_length is None:
        segment_length = len(series) // 2
    
    estimates = []
    n = len(series)
    
    for _ in range(n_bootstrap):
        start_idx = np.random.randint(0, n - segment_length + 1)
        bootstrap_sample = series[start_idx:start_idx + segment_length]
        rate_estimate = model_switching_rate_noisy_series(bootstrap_sample)
        estimates.append(rate_estimate)
    
    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates)
    
    return mean_estimate, std_estimate

# mean_rate, std_rate = bootstrap_model_switching_rate(series, segment_length=1000)
# print(f"Mean Switching Rate: {mean_rate}")
# print(f"Standard Deviation (Uncertainty): {std_rate}")

sample = [3, 5, 7, 11, 15, 20, 25]
total_switching_rate = []
total_std = []

for x in sample:
    print(x)
    series = generate_e2e_parity_series_with_noise(10000, x, 0.005)
    mean_rate, std_rate = bootstrap_model_switching_rate(series, segment_length=1000)
    switching_rate = model_switching_rate_noisy_series(series)
    total_switching_rate.append(switching_rate)
    total_std.append(std_rate)

total_switching_rate_rounded = [f"{val:.2f}" for val in total_switching_rate]
total_std_rounded = [f"{val:.2f}" for val in total_std]

fig, ax = plt.subplots()
ax.errorbar(sample, total_switching_rate, yerr=total_std, fmt='o-', capsize=5, label='Computed Switching Rate')
ax.set_xlabel('Input Switching Rate')
ax.set_ylabel('Computed Switching Rate')
ax.set_title('Computed Switching Rate with Error Bars')
ax.grid(True)
ax.legend()

table_data = {
    'Input Switching Rate': sample,
    'Computed Switching Rate': total_switching_rate_rounded,
    'Standard Deviation': total_std_rounded
}
table_df = pd.DataFrame(table_data)

cell_text = table_df.values
col_labels = table_df.columns
table = ax.table(cellText=cell_text, colLabels=col_labels, loc='bottom', cellLoc='center', bbox=[0, -0.5, 1, 0.3])

for (i, j), cell in table.get_celld().items():
    cell.set_fontsize(15)
    if i == 0:
        cell.set_text_props(weight='bold')

plt.subplots_adjust(left=0.2, bottom=0.5)
plt.show()