import numpy as np
from hmmlearn import hmm

def find_static_switching_rate_clean_series(series):
    """
    Finds the switching rate of a clean parity series

    Parameters:
    series: the parrity series
    """

    # Count transitions
    n_01 = np.sum((np.array(series[:-1]) == 0) & (np.array(series[1:]) == 1))  # Transitions from 0 to 1
    n_00 = np.sum((np.array(series[:-1]) == 0) & (np.array(series[1:]) == 0))  # Transitions from 0 to 0
    n_10 = np.sum((np.array(series[:-1]) == 1) & (np.array(series[1:]) == 0))  # Transitions from 1 to 0
    n_11 = np.sum((np.array(series[:-1]) == 1) & (np.array(series[1:]) == 1))  # Transitions from 1 to 1

    print(n_01, n_00, n_10, n_11)

    total_0 = n_01 + n_00
    total_1 = n_10 + n_11

    P = np.array([[n_00 / total_0, n_01 / total_0],
                [n_10 / total_1, n_11 / total_1]])

    switch_0_to_1 = P[0, 1]
    switch_1_to_0 = P[1, 0]

    print("Transition Matrix:")
    print(P)
    print(f"Switching Rate 0 -> 1: {switch_0_to_1}")
    print(f"Switching Rate 1 -> 0: {switch_1_to_0}")

    p_0 = np.mean(np.array(series) == 0)
    p_1 = 1 - p_0

    total_switching_probability = p_0 * switch_0_to_1 + p_1 * switch_1_to_0
    print(f"Total switching probability {total_switching_probability}")
    print(f"Total switching rate {1/total_switching_probability}")

def find_static_switching_rate_noisy_series(noisy_series):
    """
    Finds the switching rate of a noisy series using HMM

    Parameters:
    noisy_series: the noisy parity series

    Returns:
    total_switching_rate: the computed switching rate
    """

    # Reshape the series for hmmlearn
    noisy_series = np.array(noisy_series).reshape(-1, 1)

    # Use CategoricalHMM
    model = hmm.CategoricalHMM(n_components=2, init_params='tm', algorithm="MAP", n_iter=20000, random_state=123)
    model.startprob_ = np.array([1.0, 0.0])

    model.fit(noisy_series)

    # Extract transition matrix and switching rates
    transmat = model.transmat_

    # Calculate switching rates directly from the transition matrix
    switching_freq_0_to_1 = transmat[0, 1]
    switching_freq_1_to_0 = transmat[1, 0]

    # Compute stationary distribution
    total_switching = switching_freq_0_to_1 + switching_freq_1_to_0
    pi_0 = switching_freq_1_to_0 / total_switching
    pi_1 = switching_freq_0_to_1 / total_switching

    # Calculate total switching rate
    total_switching_rate = 1/(pi_0 * switching_freq_0_to_1 + pi_1 * switching_freq_1_to_0)
    
    return total_switching_rate

def find_dynamic_switching_rates_noisy_series(parity_series, segment_length):
    """
    Finds the switching rate of a clean series

    Parameters:
    series: the parrity series
    """

    # Segment the parity series into chunks of segment_length
    num_segments = len(parity_series) // segment_length
    switching_rates = []

    for i in range(num_segments):
        # Extract segment
        segment = parity_series[i * segment_length:(i + 1) * segment_length]
        if np.isnan(segment).any():
            rate = np.nan
        else:
            # Compute switching rate using the provided model function
            rate = find_static_switching_rate_noisy_series(segment)
        switching_rates.append(rate)

    return switching_rates