import numpy as np
from scipy.linalg import eig

def model_switching_rate_clean_series(series):

    # Count transitions
    n_01 = np.sum((np.array(series[:-1]) == 0) & (np.array(series[1:]) == 1))  # Transitions from 0 to 1
    n_00 = np.sum((np.array(series[:-1]) == 0) & (np.array(series[1:]) == 0))  # Transitions from 0 to 0
    n_10 = np.sum((np.array(series[:-1]) == 1) & (np.array(series[1:]) == 0))  # Transitions from 1 to 0
    n_11 = np.sum((np.array(series[:-1]) == 1) & (np.array(series[1:]) == 1))  # Transitions from 1 to 1

    print(n_01, n_00, n_10, n_11)

    # Total transitions from each state
    total_0 = n_01 + n_00
    total_1 = n_10 + n_11

    # Transition probabilities (Markov matrix)
    P = np.array([[n_00 / total_0, n_01 / total_0],
                [n_10 / total_1, n_11 / total_1]])

    # Switching rates (probabilities of switching states)
    switch_0_to_1 = P[0, 1]  # Probability of switching from 0 to 1
    switch_1_to_0 = P[1, 0]  # Probability of switching from 1 to 0

    print("Transition Matrix:")
    print(P)
    print(f"Switching Rate 0 -> 1: {switch_0_to_1}")
    print(f"Switching Rate 1 -> 0: {switch_1_to_0}")

    # Calculate the proportion of time spent in each state
    p_0 = np.mean(np.array(series) == 0)
    p_1 = 1 - p_0

    # Calculate total switching rate
    total_switching_probability = p_0 * switch_0_to_1 + p_1 * switch_1_to_0
    print(f"Total switching probability {total_switching_probability}")
    print(f"Total switching rate {1/total_switching_probability}")