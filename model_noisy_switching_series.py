from hmmlearn import hmm
import numpy as np

def model_switching_rate_noisy_series_gaussian(noisy_series):
    # Reshape the series for hmmlearn (expects 2D array)
    noisy_series = np.array(noisy_series).reshape(-1, 1)

    # Use GaussianHMM instead of MultinomialHMM
    model = hmm.GaussianHMM(n_components=2, n_iter=1000, tol=1e-4, random_state=42)

    model.fit(noisy_series)

    # Extract transition matrix and switching rates
    transmat = model.transmat_

    # Calculate switching rates directly from the transition matrix
    switching_rate_0_to_1 = transmat[0, 1]
    switching_rate_1_to_0 = transmat[1, 0]

    # Compute stationary distribution
    total_switching = switching_rate_0_to_1 + switching_rate_1_to_0
    pi_0 = switching_rate_1_to_0 / total_switching
    pi_1 = switching_rate_0_to_1 / total_switching

    # Calculate total switching rate (probability per step)
    total_switching_rate = pi_0 * switching_rate_0_to_1 + pi_1 * switching_rate_1_to_0
    
    return 1/total_switching_rate