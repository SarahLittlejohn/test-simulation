from hmmlearn import hmm
import numpy as np

def model_switching_rate_noisy_series(noisy_series):
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