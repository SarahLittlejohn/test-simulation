from computed_data.model_noisy_parity_series import model_switching_rate_noisy_series

def segment_and_compute_switching_rates(parity_series, segment_length):
    # Segment the parity series into chunks of segment_length
    num_segments = len(parity_series) // segment_length
    switching_rates = []

    for i in range(num_segments):
        # Extract segment
        segment = parity_series[i * segment_length:(i + 1) * segment_length]
        # Compute switching rate using the provided model function
        rate = model_switching_rate_noisy_series(segment)
        switching_rates.append(rate)

    return switching_rates


