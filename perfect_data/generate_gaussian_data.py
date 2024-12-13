import numpy as np
import matplotlib.pyplot as plt
import math

# Generating the perfect gaussian data

def generate_gaussian_matrix(baseline, initial_amplitude, distances, length_impact=200, noise_std=0.3, baseline_noise_std=0.3):
    """
    Generate a matrix where each row is a Gaussian dip series, with noise added to the Gaussian region
    and to the baseline row.

    Parameters:
    - baseline (float): Value of the constant baseline.
    - initial_amplitude (float): Minimum value the first Gaussian dip reaches.
    - distances (array): Array of distances controlling the spread of dips.
    - length_impact (int): Length of each Gaussian dip (default: 200).
    - noise_std (float): Standard deviation of the noise to add to the Gaussian region.
    - baseline_noise_std (float): Standard deviation of the noise to add to the baseline.

    Returns:
    - matrix (np.ndarray): Matrix where each row is a Gaussian dip or noisy baseline series.
    """
    # Compute the dip times based on the distances
    dip_times = np.array(distances) * 50  # Times for dips: t = 50 * distance
    
    # Total length of the series based on the last Gaussian dip position
    full_length = int(max(dip_times)) + length_impact + 50  # Add some buffer
    
    # Create the baseline series with noise
    baseline_series = np.random.normal(loc=baseline, scale=baseline_noise_std, size=full_length)
    
    # Initialize a list to hold all the rows
    rows = [baseline_series]  # Start with the noisy baseline as the first row
    
    # Generate Gaussian dip template
    x = np.linspace(-3, 3, length_impact)  # x-values for the Gaussian curve
    
    # Generate each Gaussian dip series
    for distance in distances:
        # Calculate the minimum value for this Gaussian dip
        min_value = baseline - (baseline - initial_amplitude) * math.exp(-distance / 8)
        
        # Scale the Gaussian dip so it dips to the correct minimum value
        gaussian_dip = baseline - (baseline - min_value) * np.exp(-0.5 * x**2)
        
        # Add Gaussian noise to the valid Gaussian region
        noise = np.random.normal(loc=0, scale=noise_std, size=length_impact)
        gaussian_dip_noisy = gaussian_dip + noise
        
        # Create a series with NaNs everywhere
        dip_series = np.full(full_length, np.nan, dtype=float)
        
        # Add the Gaussian dip with noise at the correct time
        start_index = int(distance * 50)
        end_index = start_index + length_impact
        if end_index <= full_length:
            dip_series[start_index:end_index] = gaussian_dip_noisy
        
        # Add this dip series as a new row
        rows.append(dip_series)
    
    # Convert the list of rows into a matrix
    matrix = np.vstack(rows)
    return matrix

# Parameters
baseline = 7
initial_amplitude = 3  # The first Gaussian dip reaches 3
distances = np.arange(0, 10.5, 0.5)  # Distances controlling the Gaussian dips
noise_std = 0.3  # Standard deviation of the noise for Gaussian dips
baseline_noise_std = 0.3  # Standard deviation of the noise for baseline

# Generate the matrix
# gaussian_matrix = generate_gaussian_matrix(baseline, initial_amplitude, distances, noise_std=noise_std, baseline_noise_std=baseline_noise_std)

# # Plot the rows to visualize
# plt.figure(figsize=(12, 8))
# for i, row in enumerate(gaussian_matrix):
#     label = "Baseline (Noisy)" if i == 0 else f"Gaussian Dip {i}"
#     plt.plot(row, label=label, alpha=0.7)

# plt.title("Gaussian Matrix with Noise Added to Baseline and Dips")
# plt.xlabel("Time")
# plt.ylabel("Value")
# plt.legend()
# plt.show()

def generate_gaussian_matrix_variable_impact(baseline, initial_amplitude, distances, length_impact=200, noise_std=0.3, baseline_noise_std=0.3, impact=0):
    """
    Generate a matrix where each row is a Gaussian dip series, with noise added to the Gaussian region
    and to the baseline row.

    Parameters:
    - baseline (float): Value of the constant baseline.
    - initial_amplitude (float): Minimum value the first Gaussian dip reaches.
    - distances (array): Array of distances controlling the spread of dips.
    - length_impact (int): Length of each Gaussian dip (default: 200).
    - noise_std (float): Standard deviation of the noise to add to the Gaussian region.
    - baseline_noise_std (float): Standard deviation of the noise to add to the baseline.

    Returns:
    - matrix (np.ndarray): Matrix where each row is a Gaussian dip or noisy baseline series.
    """
    # Compute the dip times based on the distances
    dip_times = np.array(distances) * 50  # Times for dips: t = 50 * distance
    
    # Total length of the series based on the last Gaussian dip position
    full_length = int(max(dip_times)) + length_impact + 50  # Add some buffer
    
    # Create the baseline series with noise
    baseline_series = np.random.normal(loc=baseline, scale=baseline_noise_std, size=int(full_length))
    
    # Initialize a list to hold all the rows
    rows = [baseline_series]  # Start with the noisy baseline as the first row
    
    # Generate Gaussian dip template
    x = np.linspace(-3, 3, int(length_impact))  # x-values for the Gaussian curve
    
    # Generate each Gaussian dip series
    for distance in distances:
        # Calculate the minimum value for this Gaussian dip
        min_value = baseline - (baseline - initial_amplitude) * math.exp(-abs(distance - impact) / 8)
        
        # Scale the Gaussian dip so it dips to the correct minimum value
        gaussian_dip = baseline - (baseline - min_value) * np.exp(-0.5 * x**2)
        
        # Add Gaussian noise to the valid Gaussian region
        noise = np.random.normal(loc=0, scale=noise_std, size=length_impact)
        gaussian_dip_noisy = gaussian_dip + noise
        
        # Create a series with NaNs everywhere
        dip_series = np.full(full_length, np.nan, dtype=float)
        
        # Add the Gaussian dip with noise at the correct time
        start_index = int(distance * 50)
        end_index = start_index + length_impact
        if end_index <= full_length:
            dip_series[start_index:end_index] = gaussian_dip_noisy
        
        # Add this dip series as a new row
        rows.append(dip_series)
    
    # Convert the list of rows into a matrix
    matrix = np.vstack(rows)
    return matrix