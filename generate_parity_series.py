import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eig

def generate_parity_series(n, switching_rate):
    series = [0]  # Start with 0 explicitly
    p_switch = 1/switching_rate
    print(p_switch)
    for _ in range(1, n):
        if random.random() < p_switch:
            # Flip the previous with probability p_switch
            next_digit = 1 - series[-1]
        else:
            # Keep the previous
            next_digit = series[-1]
        print(next_digit)
        series.append(next_digit)

    return series

def generate_parity_series_dynamic(switching_rates, num_per_rate):
    series = [0]
    for rate in switching_rates:
        p_switch = 1 / rate
        for _ in range(num_per_rate):
            if random.random() < p_switch:
                next_digit = 1 - series[-1]
            else:
                next_digit = series[-1]
            series.append(next_digit)

    return series

def generate_parity_series_with_noise(series, p_noise):
    # Introduce noise by flipping with probability p_noise
    noisy_series = []
    for digit in series:
        if random.random() < p_noise:
            # Flip to add noise
            noisy_digit = 1 - digit
        else:
            # Keep the original
            noisy_digit = digit
        noisy_series.append(noisy_digit)
    
    return noisy_series

def generate_e2e_parity_series_with_noise(n, switching_rate, p_noise):
    series = [0]  # Start with 0 explicitly
    p_switch = 1/switching_rate
    for _ in range(1, n):
        if random.random() < p_switch:
            # Flip the previous with probability p_switch
            next_digit = 1 - series[-1]
        else:
            # Keep the previous
            next_digit = series[-1]
        
        series.append(next_digit)

    # Introduce noise by flipping with probability p_noise
    noisy_series = []
    for digit in series:
        if random.random() < p_noise:
            # Flip to add noise
            noisy_digit = 1 - digit
        else:
            # Keep the original
            noisy_digit = digit
        noisy_series.append(noisy_digit)
    
    return noisy_series

series = generate_parity_series(50, 3)
noisy_series = generate_parity_series_with_noise(series, 0.01)

plt.figure(figsize=(10, 5))
plt.plot(noisy_series, marker='o', linestyle='-', markersize=4, label='Noisy Series')
plt.title('Noisy Series Plot')
plt.xlabel('Index')
plt.ylabel('Parity')
plt.legend()
plt.grid(True)
plt.show()
