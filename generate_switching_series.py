import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eig

def generate_switching_series(n, switching_rate):
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

    return series

def generate_series_with_noise(series, p_noise):
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

def generate_e2e_series_with_noise(n, switching_rate, p_noise):
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