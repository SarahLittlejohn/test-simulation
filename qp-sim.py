import random
import matplotlib.pyplot as plt

def generate_switching_series(n, switching_rate):
    series = [0]  # Start with 0 explicitly
    p = 1/switching_rate
    for _ in range(1, n):
        if random.random() < p:
            # Flip the previous digit with probability p
            next_digit = 1 - series[-1]
        else:
            # Keep the previous digit
            next_digit = series[-1]
        
        series.append(next_digit)
    
    return series

# Example usage:
binary_series = generate_switching_series(n=50, switching_rate=3)
plt.figure(figsize=(12, 6))
plt.plot(binary_series, label="Switching Series", color="blue")
plt.xlabel("Index")
plt.ylabel("Qubit Value (0 or 1)")
plt.title("Switching Series Time Series Plot")
plt.legend()
plt.show()