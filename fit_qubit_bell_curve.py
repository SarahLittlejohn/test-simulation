import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from generate_switching_rate_series import generate_switching_rate

# Generate or use your existing switching_rates data
x = np.linspace(0, 1000, 1000)  # Assuming 1000 points for x
# Replace this with your actual data:
switching_rates = generate_switching_rate()  # Placeholder for your data

# Define the reverse bell curve model
def reverse_bell_curve(x, a, b, c, d):
    return -a * np.exp(-((x - b)**2) / (2 * c**2)) + d

# Fit the model to the data
popt, pcov = curve_fit(reverse_bell_curve, x, switching_rates, p0=[10, 200, 100, 7])

# Extract fitted parameters
a, b, c, d = popt

# Generate the fitted curve
fitted_curve = reverse_bell_curve(x, a, b, c, d)

# Plot the data and the fit
plt.figure(figsize=(10, 6))
plt.plot(x, switching_rates, label="Computed Switching Rate", alpha=0.7)
plt.plot(x, fitted_curve, label="Fitted Reverse Bell Curve", color='red', linewidth=2)
plt.xlabel("Switching Rate")
plt.ylabel("Time Step")
plt.title("Inputted Switching Rate and Computed Switching Rates with Fit")
plt.legend()
plt.grid()
plt.show()


