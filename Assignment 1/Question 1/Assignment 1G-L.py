import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import linregress

# Define constants
R = 287.06  # J/kgK for air
P_min = 0.50e5  # Pa
P_max = 1.50e5  # Pa
num_steps = 400
T_initial = 281  # K
T_step = 0.025  # K

# Generate pressure values (400 equidistant steps)
P_values = np.linspace(P_min, P_max, num_steps)

# Set random seed for reproducibility
np.random.seed(42)

# Randomize pressure values
P_values_random = np.random.permutation(P_values)

# Generate temperature values (temperature still increases sequentially)
T_values_random = T_initial + np.arange(num_steps) * T_step

# Compute density using the ideal gas law: rho = p / (R * T)
rho_values_random = P_values_random / (R * T_values_random)

# Perform linear regression on randomized data
slope_random, y_intercept_random, r_value, p_value, std_err = linregress(P_values_random, rho_values_random)

# Generate regression line for plotting
rho_fit_random = y_intercept_random + slope_random * P_values_random

# Display Results
print('A = ', y_intercept_random)
print('B = ', slope_random)
print('Standard Error = ', std_err)

# Define the maximum pressure in the experiment
P_max_value_random = np.max(P_values_random)

# Compute the predicted rho at maximum pressure using the regression equation
rho_max_predicted_random = y_intercept_random + slope_random * P_max_value_random

# Compute the 95% confidence interval for rho at P_max
t_value_random = stats.t.ppf(0.975, df=len(P_values_random) - 2)  # 95% confidence level, two-tailed
rho_uncertainty_random = t_value_random * std_err

# Compute the confidence interval bounds
rho_max_lower_random = rho_max_predicted_random - rho_uncertainty_random
rho_max_upper_random = rho_max_predicted_random + rho_uncertainty_random

# Display results
print(f"Uncertainty = {rho_uncertainty_random:.10f} kg/m³")
print(f"Confidence Interval = ({rho_max_lower_random:.10f}, {rho_max_upper_random:.10f}) kg/m³")


# Plot rho vs. p with regression line
plt.figure(figsize=(8, 5))
plt.scatter(P_values_random, rho_values_random, label="Randomized Data", color='red', alpha=0.6)
plt.plot(P_values_random, rho_fit_random, label="Regression Line", color='blue', linewidth=2)
plt.xlabel("Pressure (Pa)")
plt.ylabel("Density (kg/m³)")
plt.title("Density vs. Pressure with Regression Fit (Randomized Order)")
plt.legend()
plt.grid(True)
plt.show()
