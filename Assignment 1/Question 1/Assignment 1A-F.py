import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import linregress

# Given constants
R = 287.06  # J/kgK for air
P_min = 0.5e5  # Pa
P_max = 1.50e5  # Pa
num_steps = 400
T_initial = 281  # K
T_step = 0.025  # K

# Generate pressure values (400 equidistant steps)
P_values = np.linspace(P_min, P_max, num_steps)

# Generate temperature values (starting at 281 K and increasing)
T_values = T_initial + np.arange(num_steps) * T_step

# Compute density using the ideal gas law: rho = p / (R * T)
rho_values = P_values / (R * T_values)

# Plot rho vs. p
plt.figure(figsize=(8, 5))
plt.plot(P_values, rho_values, label=r'$\rho = \frac{p}{RT}$', color='blue')
plt.xlabel("Pressure (Pa)")
plt.ylabel("Density (kg/m³)")
plt.title("Density vs. Pressure (Ideal Gas Law)")
plt.legend()
plt.grid(True)
plt.show()

# Perform linear regression using numpy and scipy
slope, y_intercept, r_value, p_value, std_err = linregress(P_values, rho_values)

# Display results
print('A = ', y_intercept)
print('B = ', slope)
print('Standard Error = ', std_err)

# Define the maximum pressure in the experiment
P_max_value = np.max(P_values)

# Compute the predicted rho at maximum pressure using the regression equation
rho_max_predicted = y_intercept + slope * P_max_value

# Compute the 95% confidence interval for rho at P_max
t_value = stats.t.ppf(0.975, df=len(P_values) - 2)  # 95% confidence level, two-tailed
rho_uncertainty = t_value * std_err

# Compute the confidence interval bounds
rho_max_lower = rho_max_predicted - rho_uncertainty
rho_max_upper = rho_max_predicted + rho_uncertainty

# Display Results
print('Uncertainty = ', rho_uncertainty)
print('95% Confidence Interval = ', (rho_max_lower, rho_max_upper))