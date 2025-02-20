import numpy as np
from scipy import stats

# -------------------------
# Parameters (from part (a))
# -------------------------
Re_x = 1e6       # Local Reynolds number
Pr = 0.7         # Prandtl number
x = 1.2          # Distance along the plate in meters
k = 0.0235       # Thermal conductivity in W/(m·K)

# Correlation (1): Nu_x = 0.029 * Re_x^(4/5) * Pr^(1/3)
def correlation1(Re_x, Pr):
    return 0.029 * (Re_x ** (4/5)) * (Pr ** (1/3))

# Correlation (2):
# Nu_x = [4*C_star/27 * Re_x^(4/5) * Pr] / [1 + (12.74*C_star/27)*(Pr^(1/3)-1)]
# with C_star = 0.0592 * Re_x^(-1/8)
def correlation2(Re_x, Pr):
    C_star = 0.0592 * (Re_x ** (-1/8))
    numerator = (4 * C_star / 27) * (Re_x ** (4/5)) * Pr
    denominator = 1 + (12.74 * C_star / 27) * ((Pr ** (1/3)) - 1)
    return numerator / denominator

# Function to compute the heat transfer coefficient h from Nu_x
def compute_h(Nu_x, k, x):
    return (k / x) * Nu_x

# Predicted Nusselt numbers and h values:
Nu1 = correlation1(Re_x, Pr)
Nu2 = correlation2(Re_x, Pr)
h1_pred = compute_h(Nu1, k, x)  # ~31.8 W/(m^2K)
h2_pred = compute_h(Nu2, k, x)  # ~1.35 W/(m^2K)

print("Predicted values:")
print(f"Correlation (1): Nu_x = {Nu1:.2f}  ->  h = {h1_pred:.2f} W/(m^2K)")
print(f"Correlation (2): Nu_x = {Nu2:.2f}  ->  h = {h2_pred:.2f} W/(m^2K)")

# -------------------------------------------
# Load measured data from heatsignal_2024.txt
# -------------------------------------------
# The file is assumed to contain one column of data with 1000 points.
data = np.loadtxt(r"Measurement-Technology\Assignment 1\Question 3\heatsignal_2024.txt", skiprows=1)  # Skip header if present

# Compute sample statistics
n = len(data)
sample_mean = np.mean(data)
sample_std = np.std(data, ddof=1)  # sample standard deviation
std_error = sample_std / np.sqrt(n)

print("\nMeasured data statistics:")
print(f"Sample size (n): {n}")
print(f"Sample mean: {sample_mean:.2f} W/(m^2K)")
print(f"Sample standard deviation: {sample_std:.2f} W/(m^2K)")
print(f"Standard error: {std_error:.2f} W/(m^2K)")

# ---------------------------------------
# Construct 99.7% confidence interval for the true mean
# ---------------------------------------
# For a large sample size the 99.7% CI is approximated by mean ± 3*std_error.
ci_lower = sample_mean - 3 * std_error
ci_upper = sample_mean + 3 * std_error

print("\n99.7% confidence interval for the true mean:")
print(f"({ci_lower:.2f}, {ci_upper:.2f}) W/(m^2K)")

# ---------------------------------------
# Hypothesis testing for each correlation
# ---------------------------------------
# We test H0: μ = h_pred vs. H1: μ ≠ h_pred.
# The test statistic is:
#   z = (sample_mean - h_pred) / std_error
# (Since n=1000 is large, we can approximate using the z-distribution.)
def hypothesis_test(sample_mean, std_error, h_pred):
    z = (sample_mean - h_pred) / std_error
    # For a two-tailed test at 99.7% confidence, the critical z-value is about 3.
    critical_value = 3.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value, critical_value

# Test for correlation (1)
z1, p_value1, crit = hypothesis_test(sample_mean, std_error, h1_pred)
# Test for correlation (2)
z2, p_value2, crit = hypothesis_test(sample_mean, std_error, h2_pred)

print("\nHypothesis test results:")
print("Correlation (1):")
print(f"  z = {z1:.2f}, p-value = {p_value1:.3e}")
if abs(z1) < crit:
    print("  -> The prediction is consistent with the measured data (fail to reject H0).")
else:
    print("  -> The prediction is NOT consistent with the measured data (reject H0).")

print("\nCorrelation (2):")
print(f"  z = {z2:.2f}, p-value = {p_value2:.3e}")
if abs(z2) < crit:
    print("  -> The prediction is consistent with the measured data (fail to reject H0).")
else:
    print("  -> The prediction is NOT consistent with the measured data (reject H0).")
