import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import math

# 1) Load your data (assuming whitespace-delimited, one header row)
data = np.loadtxt(r'Measurement-Technology\Assignment 1\Question 2\system_response_2024.txt', skiprows=1)
t = data[:, 0]
h = data[:, 1]

# 2) Use find_peaks on the 'h' data
# Adjust 'distance' or 'prominence' if needed to pick out the main peaks
peaks, properties = find_peaks(h, distance=5, prominence=1)

# 3) Plot the result
plt.plot(t, h, label='Response')
plt.plot(t[peaks], h[peaks], 'ro', label='Peaks found')
plt.xlabel('Time (s)')
plt.ylabel('Output (mm or whatever unit)')
plt.title('Step Response with Peaks')
plt.grid(True)
plt.legend()
plt.show()

# 4) Inspect the peak values
print("Peak indices:", peaks)
print("Peak times:", t[peaks])
print("Peak amplitudes:", h[peaks])

# 5) Determine natural frequency
r = (108.10585465 - 98) / (98.16125775 - 98)
zeta = 1 / (np.sqrt(1 + (2*np.pi / (np.log(r))) ** 2))
print(zeta)

# 6) Determine the natural frequency
t_first_peak = 0.032   # time of the first peak
t_second_peak = 0.060  # time of the second peak
T_d = t_second_peak - t_first_peak

omega_n = (2 * math.pi) / (T_d * math.sqrt(1 - zeta**2))
print("Natural frequency (rad/s):", omega_n)

# 7) Determine the mass 
k = 1.8e3 # N/m
m = k / omega_n**2
print("Mass (kg):", m)