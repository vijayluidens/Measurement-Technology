import numpy as np
import matplotlib.pyplot as plt

def correlation_function(x):
    """
    Computes the sample autocorrelation function R_xx(Δi) for a 1D signal x,
    for Δi = 0, 1, ..., N-1.

    Parameters
    ----------
    x : array_like
        The measured signal (1D array).

    Returns
    -------
    R : numpy.ndarray
        1D array of correlation values, where R[Δi] = R_xx(Δi).
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    x_mean = np.mean(x)
    R = np.zeros(N, dtype=float)
    
    for lag in range(N):
        s = 0.0
        for i in range(N - lag):
            s += (x[i] - x_mean) * (x[i + lag] - x_mean)
        R[lag] = s / (N - lag)
    return R

def estimate_noise_variance(x):
    """
    Estimate the noise variance assuming that the noise is white and dominates
    the small differences (x[i+1] - x[i]).

    σ_n² ≈ 0.5 * Var( x[i+1] - x[i] )
    """
    diffs = np.diff(x)
    var_diffs = np.var(diffs, ddof=1)  # sample variance
    sigma_n2 = 0.5 * var_diffs
    return sigma_n2

def estimate_independent_samples(x):
    """
    Estimate the number of effectively independent samples by computing the
    integral correlation time from the autocorrelation function.

    Returns:
        N_indep: effective number of independent samples
        rho: the normalized autocorrelation function array
    """
    R = correlation_function(x)
    R0 = R[0]
    rho = R / R0  # normalized autocorrelation
    
    # Integral correlation time: T_c = 1 + 2 * sum_{k>=1} rho(k)
    T_c = 1.0 + 2.0 * np.sum(rho[1:])
    N = len(x)
    N_indep = N / T_c
    return N_indep, rho

def main():
    # Load the measured signal from file.
    x = np.loadtxt(r"Assignment 1\Question 5\signal_x_2024.txt", skiprows=1)
    N = len(x)
    
    # (a) Estimate the noise variance.
    sigma_n2 = estimate_noise_variance(x)
    print("(a) Estimated noise variance =", sigma_n2)
    
    # (b) Compute the total variance and the physical-signal variance.
    var_x = np.var(x, ddof=1)
    sigma_s2 = var_x - sigma_n2
    print("(b) Measured total variance =", var_x)
    print("    Estimated physical-signal variance =", sigma_s2)
    
    # (c) Estimate the number of effectively independent samples.
    N_indep, rho = estimate_independent_samples(x)
    print("(c) Estimated number of independent samples =", N_indep)
    
    # Compute and plot the correlation function for the measured signal.
    R_measured = correlation_function(x)
    plt.figure(figsize=(10,6))
    
    # Plot for the measured signal
    plt.subplot(2,1,1)
    plt.plot(R_measured, marker='o', markersize=2, linestyle='-', label="Measured Signal")
    plt.title("Correlation Function of Measured Signal")
    plt.xlabel("Lag (Δi)")
    plt.ylabel("R_xx(Δi)")
    plt.grid(True)
    plt.legend()

    # (d) Generate a random signal of the same length and plot its correlation function.
    random_signal = np.random.randn(N)  # white noise signal
    R_random = correlation_function(random_signal)
    
    plt.subplot(2,1,2)
    plt.plot(R_random, marker='o', markersize=2, linestyle='-', color='red', label="Random Signal")
    plt.title("Correlation Function of Random Signal")
    plt.xlabel("Lag (Δi)")
    plt.ylabel("R_xx(Δi)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Additional Plot: Overlay measured signal vs. random signal as dots.
    plt.figure(figsize=(10,4))
    indices = np.arange(N)
    plt.scatter(indices, x, label="Measured Signal", color='blue', s=10)
    plt.scatter(indices, random_signal, label="Random Signal", color='red', s=10, alpha=0.7)
    plt.title("Overlay: Measured Signal vs. Random Signal")
    plt.xlabel("Sample Index")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
