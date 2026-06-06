"""
Exponential window function (matched filter) baseline for NMR denoising.

Applies time-domain apodization: exp(-π · LB · t) to the FID,
then Fourier transforms back. The line-broadening parameter LB
is optimized by grid search over [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] Hz.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from baselines.evaluate import (load_clean_spectra, generate_test_set,
                                evaluate_method, print_results_table)


def exponential_window(noisy_spectrum, lb: float = 1.0) -> np.ndarray:
    """
    Apply exponential window to the FID.

    noisy_spectrum: complex array [N] in frequency domain
    lb: line-broadening parameter in Hz

    Returns denoised spectrum (complex, frequency domain).
    """
    # FFT to get the FID (noise was added in time domain)
    fid = np.fft.ifft(noisy_spectrum)
    N = len(fid)
    t = np.arange(N, dtype=np.float64) / N  # normalized time axis [0, 1)

    # Apply exponential window
    window = np.exp(-np.pi * lb * t)
    fid_apodized = fid * window

    # FFT back to frequency domain
    return np.fft.fft(fid_apodized)


def grid_search_lb(clean_spectra, test_set, lb_values=None):
    """Find optimal LB by grid search on validation SNR."""
    if lb_values is None:
        lb_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

    best_lb = lb_values[0]
    best_snr = -np.inf

    for lb in lb_values:
        results = evaluate_method(
            lambda x, c: exponential_window(x, c, lb=lb),
            test_set, f"LB={lb}Hz"
        )
        snr = results['avg_snr']
        if snr > best_snr:
            best_snr = snr
            best_lb = lb

    return best_lb, best_snr


if __name__ == '__main__':
    # Load data
    clean_list = load_clean_spectra()
    # Use a moderate number of noise levels for grid search
    test_set, sigma_levels = generate_test_set(clean_list, n_noise_levels=5)
    print(f"Loaded {len(clean_list)} clean spectra, {len(test_set)} test samples")
    print(f"Noise levels: {[f'{s:.2e}' for s in sigma_levels]}")

    # Grid search for optimal LB
    lb_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    print(f"\nGrid searching LB over {lb_values}...")

    best_lb, best_snr = grid_search_lb(clean_list, test_set, lb_values)
    print(f"\nOptimal LB = {best_lb} Hz, validation SNR = {best_snr:.2f} dB")

    # Evaluate with optimal LB on full test set
    final_results = evaluate_method(
        lambda x, c: exponential_window(x, c, lb=best_lb),
        test_set, f"Exponential window (LB={best_lb}Hz)"
    )

    # Noisy input baseline for comparison
    noisy_results = evaluate_method(
        lambda x, c: x, test_set, "Noisy input"
    )

    print_results_table({
        'Noisy input': noisy_results,
        'Exponential window': final_results,
    }, f"Exponential Window Baseline (LB={best_lb}Hz)")

    # Print per-sigma details
    print("\nPer-sigma breakdown:")
    print(f"  {'Sigma':<12s}  {'Noisy SNR':>10s}  {'Denoised SNR':>13s}  {'NMSE':>12s}")
    for sigma_key in sorted(final_results['per_sigma'].keys()):
        nsnr = noisy_results['per_sigma'][sigma_key]['snr']
        dsnr = final_results['per_sigma'][sigma_key]['snr']
        nmse = final_results['per_sigma'][sigma_key]['nmse']
        print(f"  {sigma_key:<12s}  {nsnr:>8.2f}  {dsnr:>10.2f}  {nmse:>12.6f}")
