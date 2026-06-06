"""
Common evaluation infrastructure for NMR denoising baselines.

Loads clean spectra from inference/*.csv, generates noisy test data,
applies denoising, computes SNR and NMSE.
"""
import os, sys
import numpy as np
import torch

# Allow running from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import loadSpectra, splitSpectra, normalize, addNoise, SABREDataset


def load_clean_spectra(csv_dir: str = None,
                       split_range=(41308, 49500)) -> list:
    if csv_dir is None:
        csv_dir = os.path.join(os.path.dirname(__file__), '../../inference')
    csv_dir = os.path.abspath(csv_dir)
    """Load clean spectra from CSV files."""
    files = sorted([f for f in os.listdir(csv_dir) if f.endswith('.csv')])
    spectra = []
    for f in files:
        s = loadSpectra(os.path.join(csv_dir, f))
        s = splitSpectra(s, split_range)
        s, _, _ = normalize(s)
        spectra.append(s)
    return spectra


def generate_test_set(clean_spectra: list, n_noise_levels: int = 5,
                      rng: np.random.Generator = None):
    """
    Generate test set: for each clean spectrum, create noisy variants at
    multiple noise levels. Returns list of dicts:
      {'clean': np.array (8192 complex),
       'noisy': np.array (8192 complex),
       'sigma': float}
    """
    if rng is None:
        rng = np.random.default_rng(42)
    # Log-spaced noise levels covering the training range and beyond
    sigma_levels = np.logspace(-4, -1.3, n_noise_levels)  # 1e-4 to ~0.05
    test_set = []
    for clean in clean_spectra:
        for sigma in sigma_levels:
            noisy = addNoise(clean.copy(), sigma)
            test_set.append({'clean': clean, 'noisy': noisy, 'sigma': sigma})
    return test_set, sigma_levels


def complex_to_tensor(z: np.ndarray) -> torch.Tensor:
    """Convert complex numpy [N] to tensor [1, 2, N]."""
    real = np.real(z).astype(np.float32)
    imag = np.imag(z).astype(np.float32)
    stacked = np.stack([real, imag], axis=0)  # [2, N]
    return torch.from_numpy(stacked).unsqueeze(0)  # [1, 2, N]


def compute_snr(pred: np.ndarray, clean: np.ndarray) -> float:
    """SNR in dB on real part."""
    real_pred = np.real(pred)
    real_clean = np.real(clean)
    real_clean_dm = real_clean - real_clean.mean()
    real_pred_dm = real_pred - real_pred.mean()
    signal_pow = np.mean(real_clean_dm ** 2)
    noise_pow = np.mean((real_clean_dm - real_pred_dm) ** 2)
    return 10 * np.log10(signal_pow / max(noise_pow, 1e-12))


def compute_nmse(pred: np.ndarray, clean: np.ndarray) -> float:
    """NMSE on complex data."""
    err = np.sum(np.abs(pred - clean) ** 2)
    ref = np.sum(np.abs(clean) ** 2)
    return err / max(ref, 1e-12)


def evaluate_method(denoise_fn, test_set: list, method_name: str,
                    **kwargs):
    """
    Generic evaluation: apply denoise_fn to each test sample, compute metrics.

    denoise_fn: callable with signature
        fn(noisy_spectrum) -> denoised_spectrum
        or fn(noisy_spectrum, clean_spectrum) -> denoised_spectrum

    The clean spectrum is always passed; the function may accept or ignore it.

    Returns: dict with per-sigma metrics
    """
    results = {}
    for item in test_set:
        sigma_key = f"{item['sigma']:.2e}"
        if sigma_key not in results:
            results[sigma_key] = {'snr': [], 'nmse': []}

        # Always pass both; the function can accept one or two args
        denoised = denoise_fn(item['noisy'], item['clean'])

        snr = compute_snr(denoised, item['clean'])
        nmse = compute_nmse(denoised, item['clean'])
        results[sigma_key]['snr'].append(snr)
        results[sigma_key]['nmse'].append(nmse)

    # Average over all noise levels for a single summary number
    all_snr = []
    all_nmse = []
    for sigma_key in sorted(results.keys()):
        all_snr.extend(results[sigma_key]['snr'])
        all_nmse.extend(results[sigma_key]['nmse'])

    return {
        'avg_snr': float(np.mean(all_snr)),
        'avg_nmse': float(np.mean(all_nmse)),
        'per_sigma': {
            sk: {
                'snr': float(np.mean(v['snr'])),
                'nmse': float(np.mean(v['nmse']))
            }
            for sk, v in sorted(results.items())
        }
    }


def print_results_table(results: dict, title: str = "Results"):
    """Print a formatted results table."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  {'Method':<20s}  {'SNR (dB)':>10s}  {'NMSE':>10s}")
    print(f"  {'-'*42}")
    for method_name, r in sorted(results.items()):
        print(f"  {method_name:<20s}  {r['avg_snr']:>8.2f}  {r['avg_nmse']:>10.6f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Quick test
    clean_list = load_clean_spectra()
    test_set, sigma_levels = generate_test_set(clean_list, n_noise_levels=3)
    print(f"Loaded {len(clean_list)} clean spectra, {len(test_set)} test samples")
    print(f"Noise levels: {sigma_levels}")

    # Noisy baseline (no denoising)
    noisy_results = evaluate_method(
        lambda x, c=None: x,
        test_set, "Noisy (input)"
    )
