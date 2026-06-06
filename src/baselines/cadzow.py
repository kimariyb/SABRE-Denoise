"""
Cadzow filtering baseline for NMR denoising.

Constructs a Hankel matrix from the FID, performs SVD truncation,
and reconstructs the denoised FID. The rank parameter r is optimized
by grid search on a held-out validation set.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from baselines.evaluate import (load_clean_spectra, generate_test_set,
                                evaluate_method, print_results_table)


def build_hankel(signal: np.ndarray) -> np.ndarray:
    """
    Build a square-like Hankel matrix from a 1D signal.
    For an N-point FID, creates an (N-K+1) × K matrix, K ≈ N/2.
    """
    N = len(signal)
    K = N // 2
    M = N - K + 1
    H = np.zeros((M, K), dtype=signal.dtype)
    for i in range(M):
        H[i, :] = signal[i:i + K]
    return H


def hankelize(H: np.ndarray, N: int) -> np.ndarray:
    """
    Anti-diagonal averaging: recover a 1D signal from a Hankel matrix.

    对每条反对角线（对应输出信号的一个点），直接提取并求均值，
    时间复杂度从 O(M×K) 降至 O(N×min(M,K)/2)，实际加速 10~100×。
    """
    M, K = H.shape
    signal = np.zeros(N, dtype=H.dtype)
    for d in range(N):
        # 第 d 个输出点对应 H 中满足 i+j==d 的所有元素
        # i 的范围：max(0, d-K+1) ~ min(M-1, d)
        i_min = max(0, d - K + 1)
        i_max = min(M - 1, d)
        if i_min > i_max:
            continue
        i_idx = np.arange(i_min, i_max + 1)
        j_idx = d - i_idx
        signal[d] = H[i_idx, j_idx].mean()
    return signal


def cadzow_filter(noisy_spectrum: np.ndarray,
                  rank: int = 10) -> np.ndarray:
    """
    Apply Cadzow filtering to a noisy NMR spectrum.

    Steps:
    1. IFFT to get FID
    2. Build Hankel matrix from FID
    3. SVD truncation to rank r
    4. Hankelize (anti-diagonal average) to get filtered FID
    5. FFT back to frequency domain

    Args:
        noisy_spectrum: complex array [N] in frequency domain
        rank: truncation rank

    Returns:
        denoised spectrum (complex, frequency domain)
    """
    fid = np.fft.ifft(noisy_spectrum)

    # Build Hankel matrix
    H = build_hankel(fid)

    # SVD truncation
    U, S, Vh = np.linalg.svd(H, full_matrices=False)
    r = min(rank, len(S))

    # 用切片直接重建，避免 np.diag(S_trunc) 构造冗余 min(M,K)² 方阵
    H_filtered = (U[:, :r] * S[:r]) @ Vh[:r, :]

    # Anti-diagonal averaging
    fid_filtered = hankelize(H_filtered, len(fid))

    # FFT back
    return np.fft.fft(fid_filtered)


def grid_search_rank(val_set, rank_values=None):
    """
    Find optimal rank by grid search on a validation set.

    Args:
        val_set:     用于超参搜索的验证集（不得与最终评估的 test_set 重叠）
        rank_values: 待搜索的 rank 列表

    Returns:
        (best_rank, best_snr)
    """
    if rank_values is None:
        rank_values = [2, 5, 10, 20, 30, 50, 80, 100, 150, 200]

    best_rank = rank_values[0]
    best_snr = -np.inf

    for r in rank_values:
        results = evaluate_method(
            lambda x, c, _r=r: cadzow_filter(x, rank=_r),
            val_set, f"rank={r}"
        )
        snr = results['avg_snr']
        if snr > best_snr:
            best_snr = snr
            best_rank = r
            print(f"  rank={r} -> SNR={snr:.2f} dB (new best)")

    return best_rank, best_snr


if __name__ == '__main__':
    clean_list = load_clean_spectra()

    n_total = len(clean_list)
    n_val = max(1, n_total // 5)          # 20% 用于验证（超参搜索）
    val_clean   = clean_list[:n_val]
    test_clean  = clean_list[n_val:]

    val_set,  _  = generate_test_set(val_clean,  n_noise_levels=5)
    test_set, sigma_levels   = generate_test_set(test_clean, n_noise_levels=5)

    print(f"Loaded {n_total} clean spectra -> "
          f"{len(val_clean)} val / {len(test_clean)} test")
    print(f"Val samples: {len(val_set)}, Test samples: {len(test_set)}")
    print(f"Noise levels: {[f'{s:.2e}' for s in sigma_levels]}")

    # Grid search on validation set only
    rank_values = [2, 5, 10, 20, 30, 50, 80, 100, 150, 200]
    print(f"\nGrid searching rank over {rank_values} (on val set)...")

    best_rank, best_snr = grid_search_rank(val_set, rank_values)
    print(f"\nOptimal rank = {best_rank}, validation SNR = {best_snr:.2f} dB")

    # Final evaluation on held-out test set
    final_results = evaluate_method(
        lambda x, c: cadzow_filter(x, rank=best_rank),
        test_set, f"Cadzow (rank={best_rank})"
    )

    noisy_results = evaluate_method(lambda x, c: x, test_set, "Noisy input")

    print_results_table({
        'Noisy input': noisy_results,
        'Cadzow filter': final_results,
    }, f"Cadzow Filtering Baseline (rank={best_rank})")

    # Per-sigma breakdown
    print("\nPer-sigma breakdown:")
    print(f"  {'Sigma':<12s}  {'Noisy SNR':>10s}  {'Cadzow SNR':>11s}  {'NMSE':>12s}")
    for sigma_key in sorted(final_results['per_sigma'].keys()):
        nsnr = noisy_results['per_sigma'][sigma_key]['snr']
        dsnr = final_results['per_sigma'][sigma_key]['snr']
        nmse = final_results['per_sigma'][sigma_key]['nmse']
        print(f"  {sigma_key:<12s}  {nsnr:>8.2f}  {dsnr:>10.2f}  {nmse:>12.6f}")