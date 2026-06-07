import numpy as np


def fft(fid: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft(fid))


def ifft(spec: np.ndarray) -> np.ndarray:
    return np.fft.ifft(np.fft.ifftshift(spec))


def snr(spec, j=0):
    """SNR measure in the frequency domain. Specturm needs to be phased for accurate measure"""

    spec = np.real(spec)
    if j != 0:
        sn = np.abs(np.max(spec) / np.std(spec[j:-1]))
    else:
        sn = np.abs(np.max(spec) / np.std(spec[0: int(0.01 * len(spec))]))  # last 10p of data
    return sn


def snrp(spec, i, j, th=0.01):
    """Peak-to-peak SNR. Need to know the indicies of the max [i] and min[j] peaks."""
    spec = np.real(spec)
    sn = np.abs(
        (spec[i] - spec[j]) / np.std(spec[0:int(th * len(spec))])
    )

    return sn


def ssim(specref, measure):
    X = np.real(measure)
    Y = np.real(specref)
    SSIM = (2 * np.mean(X) * np.mean(Y) + 0) * (2 * np.cov(X,Y)[0][1] + 0) / ((np.mean(X)**2 + np.mean(Y)**2 +0) * (np.std(X)**2 +np.std(Y)**2 + 0))
    return SSIM


def rmse(specref, specrecon):
    """RMSE between two vectors"""
    ref = np.real(specref) / np.max(np.abs(np.real(specref)))
    result = np.real(specrecon) / np.max(np.abs(np.real(specrecon)))

    rmse = np.sqrt(np.mean((result - ref) ** 2))
    return rmse