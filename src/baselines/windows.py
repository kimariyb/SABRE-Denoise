"""
Exponential window function (matched filter) baseline for NMR denoising.

Applies time-domain apodization: exp(-π · LB · t) to the FID,
then Fourier transforms back. The line-broadening parameter LB
is optimized by grid search over [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] Hz.
"""
import numpy as np


def exponential_window(noisy_spectrum: np.ndarray, lb: float = 1.0) -> np.ndarray:
    """
    Apply exponential window to the FID.

    noisy_spectrum: complex array [N] in frequency domain
    lb: line-broadening parameter in Hz

    Returns denoised spectrum (complex, frequency domain).
    """
    # FFT to get the FID (noise was added in time domain)
    fid = np.fft.ifft(noisy_spectrum)
    td = len(fid)

    if lb != 0:
        # Apply exponential window
        sd = 1e3 / lb
        n = np.linspace(0, td, td)
        fid_apodized = np.multiply(fid, np.exp(-(n / sd)))
    else:
        fid_apodized = fid

    # FFT back to frequency domain
    return np.fft.fft(fid_apodized)


def gauss_window(noisy_spectrum: np.ndarray, lb, c=0.5, ax=0):
    """Gaussian line broadening for whole or half echo fid

    Parameters
    ----------
    noisy_spectrum : ndarray
        1D NMR FID
    lb : int or float
        Amount of line-broadening
    c : float
        center of the gaussian curve, between 0 and 1; 0.5 is symmetric (default)
    ax : int
        axis to Gaussian broaden over for 2D (0 or 1)
    """
    fid = np.fft.ifft(noisy_spectrum)

    td = len(fid)
    if lb != 0:
        sd = 1e3 / lb
        n = np.linspace(-int(c*td)/2,int((1-c)*td)/2,td)
        gauss = ((1/(2*np.pi*sd))*np.exp(-((n)**2)/(2*sd**2)))
        gbfid = np.multiply(fid, gauss)
    else:
        gbfid = fid

    return np.fft.fft(gbfid)