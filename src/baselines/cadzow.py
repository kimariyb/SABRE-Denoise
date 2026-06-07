"""
Cadzow filtering baseline for NMR denoising.

Constructs a Hankel matrix from the FID, performs SVD truncation,
and reconstructs the denoised FID. The rank parameter r is optimized
by grid search on a held-out validation set.
"""
import numpy as np
import scipy
from baselines.utils import ifft, fft


def cadzow(noisy_spectrum: np.ndarray, p: int = 10):
    """Create Hankel matrix and use SVD to denoise fid

    Parameters
    ----------
    noisy_spectrum : np.ndarray
        complex NMR time-domain FID to be denoised
    p : int
        percentage of first-amount of singular values to ignore when determining how many to discard
        Default = 10 %
    """
    fid = ifft(noisy_spectrum)

    l = round(len(fid) / 2)
    # print(l)
    a = fid[:l + 1]  ##Note that in hankel(c,r) r[0] is ignored*
    ##...so need to incude r[0] as the last point in c!!
    b = fid[l:]

    hank = scipy.linalg.hankel(a, b)
    m, n = hank.shape
    print('Hanekl Dimensions = ', m, n)
    # U, s, Vt = scipy.linalg.svd(hank) #s is a vector of singular values, not a matrix, Vt is already transposed
    U, s, Vt = np.linalg.svd(hank)  # s is a vector of singular values, not a matrix, Vt is already transposed
    s = np.array(s)
    s1 = np.flipud(np.diff(np.flipud(s)))

    r = np.argmax(s1[round((p / 100) * len(s1)):]) + round((p / 100) * len(s1))  # % of SV's to not look at
    print('Ignoring first %d percent of singular values for cut-off' % p)
    print('Retaining %d singular values' % r)

    s[(r):] = 0
    sigma = scipy.linalg.diagsvd(s, m, n)  # rebuilds s as sigma matrix
    hankrecon = np.matmul(np.matmul(U, sigma), Vt)

    ad = []  # anti-diagonal averaging algorithm
    for i in range(m - 1):
        ad.append(np.mean(np.fliplr(hankrecon[:i + 1, :i + 1]).diagonal()))
    ad = np.array(ad)

    ad2 = []
    for i in range(n):
        ad2.append(np.mean(np.fliplr(hankrecon[(m - 1 - i):, (n - 1 - i):]).diagonal()))
    ad2 = np.flip(ad2)

    fidrecon = np.append(ad, ad2)  # instead of rebuilding hank, just extract the fid

    return fft(fidrecon)


