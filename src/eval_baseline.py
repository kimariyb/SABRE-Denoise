from typing import List

from baselines.cadzow import cadzow
from baselines.windows import exponential_window, gauss_window
from baselines.utils import snr, snrp, rmse
import os
import numpy as np
import pandas as pd


def denoise(noisy_list: List[np.ndarray], method: str):
    result = []
    for noisy in noisy_list:
        if method == 'exponential_window':
            clean = exponential_window(noisy)
        elif method == 'gauss_window':
            clean = gauss_window(noisy, lb=1.0)
        elif method == 'cadzow':
            clean = cadzow(noisy)
        else:
            raise ValueError

        result.append(clean)

    return result


def evaluate(noisy_list: List[np.ndarray], clean_list: List[np.ndarray]):
    snrs, snrps, rmses = [], [], []
    for noisy, clean in zip(noisy_list, clean_list):
        snrs.append(snr(noisy))
        snrps.append(snrp(noisy, i=6600, j=6800))
        rmses.append(rmse(noisy, clean))

    return np.mean(snrs), np.mean(snrps), np.mean(rmses)


if __name__ == '__main__':
    # load dataset
    test_dif = "../data/test/"

    # get the all the files
    files = sorted([f for f in os.listdir(test_dif) if f.endswith('.csv')])
    spectras = []
    for f in files:
        df = pd.read_csv(os.path.join(test_dif, f), sep=',')
        real_part = df.iloc[:, 0]
        imag_part = df.iloc[:, 1]
        real_vals = real_part.to_numpy(dtype=np.float32)
        imag_vals = imag_part.to_numpy(dtype=np.float32)
        s = real_vals + 1j * imag_vals
        spectras.append(s)

    result = {
        "exponential_window": {},
        "gauss_window": {},
        "cadzow": {}
    }


    for method in result.keys():
        denoise_list = denoise(spectras, method)
        snr_np, snrp_np, _ = evaluate(denoise_list, spectras)
        result[method] = {
            "SNR": snr_np,
            "SNRP": snrp_np
        }


    print(result)
    #
    # {'exponential_window': {'SNR': 3.8033341291328555, 'SNRP': 1.2823205623621068},
    #  'gauss_window': {'SNR': 4.810358889094182, 'SNRP': 1.0190784085722986},
    #  'cadzow': {'SNR': 13.036697967465136, 'SNRP': 0.6980366562515076}}
