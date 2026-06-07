import os
import numpy as np

from typing import Tuple
from dataset import loadSpectra, splitSpectra, normalize, generateNoiseSpectra


def prepreocess(
    csv_root: str, save_path: str,
    split_range: Tuple[int, int] = (0, 5000),
    noise_range: Tuple[float, float] = (1e-4, 1e-2),
    pre_count: int = 50000
):
    # 参数校验
    if not os.path.isdir(csv_root):
        raise ValueError(f"csv_root 路径无效: {csv_root}")
    if split_range[0] >= split_range[1]:
        raise ValueError(f"split_range 范围无效: {split_range}")
    if noise_range[0] < 0 or noise_range[1] < 0 or noise_range[0] >= noise_range[1]:
        raise ValueError(f"noise_range 范围无效: {noise_range}")
    if pre_count <= 0:
        raise ValueError(f"pre_count 必须为正整数: {pre_count}")

    # 获取所有文件路径
    files = [f for f in os.listdir(csv_root) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError(f"csv_root路径下未找到任何CSV文件: {csv_root}")

    train_data = []
    for f in files:
        csv_path = os.path.join(csv_root, f)

        # load data
        spectra = loadSpectra(csv_path)

        # split data
        split_data = splitSpectra(spectra, split_range)

        # normalize data
        normalized_data, _, _ = normalize(split_data)

        # generate noise
        data = generateNoiseSpectra(normalized_data, pre_count, noise_range)

        train_data.extend(data)

    # save data
    np.savez(save_path, np.array(train_data), allow_pickle=False)


if __name__ == '__main__':

    # Get the Train CSV path
    TRAIN_DATA_PATH = '../data/train/'
    SAVE_DATA_PATH = '../data/train.npz'

    prepreocess(TRAIN_DATA_PATH, SAVE_DATA_PATH,
                (41308, 49500), (0.05, 1.5), 16000)

