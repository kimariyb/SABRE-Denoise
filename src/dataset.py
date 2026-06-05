import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from typing import Tuple, List
from tqdm import tqdm


def addNoise(spectra: np.ndarray, noise_level: float):
    """
    向频谱数据添加高斯白噪声

    参数:
        spectra (np.ndarray): 输入的频谱数据，复数类型数组
        noise_level (float): 噪声强度系数

    返回:
        np.ndarray: 添加噪声后的频谱数据
    """
    # 输入验证
    if not isinstance(spectra, np.ndarray):
        raise TypeError("spect must be a numpy array")
    if not np.issubdtype(spectra.dtype, np.complexfloating):
        raise TypeError("spect must be a complex array")
    if not isinstance(noise_level, (int, float)):
        raise TypeError("noise_level must be a number")

    # spectra to fid
    fid = np.fft.ifft(spectra)

    # add gaussian noise
    noise_real_imag = np.random.normal(
        loc=0,
        scale=np.sqrt(2) / 2,
        size=(len(fid), 2)
    )
    noise = (noise_real_imag[:, 0] + 1j * noise_real_imag[:, 1]).astype(fid.dtype)

    fid = fid + noise_level * noise

    return np.fft.fft(fid)


def normalize(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    # 类型检查：确保输入是复数类型数组
    if not np.iscomplexobj(data):
        raise TypeError("输入数据必须是复数类型的 numpy 数组")

    # 获取实部和虚部
    real = data.real
    imag = data.imag

    real_factor = np.max(np.abs(real))
    imag_factor = np.max(np.abs(imag))

    eps = 1e-10

    # 归一化到 [-1, 1]
    normalized_real = real / (real_factor + eps)
    normalized_imag = imag / (imag_factor + eps)

    # 重新组合数据
    normalized_data = normalized_real + 1j * normalized_imag

    return normalized_data, real_factor, imag_factor


def generateNoiseSpectra(
        spectra: np.ndarray, count: int = 1000, noise_range: Tuple[float, float] = (5e-4, 5e-3)
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    生成带噪声的频谱数据

    Parameters
    ----------
    spectra: np.ndarray
        输入的频谱数据
    count: int
        生成噪声频谱的数量，默认为1000
    noise_range: Tuple[float, float]
        噪声范围

    Returns
    -------
    Tuple[List[Tuple[np.ndarray, np.ndarray]]
        包含噪声频谱和原始频谱的元组列表
    """
    # 参数验证
    if spectra is None:
        raise ValueError("spectra cannot be None")
    if not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")

    label_data = spectra.copy()

    data = []
    for _ in tqdm(range(count), desc="Generating noise spectra", total=count):
        # 添加噪声
        noise_spectra = addNoise(spectra, np.random.uniform(*noise_range))
        # 归一化
        normalized_data, _, _ = normalize(noise_spectra)
        data.append((normalized_data, label_data))

    return data


def splitSpectra(spectra: np.ndarray, split_range: Tuple[int, int]) -> np.ndarray:
    """
    从频谱数据的后半部分中心位置提取指定长度的数据段

    Parameters
    ----------
    spectra : np.ndarray
        输入的频谱数据数组
    split_range : Tuple[int, int]
        要提取的数据段的起始和结束索引（包含起始索引，不包含结束索引）。

    Returns
    -------
    np.ndarray
        提取的频谱数据段
    """
    # 参数类型检查
    if not isinstance(spectra, np.ndarray):
        raise TypeError("参数 'spectra' 必须是 numpy.ndarray 类型")
    if not isinstance(split_range, tuple) or len(split_range) != 2:
        raise TypeError("参数 'split_range' 必须是一个包含两个整数的元组")
    if not all(isinstance(idx, int) for idx in split_range):
        raise TypeError("参数 'split_range' 中的元素必须是整数")

    start_idx, end_idx = split_range

    # 边界条件检查
    if start_idx < 0 or end_idx < 0:
        raise ValueError("参数 'split_range' 中的索引不能为负数")
    if start_idx >= end_idx:
        raise ValueError("参数 'split_range' 中的起始索引必须小于结束索引")
    if start_idx >= len(spectra) or end_idx > len(spectra):
        raise ValueError("参数 'split_range' 超出了频谱数据的索引范围")

    # 返回提取的数据段
    return spectra[start_idx: end_idx]

def loadSpectra(file_path: str) -> np.ndarray:
    """
    从指定文件路径加载光谱数据并转换为复数数组

    Parameters
    ----------
    file_path : str
        光谱数据文件的路径，文件应包含三列数据：
        第一列：化学位移
        第二列：实部
        第三列：虚部

    Returns
    -------
    np.ndarray
        包含复数光谱数据的二维数组，形状为(n, 2)，其中n为数据点数量
        第一列为实部，第二列为虚部

    Raises
    ------
    FileNotFoundError
        当指定的文件不存在时
    ValueError
        当文件格式不正确或数据无法转换为复数时
    pd.errors.EmptyDataError
        当文件为空时
    """
    # 输入验证
    if not file_path or not isinstance(file_path, str):
        raise ValueError("file_path must be a non-empty string")

    try:
        df = pd.read_csv(file_path, header=None, sep=r'\s+')
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"File is empty: {file_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing file {file_path}: {str(e)}")

    # 数据验证
    if df.empty:
        raise ValueError("Loaded dataframe is empty")

    if df.shape[1] < 3:
        raise ValueError(f"Expected at least 3 columns, but got {df.shape[1]}")

    real_part = df.iloc[:, 1]
    imag_part = df.iloc[:, 2]

    # 类型检查
    if not (np.issubdtype(real_part.dtype, np.number) and np.issubdtype(imag_part.dtype, np.number)):
        raise ValueError("Real and imaginary parts must be numeric.")

    real_vals = real_part.to_numpy(dtype=np.float32)
    imag_vals = imag_part.to_numpy(dtype=np.float32)

    # 检查 NaN 或 Inf
    if not np.isfinite(real_vals).all() or not np.isfinite(imag_vals).all():
        raise ValueError("Data contains NaN or infinite values.")

    return real_vals + 1j * imag_vals


class SABREDataset(Dataset):
    """SABRE去噪数据集基类"""
    def __init__(self, data: List[Tuple[np.ndarray, np.ndarray]]):
        self.data = data
        self.noise, self.label = zip(*self.data)

    def __getitem__(self, index):
        noise_data = self.noise[index] # [8192] Complex
        label_data = self.label[index] # [8192] Complex

        # complex to real
        noise_data = self.complex2real(noise_data) # [8192, 2]
        label_data = self.complex2real(label_data) # [8192, 2]

        # [8192, 2] -> [2, 8192]
        x = torch.FloatTensor(noise_data).permute(1, 0)
        y = torch.FloatTensor(label_data).permute(1, 0)

        return x, y

    def __len__(self):
        return len(self.data)

    @staticmethod
    def complex2real(complex_array: np.ndarray) -> np.ndarray:
        real_array = complex_array.real # [8192,]
        imag_array = complex_array.imag # [8192,]

        return np.stack([real_array, imag_array], axis=1) # [8192, 2]

    @staticmethod
    def real2complex(x: torch.Tensor) -> torch.Tensor:
        # 输入形状: [B, 2, H, W] 或 [B, 2, W]
        # 输出形状: [B, 1, H, W] 或 [B, 1, W]
        if x.dim() == 4:
            channel = x.shape[1] // 2
            return torch.complex(x[:, :channel, :, :], x[:, channel:, :, :])
        elif x.dim() == 3:
            channel = x.shape[1] // 2
            return torch.complex(x[:, :channel, :], x[:, channel:, :])
        else:
            raise ValueError("Invalid input shape")

    def plot(self, index):
        """可视化数据"""
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        # generate the x-axis (8192,)
        x = np.arange(8192)

        # generate the y-axis data for real part
        noise = self.noise[index].real
        label = self.label[index].real

        # plot the raw data
        ax[0].plot(x, noise, label='noise')
        ax[0].set_title('Noise Data')
        # plot the label data
        ax[1].plot(x, label, label='label')
        ax[1].set_title('Label Data')
        plt.show()

    def to_csv(self, index: int, save_path: str):
        noise_data = self.noise[index]

        first_col = noise_data.real
        second_col = noise_data.imag

        # save the data to csv
        df = pd.DataFrame({'real': first_col, 'imag': second_col}, columns=['real', 'imag'])
        df.to_csv(save_path, index=False)


class SABRETestDataset(Dataset):
    def __init__(self, noise_spectra: List[np.ndarray], names: List[str]):
        self.data = noise_spectra
        self.names = names
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        noise_data = self.data[index] # [8192]
        # complex to real
        noise_data = SABREDataset.complex2real(noise_data) # [8192, 2]
        x = torch.FloatTensor(noise_data).permute(1, 0) # [1, 8192, 2] -> [2, 1, 8192]
        y = x.clone()
        return x, y, self.names[index]

    def plot(self, index):
        """可视化数据"""
        plt.figure(figsize=(10, 6))
        # generate the x-axis (8192,)
        x = np.arange(8192)

        # generate the y-axis data for real part
        data = self.data[index].real

        plt.plot(x, data, label='noise')
        plt.title('Noise Data')
        plt.show()



if __name__ == '__main__':
    # load data
    spectra = loadSpectra('data/train/train1.csv')
    print(spectra.shape)
    # split data
    split_data = splitSpectra(spectra, (41308, 49500))
    print(split_data.shape)
    # normalize data
    normalized_data, _, _ = normalize(split_data)
    # generate noise
    data = generateNoiseSpectra(normalized_data, 5,  (1e-4, 1e-2))

    # build dataset
    dataset = SABREDataset(data)

    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=5, shuffle=True)
    for x, y in loader:
        print(x.shape, y.shape)
        print(SABREDataset.real2complex(x).shape)

    for i in range(len(dataset)):
        dataset.plot(i)
        # # # save data
        # dataset.to_csv(i, f'./data/test/sim_{i + 1}.csv')
