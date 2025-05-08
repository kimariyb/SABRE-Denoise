import os
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.signal import find_peaks

  
class SABREDataset(Dataset):
    r"""SABRE去噪数据集基类"""
    def __init__(self, root, nums=5000):
        self.root = root   
        self.nums = nums
        self.data = None
        
        if not os.path.exists(self.processed_path):
            os.makedirs(self.processed_dir, exist_ok=True)
            self.process()
        
        self.data = np.load(self.processed_path, allow_pickle=True)
        # 转换为 Tensor
        self.data = torch.from_numpy(self.data).type(torch.float32)
        
        # get the raw and label
        self.raw = self.data[:, :, 0].unsqueeze(1)
        self.label = self.data[:, :, 1].unsqueeze(1)
        
    def __getitem__(self, index):
        return self.raw[index], self.label[index]

    def __len__(self):
        return len(self.raw)
        
    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw")
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed")
    
    @property
    def processed_file_name(self):
        return "nmr_data.npy"
    
    @property
    def processed_path(self):
        return os.path.join(self.processed_dir, self.processed_file_name)
    
    def __len__(self):
        return len(self.data)
    
    def _load_csv_data(self, file_path):
        r"""加载并处理单个CSV文件"""
        data = pd.read_csv(file_path, header=None, sep='\s+').iloc[:, 1:].values
        return np.complex128(data[:, 0] + 1j * data[:, 1])
        
    def _split_data(self, data, height=0.008):
        r"""分割数据为固定长度的片段"""
        data = self._normalize(data)
        peaks, _ = find_peaks(np.abs(data), height=height)
        
        if len(peaks) == 0:
            raise ValueError("No peaks found in the data")
            
        start_index = peaks[-1] - 8192
        end_index = peaks[-1]
        data = data[start_index-100:end_index-100]
        return self._normalize(data)
    
    def _add_noise(self, data, noise_level):
        r"""向数据添加噪声"""
        fid = np.fft.ifft(data)
        # 增加高斯噪声
        noise = np.random.normal(
            loc=0, 
            scale=np.sqrt(2)/2, 
            size=(len(fid),2)
        ).view(np.complex128)
        noise = noise[:,0]
        fid = fid + noise_level * noise
        
        # 傅里叶变换
        noised_data = np.fft.fft(fid)
        
        return self._normalize(noised_data)
    
    def _normalize(self, data: np.ndarray):
        r"""归一化复数数据"""
        max_factor = np.max(np.abs(data.real))
        data.real = data.real / max_factor if max_factor != 0 else data.real
        data.imag = data.imag / max_factor if max_factor != 0 else data.imag

        return data
    
    def process(self):
        r"""处理原始数据并生成数据集"""
        data_list = []
        
        # 加载所有CSV文件
        csv_files = sorted(
            f for f in os.listdir(self.raw_dir) 
            if f.endswith('.csv')
        )
        
        csv_data = [
            self._split_data(
                self._load_csv_data(os.path.join(self.raw_dir, f))
            ) for f in csv_files
        ]
        
        # 生成带噪声的数据
        for _ in tqdm(range(self.nums), desc="Generating training data"):
            for clean_data in csv_data:
                noise_level = np.random.uniform(5e-4, 5e-3)
                label_data = clean_data.copy()
                noisy_data = self._add_noise(label_data, noise_level)
                
                # data[:, 0] is noisy data, data[:, 1] is clean data
                data = np.stack((noisy_data.real, label_data.real), axis=1)
                
                # save the raw data and label data to the list
                data_list.append(data)
        
        # 保存处理后的数据集
        np.save(self.processed_path, data_list, allow_pickle=True)
           
    def plot_data(self, index):
        r"""可视化数据"""
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        # generate the x-axis (8192,)
        x = np.arange(8192)
    
        # generate the y-axis
        raw_data = self.raw[index].numpy().reshape(-1)
        label_data = self.label[index].numpy().reshape(-1)
        
        # plot the raw data
        ax[0].plot(x, raw_data, label='raw')
        ax[0].set_title('Raw Data')
        # plot the label data
        ax[1].plot(x, label_data, label='label')
        ax[1].set_title('Label Data')
        plt.show()
  
  
class SABRETestDataset(SABREDataset):
    r"""SABRE测试数据集"""
    def __init__(self, root):
        super().__init__(root, nums=0)  # nums=0 表示不生成额外数据
    
    def process(self):
        r"""处理测试数据"""
        data_list = []
        
        csv_files = sorted(
            f for f in os.listdir(self.raw_dir) 
            if f.endswith('.csv')
        )
        
        for file in tqdm(csv_files, desc="Processing test data"):
            file_path = os.path.join(self.raw_dir, file)
            clean_data = self._load_csv_data(file_path)
            clean_data = self._split_data(clean_data, height=0.7)
            
            # data[:, 0] is noisy data, data[:, 1] is clean data
            data = np.stack((clean_data.real, clean_data.real), axis=1)
            
            # save the raw data and label data to the list
            data_list.append(data)
        
        # 保存处理后的数据集
        np.save(self.processed_path, data_list, allow_pickle=True)  
    
    
