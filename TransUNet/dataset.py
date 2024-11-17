import os
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.signal import find_peaks


class NMRData:
    def __init__(self, raw, label):
        self.raw = self.disassemble(raw)
        self.label = self.disassemble(label)
        
        self.is_plot = False
        if self.is_plot is True:
            self.plot()
        
        # raw 和 label 的尺寸必须相同
        assert self.raw.shape == self.label.shape, "raw and label must have the same shape"
    
    def __repr__(self):
        return f"NMRData(raw={self.raw}, label={self.label})"
    
    def __getitem__(self, index):
        return self.raw[index], self.label[index]
    
    def __len__(self):
        return len(self.raw)
    
    def disassemble(self, data):
        # 将数据分解为实部和虚部，并返回一个 tensor 列表
        return torch.stack([data.real, data.imag], dim=1) # (batch_size, 2, seq_len)
    
    def plot(self):
        plt.plot(self.raw[0], label='raw data')
        plt.plot(self.label[0], label='label data')
        plt.legend()
        plt.show()
  
  
class SABREDataset(Dataset):
    def __init__(self, root):
        self.root = root    
        self.data = None
        
        if not os.path.exists(self.processed_paths):
            # 如果没有这个文件夹，则创建
            os.makedirs(self.processed_dir, exist_ok=True)
            self.process()
        
        self.data = torch.load(self.processed_paths)
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def process(self): 
        data_list = []
        
        # 读取原始数据
        csv_dir = os.path.join(self.raw_dir)
        csv_list = os.listdir(csv_dir)
        csv_list = [file_name for file_name in csv_list if file_name.endswith('.csv')]
        
        csv_datas = []
        
        # 处理数据，加入噪声
        for csv in csv_list:
            csv_path = os.path.join(csv_dir, csv)
            csv_data = pd.read_csv(csv_path, header=None, delim_whitespace=True).iloc[:, 1:].values
            csv_data = torch.complex(
                torch.tensor(csv_data[:, 0], dtype=torch.float32),
                torch.tensor(csv_data[:, 1], dtype=torch.float32)
            )
            csv_data = self.split(csv_data, height=0.008)
            csv_datas.append(csv_data)
            
        # 生成数据
        for i in tqdm(range(3000), desc="Generating data"):
            for csv_data in csv_datas:
                label_data = csv_data.clone()
                
                # 随机添加高斯噪声
                noise_level = np.random.choice(['high','mid', 'low'])
                raw_data = self.gauss_noise(csv_data.clone(), noise_level)
                
                data = NMRData(raw=raw_data, label=label_data)
                data_list.append(data)
   
        torch.save(data_list, self.processed_paths)
        
        
    def split(self, data, height=0.008):
        data.real = data.real / torch.max(torch.abs(data.real))
        data.imag = data.imag / torch.max(torch.abs(data.imag))
        
        peaks, _ = find_peaks(torch.abs(data), height=height)
        start_index = peaks[-1] - 8192
        end_index = peaks[-1]
        data = data[start_index-100:end_index-100]

        return data      

    def gauss_noise(self, data, noise_level):
        
        if noise_level == 'high':
            scale = np.random.uniform(0.25, 0.5)   # 高噪声水平
        elif noise_level == 'mid':
            scale = np.random.uniform(0.05, 0.15)   # 中等噪声水平
        elif noise_level == 'low':
            scale = np.random.uniform(0.01, 0.05)  # 低噪声水平
        else:
            raise ValueError("Unknown noise level: Choose 'high', 'mid', or 'low'")
        
        # 给复数数据加入随机的高斯噪声
        data.real = data.real + torch.randn(data.shape) * scale
        data.imag = data.imag + torch.randn(data.shape) * scale

        # 归一化到 [-1, 1]
        data.real = data.real / torch.max(torch.abs(data.real))
        data.imag = data.imag / torch.max(torch.abs(data.imag))
        
        return data
            
    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw")
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed")
    
    @property
    def processed_file_names(self):
        return "nmr_data.pt"
    
    @property
    def processed_paths(self):
        return os.path.join(self.processed_dir, self.processed_file_names)
    
if __name__ == '__main__':
    dataset = SABREDataset(root=r'D:\project\SABRE-Denoise\data')
    print(len(dataset))
    print(dataset[0].raw)
    print(dataset[0].label)
