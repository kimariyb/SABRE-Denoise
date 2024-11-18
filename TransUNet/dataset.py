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
        return torch.stack([data.real, data.imag], dim=-2) # (2, seq_len)
    
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
        
        # 处理数据
        for csv in csv_list:
            csv_path = os.path.join(csv_dir, csv)
            csv_data = pd.read_csv(csv_path, header=None, delim_whitespace=True).iloc[:, 1:].values
            csv_data = torch.complex(
                torch.tensor(csv_data[:, 0], dtype=torch.float32),
                torch.tensor(csv_data[:, 1], dtype=torch.float32)
            )
            csv_data = self._split(csv_data, height=0.008)
            csv_data = self._zero(csv_data)
            csv_datas.append(csv_data)
            
        # 生成数据
        for i in tqdm(range(5000), desc="Generating data"):
            for csv_data in csv_datas:
                label_data = csv_data.clone()
                
                # 随机添加高斯噪声
                noise_level = np.random.uniform(0.001, 0.01)
                raw_data = self._noise(csv_data.clone(), noise_level)
                
                data = NMRData(raw=raw_data, label=label_data)
                data_list.append(data)
   
        torch.save(data_list, self.processed_paths)
        
    def _zero(self, data, threshold=0.001):
        # 给数据中的小于阈值的部分置零
        data.real = torch.where(torch.abs(data.real) < threshold, torch.zeros_like(data.real), data.real)
        data.imag = torch.where(torch.abs(data.imag) < threshold, torch.zeros_like(data.imag), data.imag)
        
        return data
        
    def _split(self, data, height=0.008):
        # 首先对数据进行归一化
        data = self.normalize(data)
        # 找到峰值，并将数据分割为 8192 个数据点
        peaks, _ = find_peaks(torch.abs(data), height=height)
        start_index = peaks[-1] - 8192
        end_index = peaks[-1]
        
        data = data[start_index-100:end_index-100]
        # 最后再归一化
        data = self.normalize(data)

        return data      

    def _noise(self, data, noise_level):
        # 首先将 data 经过 IFFT 变换得到时域信号
        fid = torch.fft.ifft(data, dim=-1)
        # 生成高斯噪声
        noise = np.random.normal(loc=0, scale=np.sqrt(2)/2, size=(len(fid),2)).view(np.complex64)
        noise = noise[:, 0]
        # 将噪声加到时域信号中
        fid = fid + noise_level * torch.tensor(noise)
        # 最后再进行 FFT 变换得到频域信号
        data = torch.fft.fft(fid, dim=-1)        
        # 归一化到 [-1, 1]
        data = self.normalize(data)
        
        return data
    
    def normalize(self, data):
        # 对数据进行归一化
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
    
    
class SABRETestDataset(SABREDataset):
    def __init__(self, root):
        super().__init__(root)
        
    def process(self):
        data_list = []
        
        # 读取原始数据
        csv_dir = os.path.join(self.raw_dir)
        csv_list = os.listdir(csv_dir)
        csv_list = [file_name for file_name in csv_list if file_name.endswith('.csv')]
                
        # 处理数据
        for csv in tqdm(csv_list, desc="Generating data"):
            csv_path = os.path.join(csv_dir, csv)
            csv_data = pd.read_csv(csv_path, header=None, delim_whitespace=True).iloc[:, 1:].values
            
            csv_data = torch.complex(
                torch.tensor(csv_data[:, 0], dtype=torch.float32),
                torch.tensor(csv_data[:, 1], dtype=torch.float32)
            )
            csv_data = self._split(csv_data, height=0.8)
            
            data = NMRData(raw=csv_data, label=csv_data)
            data_list.append(data)
            
        torch.save(data_list, self.processed_paths)
        
    
if __name__ == '__main__':
    dataset = SABREDataset(root=r'/home/kimariyb/project/SABRE-Denoise/data/')
    print(len(dataset))
    dataset[0].plot()
    
    
