
import gzip
import os
import glob
import dill
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from scipy.signal import find_peaks
from tqdm import tqdm

    
class NMSELoss(nn.Module):
    def __init__(self):
        super(NMSELoss, self).__init__()

    def forward(self, x: torch.Tensor, label: torch.Tensor):
        # 输入的维度实际上是一个 [batch_size, 2, 8192] 的张量
        # 然而我们只需要第一个通道 （实部）的 NMSE 损失，所以只取第 0 维度的值
        x = x[:, 0, :]  
        label = label[:, 0, :]
        
        loss0 = nn.MSELoss(reduction='none')
        squared_difference = loss0(x, label)
        e = torch.sqrt(torch.sum(squared_difference))
        f = torch.sqrt(torch.sum(torch.square(label)))
        nmse = e/f
        
        return nmse
    

class SaberDataset(Dataset): 
    def __init__(self, data, label):
        self.data = data   
        self.label = label
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    

class DataReader:
    def __init__(self, data_path):
        self.data_path = data_path
        
    def preprocess_data(self, data, height=0.008):
        # 预处理数据，缩放实部和虚部到 [-1, 1]
        max_abs_value = torch.max(torch.abs(data))
        data.real = data.real / max_abs_value
        data.imag = data.imag / max_abs_value
        
        # 找到数据中最右边的峰值，并从此峰值开始往左截取 8192 个数据点
        peaks, _ = find_peaks(torch.abs(data), height=height)
        start_index = peaks[-1] - 8192
        end_index = peaks[-1]
        data = data[start_index-100:end_index-100]
        
        return data

    def load_data(self, height=0.008): 
        datas = []
        files = os.path.join(self.data_path, '*.csv')
        files = sorted(glob.glob(files))

        for file in files:
            # 读取 csv 文件
            df = pd.read_csv(file, header=None, sep='\t').iloc[:, 1:]
            data = torch.complex(torch.tensor(df.iloc[:, 0].values), torch.tensor(df.iloc[:, 1].values))
            # 数据预处理
            data = self.preprocess_data(data, height)
            datas.append(data)
                
        return datas  
    
    def add_noise(self, data, noise_level):
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
        max_abs_value = torch.max(torch.abs(data))
        data.real = data.real / max_abs_value
        data.imag = data.imag / max_abs_value
        
        return data
    
    def convert_data(self, data):
        # input: (batch_size, 8192, 1)
        # output: (batch_size, 2, 8192)
        data = data.unsqueeze(1)
        data = torch.cat ((data.real, data.imag), dim=-1).T
        
        return data
    
    def generate_data(self, generated_num):
        row_datas = self.load_data()
        
        print('Generating data model, the number of data is %d' % (generated_num * len(row_datas)))
        
        data_list = []
        label_list = []
        
        for i in tqdm(range(generated_num)):
            for row_data in row_datas:
                label_data = self.convert_data(row_data.clone())
                label_list.append(label_data)
                # 加入噪声
                noise_level = np.random.choice(['high','mid', 'low'])
                noisy_data = self.add_noise(row_data.clone(), noise_level)
                data_list.append(self.convert_data(noisy_data))
        
        saber_data = SaberDataset(data_list, label_list)
        
        return saber_data
    
    
    def save_data(self, data, save_path):
        # 保存数据
        with gzip.open(save_path, 'wb') as f:
            dill.dump(data, f)
        
        print('Data saved to %s' % save_path)
        
    def split_data(self, data: SaberDataset, ratio: float):
        # 划分数据集
        train_num = int(len(data) * ratio)
        train_data, val_data = torch.utils.data.random_split(data, [train_num, len(data) - train_num])
        
        return train_data, val_data


if __name__ == '__main__':
    data_path = './data'
    save_path = './data/saber_data.gz'
    test_path = './test'
    
    data_loader = DataReader(test_path)
    data = data_loader.generate_data(10)
    
    plot_data = data.data[0]
    plot_label = data.label[0]

    plt.subplot(2, 1, 1)
    plt.plot(plot_data[0].numpy())
    plt.title('Data')
    plt.subplot(2, 1, 2)
    plt.plot(plot_label[0].numpy())
    plt.title('label')
    plt.show()