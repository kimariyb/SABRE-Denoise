
import gzip
import os
import glob
import dill
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from scipy.signal import find_peaks
from tqdm import tqdm


class NMSELoss(nn.Module):
    def __init__(self):
        super(NMSELoss, self).__init__()

    def forward(self, input, target):
        # 确保 input 和 target 的形状相同
        if input.shape != target.shape:
            raise ValueError("Input and target must have the same shape.")

        # 计算均方误差 (MSE)
        mse = torch.mean((input - target) ** 2)

        # 计算目标数据的方差
        var_target = torch.var(target)

        # 如果目标数据的方差为0，即所有值都相同，则NMSE为0
        if var_target == 0:
            return torch.tensor(0.0, dtype=torch.float32)

        # 计算归一化均方误差 (NMSE)
        nmse = mse / var_target

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

    def load_data(self): 
        datas = []
        files = os.path.join(self.data_path, '*.csv')
        files = sorted(glob.glob(files))

        for file in files:
            # 读取 csv 文件
            df = pd.read_csv(file, header=None, sep='\t').iloc[:, 1:]
            data = torch.complex(torch.tensor(df.iloc[:, 0].values), torch.tensor(df.iloc[:, 1].values))
            # 数据预处理
            data = self.preprocess_data(data)
            datas.append(data)
                
        return datas

    def preprocess_data(self, data):
        # 预处理数据，缩放实部和虚部到 [-1, 1]
        max_abs_value = torch.max(torch.abs(data))
        data.real = data.real / max_abs_value
        data.imag = data.imag / max_abs_value
        
        # 找到数据中最右边的峰值，并从此峰值开始往左截取 8192 个数据点
        peaks, _ = find_peaks(torch.abs(data), height=0.008)
        start_index = peaks[-1] - 8192
        end_index = peaks[-1]
        data = data[start_index-100:end_index-100]
        
        return data
    
    
    def add_noise(self, data, noise_level):
        # 给复数数据加入随机的高斯噪声
        data.real = data.real + torch.randn(data.shape) * noise_level
        data.imag = data.imag + torch.randn(data.shape) * noise_level
        
        # 归一化到 [-1, 1]
        max_abs_value = torch.max(torch.abs(data))
        data.real = data.real / max_abs_value
        data.imag = data.imag / max_abs_value
        
        return data
    
    def convert_data(self, data):
        # input: (batch_size, 8192, 1)
        # output: (batch_size, 2, 8192, 1)
        data = data.unsqueeze(1)
        data = torch.cat ((data.real, data.imag), dim=-1).T
        data = data.unsqueeze(2)
        
        return data
    
    def generate_data(self, generated_num):
        row_datas = self.load_data()

        data_list = []
        label_list = []
        
        print('Generating data model, the number of data is %d' % (generated_num * len(row_datas)))
        
        for row_data in row_datas:
            for i in tqdm(range(generated_num), desc='Generating data'):
                # 转换数据维度 -> (batch_size, 1, 8192, 2)
                label_data = self.convert_data(row_data)
                label_list.append(label_data)
                # 随机加入噪声
                noise_level = np.random.uniform(0.05, 0.5)
                noise_data = self.add_noise(row_data, noise_level)
                noise_data = self.convert_data(noise_data)
                data_list.append(noise_data)
        
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
    
    data_loader = DataReader(data_path)
    data = data_loader.generate_data(2500)
    print(len(data))
    print(data[0][0])
    
