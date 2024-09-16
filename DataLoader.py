
import os
import glob
import dill
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from scipy.signal import find_peaks


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


class DataLoader:
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

    def preprocess_data(self, data, noise_level=0.01):
        # 预处理数据，缩放实部和虚部到 [-1, 1]
        max_abs_value = torch.max(torch.abs(data))
        data.real = data.real / max_abs_value
        data.imag = data.imag / max_abs_value
        
        # 找到数据中最右边的峰值，并从此峰值开始往左截取 8192 个数据点
        peaks, _ = find_peaks(torch.abs(data), height=0.008)
        start_index = peaks[-1] - 8192
        end_index = peaks[-1]
        data = data[start_index-100:end_index-100]
        
        # 加入随机权重的高斯白噪声
        noise = torch.randn(data.shape) * noise_level
        data = (data.real + noise) + (1j * (data.imag + noise))
        
        return data
    
    def convert_data(self, data):
        # input: (batch_size, 8192, 1)
        # output: (batch_size, 1, 8192, 2)
        data = data.unsqueeze(1)
        data = torch.cat ((data.real, data.imag), dim=-1)
        data = data.unsqueeze(0)
        
        return data
    
    def save_data(self, save_path, generated_num):
        row_datas = self.load_data()

        data_list = []
        label_list = []
        
        for row_data in row_datas:
            for i in range(generated_num):
                # 随机生成噪声等级
                noise_level = np.random.uniform(0.005, 0.2)
                preprocessed_data = self.preprocess_data(row_data, noise_level)
                # 转换数据维度 -> (batch_size, 1, 8192, 2)
                preprocessed_data = self.convert_data(preprocessed_data)
                data_list.append(preprocessed_data)
                label_list.append(row_data)
        
        # 保存数据
        saber_data = SaberDataset(data_list, label_list)
        with open(save_path, 'wb') as f:
            dill.dump(saber_data, f)
        
        print('Data saved to', save_path)
        

class SaberDataset(Dataset): 
    def __init__(self, data, label):
        self.data = data   
        self.label = label
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]


if __name__ == '__main__':
    data_path = './data'
    save_path = './data/saber_data.pkl'
    
    data_loader = DataLoader(data_path)
    data_loader.save_data(save_path, 4000)
    
