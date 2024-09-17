import torch
import torch.nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from DataReader import DataReader


class TestDataReader(DataReader):
    def __init__(self, data_path):
        super().__init__(data_path)
        
    def get_test_loader(self, batch_size=1, shuffle=False):
        test_data = self.load_data()
        for i in range(len(test_data)):
            test_data[i] = self.add_noise(test_data[i], 0.01)
            test_data[i] = self.convert_data(test_data[i])

        test_dataset = TestDataset(test_data[0])
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        
        return test_loader
    
        
class TestDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data
        label = self.data
                
        return data, label
        

def test(model, test_loader):
    model.eval()
    
    for step, (data, label) in enumerate(test_loader):
        data = data.float().cpu()
        output = model(data)        
    
    output, data = output.cpu(), data.cpu()
    output, data = output.detach().numpy(), data.detach().numpy()
    output, data = output.reshape(2, 8192), data.reshape(2, 8192)

    denoised_data = output[0]
    noised_data = data[0]

    fig = plt.figure(figsize=(10, 10))
    axs = fig.subplots(2, 1)

    axs[0].plot(denoised_data, color='red', label='denoised data')
    axs[1].plot(noised_data,  color='blue', label='denoised data')
    
    plt.show()
    
    
if __name__ == '__main__':
    # 读取数据
    data_path = './data'
    model_path = './model/SabreUNet 2.pth'
    test_loader = TestDataReader(data_path).get_test_loader(batch_size=1, shuffle=False)
    # 读取模型
    model = torch.load(model_path, map_location='cpu').float()
    # 测试模型
    test(model=model, test_loader=test_loader)