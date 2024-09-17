
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from DataReader import *
from SabreNet import *

from time import time
from tqdm import tqdm

# 生成数据集参数
DATA_PATH = './data'
NUMS = 200
BATCH_SIZE = 32

# 训练参数
EPOCHS = 20
LR = 1e-3
LR_DECAY = 0.8
WEIGHT_DECAY = 1e-4
WORKERS = 8

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 模型参数
MODEL_NAME = 'SabreUNet'
MODEL_PATH = './model'


def train_func(model, optimizer, criterion, train_loader, device):
    # 训练模型
    model.train()
    train_loss = 0.0
        
    for inputs, labels in tqdm(train_loader, desc='Training Epoch'):
        inputs = inputs.to(device).float()
        labels = labels.to(device).float()
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 优化器更新参数
        optimizer.step()
        # 记录训练集损失和准确率
        train_loss += loss.item()

    # 计算平均损失
    train_loss /= len(train_loader)
    
    return train_loss

    
def val_func(model, criterion, val_loader, device):
    # 验证模型
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()
            # 前向传播
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            # 记录验证集损失和准确率
            val_loss += loss.item()
    
    # 计算平均损失和准确率
    val_loss /= len(val_loader)
    
    return val_loss


def main():
    # Generate data
    print('===================== Generating data =====================')
    data_loader = DataReader(data_path=DATA_PATH)
    data = data_loader.generate_data(generated_num=NUMS)
    # 分割数据集为训练集和验证集 9 : 1
    train_data, val_data = data_loader.split_data(data, ratio=0.9)
    
    print('==================== Model initialization ====================')
    
    print('Train data:', len(train_data))
    print('Val data:', len(val_data))
    
    # 将输入输入为 DataLoader
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=WORKERS, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=WORKERS, shuffle=False)
    
    # 打印参数信息
    print('Data path:', DATA_PATH)
    print('Batch size:', BATCH_SIZE)
    print('Num of workers:', WORKERS)
    print('Epochs:', EPOCHS)
    print('Learning rate:', LR)
    print('Learning rate decay:', LR_DECAY)
    print('Weight decay:', WEIGHT_DECAY)
    print('Device:', DEVICE)
    print('Model name:', MODEL_NAME)
    print('Model path:', MODEL_PATH)
        
    # 模型初始化
    model = SabreNet()
    model = model.to(DEVICE)
    model = model.float()
    
    # 记录训练过程
    writer = SummaryWriter(f'./runs/{MODEL_NAME}')

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = NMSELoss().to(DEVICE)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)
    
    # 参数初始化，使用 He    
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                init.zeros_(m.bias)
    
    # 开始训练模型
    print('===================== Start training =====================')
    for epoch in range(EPOCHS):
        # 训练模型
        train_loss = train_func(model, optimizer, criterion, train_loader, DEVICE)
        # 验证模型
        val_loss = val_func(model, criterion, val_loader, DEVICE)
        # 打印训练集和验证集的损失和准确率
        print(f'Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        # 更新学习率
        scheduler.step()
        # 保存模型
        torch.save(model, f'{MODEL_PATH}/{MODEL_NAME} {epoch+1}.pth')
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch+1) 
    
    writer.close()
    
    print('===================== Training finished =====================')
    


if __name__ == '__main__':
    try:
        start_time = time()
        main()
    except Exception as e:
        print(f'Error occurred: {e}')
    finally:
        end_time = time()
        elapsed_time = end_time - start_time

        # 计算小时、分钟和秒
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        # 格式化输出
        print(f'Running time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}')
