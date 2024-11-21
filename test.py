from TransUNet.dataset import SABREDataset, NMRData

# dataset = SABREDataset(root=r'./data/', nums=1)

# print(dataset[0].raw.shape)  # This line prints the first item in the dataset

import torch
import torch.nn as nn

from TransUNet.conv_model import ResNet
from TransUNet.encoder_model import Transformer
from TransUNet.decoder_model import DecoderCup
from TransUNet.main_model import SabreNet

class TestResNet:
    def __init__(self):
        # 创建一个 ResNet 实例
        self.model = SabreNet(
            embedding_dim=2048,
            ffn_embedding_dim=8192,
            num_heads=16,
            num_layers=6,
            patch_size=16,
            dropout=0.0,
            attn_dropout=0.0,
        )
        
        # self.model = ResNet()
        
        # 设置模型为评估模式
        self.model.eval()
        
        # 创建一个输入张量，假设批量大小为1，通道数为1，长度为128
        self.input_tensor = torch.randn(1, 1, 8192)

    def test_forward(self):
        # 进行前向传播
        with torch.no_grad():  # 禁用梯度计算
            output = self.model(self.input_tensor)
        
        # 输出结果的形状
        print("输出形状:", output.shape)
        
        # for i in feat:
        #     print(i.shape)

# 运行测试
if __name__ == "__main__":
    tester = TestResNet()
    tester.test_forward()
