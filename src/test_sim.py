import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from dataset import SABRETestDataset
from models import SabreSDN


def inference(model, loader, is_saved: bool = False):
    model.eval()

    preds = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            x, y, name = batch
            x = x.cpu()

            pred = model(x)

            # get the real part
            pred = pred[:, 0, :] # 获取实部 [1, 8192]
            label = y[:, 0, :]

            preds.append(pred.numpy())
            current_save_path =  f"./saved_{i + 1}.png" if is_saved else None

            # plot the spectra
            plot_spectra(pred, label, save_path=current_save_path)

    return preds


def plot_spectra(pred, label, save_path=None):
    # 转换输入数据为numpy数组(如果是 PyTorch 张量)
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()

    pred = np.squeeze(pred)
    label = np.squeeze(label)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))

    # generate the x-axis (8192,)
    x = np.arange(8192)

    out = np.zeros_like(pred)
    out[6600:6800] = pred[6600:6800]

    # 绘制预测光谱
    ax1.plot(x, out, 'r-', linewidth=1.0)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_title('Predicted Spectrum')

    # 绘制真实光谱
    ax2.plot(x, label, 'b-', linewidth=1.0)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_title('Real Spectrum')

    # 调整子图间距
    plt.tight_layout()

    # 保存或显示图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"光谱对比图已保存至: {save_path}")
    else:
        plt.show()

    # 关闭图形
    plt.close()



if __name__ == "__main__":
    TEST_DIR = "../data/test"
    MODEL_PATH = "../checkpoints/best_model.pth"

    # 获取当前目录
    current_dir = Path(TEST_DIR)

    # 获取特定扩展名的文件
    files = list(current_dir.glob(f"*.csv"))
    test_data = []
    names = []
    for f in files:
        # load spectra
        df = pd.read_csv(os.path.join(TEST_DIR, f.name), sep=',')

        # first column is real, second column is imaginary
        spectra = df['real'].values + 1j * df['imag'].values

        test_data.append(spectra)
        names.append(f.name)

    print(f"Loaded {len(test_data)} spectra from {len(files)} files")

    # 创建测试数据集
    test_dataset = SABRETestDataset(test_data, names)

    # 创建测试数据模块
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    model = SabreSDN().cpu()
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"成功加载模型: {MODEL_PATH}")

    # 推理
    preds = inference(model, loader, True)
    #
    # print(preds)

