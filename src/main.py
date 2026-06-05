import csv
import os
import torch
import torch.nn as nn
import numpy as np

from datetime import datetime
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import SABREDataset
from models import SabreSDN
from trainer import train_fn, eval_fn
from utils import count_parameters


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SPECTRA_PATH = '../data/train.npz'
BATCH_SIZE = 256
EPOCHES = 50
LEARNING_RATE = 1e-3
LR_MIN = 1e-6
LR_FACTOR = 0.8
LR_PATIENCE = 5
SAVE_INTERVAL = 1
CHECKPOINT_DIR = '../checkpoints'
LOG_DIR = '../logs'


def save_checkpoint(state, filename):
    """保存模型检查点"""
    torch.save(state, filename)
    print(f"模型已保存到 {filename}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """加载模型检查点"""
    if os.path.isfile(checkpoint_path):
        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"从 epoch {start_epoch} 恢复训练")
        return start_epoch, best_val_loss
    else:
        print(f"警告: 未找到检查点文件 {checkpoint_path}")
        return 0, float('inf')


def main():
    # Print info
    print(f"设备: {DEVICE}")
    # 创建目录
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"已创建目录 {CHECKPOINT_DIR}")
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"已创建目录 {LOG_DIR}")

    # 创建带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOG_DIR, f'training_log_{timestamp}.csv')

    # 创建CSV文件并写入表头
    with open(log_filename, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'train_loss', 'val_loss',
                      'train_snr', 'val_snr',           # ← 新增
                      'learning_rate', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # Load data
    with np.load(SPECTRA_PATH, allow_pickle=False) as loaded_data:
        spectra_data = loaded_data['arr_0']

    # Create dataset
    dataset = SABREDataset(spectra_data)
    print(f"数据集大小: {len(dataset)}")

    # Split train and val
    train_dataset, val_dataset = random_split(
        dataset, [int(0.8 * len(dataset)), int(0.2 * len(dataset))]
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=8)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=8)

    # Create model
    model = SabreSDN(num_blocks=1, use_attention=True,
                     use_gated_skip=True).to(DEVICE)

    loss_fn = nn.L1Loss()

    # Create optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_FACTOR,
                                  patience=LR_PATIENCE, min_lr=LR_MIN)

    best_val_loss = float('inf')

    total_params = count_parameters(model)
    print(f"模型总参数数量: {total_params:,}")
    print(f"约等于: {total_params / 1e6:.2f} 百万参数")

    for epoch in range(EPOCHES):
        epoch_start = datetime.now()

        # ── 训练 / 验证，解包 loss 和 snr ──────────────────────
        train_loss, train_snr = train_fn(model, train_loader, optimizer, loss_fn, DEVICE)
        val_loss,   val_snr   = eval_fn(model, val_loader, loss_fn, DEVICE)

        scheduler.step(val_loss)
        current_lr     = optimizer.param_groups[0]['lr']
        epoch_duration = datetime.now() - epoch_start

        # ── 打印 ────────────────────────────────────────────────
        print(f'Epoch: {epoch}/{EPOCHES}, '
              f'Train Loss: {train_loss:.6f}, '
              f'Val Loss: {val_loss:.6f}, '
              f'Train SNR: {train_snr:.2f} dB, '  # ← 新增
              f'Val SNR: {val_snr:.2f} dB, '       # ← 新增
              f'LR: {current_lr:.2e}, '
              f'Time: {epoch_duration}')

        # ── CSV 记录 ─────────────────────────────────────────────
        with open(log_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'epoch':         epoch,
                'train_loss':    train_loss,
                'val_loss':      val_loss,
                'train_snr':     round(train_snr, 4),   # ← 新增
                'val_snr':       round(val_snr,   4),   # ← 新增
                'learning_rate': current_lr,
                'timestamp':     datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        # ── 保存最佳模型 ─────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss':           train_loss,
                'val_loss':             val_loss,
                'train_snr':            train_snr,
                'val_snr':              val_snr,
                'best_val_loss':        best_val_loss,
            }, filename=os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            print(f"保存最佳模型，验证损失: {val_loss:.6f}，Val SNR: {val_snr:.2f} dB")

        # ── 定期保存 ─────────────────────────────────────────────
        if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHES - 1:
            save_checkpoint({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss':           train_loss,
                'val_loss':             val_loss,
                'train_snr':            train_snr,
                'val_snr':              val_snr,
                'best_val_loss':        best_val_loss,
            }, filename=os.path.join(CHECKPOINT_DIR, f'epoch_{epoch}.pth'))
            print(f"保存检查点: epoch_{epoch}.pth")

    # ── 训练完成后保存最终模型 ────────────────────────────────────
    save_checkpoint({
        'epoch':                EPOCHES - 1,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss':           train_loss,
        'val_loss':             val_loss,
        'train_snr':            train_snr,
        'val_snr':              val_snr,
        'best_val_loss':        best_val_loss,
    }, filename=os.path.join(CHECKPOINT_DIR, 'final_model.pth'))
    print("训练完成，保存最终模型")


if __name__ == '__main__':
    main()