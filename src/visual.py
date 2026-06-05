"""
绘制训练日志中的 train/val loss 和 train/val SNR 折线图。

用法:
    python visual.py                              # 自动读取 logs/ 中最新的 CSV
    python visual.py --log logs/training_log_xxx.csv   # 指定日志文件
    python visual.py --save my_plot.png                # 自定义保存路径
"""

import argparse
import csv
import os
import glob
import matplotlib.pyplot as plt

# ── 全局样式 ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 12,
    'axes.linewidth': 1.2,
    'lines.linewidth': 1.6,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.edgecolor': '#333333',
})


def find_latest_log(log_dir: str = '../logs') -> str | None:
    """在 log_dir 中查找修改时间最新的 CSV 日志。"""
    pattern = os.path.join(log_dir, 'training_log_*.csv')
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def read_log(path: str) -> dict:
    """读取 CSV 日志，返回包含各列数据的字典。"""
    data = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_snr': [],
        'val_snr': [],
    }
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['epoch'].append(int(row['epoch']))
            data['train_loss'].append(float(row['train_loss']))
            data['val_loss'].append(float(row['val_loss']))
            data['train_snr'].append(float(row['train_snr']))
            data['val_snr'].append(float(row['val_snr']))
    return data


def plot_metrics(data: dict, save_path: str):
    """绘制 train/val loss 和 train/val SNR 双面板折线图。"""
    epochs = data['epoch']
    epochs_label = [f'{e}' for e in epochs]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    # ── 上图: Loss ────────────────────────────────────────────────
    color_loss = '#1f77b4'
    color_val  = '#d62728'

    ax1.plot(epochs, data['train_loss'], color=color_loss, label='Train Loss', marker='o', markersize=3)
    ax1.plot(epochs, data['val_loss'],   color=color_val,  label='Val Loss',   marker='s', markersize=3)
    ax1.set_ylabel('Loss', fontsize=13)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    # 标出最佳验证损失
    best_idx = data['val_loss'].index(min(data['val_loss']))
    ax1.annotate(f'Best: {data["val_loss"][best_idx]:.4f}',
                 xy=(epochs[best_idx], data['val_loss'][best_idx]),
                 xytext=(10, -18), textcoords='offset points',
                 fontsize=10, color=color_val,
                 arrowprops=dict(arrowstyle='->', color=color_val, lw=1.2))

    # ── 下图: SNR ────────────────────────────────────────────────
    color_snr_tr = '#2ca02c'
    color_snr_val = '#ff7f0e'

    ax2.plot(epochs, data['train_snr'], color=color_snr_tr,  label='Train SNR (dB)', marker='o', markersize=3)
    ax2.plot(epochs, data['val_snr'],   color=color_snr_val, label='Val SNR (dB)',   marker='s', markersize=3)
    ax2.set_xlabel('Epoch', fontsize=13)
    ax2.set_ylabel('SNR (dB)', fontsize=13)
    ax2.set_title('Training & Validation SNR', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    # 标出最佳验证 SNR
    best_snr_idx = data['val_snr'].index(max(data['val_snr']))
    ax2.annotate(f'Best: {data["val_snr"][best_snr_idx]:.2f} dB',
                 xy=(epochs[best_snr_idx], data['val_snr'][best_snr_idx]),
                 xytext=(10, 10), textcoords='offset points',
                 fontsize=10, color=color_snr_val,
                 arrowprops=dict(arrowstyle='->', color=color_snr_val, lw=1.2))

    fig.suptitle('SABRE-Denoise Training History', fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'[✓] 图表已保存: {save_path}')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='绘制训练日志中的 Loss 和 SNR 折线图')
    parser.add_argument('--log', type=str, default=None,
                        help='CSV 日志文件路径（默认自动查找 logs/ 中最新的）')
    parser.add_argument('--save', type=str, default=None,
                        help='图片保存路径（默认 logs/training_curves.png）')
    args = parser.parse_args()

    # ── 定位日志文件 ──────────────────────────────────────────────
    log_path = args.log
    if log_path is None:
        # 脚本在 project root/，日志在 project root/logs/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(script_dir, 'logs')
        latest = find_latest_log(log_dir)
        if latest is None:
            print('[!] 未找到日志文件，请用 --log 指定路径。')
            return
        log_path = latest
        print(f'[*] 自动检测到日志: {log_path}')

    if not os.path.isfile(log_path):
        print(f'[!] 文件不存在: {log_path}')
        return

    # ── 读取 & 绘图 ───────────────────────────────────────────────
    data = read_log(log_path)

    save_path = args.save
    if save_path is None:
        log_dir = os.path.dirname(log_path)
        save_path = os.path.join(log_dir, 'training_curves.png')

    plot_metrics(data, save_path)


if __name__ == '__main__':
    main()
