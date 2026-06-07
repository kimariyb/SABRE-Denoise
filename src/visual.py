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
    }
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['epoch'].append(int(row['epoch']))
            data['train_loss'].append(float(row['train_loss']))
            data['val_loss'].append(float(row['val_loss']))

    return data


def plot_metrics(data: dict, save_path: str):
    """绘制 train/val loss 和 train/val SNR 双面板折线图。"""
    epochs = data['epoch']
    epochs_label = [f'{e}' for e in epochs]

    fig, ax = plt.subplots(figsize=(10, 6))

    color_loss = '#1f77b4'
    color_val  = '#d62728'

    ax.plot(epochs, data['train_loss'], color=color_loss, label='Train Loss', marker='o', markersize=3)
    ax.plot(epochs, data['val_loss'],   color=color_val,  label='Val Loss',   marker='s', markersize=3)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    # 标出最佳验证损失
    best_idx = data['val_loss'].index(min(data['val_loss']))
    ax.annotate(f'Best: {data["val_loss"][best_idx]:.4f}',
                 xy=(epochs[best_idx], data['val_loss'][best_idx]),
                 xytext=(10, -18), textcoords='offset points',
                 fontsize=10, color=color_val,
                 arrowprops=dict(arrowstyle='->', color=color_val, lw=1.2))

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
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
