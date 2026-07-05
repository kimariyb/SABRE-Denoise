# SABRE-Denoise

SABRE-Denoise 是一个面向低浓度 SABRE-$^{19}$F NMR 谱图的复数域深度学习去噪项目。项目核心模型 SabreSDN 使用 complex-valued 1D U-Net、CBAM 注意力、多尺度空洞卷积瓶颈和门控 skip connection，在保留谱峰结构与复数相位信息的同时抑制基线噪声。

本仓库同时包含模型训练代码、推理脚本、实验日志、示例推理结果，以及一份已整理的论文草稿。

## 主要功能

- 复数域 NMR 谱图建模：输入/输出均为 `[2, 8192]`，通道 0 为 real，通道 1 为 imaginary。
- 复数卷积 U-Net：用复数卷积耦合 real/imag 分量，而不是把两者当作普通独立通道。
- 谱峰感知去噪：损失函数显式强化 `6600:6800` 峰区域，同时约束全谱幅度一致性和基线平滑性。
- 自动训练流程：支持数据集划分、AdamW、ReduceLROnPlateau、梯度裁剪、checkpoint 和 CSV 日志。
- 推理与可视化：支持对 CSV 谱图执行去噪，并生成光谱对比图。
- 论文材料：`paper/` 中包含 LaTeX 论文、参考文献和当前 Nature 风格图件。

## 技术栈

- Python 3.10+
- PyTorch
- NumPy / Pandas / SciPy
- Matplotlib
- tqdm
- LaTeX / latexmk，用于编译论文

依赖版本见 [requirements.txt](requirements.txt)。

## 目录结构

```text
SABRE-Denoise/
├── src/
│   ├── train_sabre.py        # 主训练入口
│   ├── preprocess.py         # 原始 CSV 谱图预处理与噪声数据生成
│   ├── inference.py          # inference/ 目录 CSV 谱图推理
│   ├── test_sim.py           # data/test 测试集推理与指标计算
│   ├── visual.py             # 训练日志可视化
│   ├── trainer.py            # train/eval loop
│   ├── dataset.py            # 数据读取、归一化、FFT/IFFT、Dataset
│   ├── utils.py              # composite loss、参数统计
│   └── models/
│       └── sabre.py          # SabreSDN 模型定义
├── inference/                # 示例推理输入 CSV
├── logs/                     # 训练日志和示例可视化结果
├── paper/
│   ├── neurips_2025.tex      # 论文主文件
│   ├── neurips_2025.pdf      # 已编译论文
│   ├── references.bib
│   ├── make_nature_figures.py
│   └── figures/
│       ├── nature_fig1_architecture_gpt.png
│       ├── nature_fig2_training.pdf
│       ├── nature_fig3_results.pdf
│       └── nature_fig4_loss_deployment.pdf
├── requirements.txt
├── LICENSE
└── README.md
```

说明：`data/` 和 `checkpoints/` 通常较大，当前仓库未必包含完整训练数据和模型权重。训练前需要准备 `data/train.npz`，推理前需要准备 `checkpoints/best_model.pth`。

## 安装

建议使用独立虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

如果你使用的是非 CUDA 12.8 环境，`requirements.txt` 中的 `torch~=2.7.0+cu128` 可能需要替换为与你本机 CUDA/CPU 匹配的 PyTorch 安装命令。以 PyTorch 官网给出的命令为准。

## 数据格式

原始 NMR 谱图 CSV 默认按空白符分隔，至少包含三列：

```text
chemical_shift real imag
```

读取逻辑在 [src/dataset.py](src/dataset.py) 的 `loadSpectra()` 中：

- 第 1 列：chemical shift 或索引信息
- 第 2 列：real part
- 第 3 列：imaginary part

训练数据预处理后保存为 `data/train.npz`，其中每个样本是 `(noisy_complex_spectrum, clean_complex_spectrum)`。

## 数据预处理

将原始训练 CSV 放入：

```text
data/train/
```

然后运行：

```bash
cd src
python preprocess.py
```

默认流程：

1. 读取每个 CSV 谱图。
2. 截取感兴趣区间，当前脚本使用 `(41308, 49500)`，长度为 8192。
3. 对 real/imag 分量分别归一化。
4. 在 FID 时域注入复数高斯噪声。
5. 生成 noisy-clean pairs。
6. 保存为 `../data/train.npz`。

当前 [src/preprocess.py](src/preprocess.py) 末尾配置为：

```python
prepreocess(
    "../data/train/",
    "../data/train.npz",
    (41308, 49500),
    (0.05, 1.5),
    16000,
)
```

如需复现实验论文中的低噪声设置，可按论文描述把噪声范围调整为 `1e-4` 到 `1e-2`，并根据显存和训练时间调整 `pre_count`。

## 模型结构

SabreSDN 定义在 [src/models/sabre.py](src/models/sabre.py)。

核心模块：

| 模块 | 作用 |
| --- | --- |
| `CConv1d` | 复数 1D 卷积，按复数乘法规则耦合 real/imag 分量 |
| `CConvTranspose1d` | 复数转置卷积，用于 U-Net 解码上采样 |
| `CBAM1d` | 通道注意力 + 谱位置注意力 |
| `MultiScaleBlock` | dilation = 1, 4, 8 的多尺度瓶颈模块 |
| `GatedSkip` | 对 encoder skip feature 做门控过滤 |
| `SabreSDN` | 单个或多个 `SDNBlock` 叠加，并带全局残差和 `tanh` 输出 |

快速检查模型输入输出：

```bash
python src/models/sabre.py
```

预期输入输出形状：

```text
输入: [B, 2, 8192]
输出: [B, 2, 8192]
```

## 损失函数

训练使用 [src/utils.py](src/utils.py) 中的 `calc_loss()`：

```python
total_loss = 0.1 * spec_loss + 2.0 * peak_loss + 0.5 * base_loss
```

三部分含义：

- `spec_loss`：FFT 幅度一致性，约束整体频谱结构。
- `peak_loss`：`6600:6800` 峰区域复数重建损失，强调分析区域。
- `base_loss`：real 分量一阶差分平滑项，抑制基线抖动。

## 训练

准备好 `data/train.npz` 后运行：

```bash
cd src
python train_sabre.py
```

主要训练配置位于 [src/train_sabre.py](src/train_sabre.py) 顶部：

| 参数 | 当前值 |
| --- | --- |
| `SPECTRA_PATH` | `../data/train.npz` |
| `BATCH_SIZE` | `128` |
| `EPOCHES` | `50` |
| `LEARNING_RATE` | `1e-3` |
| `LR_FACTOR` | `0.8` |
| `LR_PATIENCE` | `5` |
| `LR_MIN` | `1e-6` |
| `CHECKPOINT_DIR` | `../checkpoints` |
| `LOG_DIR` | `../logs` |

训练输出：

- `checkpoints/best_model.pth`：验证损失最低的模型
- `checkpoints/epoch_*.pth`：逐 epoch checkpoint
- `checkpoints/final_model.pth`：最后一轮模型
- `logs/training_log_YYYYMMDD_HHMMSS.csv`：训练日志

## 可视化训练曲线

```bash
cd src
python visual.py --log ../logs/training_log_20260606_173313.csv --save ../logs/training_curves.png
```

示例训练曲线：

![训练曲线](logs/training_curves.png)

注意：`src/visual.py` 当前自动查找日志路径的默认实现可能指向 `src/logs`，更稳妥的方式是显式传入 `--log`。

## 推理

### 对 `inference/` 目录中的 CSV 推理

准备：

- 输入 CSV：放在 `inference/`
- 模型权重：`checkpoints/best_model.pth`

运行：

```bash
cd src
python inference.py
```

该脚本会读取 `../inference/*.csv`，按训练时一致的方式截取、归一化，并输出模型预测。当前脚本默认调用 `plt.show()`，如需批量保存图片，可在 `plot_spectra()` 调用处传入 `save_path`。

### 对 `data/test` 测试集计算指标

```bash
cd src
python test_sim.py
```

`test_sim.py` 读取 `../data/test/*.csv`，并尝试计算 SNR、SNRP 和 RMSE。当前文件依赖 `baselines.utils`，如果本地没有该模块，需要补齐对应工具函数或改为使用自己的指标实现。

示例推理结果：

![推理结果](logs/saved_1.png)

## 论文与图件

论文主文件：

```text
paper/neurips_2025.tex
```

当前已编译 PDF：

```text
paper/neurips_2025.pdf
```

论文中当前引用的四张图：

| 图号 | 文件 | 内容 |
| --- | --- | --- |
| Fig. 1 | `paper/figures/nature_fig1_architecture_gpt.png` | SabreSDN 网络结构与关键模块 |
| Fig. 2 | `paper/figures/nature_fig2_training.pdf` | 训练动态与优化摘要 |
| Fig. 3 | `paper/figures/nature_fig3_results.pdf` | 低浓度测试谱图去噪结果 |
| Fig. 4 | `paper/figures/nature_fig4_loss_deployment.pdf` | 损失权重与推理部署约束 |

重新生成 Fig.2-Fig.4：

```bash
python paper/make_nature_figures.py
```

重新编译论文：

```bash
cd paper
latexmk -g -pdf -synctex=1 -interaction=nonstopmode -halt-on-error neurips_2025.tex
```

如果只想清理 LaTeX 中间文件：

```bash
cd paper
latexmk -c neurips_2025.tex
```

## 常见问题

### 1. 为什么输入是 `[2, 8192]`？

SABRE-$^{19}$F NMR 谱图是复数谱，包含 real 和 imaginary 两个分量。模型使用 `[real, imag]` 两个通道表示一个复数谱图，长度 8192 来自预处理阶段截取的谱图窗口。

### 2. 为什么要在 FID 域加噪？

在时域 FID 中注入复数高斯噪声，再变换回频域，可以更接近 NMR 接收链路中的热噪声特性，避免只在频域做简单扰动导致噪声分布不自然。

### 3. 为什么损失函数重点关注 6600-6800？

该区间是当前数据中主要 SABRE-$^{19}$F 谱峰所在区域。低浓度谱图中，峰区域对定量、积分和峰形判断最关键，因此训练时显式增加其权重。

### 4. 没有 `data/train.npz` 能直接训练吗？

不能。需要先准备原始 CSV 并运行 `src/preprocess.py` 生成训练集，或自行构造与 `SABREDataset` 兼容的 noisy-clean pair 数据。

### 5. 没有 `checkpoints/best_model.pth` 能直接推理吗？

不能。推理脚本需要先加载训练好的 checkpoint。

## 复现实验建议

1. 固定 Python、PyTorch、CUDA 和随机种子版本。
2. 明确记录原始 CSV 数量、截取区间、噪声范围和生成样本数。
3. 保存 `training_log_*.csv`、`best_model.pth`、预处理配置和论文图件脚本。
4. 对每次论文改图，都优先从日志和原始推理结果重新生成，避免手工修改实验曲线。

## 引用

如果该项目对你的研究有帮助，可以引用：

```bibtex
@software{SABRE-Denoise,
  author = {YuBin Xiong and Yao Luo},
  title = {SABRE-Denoise: Complex U-Net for SABRE NMR Spectral Denoising},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/kimariyb/SABRE-Denoise}}
}
```

## License

本项目基于 [MIT License](LICENSE) 发布。
