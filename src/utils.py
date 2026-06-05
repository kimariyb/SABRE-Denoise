import torch

from dataset import SABREDataset


def calc_loss(pred, label, loss_fn,
              lambda_spec: float = 0.1,
              lambda_peak: float = 2.0,
              lambda_base: float = 0.5):
    """
    Parameters
    ----------
    pred / label     : [B, 2, W]，通道0=实部，通道1=虚部
    loss_fn          : 基础损失函数（如 nn.L1Loss()）
    lambda_spec      : 频域一致性权重
    lambda_peak      : 峰区域加权系数
    lambda_base      : 基线平滑正则权重

    Returns
    -------
    total_loss : scalar Tensor
    """
    complex_pred  = SABREDataset.real2complex(pred).squeeze(dim=1)   # [B, W] complex
    complex_label = SABREDataset.real2complex(label).squeeze(dim=1)  # [B, W] complex

    # 频域幅度一致性
    spec_loss = torch.nn.functional.l1_loss(
        torch.fft.fft(complex_pred).abs(),
        torch.fft.fft(complex_label).abs()
    )

    peak_loss = loss_fn(complex_pred[:, 6600:6800], complex_label[:, 6600:6800])

    # 基线平滑正则（作用于实部，抑制无峰区域抖动）
    real_pred = pred[:, 0, :]   # [B, W]
    base_loss = real_pred.diff(dim=-1).pow(2).mean()

    total_loss = (
        + lambda_spec  * spec_loss
        + lambda_peak  * peak_loss
        + lambda_base  * base_loss
    )

    return total_loss


def calc_snr(pred: torch.Tensor, label: torch.Tensor,
             eps: float = 1e-8) -> torch.Tensor:
    """
    只在实部计算重建 SNR（dB）。

    Parameters
    ----------
    pred / label : [B, 2, W]，通道0=实部，通道1=虚部

    Returns
    -------
    snr : [B]，单位 dB，值越高越好
    """
    real_pred = pred[:, 0, :]
    real_label = label[:, 0, :]

    # 去均值，避免基线偏置污染信号功率估计
    real_label_zm = real_label - real_label.mean(dim=-1, keepdim=True)
    real_pred_zm = real_pred - real_pred.mean(dim=-1, keepdim=True)

    signal_power = real_label_zm.pow(2).mean(dim=-1)
    noise_power = (real_label_zm - real_pred_zm).pow(2).mean(dim=-1)
    return 10 * torch.log10(signal_power / (noise_power + eps))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)