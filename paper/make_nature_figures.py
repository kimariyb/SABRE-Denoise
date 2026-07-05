from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec, patches
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
LOG_PATH = ROOT / "logs" / "training_log_20260606_173313.csv"
SAVED_DIR = ROOT / "logs"
INFERENCE_DIR = ROOT / "inference"


PALETTE = {
    "blue": "#0F4D92",
    "blue_2": "#3775BA",
    "teal": "#42949E",
    "aqua": "#77D7D1",
    "violet": "#7C6CCF",
    "lilac": "#B9A7E8",
    "red": "#B64342",
    "red_soft": "#F6CFCB",
    "gold": "#DCA73A",
    "green": "#2E9E44",
    "neutral_0": "#F7F7F7",
    "neutral_1": "#D8D8D8",
    "neutral_2": "#8F8F8F",
    "neutral_3": "#4D4D4D",
    "black": "#272727",
}


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.75,
    "legend.frameon": False,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})


def add_panel_label(ax, label, x=-0.06, y=1.03, color="black"):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=color,
    )


def save_figure(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / stem
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def rounded_box(ax, xy, width, height, text, fc, ec=None, lw=1.0, fontsize=7):
    if ec is None:
        ec = PALETTE["neutral_3"]
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=PALETTE["black"],
    )
    return box


def arrow(ax, start, end, color=None, lw=1.0):
    if color is None:
        color = PALETTE["neutral_3"]
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, shrinkA=4, shrinkB=4),
    )


def figure_01_architecture():
    fig = plt.figure(figsize=(7.2, 5.25))
    gs = fig.add_gridspec(2, 4, height_ratios=[2.0, 1.0], hspace=0.35, wspace=0.45)
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    add_panel_label(ax, "a", x=0.0, y=0.98)
    ax.text(0.04, 0.98, "Complex-valued U-Net maps noisy SABRE spectra to denoised spectra", ha="left", va="top", fontsize=8)

    levels = [
        ("Input\n2 x 8192", 0.05, 0.58, 0.12, 0.18, PALETTE["neutral_0"]),
        ("Init\nCConv", 0.20, 0.58, 0.12, 0.18, "#E0F0F0"),
        ("Encoder\n16 -> 512", 0.36, 0.62, 0.16, 0.16, "#DDEBF7"),
        ("Multi-scale\nbottleneck", 0.56, 0.48, 0.16, 0.24, "#F0E0D0"),
        ("Decoder\n512 -> 16", 0.76, 0.62, 0.16, 0.16, "#E8E3F7"),
        ("Output\n2 x 8192", 0.93, 0.58, 0.12, 0.18, PALETTE["neutral_0"]),
    ]
    centers = []
    for text, x, y, w, h, fc in levels:
        rounded_box(ax, (x, y), w, h, text, fc, fontsize=7)
        centers.append((x + w / 2, y + h / 2))
    for s, e in zip(centers[:-1], centers[1:]):
        arrow(ax, (s[0] + 0.055, s[1]), (e[0] - 0.055, e[1]))

    skip_y = [0.43, 0.36, 0.29, 0.22]
    for i, y in enumerate(skip_y):
        x0 = 0.40 + i * 0.025
        x1 = 0.82 - i * 0.025
        ax.plot([x0, x0, x1, x1], [0.58, y, y, 0.58], color=PALETTE["teal"], lw=1.0, alpha=0.65)
        ax.text((x0 + x1) / 2, y - 0.025, "gated skip" if i == 0 else "", ha="center", va="top", fontsize=6, color=PALETTE["teal"])

    rounded_box(ax, (0.10, 0.10), 0.20, 0.14, "CConv1d\nWr*xr - Wi*xi\nWr*xi + Wi*xr", "#E0F0F0", fontsize=6.5)
    rounded_box(ax, (0.37, 0.10), 0.20, 0.14, "CBAM1d\nchannel + spectral\nattention", "#DDEBF7", fontsize=6.5)
    rounded_box(ax, (0.64, 0.10), 0.20, 0.14, "Dilated branches\n1, 1, 4, 8\ncapture line scales", "#F0E0D0", fontsize=6.5)

    ax_b = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_b, "b")
    labels = ["real conv", "complex conv"]
    vals = [2.0, 1.0]
    ax_b.bar(labels, vals, color=[PALETTE["neutral_1"], PALETTE["blue"]], edgecolor=PALETTE["black"], linewidth=0.5)
    ax_b.set_ylabel("relative kernel parameters")
    ax_b.set_ylim(0, 2.25)
    ax_b.text(1, 1.08, "50% fewer", ha="center", va="bottom", fontsize=7)
    ax_b.tick_params(axis="x", rotation=25)

    ax_c = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_c, "c")
    channels = [16, 32, 64, 128, 256, 512]
    x = np.arange(len(channels))
    ax_c.plot(x, channels, "-o", color=PALETTE["teal"], markersize=3)
    ax_c.set_yscale("log", base=2)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(["8192", "4096", "2048", "1024", "512", "256"], rotation=35)
    ax_c.set_xlabel("spectral length")
    ax_c.set_ylabel("complex channels")

    ax_d = fig.add_subplot(gs[1, 2])
    add_panel_label(ax_d, "d")
    dil = [1, 1, 4, 8]
    names = ["1x1", "d1", "d4", "d8"]
    ax_d.bar(names, dil, color=[PALETTE["neutral_1"], PALETTE["aqua"], PALETTE["teal"], PALETTE["blue"]], edgecolor=PALETTE["black"], linewidth=0.5)
    ax_d.set_ylabel("dilation")
    ax_d.set_ylim(0, 9)

    ax_e = fig.add_subplot(gs[1, 3])
    add_panel_label(ax_e, "e")
    ax_e.axis("off")
    rows = [
        ("Input/output", "[2, 8192]"),
        ("Parameters", "7.71 M"),
        ("Peak mask", "6600-6800"),
        ("Inference", "~2 ms GPU"),
    ]
    for i, (k, v) in enumerate(rows):
        y = 0.88 - i * 0.22
        ax_e.text(0.02, y, k, ha="left", va="center", fontsize=7, color=PALETTE["neutral_3"])
        ax_e.text(0.98, y, v, ha="right", va="center", fontsize=7, color=PALETTE["black"], fontweight="bold")
        ax_e.axhline(y - 0.09, color=PALETTE["neutral_1"], lw=0.6)

    save_figure(fig, "nature_fig1_architecture")


def figure_02_training():
    df = pd.read_csv(LOG_PATH)
    best = df.loc[df["val_loss"].idxmin()]
    fig = plt.figure(figsize=(7.2, 3.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.48, wspace=0.48)

    ax_a = fig.add_subplot(gs[0, :2])
    add_panel_label(ax_a, "a")
    ax_a.plot(df["epoch"], df["train_loss"], color=PALETTE["neutral_2"], lw=1.4, label="train")
    ax_a.plot(df["epoch"], df["val_loss"], color=PALETTE["blue"], lw=1.6, label="validation")
    ax_a.scatter([best["epoch"]], [best["val_loss"]], s=22, color=PALETTE["red"], zorder=5)
    ax_a.annotate(
        f"best {best['val_loss']:.3f}",
        xy=(best["epoch"], best["val_loss"]),
        xytext=(best["epoch"] - 14, best["val_loss"] + 0.045),
        arrowprops=dict(arrowstyle="-|>", lw=0.8, color=PALETTE["red"]),
        fontsize=7,
        color=PALETTE["red"],
    )
    ax_a.set_xlabel("epoch")
    ax_a.set_ylabel("composite loss")
    ax_a.legend(loc="upper right")
    ax_a.grid(axis="y", color=PALETTE["neutral_1"], lw=0.5, alpha=0.6)

    ax_b = fig.add_subplot(gs[0, 2])
    add_panel_label(ax_b, "b")
    ax_b.step(df["epoch"], df["learning_rate"], where="post", color=PALETTE["teal"], lw=1.6)
    ax_b.set_yscale("log")
    ax_b.set_xlabel("epoch")
    ax_b.set_ylabel("learning rate")
    ax_b.set_ylim(6e-4, 1.3e-3)
    ax_b.grid(axis="y", color=PALETTE["neutral_1"], lw=0.5, alpha=0.6)

    ax_c = fig.add_subplot(gs[1, :2])
    add_panel_label(ax_c, "c")
    ax_c.plot(df["epoch"], df["val_loss"].rolling(5, min_periods=1).mean(), color=PALETTE["blue"], lw=1.5)
    ax_c.fill_between(df["epoch"], 0, df["val_loss"], color=PALETTE["blue"], alpha=0.12)
    ax_c.scatter([best["epoch"]], [best["val_loss"]], s=18, color=PALETTE["red"], zorder=5)
    ax_c.set_xlabel("epoch")
    ax_c.set_ylabel("rolling validation loss")
    ax_c.set_ylim(0, max(df["val_loss"]) * 1.05)

    ax_e = fig.add_subplot(gs[1, 2])
    add_panel_label(ax_e, "d")
    ax_e.axis("off")
    rows = [
        ("training pairs", "20,000"),
        ("train / val", "80 / 20"),
        ("epochs", "50"),
        ("final LR", f"{df.iloc[-1]['learning_rate']:.1e}"),
    ]
    for i, (k, v) in enumerate(rows):
        y = 0.88 - i * 0.22
        ax_e.text(0.02, y, k, ha="left", va="center", fontsize=7, color=PALETTE["neutral_3"])
        ax_e.text(0.98, y, v, ha="right", va="center", fontsize=7, color=PALETTE["black"], fontweight="bold")
        ax_e.axhline(y - 0.09, color=PALETTE["neutral_1"], lw=0.6)

    save_figure(fig, "nature_fig2_training")


def crop_saved_result(path, region):
    im = Image.open(path).convert("RGB")
    return im.crop(region)


def smooth_curve(y, window=7):
    if window <= 1:
        return y
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y, kernel, mode="same")


def load_inference_spectrum(index):
    path = INFERENCE_DIR / f"test{index}.csv"
    arr = np.loadtxt(path)
    real = arr[:, 1]
    if real.size > 8192:
        real = real[41308:49500]
    real = real.astype(float)
    real = real - np.median(real)
    scale = np.max(np.abs(real[6400:7000])) + 1e-12
    return real / scale


def extract_prediction_curve(index):
    path = SAVED_DIR / f"saved_{index}.png"
    im = np.asarray(Image.open(path).convert("RGB"))
    red = (im[:, :, 0] > 180) & (im[:, :, 1] < 100) & (im[:, :, 2] < 100)
    ys, xs = np.where(red)
    if xs.size < 20:
        raise ValueError(f"Could not extract red prediction curve from {path}")

    # Keep the upper subplot where the predicted spectrum is drawn.
    upper = ys < np.percentile(ys, 55)
    xs = xs[upper]
    ys = ys[upper]
    x_min, x_max = xs.min(), xs.max()
    columns = np.arange(x_min, x_max + 1)
    y_by_col = np.full(columns.shape, np.nan, dtype=float)
    for i, col in enumerate(columns):
        hit = ys[xs == col]
        if hit.size:
            y_by_col[i] = np.median(hit)

    valid = np.isfinite(y_by_col)
    y_by_col = np.interp(columns, columns[valid], y_by_col[valid])
    y = -(y_by_col - np.median(y_by_col))
    y = y / (np.max(np.abs(y)) + 1e-12)
    source_x = np.linspace(0, 8191, y.size)
    return np.interp(np.arange(8192), source_x, y)


def figure_03_results():
    fig = plt.figure(figsize=(7.2, 4.7))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.35, 0.85, 0.95], hspace=0.38, wspace=0.38)

    x = np.arange(8192)
    roi = slice(6350, 7000)
    raw_1 = load_inference_spectrum(1)
    pred_1 = extract_prediction_curve(1)

    ax_a = fig.add_subplot(gs[0, :])
    add_panel_label(ax_a, "a")
    ax_a.plot(x[roi], raw_1[roi], color=PALETTE["neutral_2"], lw=0.45, alpha=0.6, label="low-concentration input")
    ax_a.plot(x[roi], smooth_curve(pred_1[roi], 5), color=PALETTE["red"], lw=1.4, label="SabreSDN output")
    ax_a.axvspan(6600, 6800, color=PALETTE["red_soft"], alpha=0.35, lw=0)
    ax_a.set_xlim(6350, 7000)
    ax_a.set_ylim(-1.15, 1.15)
    ax_a.set_xlabel("spectral index")
    ax_a.set_ylabel("normalized real intensity")
    ax_a.legend(loc="upper left", ncol=2, handlelength=1.8)
    ax_a.text(0.98, 0.88, "peak mask", transform=ax_a.transAxes, ha="right", va="center", fontsize=7, color=PALETTE["red"])
    ax_a.grid(axis="y", color=PALETTE["neutral_1"], lw=0.45, alpha=0.55)

    for i, sample in enumerate([1, 2, 3, 4]):
        ax = fig.add_subplot(gs[1, i])
        if i == 0:
            add_panel_label(ax, "b")
        raw = load_inference_spectrum(sample)
        pred = extract_prediction_curve(sample)
        ax.plot(x[roi], raw[roi], color=PALETTE["neutral_2"], lw=0.35, alpha=0.45)
        ax.plot(x[roi], smooth_curve(pred[roi], 5), color=PALETTE["red"], lw=1.0)
        ax.axvspan(6600, 6800, color=PALETTE["red_soft"], alpha=0.24, lw=0)
        ax.set_xlim(6350, 7000)
        ax.set_ylim(-1.15, 1.15)
        ax.set_xticks([6400, 6800])
        ax.set_yticks([])
        ax.set_title(f"test {sample}", fontsize=7, pad=2)
        if i == 0:
            ax.set_ylabel("intensity")

    ax_d = fig.add_subplot(gs[2, :2])
    add_panel_label(ax_d, "c")
    samples = [1, 2, 3, 4, 5]
    raw_rms = []
    out_rms = []
    for sample in samples:
        raw = load_inference_spectrum(sample)
        pred = extract_prediction_curve(sample)
        baseline = np.r_[0:6200, 7050:8192]
        raw_rms.append(np.sqrt(np.mean(raw[baseline] ** 2)))
        out_rms.append(np.sqrt(np.mean(pred[baseline] ** 2)))
    width = 0.36
    pos = np.arange(len(samples))
    ax_d.bar(pos - width / 2, raw_rms, width=width, color=PALETTE["neutral_1"], edgecolor=PALETTE["black"], linewidth=0.45, label="input")
    ax_d.bar(pos + width / 2, out_rms, width=width, color=PALETTE["teal"], edgecolor=PALETTE["black"], linewidth=0.45, label="output")
    ax_d.set_xticks(pos)
    ax_d.set_xticklabels([str(s) for s in samples])
    ax_d.set_xlabel("test spectrum")
    ax_d.set_ylabel("off-peak RMS")
    ax_d.legend(loc="upper right")

    ax_e = fig.add_subplot(gs[2, 2:])
    add_panel_label(ax_e, "d")
    ax_e.plot(x[roi], smooth_curve(pred_1[roi], 5), color=PALETTE["red"], lw=1.5)
    ax_e.fill_between(x[roi], 0, smooth_curve(pred_1[roi], 5), color=PALETTE["red"], alpha=0.12)
    ax_e.set_xlim(6600, 6800)
    ax_e.set_xlabel("peak-region index")
    ax_e.set_ylabel("denoised intensity")
    ax_e.grid(axis="y", color=PALETTE["neutral_1"], lw=0.45, alpha=0.55)

    save_figure(fig, "nature_fig3_results")


def load_inference_region(csv_path):
    arr = np.loadtxt(csv_path)
    real = arr[:, 1]
    roi = real[6600:6800]
    denom = np.max(np.abs(roi)) + 1e-12
    return roi / denom


def figure_04_loss_deployment():
    fig = plt.figure(figsize=(7.2, 2.7))
    gs = fig.add_gridspec(1, 3, wspace=0.48)

    ax_b = fig.add_subplot(gs[0, 0])
    add_panel_label(ax_b, "a")
    terms = ["spectral", "peak", "baseline"]
    weights = [0.1, 2.0, 0.5]
    colors = [PALETTE["neutral_1"], PALETTE["red"], PALETTE["teal"]]
    ax_b.bar(terms, weights, color=colors, edgecolor=PALETTE["black"], linewidth=0.5)
    ax_b.set_ylabel("loss weight")
    ax_b.tick_params(axis="x", rotation=25)
    ax_b.set_ylim(0, 2.3)

    ax_c = fig.add_subplot(gs[0, 1])
    add_panel_label(ax_c, "b")
    x = np.arange(200)
    region = np.zeros_like(x, dtype=float)
    region[:] = 0.5
    region[75:125] = 2.0
    ax_c.fill_between(x, 0, region, where=region > 0.5, color=PALETTE["red"], alpha=0.55, label="peak")
    ax_c.fill_between(x, 0, region, where=region <= 0.5, color=PALETTE["neutral_1"], alpha=0.8, label="baseline")
    ax_c.set_xticks([0, 75, 125, 199])
    ax_c.set_xticklabels(["0", "6600", "6800", "8192"])
    ax_c.set_ylabel("relative emphasis")
    ax_c.set_xlabel("spectral index")
    ax_c.set_ylim(0, 2.25)

    ax_d = fig.add_subplot(gs[0, 2])
    add_panel_label(ax_d, "c")
    labels = ["GPU", "CPU", "FID acquisition"]
    vals = [2, 50, 1000]
    ax_d.barh(labels, vals, color=[PALETTE["blue"], PALETTE["teal"], PALETTE["neutral_1"]], edgecolor=PALETTE["black"], linewidth=0.5)
    ax_d.set_xscale("log")
    ax_d.set_xlabel("time per spectrum (ms)")
    ax_d.invert_yaxis()
    ax_d.axvline(1000, color=PALETTE["neutral_2"], lw=0.8, ls="--")

    save_figure(fig, "nature_fig4_loss_deployment")


def main():
    figure_01_architecture()
    figure_02_training()
    figure_03_results()
    figure_04_loss_deployment()


if __name__ == "__main__":
    main()
