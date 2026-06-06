"""
Unified training and evaluation for all denoising baselines.

Generates training data from inference/*.csv, trains Real U-Net and DnCNN,
then evaluates them alongside SabreSDN (if checkpoint available).
"""
import os, sys, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from datetime import datetime

from dataset import SABREDataset, loadSpectra, splitSpectra, normalize, \
                    addNoise, generateNoiseSpectra
from models import SabreSDN
from model_real_unet import RealUNet
from model_dncnn import DnCNN1D
from evaluate import (load_clean_spectra, generate_test_set,
                      evaluate_method, compute_snr, compute_nmse,
                      complex_to_tensor)

# ── config ──────────────────────────────────────────────────────────────
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../inference')
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '../../checkpoints')
BATCH_SIZE = 64
EPOCHES = 20         # fewer epochs for quick baseline comparison
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
LR_FACTOR = 0.8
LR_PATIENCE = 5
GRAD_CLIP = 1.0
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# ────────────────────────────────────────────────────────────────────────


def generate_training_data(csv_files, save_path, n_noisy=5000,
                           split_range=(41308, 49500), noise_range=(1e-4, 1e-2)):
    """Generate train.npz from a list of CSV files."""
    train_data = []
    for f in csv_files:
        spectra = loadSpectra(f)
        split_data = splitSpectra(spectra, split_range)
        normalized_data, _, _ = normalize(split_data)
        data = generateNoiseSpectra(normalized_data, n_noisy, noise_range)
        train_data.extend(data)
    np.savez(save_path, np.array(train_data, dtype=object), allow_pickle=True)
    print(f"Saved {len(train_data)} samples to {save_path}")
    return train_data


def train_model(model, train_loader, val_loader, epochs, model_name):
    """Train a denoising model and return best checkpoint."""
    model = model.to(DEVICE)
    loss_fn = nn.L1Loss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE,
                      weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_FACTOR,
                                  patience=LR_PATIENCE, min_lr=1e-6)
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"{model_name} E{epoch:02d}",
                         leave=False, unit="batch"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                val_loss += loss_fn(pred, y).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  [{model_name}] Epoch {epoch:3d}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}")

    # Save
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pth")
    torch.save(best_state, ckpt_path)
    print(f"  [{model_name}] Best val_loss={best_val_loss:.4f}, saved to {ckpt_path}")

    model.load_state_dict(best_state)
    return model


def evaluate_model_on_test_set(model, test_set, device='cpu'):
    """Evaluate a PyTorch model on the test set."""
    model = model.to(device)
    model.eval()
    results = {}
    with torch.no_grad():
        for item in test_set:
            sigma_key = f"{item['sigma']:.2e}"
            if sigma_key not in results:
                results[sigma_key] = {'snr': [], 'nmse': []}

            x = complex_to_tensor(item['noisy']).to(device)
            clean = item['clean']
            pred = model(x)  # [1, 2, 8192]
            pred_np = pred.cpu().squeeze(0).numpy()  # [2, 8192]
            pred_complex = pred_np[0] + 1j * pred_np[1]

            snr = compute_snr(pred_complex, clean)
            nmse = compute_nmse(pred_complex, clean)
            results[sigma_key]['snr'].append(snr)
            results[sigma_key]['nmse'].append(nmse)

    all_snr = []
    all_nmse = []
    for sk in results:
        all_snr.extend(results[sk]['snr'])
        all_nmse.extend(results[sk]['nmse'])

    return {
        'avg_snr': float(np.mean(all_snr)),
        'avg_nmse': float(np.mean(all_nmse)),
        'per_sigma': {
            sk: {'snr': float(np.mean(v['snr'])),
                 'nmse': float(np.mean(v['nmse']))}
            for sk, v in sorted(results.items())
        }
    }


if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    sep = "=" * 65

    # ── 1. Generate training data ────────────────────────────────────
    print(f"\n{sep}\n  1. Generating training data\n{sep}")
    csv_dir = DATA_DIR
    all_csv = sorted([os.path.join(csv_dir, f) for f in os.listdir(csv_dir)
                      if f.endswith('.csv')])
    # Use first 4 for training, last 1 for held-out testing
    train_csvs = all_csv[:4]
    test_csvs = all_csv[4:]

    train_npz = os.path.join(os.path.dirname(__file__), '../../data/train.npz')
    os.makedirs(os.path.dirname(train_npz), exist_ok=True)

    if not os.path.exists(train_npz):
        generate_training_data(train_csvs, train_npz, n_noisy=5000)
    else:
        print(f"Using existing {train_npz}")

    # ── 2. Load data ─────────────────────────────────────────────────
    with np.load(train_npz, allow_pickle=True) as loaded:
        spectra_data = loaded['arr_0']

    dataset = SABREDataset(spectra_data)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Train: {n_train}, Val: {n_val}")

    # ── 3. Generate test set ─────────────────────────────────────────
    print(f"\n{sep}\n  3. Generating test set\n{sep}")
    test_clean = []
    for f in test_csvs:
        s = loadSpectra(f)
        s = splitSpectra(s, (41308, 49500))
        s, _, _ = normalize(s)
        test_clean.append(s)

    test_set, sigma_levels = generate_test_set(test_clean, n_noise_levels=5)
    print(f"Test samples: {len(test_set)}, noise levels: {[f'{s:.2e}' for s in sigma_levels]}")

    # Noisy input baseline
    noisy_results = evaluate_method(lambda x, c: x, test_set, "Noisy")
    print(f"Noisy input: SNR={noisy_results['avg_snr']:.2f} dB, "
          f"NMSE={noisy_results['avg_nmse']:.4f}")

    # ── 4. Train & eval Real U-Net ───────────────────────────────────
    print(f"\n{sep}\n  4. Real U-Net\n{sep}")
    real_unet = RealUNet()
    n_params = sum(p.numel() for p in real_unet.parameters() if p.requires_grad)
    print(f"Parameters: {n_params / 1e6:.2f} M")

    real_unet = train_model(real_unet, train_loader, val_loader, EPOCHES, "RealUNet")
    real_unet_results = evaluate_model_on_test_set(real_unet, test_set, DEVICE)
    print(f"Real U-Net: SNR={real_unet_results['avg_snr']:.2f} dB, "
          f"NMSE={real_unet_results['avg_nmse']:.4f}")

    # ── 5. Train & eval DnCNN ────────────────────────────────────────
    print(f"\n{sep}\n  5. DnCNN\n{sep}")
    dncnn = DnCNN1D()
    n_params = sum(p.numel() for p in dncnn.parameters() if p.requires_grad)
    print(f"Parameters: {n_params / 1e6:.2f} M")

    dncnn = train_model(dncnn, train_loader, val_loader, EPOCHES, "DnCNN")
    dncnn_results = evaluate_model_on_test_set(dncnn, test_set, DEVICE)
    print(f"DnCNN: SNR={dncnn_results['avg_snr']:.2f} dB, "
          f"NMSE={dncnn_results['avg_nmse']:.4f}")

    # ── 6. Evaluate SabreSDN (if checkpoint exists) ──────────────────
    sabre_results = None
    sabre_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    if os.path.exists(sabre_path):
        print(f"\n{sep}\n  6. SabreSDN (loading {sabre_path})\n{sep}")
        sabre = SabreSDN(num_blocks=1, use_attention=True,
                         use_gated_skip=True).to(DEVICE)
        ckpt = torch.load(sabre_path, map_location=DEVICE)
        if 'model_state_dict' in ckpt:
            ckpt = ckpt['model_state_dict']
        sabre.load_state_dict(ckpt)
        sabre_results = evaluate_model_on_test_set(sabre, test_set, DEVICE)
        print(f"SabreSDN: SNR={sabre_results['avg_snr']:.2f} dB, "
              f"NMSE={sabre_results['avg_nmse']:.4f}")
    else:
        # Train SabreSDN from scratch for fair comparison
        print(f"\n{sep}\n  6. SabreSDN (training from scratch)\n{sep}")
        sabre = SabreSDN(num_blocks=1, use_attention=True,
                         use_gated_skip=True).to(DEVICE)
        n_params = sum(p.numel() for p in sabre.parameters() if p.requires_grad)
        print(f"Parameters: {n_params / 1e6:.2f} M")
        sabre = train_model(sabre, train_loader, val_loader, EPOCHES, "SabreSDN")
        sabre_results = evaluate_model_on_test_set(sabre, test_set, DEVICE)
        print(f"SabreSDN: SNR={sabre_results['avg_snr']:.2f} dB, "
              f"NMSE={sabre_results['avg_nmse']:.4f}")

    # ── 7. Summary ───────────────────────────────────────────────────
    print(f"\n\n{sep}")
    print(f"  FINAL RESULTS")
    print(f"{sep}")
    print(f"  {'Method':<20s}  {'SNR (dB)':>10s}  {'NMSE':>10s}  {'Params':>10s}")
    print(f"  {'-'*54}")

    all_results = {'Noisy input': noisy_results}
    if real_unet_results:
        all_results['Real U-Net'] = real_unet_results
    if dncnn_results:
        all_results['DnCNN'] = dncnn_results
    if sabre_results:
        all_results['SabreSDN'] = sabre_results

    param_counts = {
        'Real U-Net': f"{sum(p.numel() for p in real_unet.parameters() if p.requires_grad)/1e6:.1f}M",
        'DnCNN': f"{sum(p.numel() for p in dncnn.parameters() if p.requires_grad)/1e6:.1f}M",
        'SabreSDN': '6.3M',
    }

    for method, r in sorted(all_results.items()):
        p = param_counts.get(method, '--')
        print(f"  {method:<20s}  {r['avg_snr']:>8.2f}  {r['avg_nmse']:>10.6f}  {p:>10s}")

    print(f"\n  Per-sigma breakdown:")
    print(f"  {'Sigma':<12s}", end="")
    for method in sorted(all_results.keys()):
        print(f"  {method:>12s}", end="")
    print()
    for sigma_key in sorted(noisy_results['per_sigma'].keys()):
        print(f"  {sigma_key:<12s}", end="")
        for method in sorted(all_results.keys()):
            r = all_results[method]
            snr = r['per_sigma'].get(sigma_key, {}).get('snr', float('nan'))
            print(f"  {snr:>10.2f}", end="")
        print()

    # Save results JSON
    results_json = os.path.join(os.path.dirname(__file__), '../../paper/results.json')
    os.makedirs(os.path.dirname(results_json), exist_ok=True)
    with open(results_json, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_json}")
