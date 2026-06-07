import torch
from tqdm import tqdm
from utils import calc_loss


def train_fn(model, data_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(data_loader, total=len(data_loader), desc="Train", unit="batch"):
        x, y = batch
        x = x.to(device)  # [B, 2, 8192]
        y = y.to(device)  # [B, 2, 8192]

        optimizer.zero_grad()
        pred = model(x)   # [B, 2, 8192]
        loss = calc_loss(pred, y, loss_fn)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    n = len(data_loader)
    return total_loss / n


def eval_fn(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), desc="Eval", unit="batch"):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            pred = model(x)  # [B, 2, 8192]
            total_loss += calc_loss(pred, y, loss_fn).item()
    n = len(data_loader)
    return total_loss / n