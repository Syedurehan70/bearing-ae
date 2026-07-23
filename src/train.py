from __future__ import annotations

import logging
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_autoencoder(
    model: nn.Module,
    x_train: np.ndarray,
    x_val: np.ndarray,
    cfg: dict,
    device: str = "cpu",
) -> dict[str, list[float]]:
    train_cfg = cfg["training"]
    model.to(device)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train)),
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        drop_last=False,
    )
    val_tensor = torch.from_numpy(x_val).to(device)

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    criterion = nn.MSELoss()

    history: dict[str, list[float]] = {"train": [], "val": []}
    best_val = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    patience_left = train_cfg["patience"]

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        running = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimiser.step()
            running += loss.item() * batch.shape[0]
        train_loss = running / len(loader.dataset)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_tensor), val_tensor).item()

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        log.info("epoch %3d | train %.5f | val %.5f", epoch, train_loss, val_loss)

        if val_loss < best_val - train_cfg["min_delta"]:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = train_cfg["patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info("early stopping at epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    return history


@torch.no_grad()
def reconstruction_error(
    model: nn.Module, x: np.ndarray, device: str = "cpu", batch_size: int = 512
) -> np.ndarray:
    """Per-window mean squared reconstruction error."""
    model.eval().to(device)
    errors = []
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[start : start + batch_size]).to(device)
        recon = model(batch)
        errors.append(((recon - batch) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(errors) if errors else np.empty(0)
