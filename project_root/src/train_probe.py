from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

LOGGER = logging.getLogger("train_probe")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLPProbe(nn.Module):

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, num_classes: int = 5, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_embeddings_and_split(
    emb_dir: str, labels_csv: str, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Load embeddings and split data into train/val by split column."""
    # Load embeddings
    image_emb = np.load(os.path.join(emb_dir, "image_embeddings.npy"))  # (N, 512)
    text_emb = np.load(os.path.join(emb_dir, "text_embeddings.npy"))  # (N, 512)
    labels_arr = np.load(os.path.join(emb_dir, "labels.npy"))  # (N,)
    ids_arr = np.load(os.path.join(emb_dir, "ids.npy"), allow_pickle=True)  # (N,)

    # Concatenate
    fused_emb = np.hstack([image_emb, text_emb])  # (N, 1024)
    fused_tensor = torch.from_numpy(fused_emb).float().to(device)
    labels_tensor = torch.from_numpy(labels_arr).long().to(device)

    # Load processed_labels.csv to get split info
    df = pd.read_csv(labels_csv)
    
    # Filter to only rows that are in the embeddings (by uniq_id)
    valid_ids = set(ids_arr)
    df = df[df["uniq_id"].isin(valid_ids)].reset_index(drop=True)
    
    # Ensure order matches embeddings
    df = df.set_index("uniq_id").loc[ids_arr].reset_index()
    
    split = df["split"].values  # train or val

    # Separate train and val
    train_idx = np.where(split == "train")[0]
    val_idx = np.where(split == "val")[0]

    train_X = fused_tensor[train_idx]
    train_y = labels_tensor[train_idx]
    val_X = fused_tensor[val_idx]
    val_y = labels_tensor[val_idx]

    # Load label map
    with open(os.path.join(emb_dir, "label_map.json")) as f:
        label_map = json.load(f)

    LOGGER.info("Train: %d samples, Val: %d samples", len(train_X), len(val_X))

    return train_X, train_y, val_X, val_y, label_map


def train_epoch(
    model: MLPProbe, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: str
) -> float:
    """Train for one epoch. Return average loss."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X_batch)

    return total_loss / len(train_loader.dataset)


def validate(model: MLPProbe, val_loader: DataLoader, criterion: nn.Module, device: str) -> Tuple[float, float]:
    """Validate. Return (val_loss, val_accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(X_batch)

            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

    val_loss = total_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def train_probe(
    emb_dir: str,
    labels_csv: str,
    out_dir: str,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 3,
    device: str = "cpu",
    seed: int = 42,
) -> None:
    """Train MLP probe on embeddings."""
    seed_everything(seed)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    train_X, train_y, val_X, val_y, label_map = load_embeddings_and_split(
        emb_dir, labels_csv, device=device
    )

    # Create dataloaders
    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(val_X, val_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    num_classes = len(label_map)
    model = MLPProbe(input_dim=1024, hidden_dim=256, num_classes=num_classes, dropout=0.2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    # Prepare logging
    logs_dir = "logs"
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(logs_dir, "train_log.csv")
    csv_writer = None
    csv_file = open(log_file, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_file, fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy"]
    )
    csv_writer.writeheader()

    LOGGER.info("Starting training for %d epochs...", epochs)

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Log
        csv_writer.writerow(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc}
        )
        csv_file.flush()

        LOGGER.info(
            "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.4f",
            epoch,
            epochs,
            train_loss,
            val_loss,
            val_acc,
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch

            # Save checkpoint
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "label_map": label_map,
                "config": {
                    "input_dim": 1024,
                    "hidden_dim": 256,
                    "num_classes": num_classes,
                    "dropout": 0.2,
                },
            }
            checkpoint_path = os.path.join(out_dir, "multimodal_best.pth")
            torch.save(checkpoint, checkpoint_path)
            LOGGER.info("Saved best checkpoint: %s", checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                LOGGER.info("Early stopping at epoch %d (patience %d reached)", epoch, patience)
                break

    csv_file.close()
    LOGGER.info("Training complete. Best epoch: %d (val_loss: %.4f)", best_epoch, best_val_loss)
    LOGGER.info("Logs saved to %s", log_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLP probe on CLIP embeddings")
    parser.add_argument("--emb_dir", required=True, help="Directory with embeddings (.npy files)")
    parser.add_argument("--labels_csv", default="data/processed_labels.csv", help="Path to processed_labels.csv")
    parser.add_argument("--out_dir", required=True, help="Directory to save trained model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (L2 regularization)")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Debug mode: verbose logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(name)s:%(levelname)s: %(message)s",
    )

    train_probe(
        emb_dir=args.emb_dir,
        labels_csv=args.labels_csv,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
