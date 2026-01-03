from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from matplotlib import pyplot as plt
import seaborn as sns

LOGGER = logging.getLogger("evaluate")


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

def load_model_and_embeddings(model_path: str, emb_dir: str, labels_csv: str, device: str = "cpu"):
    #Loading trained model and validation embeddings.
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    label_map = checkpoint["label_map"]
    config = checkpoint["config"]
    num_classes = config["num_classes"]

    # Initialize model
    model = MLPProbe(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=num_classes,
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    # Load embeddings
    image_emb = np.load(os.path.join(emb_dir, "image_embeddings.npy"))
    text_emb = np.load(os.path.join(emb_dir, "text_embeddings.npy"))
    labels_arr = np.load(os.path.join(emb_dir, "labels.npy"))
    ids_arr = np.load(os.path.join(emb_dir, "ids.npy"), allow_pickle=True)

    fused_emb = np.hstack([image_emb, text_emb])
    fused_tensor = torch.from_numpy(fused_emb).float().to(device)
    labels_tensor = torch.from_numpy(labels_arr).long().to(device)

    # Load split info
    df = pd.read_csv(labels_csv)
    valid_ids = set(ids_arr)
    df = df[df["uniq_id"].isin(valid_ids)].reset_index(drop=True)
    df = df.set_index("uniq_id").loc[ids_arr].reset_index()

    # Filter to val split
    val_idx = np.where(df["split"].values == "val")[0]
    val_X = fused_tensor[val_idx]
    val_y = labels_tensor[val_idx]

    LOGGER.info("Loaded model and %d validation samples", len(val_X))

    return model, val_X, val_y, label_map, device


def evaluate(
    model_path: str,
    emb_dir: str,
    labels_csv: str,
    out_dir: str,
    device: str = "cpu",
) -> None:
    #Evaluating model on validation set.
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    model, val_X, val_y, label_map, device = load_model_and_embeddings(
        model_path, emb_dir, labels_csv, device
    )

    # Predict
    with torch.no_grad():
        logits = model(val_X)
        preds = logits.argmax(dim=1)

    y_true = val_y.cpu().numpy()
    y_pred = preds.cpu().numpy()

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    LOGGER.info("Accuracy: %.10f", acc)
    LOGGER.info("Macro F1: %.10f", macro_f1)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Preparing metrics dict
    reverse_label_map = {int(v): k for k, v in label_map.items()}
    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class": {
            reverse_label_map[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }
            for i in range(len(precision))
        },
    }

    # Saving metrics.json
    metrics_file = os.path.join(out_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    LOGGER.info("Metrics saved to %s", metrics_file)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    class_labels = [reverse_label_map[i] for i in range(len(precision))]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_file = os.path.join(out_dir, "confusion.png")
    plt.savefig(cm_file, dpi=100, bbox_inches="tight")
    LOGGER.info("Confusion matrix saved to %s", cm_file)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multimodal probe")
    parser.add_argument("--model", required=True, help="Path to trained model (.pth)")
    parser.add_argument("--emb_dir", required=True, help="Directory with embeddings")
    parser.add_argument("--labels_csv", default="data/processed_labels.csv", help="Path to processed_labels.csv")
    parser.add_argument("--out_dir", required=True, help="Directory to save metrics and plots")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)"
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(name)s:%(levelname)s: %(message)s",
    )

    evaluate(
        model_path=args.model,
        emb_dir=args.emb_dir,
        labels_csv=args.labels_csv,
        out_dir=args.out_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
