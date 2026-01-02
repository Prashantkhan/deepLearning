"""Inference: predict class for new image + text pair.

CLI examples:
    python src/infer.py \
        --model models/multimodal_best.pth \
        --image_url "https://example.com/image.jpg" \
        --text "Product Name || Product Description"

This script:
- Loads trained MLP probe
- Uses CLIP to embed new image + text
- Fuses embeddings (concatenate + L2-norm)
- Outputs class probabilities and top prediction
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

LOGGER = logging.getLogger("infer")


class MLPProbe(nn.Module):
    """Lightweight MLP probe: 1024 → 256 → num_classes."""

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


def load_image_from_url(image_url: str, timeout: int = 10) -> Image.Image:
    """Download image from URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    response = requests.get(image_url, timeout=timeout, headers=headers, allow_redirects=True)
    response.raise_for_status()
    img = Image.open(io.BytesIO(response.content)).convert("RGB")
    return img


def load_image_from_path(image_path: str) -> Image.Image:
    """Load image from local path."""
    return Image.open(image_path).convert("RGB")


def infer(
    model_path: str,
    image_source: str,
    text: str,
    clip_model: str = "openai/clip-vit-base-patch32",
    device: str = "cpu",
) -> Dict:
    """Perform inference on image + text pair.

    Args:
        model_path: Path to trained .pth checkpoint
        image_source: URL or local path to image
        text: Product text (name + description)
        clip_model: CLIP model name
        device: cpu or cuda

    Returns:
        Dict with predictions, probabilities, label_map
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    label_map = checkpoint["label_map"]
    config = checkpoint["config"]

    # Load model
    model = MLPProbe(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    # Load CLIP
    LOGGER.info("Loading CLIP model: %s", clip_model)
    clip_model_obj = CLIPModel.from_pretrained(clip_model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(clip_model)

    # Load image
    if image_source.startswith("http://") or image_source.startswith("https://"):
        LOGGER.info("Loading image from URL: %s", image_source)
        img = load_image_from_url(image_source)
    else:
        LOGGER.info("Loading image from path: %s", image_source)
        img = load_image_from_path(image_source)

    # Extract embeddings
    with torch.no_grad():
        # Image embedding
        img_inputs = processor(images=img, return_tensors="pt").to(device)
        img_emb = clip_model_obj.get_image_features(**img_inputs)
        img_emb = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-8)

        # Text embedding
        text_inputs = processor(text=text, return_tensors="pt", padding=True).to(device)
        text_emb = clip_model_obj.get_text_features(**text_inputs)
        text_emb = text_emb / (text_emb.norm(dim=-1, keepdim=True) + 1e-8)

        # Fuse
        fused = torch.hstack([img_emb, text_emb])  # (1, 1024)

        # Predict
        logits = model(fused)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    # Reverse label map
    reverse_map = {int(v): k for k, v in label_map.items()}
    pred_label = int(logits.argmax(dim=1).item())
    pred_name = reverse_map[pred_label]

    result = {
        "predicted_label": pred_name,
        "predicted_label_id": pred_label,
        "confidence": float(probs[pred_label]),
        "probabilities": {reverse_map[i]: float(probs[i]) for i in range(len(probs))},
        "label_map": label_map,
    }

    LOGGER.info("Prediction: %s (confidence: %.4f)", pred_name, result["confidence"])

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer class for image + text pair")
    parser.add_argument("--model", required=True, help="Path to trained model (.pth)")
    parser.add_argument(
        "--image_url",
        default=None,
        help="Image URL (if not provided, use --image_path)",
    )
    parser.add_argument(
        "--image_path",
        default=None,
        help="Local image path (if --image_url not provided)",
    )
    parser.add_argument("--text", required=True, help="Product text (name || description)")
    parser.add_argument(
        "--clip_model",
        default="openai/clip-vit-base-patch32",
        help="CLIP model name",
    )
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

    if args.image_url is None and args.image_path is None:
        raise ValueError("Provide either --image_url or --image_path")

    image_source = args.image_url or args.image_path

    result = infer(
        model_path=args.model,
        image_source=image_source,
        text=args.text,
        clip_model=args.clip_model,
        device=args.device,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
