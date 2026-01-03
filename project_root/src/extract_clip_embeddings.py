from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

LOGGER = logging.getLogger("extract_clip_embeddings")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CLIPEmbedder:
    def __init__(
        self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"
    ):
        self.device = device
        LOGGER.info("Loading CLIP model: %s on device: %s", model_name, device)
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def embed_images(self, image_paths: list[str]) -> np.ndarray:
        """Embed a batch of image paths. Returns (N, 512) embeddings."""
        images = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                LOGGER.warning("Failed to load image %s: %s", path, e)
                images.append(Image.new("RGB", (224, 224)))  # blank fallback

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        # Normalize
        embeddings = outputs / (outputs.norm(dim=-1, keepdim=True) + 1e-8)
        return embeddings.cpu().numpy()

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        #Embed a batch of texts. Returns (N, 512) embeddings.
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(
            self.device
        )
        outputs = self.model.get_text_features(**inputs)
        # Normalize
        embeddings = outputs / (outputs.norm(dim=-1, keepdim=True) + 1e-8)
        return embeddings.cpu().numpy()


def extract_embeddings(
    labels_csv: str,
    out_dir: str,
    clip_model: str = "openai/clip-vit-base-patch32",
    image_batch_size: int = 32,
    text_batch_size: int = 64,
    device: str = "cpu",
) -> None:
    #Load processed labels, extract CLIP embeddings, save .npy files.
    # Create output directory
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load processed labels
    df = pd.read_csv(labels_csv)
    LOGGER.info("Loaded %d rows from %s", len(df), labels_csv)

    # Filter out rows with NaN labels (data quality check)
    df = df[df["label"].notna()].reset_index(drop=True)
    df = df[df["category_name"].notna()].reset_index(drop=True)
    LOGGER.info("After filtering NaN labels/categories: %d rows", len(df))

    # Initialize embedder
    embedder = CLIPEmbedder(model_name=clip_model, device=device)

    # Extract embeddings in batches
    image_embeddings = []
    text_embeddings = []
    all_labels = []
    all_ids = []

    num_samples = len(df)

    # Process images
    LOGGER.info("Extracting image embeddings...")
    for i in range(0, num_samples, image_batch_size):
        batch_end = min(i + image_batch_size, num_samples)
        batch_paths = df.iloc[i:batch_end]["image_path"].tolist()
        batch_emb = embedder.embed_images(batch_paths)
        image_embeddings.append(batch_emb)

        if (i + image_batch_size) % (image_batch_size * 5) == 0 or batch_end == num_samples:
            LOGGER.info("  Processed %d/%d images", batch_end, num_samples)

    # Process texts
    LOGGER.info("Extracting text embeddings...")
    for i in range(0, num_samples, text_batch_size):
        batch_end = min(i + text_batch_size, num_samples)
        batch_texts = df.iloc[i:batch_end]["text"].tolist()
        batch_emb = embedder.embed_texts(batch_texts)
        text_embeddings.append(batch_emb)

        if (i + text_batch_size) % (text_batch_size * 5) == 0 or batch_end == num_samples:
            LOGGER.info("  Processed %d/%d texts", batch_end, num_samples)

    # Concatenate
    image_emb = np.vstack(image_embeddings)  # (N, 512)
    text_emb = np.vstack(text_embeddings)  # (N, 512)
    all_labels = df["label"].values  # (N,)
    all_ids = df["uniq_id"].values  # (N,)

    LOGGER.info("Concatenating embeddings...")
    fused_emb = np.hstack([image_emb, text_emb])  # (N, 1024)
    LOGGER.info("Fused embedding shape: %s", fused_emb.shape)

    # Save .npy files
    LOGGER.info("Saving embeddings...")
    np.save(os.path.join(out_dir, "image_embeddings.npy"), image_emb)
    np.save(os.path.join(out_dir, "text_embeddings.npy"), text_emb)
    np.save(os.path.join(out_dir, "labels.npy"), all_labels.astype(int))
    np.save(os.path.join(out_dir, "ids.npy"), all_ids, allow_pickle=True)

    # Save label map
    # Build a clean label map from non-NaN categories
    valid_cat_labels = df[df["category_name"].notna()][["category_name", "label"]].drop_duplicates()
    label_map_dict = {row["category_name"]: int(row["label"]) for _, row in valid_cat_labels.iterrows()}
    label_map_json = {str(k): v for k, v in sorted(label_map_dict.items())}
    with open(os.path.join(out_dir, "label_map.json"), "w") as f:
        json.dump(label_map_json, f, indent=2)

    LOGGER.info("Embeddings saved to %s/", out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CLIP embeddings from images and texts")
    parser.add_argument(
        "--labels_csv", required=True, help="Path to processed_labels.csv"
    )
    parser.add_argument(
        "--out_dir", required=True, help="Directory to save embeddings (.npy files)"
    )
    parser.add_argument(
        "--clip_model",
        default="openai/clip-vit-base-patch32",
        help="HuggingFace CLIP model name",
    )
    parser.add_argument(
        "--image_batch", type=int, default=32, help="Image batch size"
    )
    parser.add_argument(
        "--text_batch", type=int, default=64, help="Text batch size"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode: verbose logging"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(name)s:%(levelname)s: %(message)s",
    )
    seed_everything(args.seed)

    extract_embeddings(
        labels_csv=args.labels_csv,
        out_dir=args.out_dir,
        clip_model=args.clip_model,
        image_batch_size=args.image_batch,
        text_batch_size=args.text_batch,
        device=args.device,
    )


if __name__ == "__main__":
    main()
