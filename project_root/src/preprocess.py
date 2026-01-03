from __future__ import annotations

import argparse
import logging
import random
import re
import unicodedata
import uuid

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


LOGGER = logging.getLogger("preprocess")


def seed_everything(seed: int) -> None:
	"""Set seed for python, numpy and random to ensure reproducibility."""
	random.seed(seed)
	np.random.seed(seed)


def find_first_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
	"""Return first matching column name from candidates present in df."""
	cols = set(df.columns.str.lower())
	for c in candidates:
		if c.lower() in cols:
			# return the actual column name (case-preserving)
			for actual in df.columns:
				if actual.lower() == c.lower():
					return actual
	return None


def normalize_text(s: str) -> str:
	if pd.isna(s):
		return ""
	s = str(s)
	s = unicodedata.normalize("NFKC", s)
	s = re.sub(r"\s+", " ", s).strip()
	return s


def parse_first_image_url(field: str) -> Optional[str]:
	"""Split an image field on '|' or whitespace and return the first non-placeholder URL.

	Heuristics skip data URIs and obvious placeholders.
	"""
	if pd.isna(field):
		return None
	text = str(field)
	parts = re.split(r"\||\s+", text)
	for p in parts:
		p = p.strip()
		if not p:
			continue
		low = p.lower()
		if low.startswith("data:"):
			continue
		if "placeholder" in low or "transparent" in low or "spacer.gif" in low:
			continue
		# basic http/https check
		if low.startswith("http://") or low.startswith("https://"):
			return p
	return None


def choose_top_categories(df: pd.DataFrame, base_k: int = 5, min_per_class: int = 200) -> List[str]:
	"""Choose top categories, extending beyond base_k until each has >= min_per_class.

	Returns ordered list of chosen category names.
	"""
	counts = df["category_name"].value_counts()
	candidates = list(counts.index)
	chosen = candidates[:base_k]
	idx = base_k
	while True:
		too_small = [c for c in chosen if counts.get(c, 0) < min_per_class]
		if not too_small:
			break
		if idx >= len(candidates):
			# can't extend further
			break
		chosen.append(candidates[idx])
		idx += 1
	return chosen


def build_processed_df(raw_df: pd.DataFrame, max_per_class: int, min_examples: int) -> pd.DataFrame:
	# identify useful columns with common name fallbacks
	name_col = find_first_column(raw_df, ["Product Name", "product_name", "title", "name"])
	desc_col = find_first_column(
		raw_df,
		["Product Description", "Product Details", "About Product", "description", "product_description", "details"],
	)
	images_col = find_first_column(raw_df, ["Image", "Images", "image", "image_url", "images_url"])
	category_col = find_first_column(raw_df, ["Category", "category", "categories"])

	if category_col is None:
		raise ValueError("Could not find a Category column in the raw CSV")

	# create category_name column for counting
	raw_df = raw_df.copy()
	raw_df["category_name"] = raw_df[category_col].astype(str).str.strip()

	# select top categories as needed (initially by frequency)
	chosen = choose_top_categories(raw_df, base_k=5, min_per_class=min_examples)
	LOGGER.info("Chosen categories: %s", chosen)

	df = raw_df[raw_df["category_name"].isin(chosen)].copy()

	# cap each class at max_per_class
	frames = []
	for cat in chosen:
		sub = df[df["category_name"] == cat]
		if len(sub) > max_per_class:
			sub = sub.sample(n=max_per_class, random_state=42)
		frames.append(sub)
	df = pd.concat(frames, ignore_index=True)

	# construct text
	def make_text(row):
		parts = []
		if name_col is not None:
			parts.append(normalize_text(row.get(name_col, "")))
		else:
			parts.append("")
		if desc_col is not None:
			parts.append(normalize_text(row.get(desc_col, "")))
		else:
			parts.append("")
		return " || ".join([p for p in parts if p])

	df["text"] = df.apply(make_text, axis=1)

	# parse first image url
	if images_col is not None:
		df["image_path"] = df[images_col].apply(parse_first_image_url)
	else:
		df["image_path"] = None

	# drop rows without any valid image URL
	df = df[df["image_path"].notna()].copy()

	# label map
	label_map = {name: i for i, name in enumerate(sorted(df["category_name"].unique()))}
	df["label"] = df["category_name"].map(label_map)

	# uniq id
	df["uniq_id"] = [uuid.uuid4().hex for _ in range(len(df))]

	# stratified split
	train_idx, val_idx = train_test_split(
		df.index, test_size=0.2, stratify=df["label"], random_state=42
	)
	df.loc[train_idx, "split"] = "train"
	df.loc[val_idx, "split"] = "val"

	# final columns order
	out = df[["uniq_id", "image_path", "text", "label", "category_name", "split"]].copy()
	return out


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Preprocess Amazon product CSV for multimodal probe")
	parser.add_argument("--raw_csv", required=True, help="Path to raw CSV (downloaded from Kaggle)")
	parser.add_argument("--out_csv", required=True, help="Path to write processed CSV")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	parser.add_argument("--max_per_class", type=int, default=3000, help="Max examples per class (cap)")
	parser.add_argument("--min_per_class", type=int, default=200, help="Minimum examples per chosen class")
	parser.add_argument("--debug", action="store_true", help="Debug mode: fewer workers/verbose logging")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG)
	seed_everything(args.seed)

	raw_csv = args.raw_csv
	out_csv = args.out_csv

	LOGGER.info("Loading raw CSV: %s", raw_csv)
	df = pd.read_csv(raw_csv, low_memory=False)
	LOGGER.info("Raw rows: %d", len(df))

	processed = build_processed_df(df, max_per_class=args.max_per_class, min_examples=args.min_per_class)
	LOGGER.info("Processed rows: %d", len(processed))

	# write CSV
	processed.to_csv(out_csv, index=False)
	LOGGER.info("Wrote processed labels to %s", out_csv)


if __name__ == "__main__":
	main()

