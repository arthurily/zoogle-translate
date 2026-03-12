#!/usr/bin/env python3
"""
Prepare character-level training data from:
vimpigro/handwritten-mongolian-cyrillic-characters-database

Uses HMCC letters merged.csv (35 merged letter classes).
Output is a compact JSON for in-browser training.
"""

from __future__ import annotations

import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import kagglehub
except Exception:
    kagglehub = None

RANDOM_SEED = 109
IMAGE_SIZE = 24

# Keep this small enough for browser-side training speed.
TRAIN_PER_CLASS = 120
VAL_PER_CLASS = 40
SAMPLES_PER_CLASS = TRAIN_PER_CLASS + VAL_PER_CLASS

DATASET_SLUG = "vimpigro/handwritten-mongolian-cyrillic-characters-database"
CSV_NAME = "HMCC letters merged.csv"
OUT_PATH = Path("/Users/arthurilyasov/cs109Project/data/hmcc_russian_letters_dataset.json")
CACHED_ROOT = Path(
    "/Users/arthurilyasov/.cache/kagglehub/datasets/vimpigro/"
    "handwritten-mongolian-cyrillic-characters-database/versions/2"
)

# Class IDs 0..34 in merged-letter dataset.
FULL_LABELS = [
    "а",
    "б",
    "в",
    "г",
    "д",
    "е",
    "ё",
    "ж",
    "з",
    "и",
    "й",
    "к",
    "л",
    "м",
    "н",
    "о",
    "ө",
    "п",
    "р",
    "с",
    "т",
    "у",
    "ү",
    "ф",
    "х",
    "ц",
    "ч",
    "ш",
    "щ",
    "ъ",
    "ы",
    "ь",
    "э",
    "ю",
    "я",
]

RUSSIAN_CLASS_IDS = [
    idx for idx, label in enumerate(FULL_LABELS) if label not in {"ө", "ү"}
]
LABELS = [FULL_LABELS[idx] for idx in RUSSIAN_CLASS_IDS]


def to_24x24_ink_uint8(flat_pixels: list[str]) -> np.ndarray:
    gray = np.asarray(flat_pixels, dtype=np.uint8).reshape(28, 28)
    img = Image.fromarray(gray, mode="L").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
    )
    # App expects "ink intensity": white->0, black->255
    ink = 255 - np.asarray(img, dtype=np.uint8)
    return ink


def reservoir_insert(
    store: dict[int, list[np.ndarray]],
    seen: dict[int, int],
    cls: int,
    sample: np.ndarray,
) -> None:
    seen[cls] += 1
    arr = store[cls]
    s = seen[cls]
    if len(arr) < SAMPLES_PER_CLASS:
        arr.append(sample)
        return
    j = random.randrange(s)
    if j < SAMPLES_PER_CLASS:
        arr[j] = sample


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    env_root = os.environ.get("HMCC_DATASET_ROOT", "").strip()
    if env_root:
        root = Path(env_root).expanduser()
    elif CACHED_ROOT.exists():
        root = CACHED_ROOT
    else:
        if kagglehub is None:
            raise RuntimeError(
                "kagglehub is not installed and no cached dataset path is available."
            )
        root = Path(kagglehub.dataset_download(DATASET_SLUG))
    csv_path = root / CSV_NAME
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected dataset file not found: {csv_path}")

    samples: dict[int, list[np.ndarray]] = defaultdict(list)
    seen: dict[int, int] = defaultdict(int)
    class_map = {original: new_idx for new_idx, original in enumerate(RUSSIAN_CLASS_IDS)}

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            original_cls = int(row[0])
            if original_cls not in class_map:
                continue
            cls = class_map[original_cls]
            sample = to_24x24_ink_uint8(row[1:])
            reservoir_insert(samples, seen, cls, sample)

    train_x: list[list[int]] = []
    train_y: list[int] = []
    val_x: list[list[int]] = []
    val_y: list[int] = []

    for cls in range(len(LABELS)):
        arr = samples.get(cls, [])
        random.shuffle(arr)
        used_train = min(TRAIN_PER_CLASS, len(arr))
        used_val = min(VAL_PER_CLASS, max(0, len(arr) - used_train))

        for x in arr[:used_train]:
            train_x.append(x.reshape(-1).tolist())
            train_y.append(cls)
        for x in arr[used_train : used_train + used_val]:
            val_x.append(x.reshape(-1).tolist())
            val_y.append(cls)

    paired_train = list(zip(train_x, train_y))
    paired_val = list(zip(val_x, val_y))
    random.shuffle(paired_train)
    random.shuffle(paired_val)
    train_x = [x for x, _ in paired_train]
    train_y = [y for _, y in paired_train]
    val_x = [x for x, _ in paired_val]
    val_y = [y for _, y in paired_val]

    payload = {
        "meta": {
            "dataset_slug": DATASET_SLUG,
            "source_csv": CSV_NAME,
            "selected_original_class_ids": RUSSIAN_CLASS_IDS,
            "image_size": IMAGE_SIZE,
            "random_seed": RANDOM_SEED,
            "train_per_class_target": TRAIN_PER_CLASS,
            "val_per_class_target": VAL_PER_CLASS,
            "seen_per_class": {str(i): seen.get(i, 0) for i in range(len(LABELS))},
            "used_train_per_class": {
                str(i): int(sum(1 for y in train_y if y == i)) for i in range(len(LABELS))
            },
            "used_val_per_class": {
                str(i): int(sum(1 for y in val_y if y == i)) for i in range(len(LABELS))
            },
        },
        "labels": LABELS,
        "trainX": train_x,
        "trainY": train_y,
        "testX": val_x,
        "testY": val_y,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    print(f"Dataset root: {root}")
    print(f"Source CSV: {csv_path}")
    print(f"Wrote: {OUT_PATH}")
    print(f"Classes: {len(LABELS)}")
    print(f"Train samples: {len(train_x)}")
    print(f"Validation samples: {len(val_x)}")


if __name__ == "__main__":
    main()
