#!/usr/bin/env python3
"""
Convert a folder of letter images into a Zoogle Translate dataset.

Expected folder structure:
  dataset/
    A/
      001.png
      002.png
      ...
    B/
      001.png
      ...
    (one folder per letter A-Z)

Or flat structure with filename prefix:
  dataset/
    A_001.png
    A_002.png
    B_001.png
    ...

Output: A JSON file you can Import in the app (Import Dataset button).

Usage:
  python scripts/photo_to_dataset.py dataset/ -o my_language.json
  python scripts/photo_to_dataset.py dataset/ -o my_language.json --name "My Alien Language"
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Install Pillow: pip install Pillow")
    raise

IMAGE_SIZE = 24
INPUT_DIM = IMAGE_SIZE * IMAGE_SIZE
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def center_of_mass(vec: list[float]) -> tuple[float, float]:
    mass = 0.0
    mx, my = 0.0, 0.0
    for y in range(IMAGE_SIZE):
        for x in range(IMAGE_SIZE):
            v = vec[y * IMAGE_SIZE + x]
            mass += v
            mx += x * v
            my += y * v
    if mass < 1e-9:
        return (IMAGE_SIZE - 1) * 0.5, (IMAGE_SIZE - 1) * 0.5
    cx = mx / mass
    cy = my / mass
    return cx, cy


def shift_vec(vec: list[float], tx: int, ty: int) -> list[float]:
    out = [0.0] * INPUT_DIM
    for y in range(IMAGE_SIZE):
        for x in range(IMAGE_SIZE):
            sx, sy = x - tx, y - ty
            if 0 <= sx < IMAGE_SIZE and 0 <= sy < IMAGE_SIZE:
                out[y * IMAGE_SIZE + x] = vec[sy * IMAGE_SIZE + sx]
    return out


def center_of_mass_shift(vec: list[float]) -> list[float]:
    cx, cy = center_of_mass(vec)
    tx = round((IMAGE_SIZE - 1) * 0.5 - cx)
    ty = round((IMAGE_SIZE - 1) * 0.5 - cy)
    if tx == 0 and ty == 0:
        return vec
    return shift_vec(vec, tx, ty)


def image_to_vec(img_path: Path, invert: bool = True) -> list[float] | None:
    """Load image, resize to 24x24, convert to grayscale 0-1 vector."""
    try:
        img = Image.open(img_path).convert("L")  # grayscale
    except Exception:
        return None
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    vec = []
    for y in range(IMAGE_SIZE):
        for x in range(IMAGE_SIZE):
            p = img.getpixel((x, y)) / 255.0
            if invert:
                p = 1.0 - p
            vec.append(max(0.0, min(1.0, p)))
    return vec


def encode_sample(vec: list[float]) -> str:
    """Encode 576 floats as base64 string (same format as app.js)."""
    bytes_arr = bytes(int(round(max(0, min(1, v)) * 255)) for v in vec)
    return base64.b64encode(bytes_arr).decode("ascii")


def collect_from_folders(root: Path, invert: bool = True) -> dict[str, list[str]]:
    """Collect images from folder-per-letter structure: dataset/A/001.png"""
    samples: dict[str, list[str]] = {l: [] for l in LETTERS}
    for letter in LETTERS:
        folder = root / letter
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                continue
            vec = image_to_vec(path, invert=invert)
            if vec is None:
                continue
            centered = center_of_mass_shift(vec)
            samples[letter].append(encode_sample(centered))
    return samples


def collect_from_flat(root: Path, invert: bool = True) -> dict[str, list[str]]:
    """Collect images from flat structure: dataset/A_001.png or dataset/a_001.png"""
    samples: dict[str, list[str]] = {l: [] for l in LETTERS}
    pattern = re.compile(r"^([A-Za-z])_([^.]+)\.(png|jpg|jpeg|bmp|webp)$", re.I)
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        m = pattern.match(path.name)
        if not m:
            continue
        letter = m.group(1).upper()
        if letter not in samples:
            continue
        vec = image_to_vec(path, invert=invert)
        if vec is None:
            continue
        centered = center_of_mass_shift(vec)
        samples[letter].append(encode_sample(centered))
    return samples


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert photos to Zoogle dataset")
    ap.add_argument("input", type=Path, help="Input folder (folder-per-letter or flat)")
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output JSON file")
    ap.add_argument("--name", default="Imported Language", help="Language name")
    ap.add_argument("--flat", action="store_true", help="Use flat naming (A_001.png) instead of folders")
    ap.add_argument("--no-invert", action="store_true", help="Use if strokes are white on black (default: black on white)")
    args = ap.parse_args()

    if not args.input.is_dir():
        ap.error(f"Input must be a directory: {args.input}")

    invert = not args.no_invert
    if args.flat:
        samples = collect_from_flat(args.input, invert=invert)
    else:
        samples = collect_from_folders(args.input, invert=invert)

    total = sum(len(v) for v in samples.values())
    print(f"Collected {total} samples across {sum(1 for v in samples.values() if v)} letters")

    payload = {
        "id": "imported",
        "name": args.name,
        "samplesByLetter": samples,
        "metrics": {"valAcc": None, "valLoss": None, "trainedAt": None},
        "model": None,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    print(f"Saved to {args.output}")
    print("Use 'Import Dataset' in the app to load this file.")


if __name__ == "__main__":
    main()
