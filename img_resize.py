"""Image preprocessing helpers.

This script provides two utilities:
1) Pad images to a square canvas (keeping aspect ratio).
2) Resize images to a fixed size (e.g., 128x128).

Examples:
  python img_resize.py pad-square --input_dir data/raw --output_dir data/square
  python img_resize.py resize --input_dir data/square --output_dir data/square128 --size 128
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


def pad_to_square(img: Image.Image, fill: int = 0) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    pad_w = (m - w) // 2
    pad_h = (m - h) // 2
    padding = (pad_w, pad_h, m - w - pad_w, m - h - pad_h)
    return ImageOps.expand(img, padding, fill=fill)


def iter_images(input_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def cmd_pad_square(input_dir: Path, output_dir: Path, fill: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for p in iter_images(input_dir):
        img = Image.open(p).convert("L")
        out = pad_to_square(img, fill=fill)
        out_path = output_dir / (p.stem + ".png")
        out.save(out_path)


def cmd_resize(input_dir: Path, output_dir: Path, size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for p in iter_images(input_dir):
        img = Image.open(p).convert("L")
        out = img.resize((size, size), resample=Image.BILINEAR)
        out_path = output_dir / (p.stem + ".png")
        out.save(out_path)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Image preprocessing")
    sub = ap.add_subparsers(dest="command", required=True)

    ap_pad = sub.add_parser("pad-square", help="Pad images to a square canvas")
    ap_pad.add_argument("--input_dir", type=str, required=True)
    ap_pad.add_argument("--output_dir", type=str, required=True)
    ap_pad.add_argument("--fill", type=int, default=0, help="Padding value (0-255)")

    ap_resize = sub.add_parser("resize", help="Resize images to N x N")
    ap_resize.add_argument("--input_dir", type=str, required=True)
    ap_resize.add_argument("--output_dir", type=str, required=True)
    ap_resize.add_argument("--size", type=int, default=128)

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if args.command == "pad-square":
        cmd_pad_square(input_dir, output_dir, fill=args.fill)
    elif args.command == "resize":
        cmd_resize(input_dir, output_dir, size=args.size)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
