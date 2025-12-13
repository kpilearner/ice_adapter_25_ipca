#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path


# Edit this if you want the instruction hard-coded.
DEFAULT_PROMPT = "convert to infrared image."


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class PairRow:
    kontext_images: str
    image: str
    prompt: str


def iter_images(root: Path, recursive: bool) -> list[Path]:
    if recursive:
        files = [p for p in root.rglob("*") if p.is_file()]
    else:
        files = [p for p in root.glob("*") if p.is_file()]
    return [p for p in files if p.suffix.lower() in IMG_EXTS]


def to_rel(path: Path, base_path: Path) -> str:
    try:
        return path.relative_to(base_path).as_posix()
    except ValueError:
        return path.as_posix()


def build_ir_index(ir_files: list[Path], ir_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for p in ir_files:
        key = p.relative_to(ir_root).with_suffix("").as_posix()
        index.setdefault(key, p)
    return index


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create Kontext-like metadata CSV for paired image translation datasets."
    )
    parser.add_argument("--vis-dir", type=str, required=True, help="Visible/source images folder")
    parser.add_argument("--ir-dir", type=str, required=True, help="Infrared/target images folder")
    parser.add_argument(
        "--out",
        type=str,
        default="metadata.csv",
        help="Output CSV path (default: metadata.csv)",
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default=None,
        help="If set, write paths relative to this directory (recommended).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f'Instruction text to put in CSV (default: "{DEFAULT_PROMPT}")',
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subfolders of vis/ir dirs",
    )
    args = parser.parse_args()

    vis_root = Path(args.vis_dir).expanduser().resolve()
    ir_root = Path(args.ir_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    base_path = Path(args.base_path).expanduser().resolve() if args.base_path else None

    if not vis_root.is_dir():
        raise SystemExit(f"--vis-dir is not a directory: {vis_root}")
    if not ir_root.is_dir():
        raise SystemExit(f"--ir-dir is not a directory: {ir_root}")

    vis_files = sorted(iter_images(vis_root, recursive=args.recursive))
    ir_files = sorted(iter_images(ir_root, recursive=args.recursive))
    if not vis_files:
        raise SystemExit(f"No images found in {vis_root} (exts: {sorted(IMG_EXTS)})")
    if not ir_files:
        raise SystemExit(f"No images found in {ir_root} (exts: {sorted(IMG_EXTS)})")

    ir_index = build_ir_index(ir_files, ir_root)

    rows: list[PairRow] = []
    missing = 0
    for vis_path in vis_files:
        key = vis_path.relative_to(vis_root).with_suffix("").as_posix()
        ir_path = ir_index.get(key)
        if ir_path is None:
            missing += 1
            continue

        if base_path is None:
            rows.append(
                PairRow(
                    kontext_images=vis_path.as_posix(),
                    image=ir_path.as_posix(),
                    prompt=args.prompt,
                )
            )
        else:
            rows.append(
                PairRow(
                    kontext_images=to_rel(vis_path, base_path),
                    image=to_rel(ir_path, base_path),
                    prompt=args.prompt,
                )
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["kontext_images", "image", "prompt"])
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {"kontext_images": r.kontext_images, "image": r.image, "prompt": r.prompt}
            )

    print(f"Wrote {len(rows)} pairs to: {out_path}")
    if missing:
        print(
            f"Warning: {missing} VIS images had no matching IR (match key: relative path without extension).",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

