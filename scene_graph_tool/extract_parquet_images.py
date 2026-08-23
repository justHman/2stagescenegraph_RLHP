#!/usr/bin/env python3
"""Extract embedded PSG test images from a Hugging Face parquet shard."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def extract_images(parquet_path: Path, output_dir: Path) -> tuple[int, int]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required; install it with `pip install pyarrow`.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(columns=["image", "file_name"], batch_size=64):
        for row in batch.to_pylist():
            name = Path(row["file_name"]).name
            data = row["image"]["bytes"]
            if not name or not data:
                raise ValueError(f"Invalid embedded image record: {row['file_name']!r}")
            destination = output_dir / name
            if destination.is_file() and destination.stat().st_size == len(data):
                skipped += 1
                continue
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(data)
            temporary.replace(destination)
            written += 1
    return written, skipped


def create_image_zip(image_dir: Path, destination: Path) -> int:
    """Create an atomic, upload-ready ZIP without recompressing JPEG/PNG bytes."""
    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"No supported images found in {image_dir}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
        for image in images:
            bundle.write(image, arcname=image.name)
    temporary.replace(destination)
    return len(images)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "test-00000-of-00001.parquet")
    parser.add_argument("--output", type=Path, default=root / "test_images")
    parser.add_argument("--zip-output", type=Path, help="optional upload-ready ZIP path")
    args = parser.parse_args()
    written, skipped = extract_images(args.input, args.output)
    total = sum(1 for path in args.output.iterdir() if path.is_file())
    print(f"Extracted {written}, skipped {skipped}; {total} images available in {args.output}")
    if args.zip_output:
        archived = create_image_zip(args.output, args.zip_output)
        print(f"Archived {archived} images in {args.zip_output}")


if __name__ == "__main__":
    main()
