"""Extract a subset of real PSG test images from the HuggingFace parquet shard."""
import sys
from pathlib import Path

PARQUET = Path(r"D:\Studying\University\Ki_8\IMP\rlhp_analysis\live_demo\data\psg_test.parquet")
OUT = Path(r"D:\Studying\University\Ki_8\IMP\rlhp_analysis\live_demo\images")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def main():
    import pyarrow.parquet as pq
    OUT.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(PARQUET)
    written = 0
    for batch in pf.iter_batches(columns=["image", "file_name"], batch_size=32):
        for row in batch.to_pylist():
            name = Path(row["file_name"]).name
            data = (row["image"] or {}).get("bytes")
            if not name or not data:
                continue
            # jpg/jpeg only
            if not name.lower().endswith((".jpg", ".jpeg")):
                continue
            dest = OUT / name
            if not dest.exists():
                dest.write_bytes(data)
                written += 1
            if written >= LIMIT:
                print(f"wrote {written} images -> {OUT}")
                return
    print(f"wrote {written} images -> {OUT}")


if __name__ == "__main__":
    main()
