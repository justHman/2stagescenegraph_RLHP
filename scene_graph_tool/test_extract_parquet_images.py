import tempfile
import unittest
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from extract_parquet_images import create_image_zip, extract_images


class ExtractParquetImagesTests(unittest.TestCase):
    def test_extracts_and_skips_existing_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "images.parquet"
            pq.write_table(pa.table({
                "image": [{"bytes": b"first", "path": "one.jpg"}, {"bytes": b"second", "path": "two.jpg"}],
                "file_name": ["one.jpg", "nested/two.jpg"],
            }), source)
            output = root / "images"
            self.assertEqual(extract_images(source, output), (2, 0))
            self.assertEqual((output / "one.jpg").read_bytes(), b"first")
            self.assertEqual((output / "two.jpg").read_bytes(), b"second")
            self.assertEqual(extract_images(source, output), (0, 2))
            archive = root / "images.zip"
            self.assertEqual(create_image_zip(output, archive), 2)
            with zipfile.ZipFile(archive) as bundle:
                self.assertEqual(bundle.namelist(), ["one.jpg", "two.jpg"])
                self.assertEqual(bundle.read("one.jpg"), b"first")


if __name__ == "__main__":
    unittest.main()
