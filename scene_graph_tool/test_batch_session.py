import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from batch_session import BatchSession


class BatchSessionTests(unittest.TestCase):
    def make_archive(self, root: Path):
        archive = root / "images.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("nested/b.png", b"not-a-real-image")
            bundle.writestr("a.jpg", b"not-a-real-image")
            bundle.writestr("notes.txt", b"ignore")
        return archive

    def test_create_orders_images_and_persists_answers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = BatchSession.create(self.make_archive(root), root / "exports")
            self.assertEqual([item["id"] for item in session.data["images"]], ["a.jpg", "nested/b.png"])
            answer = {"image_id": "a.jpg", "relation_id": 3, "response": "both wrong"}
            session.record_answer("a.jpg", 3, answer)
            answer["response"] = "strong preference"
            session.record_answer("a.jpg", 3, answer)
            reopened = BatchSession(session.directory)
            self.assertEqual(len(reopened.data["answers"]), 1)
            self.assertEqual(list(reopened.data["answers"].values())[0]["response"], "strong preference")

    def test_undo_restores_previous_answer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = BatchSession.create(self.make_archive(root), root / "exports")
            session.record_answer("a.jpg", 1, {"response": "first"})
            session.record_answer("a.jpg", 1, {"response": "second"})
            self.assertTrue(session.undo())
            self.assertEqual(list(session.data["answers"].values())[0]["response"], "first")
            self.assertTrue(session.undo())
            self.assertFalse(session.data["answers"])

    def test_export_contains_batch_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = BatchSession.create(
                self.make_archive(root),
                root / "exports",
                provenance={
                    "checkpoint_path": "/models/base.pth", "checkpoint_sha256": "a" * 64,
                    "config_path": "/models/config.yml", "config_sha256": "b" * 64,
                    "label_map_path": "/data/labels.json", "label_map_sha256": "c" * 64,
                },
            )
            session.record_answer("a.jpg", 3, {
                "created_at": "2026-08-20T10:00:00", "image_id": "a.jpg", "relation_id": 3,
                "response": "strong preference: on",
                "target_tensor": [1.0, 0.0],
                "subject": {"id": 0, "label": "person", "label_id": 1, "score": 0.9,
                            "box_xyxy": [1, 2, 3, 4]},
                "object": {"id": 1, "label": "bike", "label_id": 2, "score": 0.8,
                           "box_xyxy": [5, 6, 7, 8]},
                "presented_predicates": [
                    {"predicate_id": 1, "name": "on", "probability": 0.4},
                    {"predicate_id": 2, "name": "near", "probability": 0.39},
                ],
                "selection": {"triplet_score": 0.288, "relative_margin": 0.0127,
                              "uncertainty_score": 0.9873},
                "settings": dict(session.data["settings"]),
            })
            session.set_prediction("a.jpg", {
                "objects": [
                    {"id": 0, "label": "person", "label_id": 1, "score": 0.9, "box_xyxy": [1, 2, 3, 4]},
                    {"id": 1, "label": "bike", "label_id": 2, "score": 0.8, "box_xyxy": [5, 6, 7, 8]},
                ],
                "relations": [{
                    "id": 3, "subject_id": 0, "object_id": 1,
                    "background_probability": 0.2,
                    "predicate_probabilities": {"on": 0.4, "near": 0.39},
                }],
            })
            path = session.export()
            with path.open(newline="", encoding="utf-8") as handle:
                exported = list(csv.DictReader(handle))
            self.assertEqual(path.suffix, ".csv")
            self.assertEqual(exported[0]["source_archive"], "images.zip")
            self.assertEqual(exported[0]["top_k_relations"], "3")
            self.assertEqual(exported[0]["relation_candidate_top_k"], "15")
            self.assertEqual(exported[0]["predicate_1_name"], "on")
            self.assertEqual(exported[0]["predicate_1_id"], "1")
            self.assertEqual(exported[0]["checkpoint_sha256"], "a" * 64)
            self.assertEqual(exported[0]["config_sha256"], "b" * 64)
            self.assertEqual(exported[0]["label_map_sha256"], "c" * 64)
            self.assertEqual(exported[0]["image_checksum"], session.data["images"][0]["sha256"])
            self.assertEqual(exported[0]["uncertainty_score"], "0.9873")
            self.assertEqual(exported[0]["subject_prediction_index"], "0")
            self.assertEqual(exported[0]["object_prediction_index"], "1")
            self.assertEqual(exported[0]["relation_prediction_index"], "3")
            self.assertEqual(exported[0]["relation_pair_key"], "0:1")
            distribution = json.loads(exported[0]["relation_distribution_json"])
            self.assertEqual(distribution["predicate_probabilities"]["on"], 0.4)

    def test_legacy_session_receives_uncertainty_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = BatchSession.create(self.make_archive(root), root / "exports")
            data = json.loads(session.path.read_text())
            data["schema_version"] = 2
            data["settings"] = {"top_k_relations": 2}
            session.path.write_text(json.dumps(data))
            migrated = BatchSession(session.directory)
            self.assertEqual(migrated.data["schema_version"], 6)
            self.assertEqual(migrated.data["settings"]["object_max_count"], 30)
            self.assertEqual(migrated.data["settings"]["relation_candidate_top_k"], 10)
            self.assertNotIn("relation_min_probability", migrated.data["settings"])
            self.assertNotIn("relation_shortlist_multiplier", migrated.data["settings"])
            self.assertIn("sha256", migrated.data["images"][0])

    def test_selector_migration_requeues_images_previously_without_relations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = BatchSession.create(self.make_archive(root), root / "exports")
            data = json.loads(session.path.read_text())
            data.pop("relation_selection_version", None)
            data["images"][0]["status"] = "no_eligible_relations"
            data["images"][1]["status"] = "pending"
            data["current_image_index"] = 1
            data["current_relation_id"] = 99
            session.path.write_text(json.dumps(data))

            migrated = BatchSession(session.directory)
            self.assertEqual(migrated.data["relation_selection_version"], 3)
            self.assertEqual(migrated.data["images"][0]["status"], "pending")
            self.assertEqual(migrated.data["current_image_index"], 0)
            self.assertIsNone(migrated.data["current_relation_id"])

    def test_settings_require_candidate_k_to_cover_final_m(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = BatchSession.create(self.make_archive(root), root / "exports")
            with self.assertRaisesRegex(ValueError, "at least"):
                session.set_settings({
                    "top_k_relations": 5,
                    "relation_candidate_top_k": 3,
                    "object_max_count": 30,
                    "object_min_probability": 0.5,
                })

    def test_predictions_are_cached_outside_the_hot_session_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = BatchSession.create(self.make_archive(root), root / "exports")
            prediction = {"objects": [{"id": 0, "label": "person"}], "relations": []}
            session.set_prediction("a.jpg", prediction)
            persisted = json.loads(session.path.read_text())
            self.assertEqual(persisted["predictions"], {})
            self.assertIn("a.jpg", persisted["prediction_index"])
            self.assertEqual(BatchSession(session.directory).get_prediction("a.jpg"), prediction)

    def test_rejects_zip_path_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../outside.jpg", b"unsafe")
            with self.assertRaisesRegex(ValueError, "unsafe"):
                BatchSession.create(archive, root / "exports")
            self.assertFalse((root / "outside.jpg").exists())

    def test_accepts_multiple_individual_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "one.jpg", root / "two.png"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            session = BatchSession.create([first, second], root / "exports")
            self.assertEqual(len(session.data["images"]), 2)
            self.assertEqual(session.data["source_archive"], "2 uploaded images")
            self.assertTrue((session.input_dir / "0000_one.jpg").exists())


if __name__ == "__main__":
    unittest.main()
