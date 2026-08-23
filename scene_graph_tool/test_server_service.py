import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app import _place_badge, draw_scene_graph
from batch_session import BatchSession
from server import LabelerService, relation_selection_comparison, top_k_relations


class FakePredictor:
    def predict(self, _image):
        return {
            "objects": [
                {"id": 0, "label": "person", "label_id": 1, "score": 0.9, "box_xyxy": [2, 2, 20, 20]},
                {"id": 1, "label": "bike", "label_id": 2, "score": 0.8, "box_xyxy": [30, 2, 50, 20]},
            ],
            "relations": [{"id": 7, "subject_id": 0, "object_id": 1, "best_predicate": "riding",
                           "best_probability": 0.8, "predicate_probabilities": {"riding": 0.8, "near": 0.2}}],
        }


class LabelerServiceTests(unittest.TestCase):
    def make_session(self, root: Path):
        image = root / "sample.jpg"
        cv2.imwrite(str(image), np.zeros((64, 64, 3), dtype=np.uint8))
        return BatchSession.create(image, root / "exports")

    def test_preview_does_not_change_active_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = self.make_session(root)
            service = LabelerService(FakePredictor(), root / "exports")
            before = BatchSession(session.directory).data
            preview = service.preview(session.directory.name, "0000_sample.jpg")
            after = BatchSession(session.directory).data
            self.assertEqual(preview["image"], "0000_sample.jpg")
            self.assertEqual(before["current_image_index"], after["current_image_index"])
            self.assertEqual(before["answers"], after["answers"])

    def test_undo_returns_prior_relation_and_exposes_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = self.make_session(root)
            service = LabelerService(FakePredictor(), root / "exports")
            task = service.task_payload(session.directory.name)
            self.assertFalse(task["can_undo"])
            service.respond(session.directory.name, type("Payload", (), {
                "image_id": task["image"], "relation_id": task["relation"]["id"],
                "presented_predicates": task["relation"]["choices"], "choice": "first",
            })())
            undone = service.undo(session.directory.name)
            self.assertTrue(undone["can_undo"] is False)
            self.assertEqual(undone["relation"]["id"], 7)

    def test_task_uses_cached_overlay_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = self.make_session(root)
            service = LabelerService(FakePredictor(), root / "exports")
            task = service.task_payload(session.directory.name)
            self.assertTrue(task["overlay"].startswith(f"/api/sessions/{session.directory.name}/overlay?"))
            overlay = service.overlay(session.directory.name, task["image"], task["relation"]["id"])
            self.assertTrue(overlay.is_file())
            self.assertEqual(service.overlay(session.directory.name, task["image"], task["relation"]["id"]), overlay)

    def test_badge_layout_uses_a_non_overlapping_candidate(self):
        occupied = []
        first = _place_badge([(0, 0)], (20, 10), occupied, (100, 100), 2)
        second = _place_badge([(0, 0), (30, 0)], (20, 10), occupied, (100, 100), 2)
        self.assertEqual(first, (0, 0, 20, 10))
        self.assertEqual(second, (30, 0, 50, 10))

    def test_small_overlays_are_rendered_at_display_resolution(self):
        image = np.zeros((360, 640, 3), dtype=np.uint8)
        rendered = draw_scene_graph(image, FakePredictor().predict(image)["objects"], [], False)
        self.assertEqual(rendered.shape[:2], (720, 1280))

    def test_relation_selection_shortlists_by_p1_then_ranks_by_relative_margin(self):
        result = {
            "objects": [
                {"id": 0, "label": "person", "score": 0.95},
                {"id": 1, "label": "bike", "score": 0.90},
                {"id": 2, "label": "road", "score": 0.85},
                {"id": 3, "label": "sky", "score": 0.20},
            ],
            "relations": [
                {"id": 1, "subject_id": 0, "object_id": 1, "best_predicate": "on",
                 "best_probability": 0.40, "predicate_probabilities": {"on": 0.40, "near": 0.39}},
                {"id": 2, "subject_id": 1, "object_id": 2, "best_predicate": "riding",
                 "best_probability": 0.80, "predicate_probabilities": {"riding": 0.80, "near": 0.10}},
                {"id": 3, "subject_id": 0, "object_id": 2, "best_predicate": "above",
                 "best_probability": 0.20, "predicate_probabilities": {"above": 0.20, "under": 0.199}},
                {"id": 4, "subject_id": 0, "object_id": 3, "best_predicate": "beside",
                 "best_probability": 0.90, "predicate_probabilities": {"beside": 0.90, "near": 0.05}},
            ],
        }
        settings = {
            "top_k_relations": 1, "relation_candidate_top_k": 2,
            "object_max_count": 3, "object_min_probability": 0.6,
            "relation_shortlist_multiplier": 2, "relation_min_probability": 0.05,
        }
        selected = top_k_relations(result, settings)
        self.assertEqual([item["id"] for item in selected], [1])
        self.assertGreater(selected[0]["uncertainty_score"], 0.97)

        # Expanding K admits relation 3 into the p1 shortlist; its smaller
        # relative margin then makes it the selected Top-M item.
        expanded = top_k_relations(result, {**settings, "relation_candidate_top_k": 3})
        self.assertEqual([item["id"] for item in expanded], [3])

        # Deprecated predicate thresholds must not alter the explicit K/M stages.
        extreme_legacy_settings = {
            **settings,
            "relation_shortlist_multiplier": 1,
            "relation_min_probability": 0.99,
        }
        self.assertEqual(
            [item["id"] for item in top_k_relations(result, extreme_legacy_settings)],
            [1],
        )
        comparison = relation_selection_comparison(result, settings)
        baseline = [item for item in comparison if item["method"] == "highest_probability"]
        uncertainty = [item for item in comparison if item["method"] == "uncertainty_priority"]
        self.assertEqual(baseline[0]["relation_id"], 4)
        self.assertEqual(uncertainty[0]["relation_id"], 1)
        self.assertGreater(uncertainty[0]["uncertainty_score"], baseline[0]["uncertainty_score"])


if __name__ == "__main__":
    unittest.main()
