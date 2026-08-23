import hashlib
import json
import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from batch_session import BatchSession
from plot_rlhp_training import plot_run
from rlhp import (
    DEFAULT_LABELS,
    RLHPError,
    atomic_json,
    audit_sessions,
    evaluate_pipeline,
    flatten_evaluation_metrics,
    masked_retention_loss,
    pairwise_preference_loss,
    promote_run,
    resolve_active_model,
    rollback_manifest,
    train_pipeline,
)


class RLHPPipelineTests(unittest.TestCase):
    def make_session(self, root: Path, image_names=None):
        image_names = image_names or [f"train_{index}.jpg" for index in range(5)]
        session = root / "session_mock"
        input_dir = session / "input"
        input_dir.mkdir(parents=True)
        labels = json.loads(DEFAULT_LABELS.read_text())
        predicate_ids = {name: index for index, name in enumerate(labels["predicates"])}
        first_name, second_name = "on", "beside"
        images, answers = [], {}
        for index, name in enumerate(image_names):
            path = input_dir / name
            path.write_bytes(f"image-{index}".encode())
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            images.append({"id": name, "source_name": name, "sha256": checksum, "status": "done"})
            answers[f"{name}::{index}"] = {
                "image_id": name,
                "source_name": name,
                "image_checksum": checksum,
                "relation_id": index,
                "subject": {"id": 0, "label_id": 1, "label": "person", "box_xyxy": [0, 0, 10, 10]},
                "object": {"id": 1, "label_id": 2, "label": "bicycle", "box_xyxy": [12, 0, 22, 10]},
                "presented_predicates": [
                    {"predicate_id": predicate_ids[first_name], "name": first_name, "probability": 0.2},
                    {"predicate_id": predicate_ids[second_name], "name": second_name, "probability": 0.8},
                ],
                "response": f"strong preference: {first_name}",
                "target_tensor": [1.0, 0.0],
            }
        wrong_name = image_names[0]
        answers[f"{wrong_name}::wrong"] = {
            **answers[f"{wrong_name}::0"],
            "relation_id": 99,
            "target_tensor": [0.0, 0.0],
            "response": "both wrong",
            "presented_predicates": [
                {"predicate_id": predicate_ids[second_name], "name": second_name, "probability": 0.8},
                {"predicate_id": predicate_ids[first_name], "name": first_name, "probability": 0.2},
            ],
        }
        payload = {
            "schema_version": 5,
            "session_id": session.name,
            "provenance": {
                "checkpoint_sha256": "a" * 64,
                "config_sha256": "b" * 64,
                "label_map_sha256": "c" * 64,
            },
            "images": images,
            "answers": answers,
            "predictions": {},
            "prediction_index": {},
        }
        atomic_json(session / "session.json", payload)
        return session

    def test_pairwise_loss_and_zero_target(self):
        probabilities = torch.tensor([[0.8, 0.2], [0.4, 0.4]], requires_grad=True)
        targets = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        loss = pairwise_preference_loss(probabilities, targets, tau=10.0)
        loss.backward()
        self.assertTrue(math.isfinite(float(loss.detach())))
        self.assertLess(float(loss.detach()), 0.01)
        self.assertTrue(torch.equal(probabilities.grad[1], torch.zeros(2)))

    def test_pairwise_loss_matches_paper_bradley_terry_equations(self):
        rewards = torch.tensor([[0.23, 0.17], [0.41, 0.52]], dtype=torch.float64)
        targets = torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=torch.float64)
        tau = 10.0
        preference_probability = torch.exp(tau * rewards)
        preference_probability = preference_probability / preference_probability.sum(dim=-1, keepdim=True)
        expected = -(targets * preference_probability.log()).sum(dim=-1).mean()
        actual = pairwise_preference_loss(rewards, targets, tau=tau)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))

    def test_raw_logit_pairwise_loss_does_not_touch_unlabelled_output_logits(self):
        logits = torch.tensor([[2.0, -0.5, 1.2, 0.3]], requires_grad=True)
        pair_ids = torch.tensor([0, 2])
        pair_rewards = logits.index_select(1, pair_ids)
        loss = pairwise_preference_loss(
            pair_rewards,
            torch.tensor([[0.0, 1.0]]),
            tau=1.0,
        )
        loss.backward()
        self.assertEqual(float(logits.grad[0, 1]), 0.0)
        self.assertEqual(float(logits.grad[0, 3]), 0.0)
        self.assertGreater(float(logits.grad[0, 0].abs()), 0.0)
        self.assertGreater(float(logits.grad[0, 2].abs()), 0.0)

    def test_retention_ignores_pair_reordering_but_preserves_old_classes(self):
        teacher = torch.tensor([[0.10, 0.20, 0.30, 0.40]], dtype=torch.float64)
        pair_ids = torch.tensor([[0, 1]])
        reordered_pair = torch.tensor([[0.20, 0.10, 0.30, 0.40]], dtype=torch.float64)
        changed_old = torch.tensor(
            [[0.10, 0.20, 0.40, 0.30]], dtype=torch.float64, requires_grad=True
        )

        free_pair_loss = masked_retention_loss(reordered_pair, teacher, pair_ids)
        old_class_loss = masked_retention_loss(changed_old, teacher, pair_ids)
        self.assertAlmostEqual(float(free_pair_loss), 0.0, places=12)
        self.assertGreater(float(old_class_loss.detach()), 0.0)
        old_class_loss.backward()
        self.assertTrue(torch.isfinite(changed_old.grad).all())
        self.assertGreater(float(changed_old.grad[:, 2:].abs().sum()), 0.0)

    def test_preference_forward_keeps_prototype_projection_differentiable(self):
        from sgg_benchmark.modeling.roi_heads.relation_head.predictors.new_react_predictor import (
            REACTPlusPlusPredictor,
        )

        class PrototypeStub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.obj_embed = torch.nn.Embedding.from_pretrained(torch.eye(2), freeze=True)
                self.obj_lift = torch.nn.Identity()
                self.use_union = False
                self.proj_head = torch.nn.Linear(2, 2, bias=False)
                self.proto_weight = torch.nn.Parameter(torch.tensor([[1.0, 0.2], [0.1, 1.0]]))
                self._cache_valid = False
                self.preference_forward = False

            def _proto_bank(self):
                return self.proto_weight

        stub = PrototypeStub().eval()
        _, _, inference_protos, inference_projection = REACTPlusPlusPredictor._get_cached_protos(stub)
        self.assertFalse(inference_protos.requires_grad)
        self.assertIsNone(inference_projection)

        stub.preference_forward = True
        stub._cache_valid = False
        _, _, preference_protos, preference_projection = REACTPlusPlusPredictor._get_cached_protos(stub)
        self.assertTrue(preference_protos.requires_grad)
        self.assertIsNotNone(preference_projection)
        preference_protos[0, 0].backward()
        self.assertIsNotNone(stub.proj_head.weight.grad)
        self.assertGreater(float(stub.proj_head.weight.grad.abs().sum()), 0.0)

    def test_evaluator_summary_is_normalized_without_metric_collisions(self):
        metrics = flatten_evaluation_metrics({
            "sgdet_recall": {20: [0.1, 0.3], 50: [0.2, 0.4]},
            "sgdet_mean_recall": {50: 0.1},
            "sgdet_mean_recall_collect": {50: [[0.1], [0.2]]},
            "sgdet_mean_recall_list": {50: [0.1, 0.2]},
            "sgdet_zeroshot_recall": {50: 0.05},
            "sgdet_f1_score": {50: 0.15},
            "sgdet_bbox_mAP": 0.4,
        })
        self.assertAlmostEqual(metrics["R@50"], 0.3)
        self.assertEqual(metrics["mR@50"], 0.1)
        self.assertEqual(metrics["zR@50"], 0.05)
        self.assertEqual(metrics["F1@50"], 0.15)
        self.assertEqual(metrics["bbox_mAP"], 0.4)

    def test_audit_splits_by_image_and_keeps_both_wrong_for_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = self.make_session(root)
            test_dir = root / "test"
            test_dir.mkdir()
            report, samples = audit_sessions([session], DEFAULT_LABELS, test_dir, root / "audit")
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["audit_only_samples"], 1)
            self.assertEqual(report["train_samples"] + report["validation_samples"], 5)
            train_images = {item["image_checksum"] for item in samples if item["split"] == "train"}
            validation_images = {item["image_checksum"] for item in samples if item["split"] == "validation"}
            self.assertFalse(train_images & validation_images)
            self.assertEqual(report["model_matching_deferred"], 6)

    def test_audit_rejects_psg_test_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = self.make_session(root, ["leaked.jpg", "train_a.jpg"])
            test_dir = root / "test"
            test_dir.mkdir()
            (test_dir / "leaked.jpg").write_bytes(b"test")
            with self.assertRaisesRegex(RLHPError, "PSG test"):
                audit_sessions([session], DEFAULT_LABELS, test_dir, root / "audit")

    def test_standalone_preference_json_can_use_external_image_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = self.make_session(root)
            standalone = root / "val_preferences.json"
            standalone.write_text((session / "session.json").read_text())
            test_dir = root / "test"
            test_dir.mkdir()
            report, samples = audit_sessions(
                [standalone], DEFAULT_LABELS, test_dir, root / "audit",
                image_root=session / "input",
            )
            self.assertEqual(report["status"], "passed")
            self.assertTrue(all(Path(item["image_path"]).is_file() for item in samples))

    def test_csv_export_is_directly_auditable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "train.jpg"
            image.write_bytes(b"preference-image")
            provenance = {
                "checkpoint_sha256": "a" * 64,
                "config_sha256": "b" * 64,
                "label_map_sha256": "c" * 64,
            }
            session = BatchSession.create(image, root / "exports", provenance=provenance)
            image_meta = session.data["images"][0]
            labels = json.loads(DEFAULT_LABELS.read_text())["predicates"]
            session.record_answer(image_meta["id"], 1, {
                "image_id": image_meta["id"], "source_name": image_meta["source_name"],
                "image_checksum": image_meta["sha256"], "relation_id": 1,
                "response": "strong preference: on", "target_tensor": [1.0, 0.0],
                "subject": {"id": 0, "label_id": 1, "label": "person", "box_xyxy": [0, 0, 10, 10]},
                "object": {"id": 1, "label_id": 2, "label": "bicycle", "box_xyxy": [12, 0, 22, 10]},
                "presented_predicates": [
                    {"predicate_id": labels.index("on"), "name": "on", "probability": 0.4},
                    {"predicate_id": labels.index("beside"), "name": "beside", "probability": 0.39},
                ],
            })
            test_dir = root / "test"
            test_dir.mkdir()
            report, samples = audit_sessions(
                [session.export()], DEFAULT_LABELS, test_dir, root / "audit",
                image_root=session.input_dir,
            )
            self.assertEqual(report["status"], "passed")
            self.assertEqual(len(samples), 1)
            self.assertEqual(report["model_matching_deferred"], 0)
            self.assertEqual({item["name"] for item in samples[0]["predicates"]}, {"on", "beside"})

    def test_mock_end_to_end_gate_promote_and_rollback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = self.make_session(root)
            test_dir = root / "test"
            test_dir.mkdir()
            args = Namespace(
                sessions=[session], labels=DEFAULT_LABELS, psg_test=test_dir,
                allow_legacy_provenance=False, iou_threshold=0.8, min_match_rate=0.95, seed=42,
                image_root=None,
                runs_root=root / "runs", run_id="mock_run", mock=True,
                checkpoint=None, config=None, yolo_checkpoint=None, device="cpu",
                tau=10.0, learning_rate=0.1, weight_decay=0.01,
                gradient_accumulation=4, clip_norm=1.0, epochs=5, early_stopping_patience=2,
            )
            run_dir = train_pipeline(args)
            run = json.loads((run_dir / "run.json").read_text())
            self.assertEqual(run["status"], "trained")
            self.assertTrue((run_dir / "best_candidate.pth").is_file())
            metrics = json.loads((run_dir / "training_metrics.json").read_text())
            self.assertIn("baseline_train", metrics)
            plot_path = plot_run(run_dir, run_dir / "training_curves.png")
            self.assertGreater(plot_path.stat().st_size, 0)

            comparison = evaluate_pipeline(Namespace(run_dir=run_dir, mock=True))
            self.assertTrue(comparison["gate"]["passed"])

            baseline_checkpoint = root / "baseline.pth"
            baseline_config = root / "config.yml"
            baseline_checkpoint.write_bytes(b"baseline")
            baseline_config.write_text("seed: 42\n")
            manifest = root / "active_model.json"
            atomic_json(manifest, {
                "active": {"run_id": "baseline", "checkpoint": str(baseline_checkpoint),
                           "config": str(baseline_config)},
                "history": [],
            })
            with self.assertRaisesRegex(RLHPError, "Mock checkpoints"):
                promote_run(run_dir, manifest)
            promoted = promote_run(run_dir, manifest, allow_mock=True)
            self.assertEqual(promoted["active"]["run_id"], "mock_run")
            active_checkpoint, _ = resolve_active_model(manifest)
            self.assertEqual(active_checkpoint, run_dir / "best_candidate.pth")
            rolled_back = rollback_manifest(manifest)
            self.assertEqual(rolled_back["active"]["run_id"], "baseline")


if __name__ == "__main__":
    unittest.main()
