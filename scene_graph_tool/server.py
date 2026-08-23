#!/usr/bin/env python3
"""FastAPI backend for the React SceneGraph Labeler."""

from __future__ import annotations

import argparse
import base64
import random
import shutil
import sys
import tempfile
import time
from urllib.parse import quote
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from batch_session import BatchSession, DEFAULT_SETTINGS
from app import (
    PROJECT_ROOT, Predictor, discover_checkpoint, discover_yolo, draw_scene_graph,
    find_config, top_two_predicates,
)


OVERLAY_RENDER_VERSION = 2


def _relation_metrics(result: dict, relation: dict) -> dict | None:
    predicates = top_two_predicates(relation)
    if len(predicates) != 2:
        return None
    (_, p1), (_, p2) = predicates
    subject_score = result["objects"][relation["subject_id"]]["score"]
    object_score = result["objects"][relation["object_id"]]["score"]
    relative_margin = (p1 - p2) / max(p1 + p2, 1e-12)
    return {
        **relation,
        "relation_score": p1,
        "triplet_score": subject_score * object_score * p1,
        "relative_margin": relative_margin,
        "uncertainty_score": 1.0 - relative_margin,
    }


def highest_probability_relations(result: dict, top_k: int) -> list[dict]:
    """Baseline: the K relations with the largest top foreground probability."""
    candidates = [
        metrics
        for relation in result["relations"]
        if (metrics := _relation_metrics(result, relation)) is not None
    ]
    return sorted(candidates, key=lambda item: (-item["relation_score"], item["id"]))[:top_k]


def top_k_relations(result: dict, settings: dict | int) -> list[dict]:
    """Shortlist Top-K pairs by p1, then select Top-M by relative margin."""
    if isinstance(settings, int):
        settings = {**DEFAULT_SETTINGS, "top_k_relations": settings}

    objects = result["objects"]
    reliable_object_ids = [
        item["id"]
        for item in sorted(objects, key=lambda value: (-value["score"], value["id"]))
        if item["score"] >= settings["object_min_probability"]
    ][:settings["object_max_count"]]
    reliable_object_ids = set(reliable_object_ids)

    candidates = []
    for relation in result["relations"]:
        if relation["subject_id"] not in reliable_object_ids or relation["object_id"] not in reliable_object_ids:
            continue
        metrics = _relation_metrics(result, relation)
        if metrics is None:
            continue
        candidates.append(metrics)

    shortlist = sorted(
        candidates,
        key=lambda item: (-item["relation_score"], item["id"]),
    )[:settings["relation_candidate_top_k"]]
    return sorted(
        shortlist,
        key=lambda item: (item["relative_margin"], -item["relation_score"], item["id"]),
    )[:settings["top_k_relations"]]


def relation_selection_comparison(result: dict, settings: dict) -> list[dict]:
    """Return comparable baseline/new rows for a single image prediction."""
    top_k = settings["top_k_relations"]
    methods = {
        "highest_probability": highest_probability_relations(result, top_k),
        "uncertainty_priority": top_k_relations(result, settings),
    }
    selected_ids = {name: {item["id"] for item in relations} for name, relations in methods.items()}
    rows = []
    for method, relations in methods.items():
        other = "uncertainty_priority" if method == "highest_probability" else "highest_probability"
        for rank, relation in enumerate(relations, start=1):
            predicates = top_two_predicates(relation)
            rows.append({
                "method": method,
                "rank": rank,
                "relation_id": relation["id"],
                "subject": result["objects"][relation["subject_id"]]["label"],
                "object": result["objects"][relation["object_id"]]["label"],
                "predicate_1": predicates[0][0], "p1": predicates[0][1],
                "predicate_2": predicates[1][0], "p2": predicates[1][1],
                "relation_score": relation["relation_score"],
                "triplet_score": relation["triplet_score"],
                "relative_margin": relation["relative_margin"],
                "uncertainty_score": relation["uncertainty_score"],
                "selected_by_both": relation["id"] in selected_ids[other],
            })
    return rows


def read_rgb(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path.name}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def image_data_url(image) -> str:
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("Could not encode annotation overlay.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


class ResponsePayload(BaseModel):
    image_id: str
    relation_id: int
    presented_predicates: list[dict]
    choice: str


class SettingsPayload(BaseModel):
    top_k_relations: int = Field(ge=1, le=10)
    relation_candidate_top_k: int = Field(default=15, ge=1, le=1000)
    object_max_count: int = Field(ge=2, le=100)
    object_min_probability: float = Field(ge=0, le=1)


class LabelerService:
    def __init__(self, predictor, exports_root: Path):
        self.predictor = predictor
        self.exports_root = Path(exports_root)

    def _predicate_id(self, name: str):
        names = getattr(self.predictor, "predicate_names", None)
        if names is None:
            return None
        try:
            return names.index(name)
        except ValueError:
            return None

    def _task(self, session: BatchSession):
        for index in range(session.data["current_image_index"], len(session.data["images"])):
            session.data["current_image_index"] = index
            image_id = session.data["images"][index]["id"]
            if not session.has_prediction(image_id):
                session.set_prediction(image_id, self.predictor.predict(read_rgb(session.image_path(image_id))))
            result = session.get_prediction(image_id)
            eligible = top_k_relations(result, session.data["settings"])
            for relation in eligible:
                if session.answer_key(image_id, relation["id"]) not in session.data["answers"]:
                    session.data["current_relation_id"] = relation["id"]
                    session.save()
                    return image_id, result, relation, eligible
            session.data["images"][index]["status"] = "done" if eligible else "no_eligible_relations"
            session.save()
        session.data["current_relation_id"] = None
        session.save()
        return None

    @staticmethod
    def _overlay_path(session: BatchSession, image_id: str, relation_id: int) -> Path:
        """Use a filesystem-safe, persistent location for a relation overlay."""
        safe_image_id = quote(image_id, safe="").replace("%", "_")
        return session.directory / "overlays" / f"{safe_image_id}_{relation_id}_v{OVERLAY_RENDER_VERSION}.jpg"

    def overlay(self, session_id: str, image_id: str, relation_id: int) -> Path:
        """Create an overlay once, then serve the cached JPEG for future visits."""
        session = BatchSession(self.exports_root / session_id)
        result = session.get_prediction(image_id)
        if result is None:
            raise HTTPException(404, "Image prediction is missing.")
        relation = next((item for item in result["relations"] if item["id"] == relation_id), None)
        if relation is None:
            raise HTTPException(404, "Relation does not exist.")
        destination = self._overlay_path(session, image_id, relation_id)
        if not destination.is_file():
            destination.parent.mkdir(exist_ok=True)
            subject = result["objects"][relation["subject_id"]]
            obj = result["objects"][relation["object_id"]]
            rendered = draw_scene_graph(read_rgb(session.image_path(image_id)), [subject, obj], [relation],
                                        show_probability=False)
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise ValueError("Could not encode annotation overlay.")
            destination.write_bytes(encoded.tobytes())
        return destination

    def task_payload(self, session_id: str):
        session = BatchSession(self.exports_root / session_id)
        task = self._task(session)
        completed = len(session.data["answers"])
        navigator = {
            "image_ids": [item["id"] for item in session.data["images"]],
            "active_image_id": session.current_image()["id"] if session.current_image() else None,
        }
        if task is None:
            return {"completed": True, "progress": {"images": len(session.data["images"]), "labels": completed},
                    "settings": session.data["settings"],
                    "can_undo": bool(session.data["history"]), "navigator": navigator}
        image_id, result, relation, eligible = task
        subject = result["objects"][relation["subject_id"]]
        obj = result["objects"][relation["object_id"]]
        choices = [
            {"name": name, "predicate_id": self._predicate_id(name), "probability": probability}
            for name, probability in top_two_predicates(relation)
        ]
        random.shuffle(choices)
        qa = [{"id": item["id"], "subject": result["objects"][item["subject_id"]]["label"],
               "predicate": item["best_predicate"], "object": result["objects"][item["object_id"]]["label"],
               "confidence": item["best_probability"], "relation_score": item["relation_score"],
               "triplet_score": item["triplet_score"],
               "uncertainty_score": item["uncertainty_score"]} for item in eligible]
        return {
            "completed": False, "image": image_id,
            "overlay": (f"/api/sessions/{session_id}/overlay?image_id={quote(image_id, safe='')}"
                        f"&relation_id={relation['id']}&render={OVERLAY_RENDER_VERSION}"),
            "relation": {
                "id": relation["id"], "subject": subject, "object": obj, "choices": choices,
                "relation_score": relation["relation_score"],
                "triplet_score": relation["triplet_score"],
                "relative_margin": relation["relative_margin"],
                "uncertainty_score": relation["uncertainty_score"],
            },
            "progress": {"image": session.data["current_image_index"] + 1, "images": len(session.data["images"]),
                         "relation": eligible.index(relation) + 1, "relations": len(eligible), "labels": completed},
            "qa": qa, "comparison": relation_selection_comparison(result, session.data["settings"]),
            "settings": session.data["settings"], "can_undo": bool(session.data["history"]),
            "navigator": navigator,
        }

    def preview(self, session_id: str, image_id: str):
        """Return an image preview without changing annotation progress."""
        session = BatchSession(self.exports_root / session_id)
        index = next((i for i, item in enumerate(session.data["images"]) if item["id"] == image_id), None)
        if index is None:
            raise HTTPException(404, "Image does not exist in this session.")
        if not session.has_prediction(image_id):
            session.set_prediction(image_id, self.predictor.predict(read_rgb(session.image_path(image_id))))
        result = session.get_prediction(image_id)
        eligible = top_k_relations(result, session.data["settings"])
        relation = eligible[0] if eligible else None
        objects = []
        relations = []
        if relation:
            objects = [result["objects"][relation["subject_id"]], result["objects"][relation["object_id"]]]
            relations = [relation]
        return {
            "image": image_id, "index": index + 1, "images": len(session.data["images"]),
            "overlay": image_data_url(draw_scene_graph(read_rgb(session.image_path(image_id)), objects, relations,
                                                         show_probability=False)),
            "relation": None if relation is None else {
                "subject": objects[0], "object": objects[1],
                "predicate": relation["best_predicate"], "probability": relation["best_probability"],
            },
        }

    def respond(self, session_id: str, payload: ResponsePayload):
        session = BatchSession(self.exports_root / session_id)
        choices = {"first": [1.0, 0.0], "second": [0.0, 1.0], "neutral": [0.5, 0.5], "wrong": [0.0, 0.0], "skip": None}
        if payload.choice not in choices or len(payload.presented_predicates) != 2:
            raise HTTPException(422, "Invalid label response.")
        labels = {"first": f"strong preference: {payload.presented_predicates[0]['name']}",
                  "second": f"strong preference: {payload.presented_predicates[1]['name']}",
                  "neutral": "no preference", "wrong": "both wrong", "skip": "skipped"}
        result = session.get_prediction(payload.image_id)
        if result is None:
            raise HTTPException(409, "Image prediction is missing.")
        relation = next((item for item in result["relations"] if item["id"] == payload.relation_id), None)
        if relation is None:
            raise HTTPException(404, "Relation does not exist.")
        selected = next(
            (item for item in top_k_relations(result, session.data["settings"])
             if item["id"] == payload.relation_id),
            None,
        )
        if selected is None:
            raise HTTPException(409, "Relation is no longer eligible under the current settings.")
        image_metadata = next(
            (item for item in session.data["images"] if item["id"] == payload.image_id),
            {},
        )
        session.record_answer(payload.image_id, payload.relation_id, {
            "session_id": session.data.get("session_id", session.directory.name),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "image_id": payload.image_id, "relation_id": payload.relation_id,
            "source_name": image_metadata.get("source_name", payload.image_id),
            "image_checksum": image_metadata.get("sha256"),
            "subject": result["objects"][relation["subject_id"]], "object": result["objects"][relation["object_id"]],
            "presented_predicates": payload.presented_predicates, "response": labels[payload.choice],
            "target_tensor": choices[payload.choice],
            "prediction_reference": {
                "schema_version": 1,
                "relation_index": int(relation["id"]),
                "subject_index": int(relation["subject_id"]),
                "object_index": int(relation["object_id"]),
                "object_count": len(result.get("objects", [])),
                "relation_count": len(result.get("relations", [])),
                "background_probability": relation.get("background_probability"),
                "predicate_probabilities": relation.get("predicate_probabilities", {}),
            },
            "selection": {
                "model_best_predicate": selected["best_predicate"],
                "model_best_probability": selected["best_probability"],
                "background_probability": selected.get("background_probability"),
                "relation_score": selected["relation_score"],
                "triplet_score": selected["triplet_score"],
                "relative_margin": selected["relative_margin"],
                "uncertainty_score": selected["uncertainty_score"],
            },
            "settings": dict(session.data["settings"]),
        })
        return self.task_payload(session_id)

    def undo(self, session_id: str):
        session = BatchSession(self.exports_root / session_id)
        if session.undo() and session.last_undone:
            image_id = session.last_undone["key"].split("::", 1)[0]
            session.data["current_image_index"] = next(i for i, item in enumerate(session.data["images"]) if item["id"] == image_id)
            session.save()
        return self.task_payload(session_id)


def create_app(predictor, exports_root: Path) -> FastAPI:
    service = LabelerService(predictor, exports_root)
    api = FastAPI(title="SceneGraph Labeler API")

    @api.post("/api/sessions")
    async def create_session(
        files: list[UploadFile] = File(...),
        top_k_relations: int = 3,
        relation_candidate_top_k: int = 15,
    ):
        if not 1 <= top_k_relations <= 10:
            raise HTTPException(422, "M must be between 1 and 10.")
        if not top_k_relations <= relation_candidate_top_k <= 1000:
            raise HTTPException(422, "K must be between M and 1000.")
        temporary = Path(tempfile.mkdtemp(prefix="sgg_upload_"))
        try:
            paths = []
            for index, upload in enumerate(files):
                suffix = Path(upload.filename or "upload").suffix.lower()
                destination = temporary / f"{index:04d}_{Path(upload.filename or 'upload').name}"
                with destination.open("wb") as output:
                    shutil.copyfileobj(upload.file, output)
                paths.append(destination)
            session = BatchSession.create(
                paths,
                exports_root,
                top_k_relations,
                relation_candidate_top_k,
                provenance=getattr(predictor, "provenance", None),
            )
            return {"id": session.directory.name, "images": len(session.data["images"]), "task": service.task_payload(session.directory.name)}
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    @api.get("/api/sessions/{session_id}/task")
    def get_task(session_id: str):
        return service.task_payload(session_id)

    @api.post("/api/sessions/{session_id}/responses")
    def save_response(session_id: str, payload: ResponsePayload):
        return service.respond(session_id, payload)

    @api.post("/api/sessions/{session_id}/undo")
    def undo(session_id: str):
        return service.undo(session_id)

    @api.get("/api/sessions/{session_id}/preview")
    def preview(session_id: str, image_id: str):
        return service.preview(session_id, image_id)

    @api.get("/api/sessions/{session_id}/overlay")
    def overlay(session_id: str, image_id: str, relation_id: int):
        return FileResponse(service.overlay(session_id, image_id, relation_id), media_type="image/jpeg",
                            headers={"Cache-Control": "private, max-age=31536000, immutable"})

    @api.patch("/api/sessions/{session_id}/settings")
    def update_settings(session_id: str, payload: SettingsPayload):
        session = BatchSession(exports_root / session_id)
        try:
            session.set_settings(payload.model_dump())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return service.task_payload(session_id)

    @api.get("/api/sessions/{session_id}/export")
    def export(session_id: str):
        return FileResponse(
            BatchSession(exports_root / session_id).export(),
            filename="scene_graph_preferences.csv",
            media_type="text/csv",
        )

    frontend_dist = Path(__file__).parent / "frontend" / "dist"
    if frontend_dist.is_dir():
        api.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")
        @api.get("/")
        def frontend():
            return FileResponse(frontend_dist / "index.html")
    return api


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config")
    parser.add_argument("--yolo-root", type=Path)
    parser.add_argument("--yolo-checkpoint", type=Path)
    parser.add_argument("--psg-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "scene_graph_tool" / "exports")
    parser.add_argument(
        "--active-manifest",
        type=Path,
        default=PROJECT_ROOT / "scene_graph_tool" / "rlhp_runs" / "active_model.json",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    active_config = None
    checkpoint = args.checkpoint
    if checkpoint is None:
        from rlhp import resolve_active_model
        active = resolve_active_model(args.active_manifest)
        if active is not None:
            checkpoint, active_config = active
    checkpoint = checkpoint or discover_checkpoint(args.checkpoint_root)
    config = Path(args.config) if args.config else (active_config or find_config(checkpoint, None))
    predictor = Predictor(
        checkpoint,
        config,
        args.yolo_checkpoint or discover_yolo(args.yolo_root),
        args.psg_dir,
        args.device,
    )
    import uvicorn
    uvicorn.run(create_app(predictor, args.output_dir), host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
