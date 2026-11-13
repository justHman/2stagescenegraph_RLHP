"""Create an RLHP-aligned ONNX model by patching the projection head weights.

RLHP fine-tuning only changes 4 tensors (proj_head.0.weight, proj_head.1.scale,
proj_head.3.weight, logit_scale — the last by <3e-4). We overwrite the matching
ONNX initializers with the tuned values, producing react_pp_yolo12m_rlhp.onnx.
"""
import torch
import onnx
from onnx import numpy_helper
from pathlib import Path

CKPT = Path(r"D:\Studying\University\Ki_8\IMP\rlhp_analysis\live_demo\checkpoint")
BASE_PTH = CKPT / "best_model.pth"
TUNED_PTH = Path(r"D:\Studying\University\Ki_8\IMP\2stagescenegraph_RLHP\scene_graph_tool"
                 r"\rlhp_runs\extracted\preference3_tuned_anchor_ret20_tau01\best_candidate.pth")
ONNX_IN = CKPT / "react_pp_yolo12m.onnx"
ONNX_OUT = CKPT / "react_pp_yolo12m_rlhp.onnx"

PATCH = {
    # onnx initializer name -> (state-dict key, transpose?)
    "onnx::MatMul_5097": ("roi_heads.relation.predictor.proj_head.0.weight", True),
    "onnx::Mul_5098": ("roi_heads.relation.predictor.proj_head.1.scale", False),
    "onnx::MatMul_5099": ("roi_heads.relation.predictor.proj_head.3.weight", True),
}


def main():
    print("loading pth ...")
    base = torch.load(BASE_PTH, map_location="cpu", weights_only=False)["model"]
    tuned = torch.load(TUNED_PTH, map_location="cpu", weights_only=False)["model"]

    print("loading onnx ...")
    m = onnx.load(ONNX_IN, load_external_data=False)
    inits = {i.name: i for i in m.graph.initializer}

    # sanity: baseline values must match the current ONNX (up to transpose)
    for onnx_name, (key, do_T) in PATCH.items():
        ref = base[key].numpy()
        if do_T:
            ref = ref.T
        cur = numpy_helper.to_array(inits[onnx_name])
        frac = float((abs(cur - ref) < 1e-4).mean())
        print(f"verify {onnx_name} vs baseline {key}: match={frac:.4f}")
        if frac < 0.99:
            raise SystemExit(f"baseline mismatch for {onnx_name} — aborting, wrong tensor")

    # patch with tuned values
    for onnx_name, (key, do_T) in PATCH.items():
        val = tuned[key].numpy()
        if do_T:
            val = val.T
        old = numpy_helper.to_array(inits[onnx_name])
        assert val.shape == old.shape, (onnx_name, val.shape, old.shape)
        delta = float(np.abs(val - old).max())
        print(f"patch {onnx_name}: max |tuned-onnx| = {delta:.5f}")
        new = numpy_helper.from_array(val.astype(old.dtype), name=onnx_name)
        inits[onnx_name].CopyFrom(new)

    onnx.save(m, str(ONNX_OUT), save_as_external_data=True,
              all_tensors_to_one_file=True, location="react_pp_yolo12m_rlhp.onnx.data")
    print("saved", ONNX_OUT)


import numpy as np  # noqa: E402

if __name__ == "__main__":
    main()
