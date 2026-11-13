"""Expose extra graph tensors (rel distribution, pair map, box remap) from the
baseline and RLHP ONNX models — needed for the uncertainty-based annotation
selector (relative margin between top-2 predicates per pair).

Outputs added:
  rel_probs       (num_pairs, 57)  predicate softmax over ALL candidate pairs
  pair_map        (num_pairs, 2)   raw box indices per pair row
  raw_rels_inds   (num_rels, 2)    raw box indices of the rels output rows
  box_remap       (num_raw,)       raw box index -> filtered box index (or -1/dropped)
  raw_det         (num_raw, ...)   raw detections used for pair enumeration
"""
import onnx
from pathlib import Path

CKPT = Path(r"D:\Studying\University\Ki_8\IMP\rlhp_analysis\live_demo\checkpoint")
EXTRA = [
    ("/roi_heads/relation/post_processor/Softmax_output_0", "rel_probs"),
    ("/roi_heads/relation/Reshape_output_0", "pair_map"),
    ("/GatherND_7_output_0", "raw_rels_inds"),
    ("/Sub_3_output_0", "box_remap"),
    ("/GatherND_2_output_0", "raw_det"),
]
NEW_OUTPUTS = tuple(n for _, n in EXTRA)

for name in ("react_pp_yolo12m.onnx", "react_pp_yolo12m_rlhp.onnx"):
    p = CKPT / name
    m = onnx.load(str(p), load_external_data=False)
    g = m.graph
    # clean any previous attempt
    if any(o.name in NEW_OUTPUTS for o in g.output):
        keep = [o for o in g.output if o.name not in NEW_OUTPUTS]
        del g.output[:]
        g.output.extend(keep)
        keep_nodes = [n for n in g.node if n.output and n.output[0] not in NEW_OUTPUTS]
        del g.node[:]
        g.node.extend(keep_nodes)

    for tensor, out_name in EXTRA:
        found = any(n.output and tensor in n.output for n in g.node)
        assert found, f"{tensor} not found in {name}"
        g.node.append(onnx.helper.make_node(
            "Identity", inputs=[tensor], outputs=[out_name], name=f"{out_name}_output"))

    for out_name, dtype, dims in [
        ("rel_probs", onnx.TensorProto.FLOAT, [("num_pairs", None), ("classes", 57)]),
        ("pair_map", onnx.TensorProto.INT64, [("num_pairs", None), ("two", 2)]),
        ("raw_rels_inds", onnx.TensorProto.INT64, [("num_rels", None), ("two", 2)]),
        ("box_remap", onnx.TensorProto.INT64, [("num_raw", None)]),
        ("raw_det", onnx.TensorProto.FLOAT, [("num_raw", None), ("det_fields", None)]),
    ]:
        out = g.output.add()
        out.name = out_name
        out.type.tensor_type.elem_type = dtype
        for dim_param, dim_value in dims:
            d = out.type.tensor_type.shape.dim.add()
            if dim_value is not None:
                d.dim_value = dim_value
            else:
                d.dim_param = dim_param or f"{out_name}_d{len(dims)}"

    onnx.save(m, str(p))
    print(name, "-> outputs added:", ", ".join(NEW_OUTPUTS))
