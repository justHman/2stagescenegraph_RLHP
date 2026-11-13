"""Inspect checkpoint structure: full model or projection-only?"""
import torch
from pathlib import Path

FILES = [
    Path(r"D:\Studying\University\Ki_8\IMP\rlhp_analysis\live_demo\checkpoint\best_model.pth"),
    Path(r"D:\Studying\University\Ki_8\IMP\2stagescenegraph_RLHP\scene_graph_tool\rlhp_runs\extracted\preference3_tuned_anchor_ret20_tau01\best_candidate.pth"),
]

for f in FILES:
    print("===", f.name, f.parent.name)
    d = torch.load(f, map_location="cpu", weights_only=False)
    if isinstance(d, dict):
        print(" top keys:", list(d.keys())[:15])
        sd = d.get("model") or d.get("state_dict") or d
        if hasattr(sd, "keys"):
            print(" n tensors:", len(sd))
            for k in list(sd.keys())[:10]:
                v = sd[k]
                print("   ", k, tuple(v.shape) if hasattr(v, "shape") else type(v))
    elif hasattr(d, "state_dict"):
        print(" nn.Module:", type(d).__name__)
    else:
        print(" type:", type(d))
    print()
