# Pretrained Checkpoints & Experiment Runs (Google Drive)

All trained model weights, evaluation metrics, and training diagnostics (~5.4 GB) are hosted on Google Drive:

🔗 **[Download Pretrained Checkpoints & RLHP Runs (Google Drive)](https://drive.google.com/drive/folders/1CRyhpJNsR1Ht4Kay5SEpRqtrl-uTk-Bw?usp=sharing)**

---

## Folder Contents & Model Breakdown

The Google Drive directory contains the complete multi-epoch checkpoints and evaluation summaries for all models evaluated in the research report:

| Model Directory | Description | SGDet mR@50 | SGDet mR@100 | SGDet F1@50 | Bbox mAP |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`preference3_tuned_anchor_ret20_tau01/`** | **Full RLHP (Tuned - Main Model)**<br/>$\mathcal{L}_{\text{pref}} + 20\mathcal{L}_{\text{KL}} + 20\mathcal{L}_{\text{proto}} \ (\tau=0.1)$ | **26.95%** | **28.69%** | **31.58%** | **56.57%** |
| `preference3_pure_pref_setting_tau01/` | **Ablation 1**: Pure Bradley-Terry ($\mathcal{L}_{\text{pref}}, \tau=0.1$) | 21.21% | 22.57% | 26.64% | 56.57% |
| `preference3_proto_anchor_only_tau01/` | **Ablation 2**: Preference + Proto Anchor ($\mathcal{L}_{\text{pref}} + 20\mathcal{L}_{\text{proto}}$) | 24.51% | 26.18% | 29.58% | 56.57% |
| `preference3_retention_only_tau01/` | **Ablation 3**: Preference + KL Retention ($\mathcal{L}_{\text{pref}} + 20\mathcal{L}_{\text{KL}}$) | 26.21% | 27.88% | 30.87% | 56.57% |
| `preference3_audit_v6_loader/` | Audited preference splits & validation provenance | — | — | — | — |
| `pure_pref_run1/` | Preliminary pure preference diagnostic baseline | — | — | — | — |

---

## Directory Structure After Download

Place or extract the downloaded folders into `scene_graph_tool/rlhp_runs/`:

```
scene_graph_tool/
├── rlhp_runs/
│   ├── preference3_tuned_anchor_ret20_tau01/    # Main Full RLHP model (Epochs 1-5)
│   │   ├── epoch_1/ ... epoch_5/
│   │   │   ├── best_model.pth                   # Relation projection checkpoint
│   │   │   ├── training_curves.png              # Epoch diagnostics
│   │   │   └── eval_summary.json                # Evaluated PSG test metrics
│   │   └── run_config.json
│   │
│   ├── preference3_pure_pref_setting_tau01/     # Ablation 1 (Pure Preference)
│   ├── preference3_proto_anchor_only_tau01/     # Ablation 2 (+Proto Anchor)
│   ├── preference3_retention_only_tau01/        # Ablation 3 (+KL Retention)
│   └── preference3_audit_v6_loader/             # Audit splits
```

---

## Automated Download

You can download all folders directly into `scene_graph_tool/rlhp_runs/` using the provided script:

```bash
chmod +x scene_graph_tool/checkpoints/download_runs.sh
./scene_graph_tool/checkpoints/download_runs.sh
```

---

## Evaluation & Inference with Pretrained Checkpoints

### 1. Launching GUI with Aligned Model
```bash
python scene_graph_tool/server.py \
  --checkpoint scene_graph_tool/rlhp_runs/preference3_tuned_anchor_ret20_tau01/epoch_5/best_model.pth \
  --config checkpoint/config.yml \
  --yolo-checkpoint checkpoint/PSG_yolo12m_backbone.pt
```

### 2. Evaluating on PSG Test Benchmark
```bash
python scene_graph_tool/evaluate_rlhp_epochs.py \
  --run-dir scene_graph_tool/rlhp_runs/preference3_tuned_anchor_ret20_tau01 \
  --config checkpoint/config.yml \
  --yolo-checkpoint checkpoint/PSG_yolo12m_backbone.pt
```
