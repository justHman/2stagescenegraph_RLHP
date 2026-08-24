# Scene Graph Preference Alignment & Active Learning Toolkit

<p align="center">
  <img src="report/figures/GUI_image.png" alt="GUI Annotation Interface" width="700"/>
</p>

An end-to-end human preference alignment and active uncertainty-guided annotation framework for **Two-Stage Scene Graph Generation (SGG)** on the **Panoptic Scene Graph (PSG)** benchmark.

This toolkit enables:
- 🖥️ **Active Uncertainty-Guided Annotation GUI**: Interactive React + Vite frontend backed by a high-performance FastAPI server running two-stage YOLO12m-REACT++ inference to query human preferences on the most ambiguous predicate candidates.
- 🎯 **Reinforcement Learning from Human Preferences (RLHP)**: Fine-tunes relation projection layers using a tripartite objective (Bradley-Terry preference loss + masked KL retention + prototype anchoring) to eliminate catastrophic semantic drift.
- 📊 **Comprehensive Analysis & Benchmark Evaluation**: Complete suite of tools to evaluate SGDet / PredCLS on PSG test, generate publication-grade training dynamics, per-predicate recall shifts, and qualitative scene graph grids.
- 📄 **Academic Publication Report**: A full 9-page research report with complete LaTeX sources and precompiled PDF under [`report/`](report/).

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Directory Structure](#directory-structure)
- [Pretrained Checkpoints & Runs (Google Drive)](#pretrained-checkpoints--runs-google-drive)
- [Mathematical Formulation (RLHP)](#mathematical-formulation-rlhp)
- [Installation](#installation)
- [Quick Start: Annotation GUI](#quick-start-annotation-gui)
- [RLHP Training & Ablation Pipeline](#rlhp-training--ablation-pipeline)
- [PSG Benchmark Evaluation](#psg-benchmark-evaluation)
- [Analysis & Visualization Suite](#analysis--visualization-suite)
- [Academic Report](#academic-report)
- [Citation & Contributor Attribution](#citation--contributor-attribution)

---

## Pretrained Checkpoints & Runs (Google Drive)

All trained model checkpoints, epoch diagnostics, and ablation logs (~5.4 GB) are hosted on Google Drive:

🔗 **[Google Drive Checkpoints & RLHP Runs Folder](https://drive.google.com/drive/folders/1CRyhpJNsR1Ht4Kay5SEpRqtrl-uTk-Bw?usp=sharing)**

To download them automatically into `scene_graph_tool/rlhp_runs/`:
```bash
chmod +x scene_graph_tool/checkpoints/download_runs.sh
./scene_graph_tool/checkpoints/download_runs.sh
```
Detailed model breakdowns and evaluation guides are available under [`checkpoints/README.md`](checkpoints/README.md).

---

## System Architecture

```
                                  ┌────────────────────────┐
                                  │  Input Scene Image(s)  │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                 Stage 1: Frozen YOLO12m
                                 (Box Localization & mAP)
                                              │
                                              ▼
                             Stage 2: REACT++ Relation Encoder
                                              │
                                              ▼
                        ┌──────────────────────────────────────────┐
                        │   Active Uncertainty-Selection Module    │
                        │ (Relative Margin: (p1 - p2) / (p1 + p2)) │
                        └─────────────────────┬────────────────────┘
                                              │
                                              ▼
                               ┌────────────────────────────┐
                               │  Interactive Labeler GUI   │
                               │   (FastAPI + React/Vite)   │
                               └──────────────┬─────────────┘
                                              │ Human Feedback
                                              ▼
                         ┌────────────────────────────────────────┐
                         │       Audited Preference Dataset       │
                         └────────────────────┬───────────────────┘
                                              │
                                              ▼
                         ┌────────────────────────────────────────┐
                         │        Tripartite RLHP Trainer         │
                         │  - Bradley-Terry Preference Loss       │
                         │  - Masked KL Retention (55 classes)    │
                         │  - Prototype Cosine Anchor (57 classes)│
                         └────────────────────┬───────────────────┘
                                              │
                                              ▼
                         ┌────────────────────────────────────────┐
                         │   Aligned YOLO12m-REACT++ Checkpoint   │
                         │  (mR@50: +0.57%, F1@50: +0.36%,        │
                         │   Bbox mAP strictly invariant: 56.57%) │
                         └────────────────────────────────────────┘
```

---

## Directory Structure

```
scene_graph_tool/
├── README.md                          # Toolkit guide & quickstart
├── requirements.txt                   # Python dependencies for API & tools
├── psg_labels.json                    # PSG object & predicate taxonomy dictionary
├── scene_graph_preferences3.csv       # Curated & audited human preference dataset (510 records)
├── relation_selection_comparison.csv  # Uncertainty vs. probability selection benchmark data
│
├── server.py                          # FastAPI backend serving inference & React static UI
├── batch_session.py                   # Batch inference engine & relative-margin selector
├── app.py                             # Standalone / fallback annotation GUI
├── frontend/                          # React 18 + Vite + TailwindCSS interactive UI
│   ├── package.json
│   ├── src/                           # UI components, canvas overlays, shortcuts
│   └── vite.config.ts
│
├── rlhp.py                            # CLI Dispatcher (audit, train, evaluate, promote, rollback)
├── train_rlhp_custom.py               # Full RLHP engine (BT loss + Masked KL + Proto Anchor)
├── train_rlhp_pure_preference.py      # Baseline ablation trainer (unregularized preference)
├── evaluate_rlhp_epochs.py            # Checkpoint evaluator on PSG test benchmark (SGDet/PredCLS)
├── run_ablation_tau01.sh              # 4-way loss ablation reproduction runner
│
├── plot_rlhp_training.py              # Figure 3: 4-panel optimization dynamics generator
├── plot_pure_pref_retention.py        # Figure 5: Catastrophic semantic drift generator
├── visualize_preference_dataset.py    # Figure 2: Class distribution & uncertainty pairs
├── visualize_epoch_metrics.py         # Figure 4: PSG test mR@K trajectories
├── visualize_predicate_improvements.py# Figure 6: 56 categories recall shifts
├── visualize_qualitative_results.py   # Figure 7: Qualitative comparison grid generator
├── compare_relation_selection.py      # Table 1: Uncertainty vs Top-Prob selection evaluator
├── compare_psg_checkpoints.py         # Multi-checkpoint comparison tool
├── extract_parquet_images.py          # HuggingFace Parquet dataset image extractor
├── make_label.py                      # Category dictionary mapping helper
│
├── test_batch_session.py              # Unit tests for batch session inference
├── test_server_service.py             # Integration tests for FastAPI endpoints
├── test_rlhp.py                       # Unit tests for RLHP mathematical objectives & audits
├── test_extract_parquet_images.py     # Unit tests for Parquet extraction
│
├── doc/
│   ├── REACTPP_SHAPE_FLOW.md          # Tensor shape flow and projection architecture
│   └── plan_rlhp.md                   # RLHP mathematical formulation & gating spec
│
├── checkpoints/                       # Pretrained checkpoints & Google Drive links
│   ├── README.md                      # Download links & model breakdown
│   ├── download_runs.sh               # Automated Google Drive downloader
│   └── package_runs.sh                # Archive packager
│
└── report/                            # Academic Research Report
    ├── main.tex                       # LaTeX source (9-page publication layout)
    ├── main.pdf                       # Compiled publication report PDF
    ├── references.bib                 # BibTeX citations
    └── figures/                       # High-resolution publication figures (Fig 1–7)
```

---

## Mathematical Formulation (RLHP)

### 1. Active Uncertainty Querying
To maximize label efficiency under human budget constraints, the toolkit identifies the most informative relation queries using the **relative margin** between the top two candidate predicates ($p_1 \ge p_2$):

$$
\Delta_{\text{rel}} = \frac{p_1 - p_2}{p_1 + p_2}, \qquad U = 1 - \Delta_{\text{rel}} = \frac{2p_2}{p_1 + p_2}
$$

The selector first keeps a confident candidate shortlist ($\text{Top-}K$ by $p_1$), then ranks and presents candidate queries with the smallest relative margin $\Delta_{\text{rel}}$ (highest uncertainty $U$).

### 2. Bradley-Terry Preference Objective
For an annotated sample $m = (I_m, s, o, p_a, p_b) \in \mathcal{D}$, let $r_a, r_b \in \mathbb{R}$ denote the pre-softmax relation logits for candidate predicates $p_a$ and $p_b$ on the directed object pair $(s, o)$. Under the Bradley-Terry preference model with temperature scaling $\tau = 0.1$, the predicted choice probability $\pi_{m,k}$ for candidate $k \in \{a, b\}$ is:

$$
\pi_{m,k} = \frac{\exp(\tau r_{m,k})}{\exp(\tau r_{m,a}) + \exp(\tau r_{m,b})}
$$

Given the human target distribution $\mathbf{y}_m = [y_{m,a}, y_{m,b}]^\top \in \Delta^1$ (where $[1, 0]^\top$ denotes strict preference for $p_{m,a}$, and $[0.5, 0.5]^\top$ denotes a neutral tie), the cross-entropy sample loss is:

$$
\ell_{\text{pref}}(m) = -\sum_{k\in\{a,b\}} y_{m,k} \log \pi_{m,k}
$$

For a strict choice ($p_w \succ p_l$), this reduces to the classical Bradley-Terry negative log-likelihood:

$$
\ell_{\text{BT}}(m) = -\log \sigma\big(\tau(r_{m,w} - r_{m,l})\big)
$$

The weighted batch preference loss across $M$ samples is:

$$
\mathcal{L}_{\text{pref}} = -\frac{\sum_{m=1}^M w_m \sum_{k\in\{a,b\}} y_{m,k}\log \pi_{m,k}}{\sum_{m=1}^M w_m}
$$

### 3. Masked KL Retention Regularization
To protect baseline visual relation knowledge across the $55$ unlabelled predicate classes $\mathcal{O} = \{1, \dots, C\} \setminus \{p_a, p_b\}$ ($C = 57$ total PSG classes), the full softmax distribution is decomposed into unlabelled total mass $M_m, M_m^0$ and conditional distributions $\bar{q}_{m,c}, \bar{q}_{m,c}^0$:

$$
M_m^0 = \sum_{c\in\mathcal{O}_m} q_{m,c}^0, \quad M_m = \sum_{c\in\mathcal{O}_m} q_{m,c}, \quad \bar{q}_{m,c}^0 = \frac{q_{m,c}^0}{M_m^0}, \quad \bar{q}_{m,c} = \frac{q_{m,c}}{M_m}
$$

By the chain rule of KL divergence, the retention loss per sample is:

$$
\ell_{\text{KL}}(m) = D_{\text{KL}}^{\text{mass}}(m) + M_m^0 \cdot D_{\text{KL}}^{\text{cond}}(m)
$$

where:

$$
D_{\text{KL}}^{\text{mass}}(m) = M_m^0 \log \frac{M_m^0}{M_m} + (1-M_m^0) \log \frac{1-M_m^0}{1-M_m}
$$

$$
D_{\text{KL}}^{\text{cond}}(m) = \sum_{c\in\mathcal{O}_m} \bar{q}_{m,c}^0 \log \frac{\bar{q}_{m,c}^0}{\bar{q}_{m,c}}
$$

$$
\mathcal{L}_{\text{KL}} = \frac{\sum_{m=1}^M w_m \ell_{\text{KL}}(m)}{\sum_{m=1}^M w_m}
$$

### 4. Prototype Manifold Cosine Anchor
To prevent global coordinate frame distortion and prototype crowding across all $57$ predicate categories ($C = 57$), normalized predicate prototype embeddings $\hat{\mathbf{p}}_c$ are anchored to frozen baseline prototypes $\hat{\mathbf{p}}_c^0$:

$$
\mathcal{L}_{\text{proto}} = \frac{1}{C} \sum_{c=1}^{C} \left( 1 - \langle \hat{\mathbf{p}}_c, \hat{\mathbf{p}}_c^0 \rangle \right), \qquad \hat{\mathbf{p}}_c = \frac{\phi_{\text{proj}}(\mathbf{P}_c)}{\|\phi_{\text{proj}}(\mathbf{P}_c)\|_2}
$$

### 5. Tripartite Total Optimization Objective

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pref}} + \lambda_{\text{KL}}\mathcal{L}_{\text{KL}} + \lambda_{\text{proto}}\mathcal{L}_{\text{proto}}
$$

Calibrated hyperparameters: $\lambda_{\text{KL}} = 20.0$, $\lambda_{\text{proto}} = 20.0$, $\tau = 0.1$, and learning rate $\eta = 2\times 10^{-6}$.

---

## Installation

### 1. Python Environment Setup
From the repository root:
```bash
# Recommended Python 3.10+
pip install -r requirements.txt
pip install -r scene_graph_tool/requirements.txt
```

### 2. Build Frontend UI (React + Vite)
```bash
cd scene_graph_tool/frontend
npm install
npm run build
cd ../..
```

---

## Quick Start: Annotation GUI

Serve both the REST API and the interactive web interface with a single command:

```bash
python scene_graph_tool/server.py \
  --checkpoint checkpoint/best_model.pth \
  --config checkpoint/config.yml \
  --yolo-checkpoint checkpoint/PSG_yolo12m_backbone.pt \
  --psg-dir scene_graph_tool/psg_labels.json
```

Navigate to `http://localhost:7860` in your browser.

### Features:
- Upload images as individual files or bundled ZIP archives.
- Automatic **Stage 1 YOLO12m** object detection and **Stage 2 REACT++** relation candidate ranking.
- Active uncertainty sampling selects candidate pairs with minimal relative margin: $\Delta_{\text{rel}} = \frac{p_1 - p_2}{p_1 + p_2}$.
- Autosaved annotations with provenance checksums, model hashes, and export to CSV/JSON.

### Keyboard Shortcuts:
| Key | Action |
| :---: | :--- |
| `1` / `2` | Select Candidate Predicate 1 / 2 |
| `0` | Neutral preference (Tie: `[0.5, 0.5]`) |
| `X` | Both Wrong / No Relation (`[0.0, 0.0]`) |
| `S` | Skip Sample |
| `Ctrl+Z` / `Z` | Undo previous action |

---

## RLHP Training & Ablation Pipeline

### 1. Audit Preference Dataset
Verify data integrity and prevent PSG test set contamination:
```bash
python scene_graph_tool/rlhp.py audit \
  --sessions scene_graph_tool/scene_graph_preferences3.csv \
  --output scene_graph_tool/rlhp_runs/audit
```

### 2. Train Full RLHP Model
Fine-tune relation projection head with Bradley-Terry preference loss, masked KL retention ($\lambda_{\mathrm{KL}}=20$), and prototype anchor ($\lambda_{\mathrm{proto}}=20$):
```bash
python scene_graph_tool/train_rlhp_custom.py \
  --checkpoint checkpoint/best_model.pth \
  --config checkpoint/config.yml \
  --yolo-checkpoint checkpoint/PSG_yolo12m_backbone.pt \
  --preference-csv scene_graph_tool/scene_graph_preferences3.csv \
  --output-dir scene_graph_tool/rlhp_runs/full_rlhp \
  --epochs 5 \
  --lr 2e-6 \
  --tau 0.1 \
  --lambda-kl 20.0 \
  --lambda-proto 20.0
```

### 3. Run Complete 4-Way Loss Ablation
Reproduce all ablation variants (Pure Preference, +Proto Anchor, +KL Retention, Full RLHP):
```bash
bash scene_graph_tool/run_ablation_tau01.sh
```

---

## PSG Benchmark Evaluation

Evaluate checkpoint across all PSG test images ($2,177$ images) on SGDet and PredCLS:
```bash
python scene_graph_tool/evaluate_rlhp_epochs.py \
  --run-dir scene_graph_tool/rlhp_runs/full_rlhp \
  --config checkpoint/config.yml \
  --yolo-checkpoint checkpoint/PSG_yolo12m_backbone.pt
```

### Benchmark Summary (PSG Test SGDet):
| Model Configuration | mR@20 | mR@50 | mR@100 | R@50 | F1@50 | Bbox mAP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline YOLO12m-REACT++ | 23.28 | 26.37 | 28.28 | **38.25** | 31.22 | **56.57** |
| Preference Only (Unregularized) | 18.92 | 21.21 | 22.57 | 35.82 | 26.64 | 56.57 |
| Preference + Prototype Anchor | 21.24 | 24.51 | 26.18 | 37.31 | 29.58 | 56.57 |
| Preference + KL Retention | 22.93 | 26.21 | 27.88 | 37.56 | 30.87 | 56.57 |
| **Full RLHP (Tuned)** | **23.57** | **26.95** | **28.69** | 38.14 | **31.58** | **56.57** |

---

## Analysis & Visualization Suite

Generate all figures and tables included in the research report:

```bash
# Figure 2: Preference Dataset & Uncertainty Pairs Distribution
python scene_graph_tool/visualize_preference_dataset.py

# Figure 3: Four-Panel Optimization Diagnostics
python scene_graph_tool/plot_rlhp_training.py

# Figure 4: Epoch-wise Macro Recall Curves
python scene_graph_tool/visualize_epoch_metrics.py

# Figure 5: Catastrophic Semantic Drift Dynamics
python scene_graph_tool/plot_pure_pref_retention.py

# Figure 6: Per-Predicate SGDet Recall Shifts (All 56 classes)
python scene_graph_tool/visualize_predicate_improvements.py

# Figure 7: Qualitative Visual Comparison Grid
python scene_graph_tool/visualize_qualitative_results.py

# Table 1: Selection Strategy Comparison
python scene_graph_tool/compare_relation_selection.py
```

---

## Academic Report

The complete scientific report is available under [`report/`](report/):
- **[`report/main.pdf`](report/main.pdf)**: Precompiled 9-page research paper.
- **[`report/main.tex`](report/main.tex)**: Complete LaTeX source code.

To recompile the document locally:
```bash
cd scene_graph_tool/report
tectonic main.tex
# or pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## Citation & Contributor Attribution

If you use this toolkit, preference learning framework, or benchmark code in your research, please cite:

```bibtex
@article{phan2026humanpref_sgg,
  title={Human Preference Alignment for Two-Stage Scene Graph Generation: YOLO12m-REACT++ on Panoptic Scene Graphs},
  author={Phan, Quoc Anh},
  institution={Vision \& Language Navigation Team, VinRobotics},
  year={2026}
}

@inproceedings{neau2024reactpp,
  title={REACT++: Efficient Cross-Attention for Real-Time Scene Graph Generation},
  author={Neau, Maelic and others},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2024}
}

@inproceedings{yang2022psg,
  title={Panoptic Scene Graph Generation},
  author={Yang, Jingkang and Ang, Yi Zhe and Guo, Zujin and Zhou, Kaiyang and Zhang, Wayne and Liu, Ziwei},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

---

## License

This project is licensed under the [MIT License](../LICENSE). Upstream components are credited to their respective original authors and contributors.
