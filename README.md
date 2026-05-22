# lrm-code-chapter-3

Companion code for **Chapter 3** of *Build a Large Robot Model (From Scratch)* (Manning).

This chapter builds the **VLA backbone**: a frozen SigLIP vision encoder + a SmolLM-135M language backbone + a state encoder + a multimodal fusion transformer. The output is a sequence of contextualized hidden states ready for an action head (added in chapter 4).

## What you build

```
image  ──▶  VisionEncoder (SigLIP, frozen)         ──▶  [B, 196, 512]
text   ──▶  LanguageBackbone (SmolLM-135M)         ──▶  [B,  L,  512]
state  ──▶  StateEncoder (MLP)                     ──▶  [B,  1,  512]
              concat + causal self-attention
                          ▼
                    [B, 196+L+1, 512]
```

Plus a **VQA training loop** that proves the backbone learned vision-language alignment before any action head is attached.

## Locked architecture

| Component | Choice |
|---|---|
| Vision encoder | SigLIP-base/16 (frozen) |
| Language backbone | SmolLM-135M |
| Fusion | Concat + causal self-attention, 6 layers |
| Hidden dim | 512 |
| Action tokens reserved | 1,536 (256 bins × 6 joint dims) |
| Robot | SO-100 (6-DOF) |

## Setup

```bash
# Clone
git clone https://github.com/Large-Robotics-Models-From-Scratch/lrm-code-chapter-3.git
cd lrm-code-chapter-3

# Install
uv venv
source .venv/bin/activate
pip install -e .

# Wire up code-style agents (optional but recommended)
git clone https://github.com/Large-Robotics-Models-From-Scratch/lrm-code-agents.git ../lrm-code-agents
ln -s ../lrm-code-agents/agents .claude/agents
ln -s ../lrm-code-agents/CLAUDE.md .claude/CLAUDE.md
```

## Repository layout

```
lrm-code-chapter-3/
├── models/
│   ├── vision_encoder.py       # SigLIP wrapper, frozen, projects 768 → 512
│   ├── language_backbone.py    # SmolLM wrapper + vocab expansion
│   ├── state_encoder.py        # 6 → 512 MLP, one token output
│   ├── fusion_transformer.py   # Causal self-attention over multimodal sequence
│   └── vla_backbone.py         # Composes the above
├── data/
│   └── preprocess.py           # Center-crop 480×640 → 224×224
├── train_vqa.py                # VQA pretraining loop
├── viz_attention.py            # Attention rollout for SigLIP
├── tests/                      # Shape, dtype, device-placement assertions
├── ARCHITECTURE_LOG.md         # Cross-chapter decision log
├── pyproject.toml
└── README.md
```

## Hand-off contract to chapter 4

```python
from lrm_ch03 import VLABackbone

backbone = VLABackbone(hidden_dim=512)
hidden = backbone(image, instruction, state)
# image:       [B, 3, 224, 224]
# instruction: List[str]
# state:       [B, 6]
# hidden:      [B, 196 + L + 1, 512]
```

## Reader tracks (see book chapter 1)

- **Track A (sim-only, free)**: this chapter runs end-to-end on a laptop CPU. SigLIP and SmolLM forward passes are slow but functional.
- **Track B (sim + GPU)**: full pace on Colab T4 (12 GB) or any consumer GPU.
- **Track C (full hardware)**: same as A or B for chapter 3 — hardware only matters from chapter 9.

## Built on

- **Chapter 2 repo**: [`lrm-code-chapter-2`](https://github.com/Large-Robotics-Models-From-Scratch/lrm-code-chapter-2) — the SO-100 simulation env and the `lerobot/svla_so100_pickplace` data loader

## License

Apache 2.0 — code samples in the book are open for reader use and modification.
