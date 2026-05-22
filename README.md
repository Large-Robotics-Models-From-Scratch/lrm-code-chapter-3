# lrm-code-chapter-3

Companion code for **Chapter 3** of *Build a Large Robot Model (From Scratch)* (Manning).

This chapter builds the **VLA backbone**: a frozen SigLIP vision encoder + a SmolLM-135M language backbone + a state encoder + a multimodal fusion transformer. The output is a sequence of contextualized hidden states ready for an action head (added in Chapter 4).

## What you build

```
image  ──▶  VisionEncoder (SigLIP, frozen)         ──▶  [B, 196, 512]
text   ──▶  LanguageBackbone (SmolLM-135M, native) ──▶  [B,  L,  512]
state  ──▶  StateEncoder (MLP)                     ──▶  [B,  1,  512]
              concat + causal self-attention
                          ▼
                    [B, 196+L+1, 512]
```

Plus a **prompted attention visualization** that shows the backbone routes attention to instruction-relevant patches before any action head is attached — diagnostic only, no training.

## Locked architecture (Ch 3 plan v3)

| Component | Choice |
|---|---|
| Vision encoder | SigLIP-base/16 (frozen) |
| Language backbone | SmolLM-135M (native tokenizer; **no vocab expansion** — that's Ch 4) |
| Fusion | Concat + causal self-attention, 6 layers, 8 heads, dropout 0.1 |
| Hidden dim | 512 |
| Robot | SO-100 (6-DOF arm + 1 gripper) |
| Camera input | `image_top` only (wrist added in Ch 6) |

## Setup

### Local (Linux / macOS / WSL)

```bash
git clone git@github.com:Large-Robotics-Models-From-Scratch/lrm-code-chapter-3.git
cd lrm-code-chapter-3
pip install -e ".[dev,data]"

# Wire up code-style agents (assumes ../lrm-code-agents cloned alongside)
mkdir -p .claude/agents
for f in ../lrm-code-agents/agents/*; do ln -sf "../$f" .claude/agents/; done
ln -s ../../lrm-code-agents/CLAUDE.md .claude/CLAUDE.md

# Reader-facing companion agent (Ch 3 specific)
ln -s ../../agents/chapter-03-guide.md .claude/agents/chapter-03-guide.md

# Optional: override agent defaults
cp ../lrm-code-agents/defaults.yml .lrm-agents.yml
```

### Colab

The notebook (`notebooks/ch03.ipynb`) auto-installs from this repo's git URL in its first cell. Same pattern as Ch 2.

### Tests

```bash
pytest -m "not integration"   # unit tests; what CI runs
pytest -m integration          # downloads HF models + tests full forward
```

### Pre-commit hooks

```bash
pre-commit install
```

Installs `nbstripout` (clears notebook outputs before commit) and `ruff` (Python lint + autofix).

## Repository layout

```
lrm-code-chapter-3/
├── README.md
├── CLAUDE.md                       # Project guide for Claude Code
├── program.md                      # Operating manual (how to use this repo)
├── pyproject.toml
├── ARCHITECTURE_LOG.md             # Cross-chapter decision log
├── .lrm-agents.yml                 # lrm-code-agents config overrides
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── src/ch03/                       # Importable Python package (the export contract)
│   ├── __init__.py
│   ├── vision_encoder.py           # PR 2 — SigLIP load + freeze + project
│   ├── language_backbone.py        # PR 3 — SmolLM, native tokenizer
│   ├── state_encoder.py            # PR 4 — 6→512 MLP
│   ├── fusion_transformer.py       # PR 4 — concat + causal self-attention
│   ├── vla_backbone.py             # PR 5 — composes the above
│   ├── viz_attention.py            # PR 2 — attention rollout
│   ├── viz_prompted_attention.py   # PR 6 — 3-prompt attention grid
│   └── preprocess.py               # PR 5 — image preprocessing
├── notebooks/
│   └── ch03.ipynb                  # Reader's canonical walkthrough
├── agents/
│   └── chapter-03-guide.md         # Reader-facing chapter companion agent
├── docs/
│   ├── chapter_3_plan.md           # Synced copy of lrm-book/chapter_3/chapter_3_structure_and_plan.md
│   ├── design.md                   # Module APIs, test strategy, dep pins
│   ├── MANNING_STYLE.md            # Manning conventions distilled
│   └── decisions/
│       ├── vision-encoder-choice.md
│       └── fusion-design.md
├── scripts/                        # Reader sanity scripts (check_<module>.py, added per PR)
├── tests/
│   ├── conftest.py
│   ├── test_smoke.py
│   ├── test_vision_encoder.py
│   ├── test_language_backbone.py
│   ├── test_state_encoder.py
│   ├── test_fusion_transformer.py
│   └── test_vla_backbone.py
└── figures/                        # Rendered figures (figure_3_*.png)
```

## Hand-off contract to chapter 4

```python
from ch03 import VLABackbone

backbone = VLABackbone(hidden_dim=512)
hidden = backbone(image, instruction, state)
# image:       [B, 3, 224, 224]   top camera, preprocessed
# instruction: list[str]          batch of B instructions
# state:       [B, 7]             SO-100 state (6 joint positions + 1 gripper) per Ch 2
# hidden:      [B, 196 + L + 1, 512]
# tokenizer:   native SmolLM (49,152 vocab) — no expansion in Ch 3
```

Chapter 4's first step: expand the vocab with 1,536 action token IDs and `resize_token_embeddings`. That belongs to Ch 4, not here.

## Reader tracks (see book chapter 1)

- **Track A (laptop CPU)**: this chapter runs end-to-end. SigLIP and SmolLM forward passes are slow but functional.
- **Track B (Colab Pro / consumer GPU)**: full pace on T4 (12 GB).
- **Track C (full hardware)**: same as A or B for Ch 3 — hardware only matters from Ch 9.

## Built on

- **Chapter 2 repo**: [`lrm-code-chapter-2`](https://github.com/Large-Robotics-Models-From-Scratch/lrm-code-chapter-2) — `from ch02 import make_pickplace_dataloader, normalize, denormalize`

## License

Apache 2.0 — code samples in the book are open for reader use and modification.
