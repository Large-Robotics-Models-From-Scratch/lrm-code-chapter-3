# lrm-code-chapter-3

Companion code for **Chapter 3** of *Build a Large Robot Model (From Scratch)* (Manning).

This chapter builds the **VLA backbone**: a frozen SigLIP vision encoder + a SmolLM2-135M language backbone + a state encoder, fused by **token-level fusion**. The two camera views and the state are projected and spliced into the language backbone's own token stream, so the pretrained backbone is the fuser - there is no separate fusion module on the main path. The output is a sequence of contextualized hidden states ready for an action head (added in Chapter 4).

## What you build

```
images ──▶  VisionEncoder (SigLIP, frozen, 2 cams) ──▶  [B, 392, 576]
text   ──▶  LanguageBackbone (SmolLM2-135M)        ──▶  [B,  L,  576]
state  ──▶  StateEncoder (MLP)                     ──▶  [B,  1,  576]
        concatenated into one sequence, fed via inputs_embeds
        (the pretrained backbone fuses; no separate fusion module)
                          ▼
                    [B, 392+L+1, 576]
```

The backbone composes those three modules rather than rebuilding them. `VisionEncoder` is used as-is, so the frozen SigLIP, the 768->576 projection, and SigLIP's `[0,1]`->`[-1,1]` pixel normalization exist in exactly one place, and the grown input embedding table is handed back to the language backbone so the model carries one embedding table (135,295,488 trainable / 92,884,224 frozen parameters).

Plus a **patch self-similarity visualization** that shows the frozen vision encoder localizes objects before any action head is attached - diagnostic only, no training.

One **real sample** from the Chapter 2 dataset (`lerobot/svla_so101_pickplace`) ships inside the package, so every listing and the notebook run on real data rather than `torch.rand` noise, with no dataloader, video decoder, or network access required:

```python
from ch03 import load_sample

images, state, instruction = load_sample()
# images: [1, 2, 3, 480, 640] float32 in [0, 1]   both camera views, native resolution
# state:  [1, 6] float32                          z-scored SO-101 state
# instruction: "pink lego brick into the transparent box"
```

## Locked architecture (Ch 3 plan v5)

| Component | Choice |
|---|---|
| Vision encoder | SigLIP-base/16 (frozen) |
| Language backbone | SmolLM2-135M (native tokenizer; **no vocab expansion** - that's Ch 4) |
| Fusion | Token-level fusion: image, language, and state embeddings concatenated into one sequence, fed via `inputs_embeds`; the pretrained backbone fuses (no separate fusion module, no placeholder IDs) |
| Hidden dim | 576 (the backbone's native width) |
| Robot | SO-100 (6-DOF arm + 1 gripper) |
| Camera input | both `observation.images.up` and `observation.images.side` (two cameras, 392 image tokens = 2 x 196) |

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
│   ├── language_backbone.py        # PR 3 — SmolLM2, native tokenizer
│   ├── state_encoder.py            # PR 4 — 6→576 MLP
│   ├── fusion_transformer.py       # optional separate-encoder fusion exercise (off main path)
│   ├── vla_backbone.py             # PR 5 — VLABackbone, composes the above
│   ├── viz_attention.py            # PR 2 — attention rollout
│   ├── viz_prompted_attention.py   # PR 6 — 3-prompt attention grid
│   ├── preprocess.py               # PR 5 — image preprocessing
│   ├── sample.py                   # load_sample() — one real Ch 2 sample
│   └── assets/                     # frame_up.png, frame_side.png, sample.json
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
│   ├── test_sample.py
│   └── test_vla_backbone.py
└── figures/                        # Rendered figures (figure_3_*.png)
```

## Hand-off contract to chapter 4

```python
from ch03 import VLABackbone

backbone = VLABackbone()
tokens = backbone.tokenizer(
    [instruction], return_tensors="pt", padding=True
)
hidden = backbone(
    images, tokens.input_ids, state, tokens.attention_mask
)
# images:  [B, 2, 3, H, W] in [0, 1]  two cameras (up + side); the
#                                     vision module resizes to 224
# input_ids: [B, L] long tensor       HF text ids (pad token = eos)
# state:   [B, 6]                     SO-101 state per Ch 2 Table 2.2
# hidden:  [B, N, 576]                N = 392 + L + 1
# Two-stage access for heads that extend the sequence:
#   emb, mask, pos = backbone.embed_inputs(images, ids, state, m)
#   hidden = backbone.contextualize(emb, mask, pos)
# tokenizer: native SmolLM2 (49,152 vocab) - no expansion in Ch 3
```

`fusion_transformer.py` is the optional separate-encoder fusion exercise (Exercise 3.4), kept off the main path; the shipped backbone fuses by splicing image and state tokens into the language backbone's own stream.

Chapter 4's first step: expand the vocab with 1,536 action token IDs and `resize_token_embeddings`. That belongs to Ch 4, not here.

## Reader tracks (see book chapter 1)

- **Track A (laptop CPU)**: this chapter runs end-to-end. SigLIP and SmolLM forward passes are slow but functional.
- **Track B (Colab Pro / consumer GPU)**: full pace on T4 (12 GB).
- **Track C (full hardware)**: same as A or B for Ch 3 — hardware only matters from Ch 9.

## Built on

- **Chapter 2 repo**: [`lrm-code-chapter-2`](https://github.com/Large-Robotics-Models-From-Scratch/lrm-code-chapter-2) — `from ch02 import make_pickplace_dataloader, normalize, denormalize`

## License

Apache 2.0 — code samples in the book are open for reader use and modification.
