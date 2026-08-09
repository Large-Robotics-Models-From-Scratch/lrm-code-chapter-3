# Chapter 3 Software Design

> **Superseded by v5**: the shipped backbone is `VLABackbone` (the v3 name, kept), language model is `SmolLM2-135M`, hidden width is **576** (the backbone's native width, no down-projection to 512), two cameras give **392** image tokens, and fusion is a token-level concatenation fed to the language backbone via `inputs_embeds` (concat migration 2026-08-09; the splice is retired) (the pretrained backbone is the fuser; `fusion_transformer.py` is only the optional separate-encoder exercise). Output contract: `[B, 392 + L + 1, 576]`. The APIs below are corrected for the load-bearing facts; some prose still reflects the older single-camera / separate-fusion plan.

Companion to `chapter_3_plan.md`. The plan locks the *what* — listings, exports, prose mapping. This doc locks the *how* — module APIs, notebook architecture, test strategy, dependency pinning, agent prompt.

If you change anything in this doc that affects the Ch 4 export contract, update `chapter_3_plan.md` and `../lrm-book/chapter_3/chapter_3_structure_and_plan.md` in the same commit.

**Authoritative upstream sources** per `MANNING_STYLE.md`: `../lrm-book/STYLEGUIDE.md`, `../lrm-book/FIGURE_STYLE_GUIDE.md`, `../lrm-book/writing_instructions/`. When this doc conflicts with any upstream, upstream wins.

---

## 1. Reader Experience Workflow

Two paths, same end state — the reader runs `notebooks/ch03.ipynb` against the `ch03` package.

**Local:** `git clone`, `pip install -e ".[dev,data]"`, open the notebook in Jupyter.

**Colab:** The README documents the install recipe; first cell auto-installs from this repo's git URL.

### Notebook ↔ package

**Type-along** listings (vision encoder, language backbone, state encoder, fusion transformer, VLA backbone) live in the notebook namespace — the reader writes them, subsequent cells call their version. A byte-for-byte equivalent exists in `src/ch03/<module>.py` but is independent (no live binding) and serves tests, Ch 4 imports, and editor cross-reference.

**Provided utility** listings (patch self-similarity viz, tracking viz) are one-line imports from the package.

### Figures

Inline in both paths. `fig.savefig("figures/...")` is author-only — used locally to regenerate print-quality PNGs.

---

## 2. Module APIs

The chapter plan's module mapping is the starting point. This section pins every public function signature.

### `src/ch03/__init__.py`

Re-exports the Ch 4 contract symbols at the package root.

```python
from ch03.vla_backbone import VLABackbone
from ch03.vision_encoder import VisionEncoder
from ch03.language_backbone import LanguageBackbone
from ch03.state_encoder import StateEncoder
from ch03.fusion_transformer import FusionTransformer  # optional exercise only

__all__ = [
    "VLABackbone",
    "VisionEncoder",
    "LanguageBackbone",
    "StateEncoder",
    "FusionTransformer",
]
```

### `src/ch03/vision_encoder.py` (listing 3.1)

```python
class VisionEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        hidden_dim: int = 576,
    ) -> None: ...

    def forward(
        self,
        images: torch.Tensor,  # [B, 3, 224, 224] float in [0, 1]
        output_attentions: bool = False,
    ) -> torch.Tensor:  # [B, 196, 576] per camera
        ...
```

SigLIP weights frozen via `requires_grad = False`. Single `nn.Linear(768, 576)` projection trainable.

### `src/ch03/viz_similarity.py` (listings 3.2 and 3.7)

Patch self-similarity on the raw frozen SigLIP features (chosen over
attention rollout per the 2026-06-15 deep-research verdict: rollout has
no CLS row on a CLS-less encoder and deep-layer token mixing breaks it).

```python
def patch_self_similarity(
    features: torch.Tensor,  # [196, 768] raw SigLIP features for one frame
    query_row: int, query_col: int,
) -> torch.Tensor:  # [14, 14] cosine sim of the query patch to all patches

def similarity_grid(          # listing 3.2 / figure 3.3
    vision_encoder, image,    # [3, H, W] in [0, 1]
    queries,                  # list of (row, col, label), e.g. cube + arm
    save_path=None,
) -> matplotlib.figure.Figure: ...

def tracking_grid(            # listing 3.7 / figure 3.6
    vision_encoder, frames,   # list of [3, H, W]
    queries,                  # one (row, col) per frame; the object moves
    labels=None, save_path=None,
) -> matplotlib.figure.Figure: ...
```

### `src/ch03/language_backbone.py` (listing 3.3)

```python
class LanguageBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-135M",
        hidden_dim: int = 576,
    ) -> None: ...

    def tokenize(
        self,
        instructions: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:  # (input_ids [B,L], attention_mask [B,L])
        ...

    def forward(
        self,
        instructions: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:  # (hidden [B,L,576], mask [B,L])
        ...
```

**No vocab expansion** — uses native SmolLM tokenizer (49,152 vocab). Vocab expansion is Chapter 4's first step.

### `src/ch03/state_encoder.py` (listing 3.4)

```python
class StateEncoder(nn.Module):
    def __init__(
        self,
        state_dim: int = 6,  # SO-100: 5 arm joints + 1 gripper (per Ch 2)
        hidden_dim: int = 576,
    ) -> None: ...

    def forward(
        self,
        state: torch.Tensor,  # [B, state_dim]
    ) -> torch.Tensor:  # [B, 1, 576]
        ...
```

2-layer MLP (`Linear → GELU → Linear`). `state_dim=6` is locked by Ch 2 Table 2.2 (`observation.state: (6,)`, verified pr-7).

### `src/ch03/fusion_transformer.py` (optional separate-encoder exercise)

> Off the main path. The shipped backbone fuses via token-level splice; this module is kept only as the optional separate-encoder fusion exercise.

```python
class FusionTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 576,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None: ...

    def forward(
        self,
        tokens: torch.Tensor,  # [B, N, 576]
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:  # [B, N, 576]
        ...
```

Causal self-attention via upper-triangular mask. Pre-norm.

### `src/ch03/vla_backbone.py` (listing 3.6)

Fusion is a token-level splice: image and state tokens are written into the language backbone's own input-embedding stream via `masked_scatter`, so the pretrained backbone is the fuser (no separate fusion module on the main path). The backbone grows its input embedding table by two inert placeholder rows for the image and state splice markers - this is NOT `resize_token_embeddings`; the tokenizer is untouched and `config.vocab_size` stays 49152.

Two things the backbone deliberately does NOT own, because owning them once is the point:

- **The vision path.** `self.vision_encoder = VisionEncoder(hidden_dim=576)` composes listing 3.1's encoder as-is. Frozen SigLIP, the 768->576 projection, and the `[0,1]`->`[-1,1]` pixel normalization buffers all live in `vision_encoder.py`. `forward` calls it once and gets `[B*2, 196, 576]` back already projected, so there is no `img_proj` on the backbone.
- **A second embedding table.** After growing the table, `self.language_backbone.set_input_embeddings(self.embed_tokens)` hands it back, so the model holds one table. Keeping the language backbone's original alongside it would add 28,311,552 trainable parameters (49,152 x 576) that `forward` never reads, because it passes `inputs_embeds`. Measured: 135,295,488 trainable / 92,884,224 frozen. `tests/test_vla_backbone.py` guards both with an identity check and a `< 140_000_000` trainable budget.

```python
class VLABackbone(nn.Module):
    def __init__(self) -> None: ...   # hidden width 576 (backbone native)

    def tokenize_instruction(
        self,
        instruction: str,
    ) -> list[int]:                  # L native SmolLM2 text ids

    def build_sequence_ids(
        self,
        text_ids: list[int],
    ) -> list[int]:                  # image + text + state template

    def forward(
        self,
        images: torch.Tensor,        # [B, 2, 3, 224, 224] two cameras
        sequence_ids: torch.Tensor,  # [B, N] from build_sequence_ids
        state: torch.Tensor,         # [B, state_dim]
    ) -> torch.Tensor:               # [B, 392 + L + 1, 576]
        ...
```

**Frozen export contract for Chapter 4.** Do not rename, do not change signature.

### Listing 3.7 / figure 3.6 — `tracking_grid` (in `viz_similarity.py`)

`tracking_grid` (signature above) runs patch self-similarity on the cube
query in several frames where the cube spawns in different positions; the
highlight tracks the object. Lineage: this replaced the v4 prompted-
attention figure (could not reproduce on an untrained causal backbone),
then the v5 object-attention rollout (blurry on a CLS-less encoder per
the deep-research verdict). The "language doesn't steer vision yet" point
stays in prose. No `output_attentions`/eager needed.

### `src/ch03/preprocess.py`

```python
def preprocess_image(
    image: torch.Tensor,  # raw from Ch 2 dataloader, already [3, 224, 224] float in [0, 1]
) -> torch.Tensor:  # [3, 224, 224] (passthrough or minimal adjustment)
    ...
```

Likely a near-passthrough — Ch 2's `make_pickplace_dataloader` already yields `[3, 224, 224]` float images scaled to `[0, 1]`. This module exists to centralize any per-model normalization SigLIP might need (e.g., ImageNet mean/std).

---

## 3. Notebook Architecture

### Layout

One notebook: `notebooks/ch03.ipynb`. Section headers mirror the chapter.

Within each section:
- Markdown cell introducing the listing — caption format **noun phrase, no colon** (e.g., `### Listing 3.1 Loading and freezing SigLIP`)
- A **lead-in sentence** before the code cell, keyed to the listing's role
- Code cell containing the listing
- Output cell (cleared before commit)

### What lives inline vs imported

| Role | Notebook cell contents |
|---|---|
| Type-along (3.1, 3.3, 3.4, 3.5, 3.6) | Full code from the book listing, inline |
| Provided utility (3.2, 3.7) | `from ch03.viz_similarity import ...` + a usage call |

### Figure conventions

Every figure-producing cell:

```python
fig = plot_X(...)
fig.savefig("../figures/figure_3_<n>_<slug>.png", dpi=300, bbox_inches="tight")
```

Figure 3.1 is reused from Chapter 1's roadmap recap — no code cell.

### Reproducibility

- Every random call seeded
- HuggingFace model loads pinned via `pyproject.toml` transformers version

### Output discipline

- Pre-commit hook: `jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`
- Notebook diffs stay reviewable

---

## 4. Test Strategy

### Layout

```
tests/
├── conftest.py            # shared fixtures (dummy tensors, mock dataloader output)
├── test_smoke.py          # ch03 imports cleanly
├── test_vision_encoder.py # frozen-param + shape + dtype
├── test_language_backbone.py # tokenize + forward shape + vocab size assertion
├── test_state_encoder.py  # shape + nonlinearity smoke
├── test_fusion_transformer.py # mask shape + causality + grad flow
└── test_vla_backbone.py   # integration: dummy batch → expected output shape
```

### Coverage by module

| Module | Approach |
|---|---|
| `vision_encoder.py` | Assert all SigLIP params have `requires_grad=False`, output shape `[2, 196, 576]` per camera, dtype `float32` |
| `language_backbone.py` | Assert vocab size == 49,152 (catches silent SmolLM revisions), tokenize+forward shape, projection trainable |
| `state_encoder.py` | Shape `[B, state_dim] → [B, 1, 576]`, GELU nonlinearity present |
| `fusion_transformer.py` | Causal mask is upper-triangular `-inf`, output shape == input shape, gradient flows (optional exercise module) |
| `vla_backbone.py` | End-to-end forward on dummy batch returns `[B, 392+L+1, 576]`; integration-marked |

### Fixtures (`conftest.py`)

```python
@pytest.fixture
def dummy_image_batch():
    return torch.rand(2, 3, 224, 224)

@pytest.fixture
def dummy_instructions():
    return ["pick up the cube", "place it in the target"]

@pytest.fixture
def dummy_state(state_dim=6):
    return torch.rand(2, state_dim)
```

### CI

GitHub Actions: `pytest tests/ -m "not integration"` on push. Same CPU-torch-first trick as Sid's CI.

---

## 5. Dependency Strategy

### Python version

**3.12 only** (`>=3.12,<3.13`) to match `lerobot==0.5.1` and `lrm-code-agents/defaults.yml`.

### Packaging

```toml
[project]
name = "lrm-ch03"
requires-python = ">=3.12,<3.13"
dependencies = [
    "torch>=2.1,<3.0",
    "transformers>=4.45,<5.0",
    "numpy>=1.24,<3.0",
    "pillow>=10,<12",
    "matplotlib>=3.7,<4.0",
]

[project.optional-dependencies]
data = [
    "lerobot==0.5.1",
    "huggingface-hub>=0.25",
]
sim = [
    # transitive via lrm-ch02 if integration tests run full pipeline
    "mani-skill==3.0.1",
]
dev = [
    "pytest>=7.4",
    "pytest-cov",
    "ruff",
    "jupyter",
    "nbconvert",
    "nbstripout",
    "pre-commit",
]
```

Pins:
- **`transformers>=4.45`** — vocab expansion APIs stable in 4.45+; SigLIP-base+SmolLM2-135M both supported
- **`lerobot==0.5.1`** — matches Ch 2 hand-off contract
- **`torch>=2.1`** — matches Ch 2

### Lockfile

Generated as a release-time artifact, not maintained during development. Strict pins on `transformers` and `lerobot` plus loose pins on stable foundations.

---

## 6. Chapter-3 Agent Prompt

Lives in `agents/chapter-03-guide.md`. See that file. Symlinked into `.claude/agents/` via the setup in `program.md` §2.

The agent loads files lazily via `Read` rather than stuffing them into the system prompt.

---

## Open Questions

1. **`state_dim`** — RESOLVED: 6 (Ch 2 Table 2.2, `observation.state: (6,)`, verified pr-7). `StateEncoder` defaults to `state_dim=6`.
2. **`preprocess.py` scope** — RESOLVED: Ch 2 ships `[3, 480, 640]` float in `[0,1]` (Table 2.2), so `preprocess_image` resizes to `[3, 224, 224]`. SigLIP's `[-1,1]` normalization lives inside `VisionEncoder`.
3. **Notebook integration with Ch 2's dataloader** — confirm import path (`from ch02 import make_pickplace_dataloader`) works locally once Sid's PRs 4-6 land.
4. **CI integration suite** — wait for Vulkan setup in GHA or skip integration tests in CI like Sid does.
