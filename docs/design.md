# Chapter 3 Software Design

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

**Provided utility** listings (attention rollout viz, prompted attention viz) are one-line imports from the package.

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
from ch03.fusion_transformer import FusionTransformer

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
        hidden_dim: int = 512,
    ) -> None: ...

    def forward(
        self,
        images: torch.Tensor,  # [B, 3, 224, 224] float in [0, 1]
        output_attentions: bool = False,
    ) -> torch.Tensor:  # [B, 196, 512]
        ...
```

SigLIP weights frozen via `requires_grad = False`. Single `nn.Linear(768, 512)` projection trainable.

### `src/ch03/viz_attention.py` (listing 3.2)

```python
def attention_rollout(
    attentions: list[torch.Tensor],  # one per layer, each [B, H, N, N]
    discard_ratio: float = 0.0,
) -> torch.Tensor:  # [B, N, N] composed
    ...

def overlay_heatmap(
    image: torch.Tensor,  # [3, 224, 224]
    rollout: torch.Tensor,  # [N, N], use row 0 (CLS) → columns 1: are patches
    save_path: str | None = None,
) -> matplotlib.figure.Figure: ...
```

### `src/ch03/language_backbone.py` (listing 3.3)

```python
class LanguageBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM-135M",
        hidden_dim: int = 512,
    ) -> None: ...

    def tokenize(
        self,
        instructions: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:  # (input_ids [B,L], attention_mask [B,L])
        ...

    def forward(
        self,
        instructions: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:  # (hidden [B,L,512], mask [B,L])
        ...
```

**No vocab expansion** — uses native SmolLM tokenizer (49,152 vocab). Vocab expansion is Chapter 4's first step.

### `src/ch03/state_encoder.py` (listing 3.4)

```python
class StateEncoder(nn.Module):
    def __init__(
        self,
        state_dim: int = 7,  # SO-100: 6 joints + 1 gripper position (per Ch 2)
        hidden_dim: int = 512,
    ) -> None: ...

    def forward(
        self,
        state: torch.Tensor,  # [B, state_dim]
    ) -> torch.Tensor:  # [B, 1, 512]
        ...
```

2-layer MLP (`Linear → GELU → Linear`). Note `state_dim=7` matches Ch 2's hand-off observation shape — to be verified against Sid's actual export.

### `src/ch03/fusion_transformer.py` (listing 3.5)

```python
class FusionTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None: ...

    def forward(
        self,
        tokens: torch.Tensor,  # [B, N, 512]
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:  # [B, N, 512]
        ...
```

Causal self-attention via upper-triangular mask. Pre-norm.

### `src/ch03/vla_backbone.py` (listing 3.6)

```python
class VLABackbone(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 512,
        num_fusion_layers: int = 6,
        num_fusion_heads: int = 8,
    ) -> None: ...

    def forward(
        self,
        image: torch.Tensor,  # [B, 3, 224, 224]
        instruction: list[str],
        state: torch.Tensor,  # [B, state_dim]
    ) -> torch.Tensor:  # [B, 196 + L + 1, 512]
        ...
```

**Frozen export contract for Chapter 4.** Do not rename, do not change signature.

### `src/ch03/viz_prompted_attention.py` (listing 3.7)

```python
def prompted_attention_grid(
    backbone: VLABackbone,
    image: torch.Tensor,  # [3, 224, 224]
    instructions: list[str],
    save_path: str | None = None,
) -> matplotlib.figure.Figure: ...
```

Three forward passes with different instructions; three attention rollout heatmaps; 1×3 panel grid.

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
| Provided utility (3.2, 3.7) | `from ch03.viz_attention import ...` + a usage call |

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
| `vision_encoder.py` | Assert all SigLIP params have `requires_grad=False`, output shape `[2, 196, 512]`, dtype `float32` |
| `language_backbone.py` | Assert vocab size == 49,152 (catches silent SmolLM revisions), tokenize+forward shape, projection trainable |
| `state_encoder.py` | Shape `[B, state_dim] → [B, 1, 512]`, GELU nonlinearity present |
| `fusion_transformer.py` | Causal mask is upper-triangular `-inf`, output shape == input shape, gradient flows |
| `vla_backbone.py` | End-to-end forward on dummy batch returns `[B, 196+L+1, 512]`; integration-marked |

### Fixtures (`conftest.py`)

```python
@pytest.fixture
def dummy_image_batch():
    return torch.rand(2, 3, 224, 224)

@pytest.fixture
def dummy_instructions():
    return ["pick up the cube", "place it in the target"]

@pytest.fixture
def dummy_state(state_dim=7):
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
- **`transformers>=4.45`** — vocab expansion APIs stable in 4.45+; SigLIP-base+SmolLM-135M both supported
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

1. **`state_dim` is 7 (Ch 2 ships 7-dim observation.state) or 6?** Verify against Sid's final `make_pickplace_dataloader` output before PR 4 lands. Likely 7 based on Ch 2 plan; `StateEncoder` defaults to 7 here.
2. **`preprocess.py` scope** — if Ch 2 already ships `[3, 224, 224]` float in `[0,1]`, this module shrinks to a comment + passthrough.
3. **Notebook integration with Ch 2's dataloader** — confirm import path (`from ch02 import make_pickplace_dataloader`) works locally once Sid's PRs 4-6 land.
4. **CI integration suite** — wait for Vulkan setup in GHA or skip integration tests in CI like Sid does.
