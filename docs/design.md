# Chapter 3 Software Design

> **Superseded by v5**: the shipped backbone is `VLABackbone` (the v3 name, kept), language model is `SmolLM2-135M`, hidden width is **576** (the language backbone's native width, no down-projection to 512), two cameras give **392** visual positions, and fusion is direct concatenation of the visual, language, and state input embeddings into one observation prefix fed to the language backbone via `inputs_embeds` (concat migration 2026-08-09). The language backbone is the fuser; `fusion_transformer.py` is only the optional separate-encoder exercise. Output contract: `[B, 392 + L + 1, 576]` contextualized hidden states. The APIs below are corrected for the load-bearing facts; some prose still reflects the older single-camera / separate-fusion plan.

Companion to `chapter_3_plan.md`. The plan locks the *what* — listings, exports, prose mapping. This doc locks the *how* — module APIs, notebook architecture, test strategy, dependency pinning, agent prompt.

If you change anything in this doc that affects the Ch 4 export contract, update `chapter_3_plan.md` and `../lrm-book/chapter_3/chapter_3_structure_and_plan.md` in the same commit.

**Authoritative upstream sources** per `MANNING_STYLE.md`: `../lrm-book/STYLEGUIDE.md`, `../lrm-book/FIGURE_STYLE_GUIDE.md`, `../lrm-book/writing_instructions/`. When this doc conflicts with any upstream, upstream wins.

---

## 1. Reader Experience Workflow

Two paths, same end state — the reader runs `notebooks/ch03.ipynb` against the `ch03` package.

**Local:** `git clone`, `pip install -e ".[dev,data]"`, open the notebook in Jupyter.

**Colab:** The README documents the install recipe; first cell auto-installs from this repo's git URL.

### Notebook ↔ package

**Type-along** listings (vision encoder, language backbone embeddings, state encoder, VLA backbone) live in the notebook namespace — the reader writes them, subsequent cells call their version. A byte-for-byte equivalent exists in `src/ch03/<module>.py` but is independent (no live binding) and serves tests, Ch 4 imports, and editor cross-reference.

**Provided utility** listings (patch self-similarity viz, plus the bonus tracking grid) are one-line imports from the package.

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
from ch03.preprocess import preprocess_image
from ch03.sample import load_sample

__all__ = [
    "VLABackbone",
    "VisionEncoder",
    "LanguageBackbone",
    "StateEncoder",
    "preprocess_image",
    "load_sample",
]
```

`FusionTransformer` is off the main path, so it is not re-exported; the optional exercise imports it from `ch03.fusion_transformer` directly.

### `src/ch03/vision_encoder.py` (listing 3.1)

```python
class VisionEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 576) -> None: ...

    def forward(
        self,
        images: torch.Tensor,  # [B, 3, 224, 224] float in [0, 1]
        output_attentions: bool = False,
    ) -> torch.Tensor:  # [B, 196, 576] per camera
        ...
```

SigLIP weights frozen via `requires_grad = False`. Single `nn.Linear(768, 576)` projection trainable.

### `src/ch03/viz_similarity.py` (listing 3.2, plus a bonus grid)

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
    queries,                  # list of (row, col, label)
    save_path=None, similarity_range=(-0.1, 1.0),
) -> matplotlib.figure.Figure: ...

def tracking_grid(            # bonus notebook visualization
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
    ) -> None: ...   # width is SmolLM2's native 576, not configurable

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

**No vocabulary expansion** — uses the native SmolLM tokenizer (49,152 vocab). The revised Chapter 4 contract does not expand it either: action architectures attach their own components instead.

### `src/ch03/state_encoder.py` (listing 3.4)

```python
class StateEncoder(nn.Module):
    def __init__(
        self,
        state_dim: int = 6,  # SO-100: 5 arm joints + 1 gripper (per Ch 2)
        width: int = 576,
    ) -> None: ...

    def forward(
        self,
        state: torch.Tensor,  # [B, state_dim]
    ) -> torch.Tensor:  # [B, 1, 576]
        ...
```

2-layer MLP (`Linear → GELU → Linear`). `state_dim=6` is locked by Ch 2 Table 2.2 (`observation.state: (6,)`, verified pr-7).

### `src/ch03/fusion_transformer.py` (optional separate-encoder exercise)

> Off the main path. The shipped backbone fuses by direct concatenation; this module is kept only as the optional separate-encoder fusion exercise.

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

Fusion is direct concatenation: the visual and state input embeddings are concatenated with the language input embeddings looked up from SmolLM2's own table, forming the observation prefix that the language backbone reads through `inputs_embeds`. The pretrained language backbone is the fuser (no separate fusion module on the main path). Only the language stream carries vocabulary IDs, so the tokenizer, the embedding table (49,152 rows), and `config.vocab_size` are all untouched.

Two things the backbone deliberately does NOT own, because owning them once is the point:

- **The vision path.** `self.vision_encoder = VisionEncoder(hidden_dim=576)` composes listing 3.1's encoder as-is. Frozen SigLIP, the bicubic antialiased resize to 224, the 768->576 projection, and the `[0,1]`->`[-1,1]` pixel normalization buffers all live in `vision_encoder.py`. `embed_inputs` calls it once and gets `[B*2, 196, 576]` back already projected, so there is no `img_proj` on the backbone.
- **A second embedding table.** The language stream is embedded through `self.language_backbone.get_input_embeddings()`, so the model holds exactly one table, the native one. Measured: 135,294,336 trainable / 92,884,224 frozen. `tests/test_vla_backbone.py` guards the layout with a per-position identity check and a `< 140_000_000` trainable budget.

```python
class VLABackbone(nn.Module):
    def __init__(self) -> None: ...  # width 576 (language-backbone native)

    def embed_inputs(
        self,
        images: torch.Tensor,        # [B, 2, 3, H, W] in [0, 1]
        input_ids: torch.Tensor,     # [B, L] HF text ids (pad = eos)
        state: torch.Tensor,         # [B, state_dim]
        text_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...  # (input_embeddings [B,N,576], attention_mask, position_ids)

    def contextualize(
        self,
        input_embeddings: torch.Tensor,   # [B, N, 576]
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:               # [B, N, 576] hidden states

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        state: torch.Tensor,
        text_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:               # [B, 392 + L + 1, 576]
        ...
```

Position IDs are derived from the attention mask (`cumsum - 1`) rather than from the physical index, so padded instruction slots never shift the state position's rotary index: it is `392 + L_valid` in every row.

**Frozen export contract for Chapter 4.** Do not rename, do not change these three signatures.

### `tracking_grid` (in `viz_similarity.py`) — bonus visualization

`tracking_grid` (signature above) runs patch self-similarity on the brick
query in several frames where the brick sits in different positions; the
highlight tracks the object. It is a notebook bonus, not a numbered
chapter listing or figure. Lineage: this replaced the v4 prompted-
attention figure (could not reproduce on an untrained causal backbone),
then the v5 object-attention rollout (blurry on a CLS-less encoder per
the deep-research verdict). The "language doesn't steer vision yet" point
stays in prose. No `output_attentions`/eager needed.

### `src/ch03/preprocess.py`

```python
def preprocess_image(
    image: torch.Tensor,  # [3, H, W] or [B, 3, H, W] float in [0, 1]
) -> torch.Tensor:        # same rank, spatial dims 224 x 224
    ...
```

Ch 2's `make_pickplace_dataloader` yields `[3, 480, 640]` float frames in `[0, 1]`, so this resizes to SigLIP's `224 x 224` with `mode="bicubic"`, `align_corners=False`, `antialias=True` — the resampling the pinned SigLIP image processor uses. This is the chapter's only resize: `VisionEncoder.siglip_features` calls this helper for frames that are not already 224, so the probe path and the encoder path cannot drift apart, and `tests/test_preprocess.py` pins both facts. SigLIP's `[-1, 1]` normalization stays inside `VisionEncoder`.

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
| Type-along (3.1, 3.3, 3.4, 3.5, 3.6, 3.7) | Full code from the book listing, inline |
| Provided utility (3.2) | `from ch03.viz_similarity import ...` + a usage call |

### Figure conventions

Every figure-producing cell:

```python
fig = plot_X(...)
fig.savefig("../figures/figure_3_<n>_<slug>.png", dpi=300, bbox_inches="tight")
```

Figures 3.1 and 3.7 are diagrams, not rendered outputs — no code cell.

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
├── test_preprocess.py     # shapes + bicubic/antialias parity + one shared resize
├── test_vision_encoder.py # frozen-param + shape + dtype
├── test_language_backbone.py # tokenize + forward shape + vocab size assertion
├── test_state_encoder.py  # shape + nonlinearity smoke
├── test_fusion_transformer.py # mask shape + causality + grad flow
├── test_viz_similarity.py # probe helper shapes + query indexing
├── test_guardrails.py     # source scan: no add_tokens / resize_token_embeddings / masked_scatter
├── test_migration_parity.py # pins the retired construction against the shipped one
├── test_sample.py         # the shipped real sample loads and runs
└── test_vla_backbone.py   # observation prefix, mask, position ids, padded batches, forward
```

### Coverage by module

| Module | Approach |
|---|---|
| `preprocess.py` | Shapes for single and batched frames, values stay in `[0, 1]`, the resize equals bicubic + `align_corners=False` + `antialias=True` exactly and differs from bilinear or non-antialiased variants, and `VisionEncoder`'s internal resize is the same implementation (identity check on the imported helper, plus feature-level equality between native and pre-resized frames) |
| `vision_encoder.py` | Assert all SigLIP params have `requires_grad=False`, output shape `[2, 196, 576]` per camera, dtype `float32` |
| `language_backbone.py` | Assert vocab size == 49,152 (catches silent SmolLM revisions), tokenize+forward shape, projection trainable |
| `state_encoder.py` | Shape `[B, state_dim] → [B, 1, 576]`, GELU nonlinearity present |
| `fusion_transformer.py` | Causal mask is upper-triangular `-inf`, output shape == input shape, gradient flows (optional exercise module) |
| `vla_backbone.py` | `embed_inputs` returns `(input_embeddings, attention_mask, position_ids)` with every vector at the right observation-prefix position, padded batches keep the state position's logical index at `392 + L_valid`, and the ordinary `forward` returns `[B, 392+L+1, 576]`; integration-marked |

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
- **`transformers>=4.45`** — SigLIP-base and SmolLM2-135M both supported, and the `inputs_embeds` + `position_ids` path behaves consistently
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
