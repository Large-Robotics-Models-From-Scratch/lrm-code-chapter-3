# lrm-code-chapter-3 — Claude Code project guide

You are working on **Chapter 3** of "Build a Large Robot Model (From Scratch)" (Manning).

## Chapter scope

This repo contains the code for **the VLA backbone**: a frozen SigLIP vision encoder + SmolLM2-135M language backbone + state encoder, joined by direct concatenation. The two camera views and the state are projected to the language backbone's width and concatenated with the language input embeddings into one observation prefix, which the language backbone reads through `inputs_embeds` - so the language backbone is the fuser, and there is no separate fusion transformer on the main path. Output: contextualized hidden states ready for an action head (Chapter 4 adds it).

## Locked architectural decisions (Ch 3 plan v3)

| Component | Choice | Source of truth |
|---|---|---|
| Vision encoder | SigLIP-base/16 (frozen) | `src/ch03/vision_encoder.py` |
| Language backbone | SmolLM2-135M (**native tokenizer; no vocab expansion**) | `src/ch03/language_backbone.py` |
| Fusion | Direct concatenation: visual, language, and state input embeddings concatenated into one observation prefix, fed to the language backbone via `inputs_embeds`; the pretrained language backbone fuses (no separate fusion module on the main path, no placeholder IDs, no `masked_scatter`) | `src/ch03/vla_backbone.py` (source of truth); `fusion_transformer.py` kept only as the optional "separate-encoder fusion" exercise |
| Hidden dim | 576 | The language backbone's native width; all projections target 576 |
| Robot | SO-100 sim env (PickCubeSO100-v1); SO-101 teleop dataset (6-DOF: 5 arm + gripper) | Ch 2 hand-off |
| Camera input | both `observation.images.up` and `observation.images.side` (two cameras, 392 visual positions = 2 x 196) | Ch 2 hand-off |
| State dim | **6** (5 SO-101 joint positions + gripper position; verified per Ch 2 pr-7 Table 2.2) | `src/ch03/state_encoder.py` |
| Action dim (consumed by Ch 4) | 6 (matches dataset action format) | Ch 4 owns |

**Deferred to Chapter 4** (do NOT introduce here):
- Action head + autoregressive prediction loop
- Action discretization (256 bins × 6 dims), owned by Ch 4's own components

Never in either chapter: vocabulary expansion (`tokenizer.add_tokens`, `model.resize_token_embeddings`). The revised Ch 4 contract does not expand SmolLM2's vocabulary.

> Note (2026-08-09 concat migration): the placeholder-ID + `masked_scatter` splice is RETIRED. Only the language stream carries vocabulary IDs; image and state streams enter as vectors and the three are concatenated, so the embedding table stays native (49,152 rows, no grown copy, no `set_input_embeddings`). The only sanctioned copy of the old splice lives in `tests/test_migration_parity.py`, which pins numerical equivalence between the two constructions. `tests/test_guardrails.py` bans `masked_scatter` from `src/`. Never reach for `resize_token_embeddings`: it rewrites `config.vocab_size`, which Ch 3 asserts.

> Note: the backbone exposes **two representation levels**: `embed_inputs(images, input_ids, state, text_attention_mask=None) -> (input_embeddings, attention_mask, position_ids)` stops before the Transformer; `contextualize(...)` runs SmolLM2 over prepared embeddings via `inputs_embeds`; `forward` chains the two. Padding reuses SmolLM2's end-of-text token (`tokenizer.pad_token = eos`), and position IDs are derived from the attention mask so padded text slots never shift the state position's rotary index (state = 392 + L_valid).

> Note: `VLABackbone` **composes** `VisionEncoder` (`self.vision_encoder = VisionEncoder(hidden_dim=576)`); it does not rebuild the vision path. The frozen SigLIP, the resize to 224 (any input resolution accepted, per manuscript 3.2.4 "the vision module resizes"), the 768->576 projection, and SigLIP's `[0,1]`->`[-1,1]` pixel normalization all live in `vision_encoder.py` only. There is no `img_proj` on the backbone.

## Hand-off contracts

**From Chapter 2** (`from ch02 import make_pickplace_dataloader, normalize, denormalize`):
- Batches with `observation.state: (B, 6) float32` z-scored (5 arm joints + gripper)
- `observation.images.up: (B, 3, 480, 640) float32` in `[0, 1]` (Ch 3 resizes to 224x224)
- `observation.images.side: (B, 3, 480, 640) float32` in `[0, 1]` (Ch 3 now consumes BOTH cameras - up and side - each resized to 224x224)
- `action: (B, 6) float32` z-scored — Ch 3 doesn't predict actions, just hands hidden states to Ch 4
- `task: list[str]` — the instruction, fed to the language backbone

> **Real sample shipped in-repo:** Chapter 2's loader does not install alongside Chapter 3 (lerobot 0.5.1 wants huggingface-hub>=1.0, transformers<5.0 wants <1.0) and video decode needs FFmpeg, so one real timestep travels with the package: `from ch03 import load_sample` returns `(images [1, 2, 3, 480, 640] float32 in [0,1], state [1, 6] float32, instruction)`, backed by lossless PNGs in `src/ch03/assets/`. Use it instead of `torch.rand` in listings, notebooks, and demos. The dataset's real task string is `"pink lego brick into the transparent box"` - do not paraphrase it.

**To Chapter 4**:
```python
from ch03 import VLABackbone

backbone = VLABackbone()
tokens = backbone.tokenizer(
    [instruction], return_tensors="pt", padding=True
)  # HF input_ids [B, L] + attention_mask; pad token reuses eos

# Simple path (factorized head reads contextualized states):
hidden = backbone(
    images, tokens.input_ids, state, tokens.attention_mask
)  # images: [B, 2, 3, H, W] in [0, 1]; hidden: [B, N, 576], N = 392 + L + 1

# Extended path (parallel head appends action slots pre-Transformer):
embeddings, mask, position_ids = backbone.embed_inputs(
    images, tokens.input_ids, state, tokens.attention_mask
)
# ...append slot vectors to embeddings, extend mask/position_ids,
# build the block mask INCORPORATING `mask` (do not replace it), then
# run backbone.contextualize(...) or the language model directly.
```

> **Cross-chapter API change (concat migration, 2026-08-09):**
> `tokenize_instruction` and `build_sequence_ids` are REMOVED. The
> forward signature is now `(images, input_ids, state,
> text_attention_mask=None)` where `input_ids` is Hugging Face's
> text-only tokenizer output `[B, L]`, and `embed_inputs` returns a
> 3-tuple `(input_embeddings, attention_mask, position_ids)`.
> **Chapter 4 must adopt this contract**: its parallel head's custom
> block mask must incorporate the returned prefix `attention_mask`
> (padded instruction slots are invalid) rather than building its own
> from scratch, and its autoregressive head embeds action-token IDs
> through its own expanded table and concatenates at the embedding
> level. Vatsal owns Ch 4 and needs to sync this contract.

## Code-style agents

This repo uses [`lrm-code-agents`](https://github.com/Large-Robotics-Models-From-Scratch/lrm-code-agents):

- `style-check`: line length, naming, banned abbreviations
- `chapter-continuity`: cross-chapter import resolution, locked names, tensor shape boundaries
- `listing-check`: Manning annotation format
- `test-gen`: pytest stubs with shape assertions
- `resource-check`: Colab T4 budget compliance

Symlinked into `.claude/agents/` via the setup in `program.md` §2. See `../lrm-code-agents/README.md`.

## Reader-facing companion agent

`agents/chapter-03-guide.md` — a Claude Code subagent that walks the reader conversationally through the 7 listings. Symlinked into `.claude/agents/chapter-03-guide.md` at reader setup time. Scope is strictly Ch 3 — it declines to wander into Ch 4 material.

## When editing code

- Locked component names: `vision_encoder`, `language_backbone`, `action_head`, `fusion_transformer` - never abbreviate or rename. Note: `fusion_transformer` is a valid name but only for the optional separate-encoder fusion exercise; the main path fuses by direct concatenation in `vla_backbone.py`
- Banned: JAX, TensorFlow imports
- Line length: 76 chars (55 for annotated lines)
- Python 3.12, indent 4 spaces
- Tests must include shape and dtype assertions. For the observation prefix, shapes are not enough: `tests/test_vla_backbone.py::test_embed_inputs_layout` pins which vector lands at which position (a swapped camera or an interleaving reshape keeps every shape correct), so keep that test green rather than relaxing it. Likewise `tests/test_preprocess.py` pins the bicubic antialiased resize and the fact that `VisionEncoder` reuses the one `preprocess_image` implementation
- Save the tokenizer alongside the model if you persist anything (we use native SmolLM tokenizer; no expansion)

## When writing prose for the book chapter

- Prose drafts go in `../lrm-book/chapter_3/drafts/` (pre-Erik markdown), then Google Docs (post-Erik)
- Use `/book` skill for linting (em dashes, marketing words, meta-language, code-listing format, captions)
- Reference `../lrm-book/STYLEGUIDE.md` for locked terminology
- Erik (Manning DE) reviews after lint + structure pass

## Cross-references

- Wiki: `~/Desktop/wiki/lrm-book/ch3-outline-detailed.md` — full outline with concept explainers
- Book repo: `../lrm-book/chapter_3/chapter_3_structure_and_plan.md` — section-by-section plan (this repo's `docs/chapter_3_plan.md` is a synced copy)
- Style guide: `../lrm-book/STYLEGUIDE.md`
- Process doc: `../lrm-book/PROCESS.md`
- Operating manual for this repo: `program.md`
- Design doc for this repo: `docs/design.md`
- Decision records: `docs/decisions/{vision-encoder-choice,fusion-design}.md`
