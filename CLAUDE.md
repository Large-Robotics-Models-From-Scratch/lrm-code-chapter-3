# lrm-code-chapter-3 — Claude Code project guide

You are working on **Chapter 3** of "Build a Large Robot Model (From Scratch)" (Manning).

## Chapter scope

This repo contains the code for **the VLA backbone**: a frozen SigLIP vision encoder + SmolLM2-135M language backbone + state encoder, fused by unified embedding. The two camera views are projected and spliced into the language backbone's own token stream, so the pretrained backbone is the fuser - there is no separate fusion transformer on the main path. Output: contextualized hidden states ready for an action head (Chapter 4 adds it).

## Locked architectural decisions (Ch 3 plan v3)

| Component | Choice | Source of truth |
|---|---|---|
| Vision encoder | SigLIP-base/16 (frozen) | `src/ch03/vision_encoder.py` |
| Language backbone | SmolLM2-135M (**native tokenizer; no vocab expansion**) | `src/ch03/language_backbone.py` |
| Fusion | Unified-embedding fusion: image tokens spliced into the language backbone's stream via `masked_scatter`; the pretrained backbone fuses (no separate fusion module on the main path) | `src/ch03/vla_backbone.py` (source of truth); `fusion_transformer.py` kept only as the optional "separate-encoder fusion" exercise |
| Hidden dim | 576 | The language backbone's native width; all projections target 576 |
| Robot | SO-100 sim env (PickCubeSO100-v1); SO-101 teleop dataset (6-DOF: 5 arm + gripper) | Ch 2 hand-off |
| Camera input | both `observation.images.up` and `observation.images.side` (two cameras, 392 image tokens = 2 x 196) | Ch 2 hand-off |
| State dim | **6** (5 SO-101 joint positions + gripper position; verified per Ch 2 pr-7 Table 2.2) | `src/ch03/state_encoder.py` |
| Action dim (consumed by Ch 4) | 6 (matches dataset action format) | Ch 4 owns |

**Deferred to Chapter 4** (do NOT introduce here):
- Vocab expansion (`tokenizer.add_tokens`, `model.resize_token_embeddings`)
- Action token reservation (256 bins × 6 dims = 1,536 IDs)
- Action head + autoregressive prediction loop

> Note: the backbone grows its input embedding table by two inert placeholder rows for the image and state splice markers. This is **not** vocabulary expansion - the rows are overwritten by `masked_scatter` before the backbone runs, the tokenizer is untouched, and `config.vocab_size` stays 49152. The grown table is handed back with `set_input_embeddings`, so the model holds exactly **one** embedding table (a second, unread copy would add ~28.3M trainable parameters). Never reach for `resize_token_embeddings`: it rewrites `config.vocab_size`, which Ch 3 asserts.

> Note: `UnifiedEmbeddingBackbone` **composes** `VisionEncoder` (`self.vision_encoder = VisionEncoder(hidden_dim=576)`); it does not rebuild the vision path. The frozen SigLIP, the 768->576 projection, and SigLIP's `[0,1]`->`[-1,1]` pixel normalization all live in `vision_encoder.py` only. There is no `img_proj` on the backbone.

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
import torch
from ch03 import UnifiedEmbeddingBackbone

backbone = UnifiedEmbeddingBackbone()
text_ids = backbone.tokenize_instruction(instruction)  # L native SmolLM2 ids
sequence_ids = torch.tensor(
    [backbone.build_sequence_ids(text_ids)], dtype=torch.long
)  # [1, N]
hidden = backbone(images, sequence_ids, state)  # images: [B, 2, 3, 224, 224]
# hidden: [B, N, 576]  where N = 392 + L + 1
```

> **Cross-chapter API change (this PR):** the fused-layout builder is
> `build_sequence_ids` and the `forward` argument is `sequence_ids` (was
> `build_input_ids` / `input_ids`). This renames only OUR fused layout;
> Hugging Face's `input_ids` (tokenizer/model API) is untouched. The
> fused sequence length is `N = 392 + L + 1`. **Chapter 4's repo must
> adopt `build_sequence_ids` / `sequence_ids`** in the hand-off call
> above; Vatsal owns Ch 4 and needs to sync this contract.

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

- Locked component names: `vision_encoder`, `language_backbone`, `action_head`, `fusion_transformer` - never abbreviate or rename. Note: `fusion_transformer` is a valid name but only for the optional separate-encoder fusion exercise; the main fusion is the unified-embedding splice in `vla_backbone.py`
- Banned: JAX, TensorFlow imports
- Line length: 76 chars (55 for annotated lines)
- Python 3.12, indent 4 spaces
- Tests must include shape and dtype assertions. For the fused sequence, shapes are not enough: `tests/test_vla_backbone.py::test_splice_puts_every_vector_at_the_right_position` pins which vector lands at which position (a swapped camera or an interleaving reshape keeps every shape correct), so keep that test green rather than relaxing it
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
