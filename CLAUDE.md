# lrm-code-chapter-3 — Claude Code project guide

You are working on **Chapter 3** of "Build a Large Robot Model (From Scratch)" (Manning).

## Chapter scope

This repo contains the code for **the VLA backbone**: a frozen SigLIP vision encoder + SmolLM-135M language backbone + state encoder + multimodal fusion transformer. Output: contextualized hidden states ready for an action head (Chapter 4 adds it).

## Locked architectural decisions (Ch 3 plan v3)

| Component | Choice | Source of truth |
|---|---|---|
| Vision encoder | SigLIP-base/16 (frozen) | `src/ch03/vision_encoder.py` |
| Language backbone | SmolLM-135M (**native tokenizer; no vocab expansion**) | `src/ch03/language_backbone.py` |
| Fusion | Concat + causal self-attention, 6 layers, 8 heads, dropout 0.1 | `src/ch03/fusion_transformer.py` |
| Hidden dim | 512 | All projections target 512 |
| Robot | SO-100 (6-DOF arm + 1 gripper) | Ch 2 hand-off |
| Camera input | `observation.images.up` only | Wrist camera added in Chapter 6 |
| State dim | **6** (5 SO-101 joint positions + gripper position; verified per Ch 2 pr-7 Table 2.2) | `src/ch03/state_encoder.py` |
| Action dim (consumed by Ch 4) | 6 (matches dataset action format) | Ch 4 owns |

**Deferred to Chapter 4** (do NOT introduce here):
- Vocab expansion (`tokenizer.add_tokens`, `model.resize_token_embeddings`)
- Action token reservation (256 bins × 6 dims = 1,536 IDs)
- Action head + autoregressive prediction loop

## Hand-off contracts

**From Chapter 2** (`from ch02 import make_pickplace_dataloader, normalize, denormalize`):
- Batches with `observation.state: (B, 7) float32` z-scored
- `observation.images.top: (B, 3, 224, 224) float32` in `[0, 1]`
- `observation.images.wrist: (B, 3, 224, 224) float32` (available; we use `top` only in Ch 3)
- `action: (B, 6) float32` z-scored — Ch 3 doesn't predict actions, just hands hidden states to Ch 4

**To Chapter 4**:
```python
from ch03 import VLABackbone
backbone = VLABackbone(hidden_dim=512)
hidden = backbone(image, instruction, state)
# hidden: [B, 196 + L + 1, 512]
```

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

- Locked component names: `vision_encoder`, `language_backbone`, `action_head`, `fusion_transformer` — never abbreviate or rename
- Banned: JAX, TensorFlow imports
- Line length: 76 chars (55 for annotated lines)
- Python 3.12, indent 4 spaces
- Tests must include shape and dtype assertions
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
