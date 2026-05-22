# lrm-code-chapter-3 — Claude Code project guide

You are working on **Chapter 3** of "Build a Large Robot Model (From Scratch)" (Manning).

## Chapter scope

This repo contains the code for **the VLA backbone**: a frozen SigLIP vision encoder + SmolLM-135M language backbone + state encoder + multimodal fusion transformer. Output: contextualized hidden states ready for an action head (chapter 4 adds it).

## Locked architectural decisions

| Component | Choice | Source of truth |
|---|---|---|
| Vision encoder | SigLIP-base/16 (frozen) | `models/vision_encoder.py` |
| Language backbone | SmolLM-135M | `models/language_backbone.py` |
| Fusion | Concat + causal self-attention, 6 layers | `models/fusion_transformer.py` |
| Hidden dim | 512 | All projections target 512 |
| Action tokens reserved | 1,536 (256 bins × 6 joint dims) | Vocab expansion in `language_backbone.py` |
| Robot | SO-100 (6-DOF) | `data/preprocess.py` |
| Camera input | `image_top` only | Wrist camera added in chapter 6 |

## Hand-off contracts

- **From chapter 2**: `lerobot/svla_so100_pickplace` dataset (50 episodes, two cameras at 480×640, 6-DOF state and action)
- **To chapter 4**: `VLABackbone.forward(image, instruction, state) → [B, 196+L+1, 512]`

## Code-style agents

This repo uses [`lrm-code-agents`](https://github.com/Large-Robotics-Models-From-Scratch/lrm-code-agents) for code style enforcement:

- `style-check`: line length, naming, banned abbreviations
- `chapter-continuity`: cross-chapter import resolution, locked names, tensor shape boundaries
- `listing-check`: Manning annotation format
- `test-gen`: pytest stubs with shape assertions
- `resource-check`: Colab T4 budget compliance

These run via the symlink at `.claude/agents/`. See `lrm-code-agents/README.md` for details.

## When editing code

- Locked component names: `vision_encoder`, `language_backbone`, `action_head`, `fusion_transformer` — never abbreviate or rename
- Banned: JAX, TensorFlow imports
- Line length: 76 chars (55 for annotated lines)
- Python 3.12, indent 4 spaces
- Test must include shape and dtype assertions
- All forward pass tensors must trace back to documented input shapes

## When writing prose for the book chapter

- Prose lives in `/Users/krishnamgupta/Desktop/projects/manning/lrm-book/chapter_3/`
- Use `/book` skill for linting (em dashes, marketing words, meta-language, code-listing format, captions)
- Reference `STYLEGUIDE.md` at the lrm-book repo root for locked terminology
- Erik (Manning DE) reviews after lint + structure pass

## Cross-references

- Wiki: `~/Desktop/wiki/lrm-book/ch3-outline-detailed.md` — full outline with concept explainers
- Book repo: `~/Desktop/projects/manning/lrm-book/chapter_3/chapter_3_structure_and_plan.md` — section-by-section plan
- Style guide: `~/Desktop/projects/manning/lrm-book/STYLEGUIDE.md`
