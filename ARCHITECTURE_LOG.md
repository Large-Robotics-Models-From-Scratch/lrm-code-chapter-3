# Architecture log — chapter 3

Cross-chapter decisions tracked here. Update with each significant architectural change. Maintained by `lrm-code-agents/chapter-continuity` agent.

## Locked decisions inherited from earlier chapters

| Decision | Set in | Notes |
|---|---|---|
| Robot platform | Ch 2 | SO-100, 6-DOF (5 arm + 1 gripper) |
| Sim engine | Ch 2 | ManiSkill3 + SAPIEN (Sid's choice; see Ch 2 `docs/decisions/simulator-choice.md`) |
| Anchor dataset | Ch 2 | `lerobot/svla_so101_pickplace` (50 ep, two cams, 30 FPS) |
| Dataset format | Ch 2 | LeRobotDataset v2.1 (Parquet + AV1) |
| Action dim consumed by Ch 4 | Ch 2 | 6 (dataset native action format) |
| State dim consumed by Ch 3 | Ch 2 | **6** (5 SO-101 joint positions + gripper position; verified per Ch 2 pr-7 Table 2.2) |
| Camera tensors (post-Ch 2 collate) | Ch 2 | `observation.images.up: [B, 3, 480, 640] float32` in `[0,1]`; `observation.images.side: [B, 3, 480, 640] float32` in `[0,1]` — Ch 3 resizes to (3, 224, 224) for SigLIP |
| Single task in dataset | Ch 2 | `"pink lego brick into the transparent box"` — `batch["task"]` is `list[str]`. Language conditioning meaningfully exercised from Ch 6 onward. |
| Sim env vs dataset distinction | Ch 2 | `PickCubeSO100-v1` is the SIM env (ManiSkill3 ships SO-100); `svla_so101_pickplace` is the DATASET (SO-101 real teleoperated); SO-101 hardware in Ch 9. Same 6-DOF action interface. |
| State / action normalization | Ch 2 | Z-scored at dataloader collate; Ch 3 receives normalized; Ch 3+ must call `denormalize` before `env.step()` |

## Locked decisions made in chapter 3 (v3 plan, 2026-05-21)

| Decision | Why | Affects downstream |
|---|---|---|
| Vision encoder = SigLIP-base/16 (224) | Image-text aligned, 196 patch tokens | Ch 4-10; see `docs/decisions/vision-encoder-choice.md` |
| Vision encoder frozen | Small dataset, inherit pretrain | Ch 4-5 (Ch 6 may LoRA-fine-tune) |
| Camera input = `observation.images.up` only | Single-stream simplicity | Ch 6 adds `observation.images.side` |
| Image preprocessing | Near-passthrough — Ch 2 already ships `[3, 224, 224]` float in `[0, 1]` | All later chapters |
| Language backbone = SmolLM-135M | Small, OSS, T4-fit | Ch 4-10 |
| **No vocab expansion in Ch 3** | Vocab expansion + action tokens belong to Ch 4 (action head story) | Ch 4 calls `add_tokens` + `resize_token_embeddings` as its first step |
| Hidden dim = 512 | Bridges SigLIP 768 + SmolLM 576; multiple of 64 for 8 heads | All later chapters |
| Attention heads = 8 (64-dim each) | Standard for d=512 | All later chapters |
| Fusion = concat + causal self-attention (RT-2 / OpenVLA pattern) | Simplest fusion supporting Ch 4's autoregressive generation | All later chapters; see `docs/decisions/fusion-design.md` |
| Fusion layers = 6 | Fits T4 with headroom | Ch 6 may scale |
| Dropout = 0.1 in fusion transformer | Standard regularization | All later chapters |
| Token order = `[image_patches (196), lang_tokens (L), state (1)]` | Causal attention left-to-right; Ch 4 appends action positions at the right | Ch 4 appends 6×K action token slots |
| Sanity check = prompted attention visualization (no training) | Replaces VQA training loop from v1/v2; lighter, same pedagogical payoff | — |

## Hand-off contracts

### To chapter 4

```python
from ch03 import VLABackbone

backbone = VLABackbone(hidden_dim=512)
hidden = backbone(image, instruction, state)
# image:       [B, 3, 224, 224]   top camera, preprocessed (already [0,1] from Ch 2)
# instruction: list[str]          batch of B instructions
# state:       [B, 6]             SO-101 state (5 joint positions + gripper) — verified per Ch 2 pr-7 Table 2.2
# hidden:      [B, 196 + L + 1, 512]
# tokenizer:   native SmolLM (49,152 vocab) — no expansion in Ch 3
```

Chapter 4's first step: expand the vocab with 1,536 action token IDs (256 bins × 6 dims) and call `model.resize_token_embeddings(50688)`. Then Ch 4's action head appends action token positions to the rightmost end of the sequence and trains autoregressive prediction over them.

### From chapter 2

```python
from ch02 import make_pickplace_dataloader, normalize, denormalize
from ch02.env import make_env  # optional, for eval rollouts

loader, stats = make_pickplace_dataloader(batch_size=32)
batch = next(iter(loader))
# batch["observation.images.up"]:   [B, 3, 480, 640] float32 in [0, 1]
# batch["observation.images.side"]: [B, 3, 480, 640] float32 in [0, 1]  (unused in Ch 3, Ch 6 adds)
# batch["observation.state"]:       [B, 6] float32 z-scored
# batch["action"]:                  [B, 6] float32 z-scored  (consumed by Ch 4, not Ch 3)
# batch["task"]:                    list[str] of length B (single instruction in this dataset)
```

Ch 3 does NOT predict actions — it produces backbone hidden states. The `action` key in the batch is for Ch 4 to learn from.

## Open decisions

- **state_dim resolved**: Ch 2 pr-7 Table 2.2 locks `observation.state: (6,)` and `action: (6,)`. `StateEncoder` should default to `state_dim=6`.
- **LoRA-fine-tune SigLIP in Ch 6** vs keep frozen all the way — Ch 6 author decides.
- **Scale fusion layers** (6 → 12 or more) in Ch 6 — Ch 6 author decides.
- **`observation.images.side` fusion pattern** (concat vs separate tower) — Ch 6 author decides.
- **Image renormalization for SigLIP** — does Ch 3's `preprocess.py` need to apply ImageNet mean/std on top of Ch 2's `/255`? Verify with the SigLIP preprocessor.
