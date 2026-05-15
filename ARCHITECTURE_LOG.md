# Architecture log — chapter 3

Cross-chapter decisions tracked here. Update with each significant architectural change. Maintained by `lrm-code-agents/chapter-continuity` agent.

## Locked decisions inherited from earlier chapters

| Decision | Set in | Notes |
|---|---|---|
| Robot platform | Ch 2 | SO-100, 6-DOF (5 arm + 1 gripper) |
| Sim engine | Ch 2 | MuJoCo via LeRobot |
| Anchor dataset | Ch 2 | `lerobot/svla_so100_pickplace` (50 ep, two cams, 30 FPS) |
| Dataset format | Ch 2 | LeRobotDataset v2.1 (Parquet + AV1) |
| Action dim | Ch 2 | 6 |
| State dim | Ch 2 | 6 |
| Camera tensors | Ch 2 | `image_top: [H, W, 3] uint8`, `image_wrist: [H, W, 3] uint8` |

## Locked decisions made in chapter 3

| Decision | Why | Affects downstream |
|---|---|---|
| Vision encoder = SigLIP-base/16 (224) | Image-text aligned, fits VQA | Ch 4-10 |
| Vision encoder frozen | Small dataset, inherit pretrain | Ch 4-5 (Ch 6 may LoRA-fine-tune) |
| Camera input = `image_top` only | Single-stream simplicity | Ch 6 adds `image_wrist` |
| Image preprocessing = center-crop + resize 224×224 | SigLIP input size | All later chapters |
| Language backbone = SmolLM-135M | Small, OSS, T4-fit | Ch 4-10 |
| Hidden dim = 512 | Bridge SigLIP 768 + SmolLM 576 | All later chapters |
| Fusion = concat + causal self-attention | Simplest fusion that supports generation | All later chapters |
| Fusion layers = 6 | Fits T4 with headroom | Ch 6 may scale |
| Action tokens reserved = 1,536 (256 × 6) | Discrete BC vocab in Ch 4 | Ch 4 uses these tokens |
| Token order = `[image, language, state, (action)]` | Causal attention left-to-right | Ch 4 appends action positions |

## Hand-off contracts

### To chapter 4

```python
from lrm_ch03_vla_backbone import VLABackbone

backbone = VLABackbone(hidden_dim=512)
hidden = backbone(image, instruction, state)
# image:       [B, 3, 224, 224]   top camera, preprocessed
# instruction: List[str]          batch of B instructions
# state:       [B, 6]             SO-100 joint positions
# hidden:      [B, 196 + L + 1, 512]
```

Chapter 4's action head reads the rightmost K positions and predicts action tokens in the reserved vocab range `[vocab_size - 1536, vocab_size)`.

### From chapter 2

```python
from lrm_ch02_simulation import SO100Env  # to be confirmed when Ch 2 repo lands

env = SO100Env()
obs, info = env.reset()
# obs["image_top"]:   [H, W, 3] uint8
# obs["image_wrist"]: [H, W, 3] uint8
# obs["state"]:       [6] float32
```

## Open decisions

- Whether to LoRA-fine-tune SigLIP in Ch 6 vs keep frozen all the way (Ch 6 author decides)
- Whether to scale fusion layers (6 → 12 or more) in Ch 6 (Ch 6 author decides)
- Whether `image_wrist` enters the fusion via concat or a separate vision tower (Ch 6 author decides)
