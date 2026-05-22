# Decision: Vision Encoder Choice for Chapter 3

**Date:** 2026-05-21
**Status:** Accepted
**Scope:** Chapter 3 backbone vision component; influences Chapters 4-10 since the backbone is reused.

## TL;DR

**Chosen:** SigLIP-base/16 at 224×224 input (`google/siglip-base-patch16-224`), frozen during Chapter 3.

## Constraints

The vision encoder must:
1. Be **pre-trained** at internet scale (we have 50 episodes of SO-100 data, not enough to train one)
2. Produce representations **aligned with language** (the policy must ground "red cube" in image regions)
3. Be **small enough** to load alongside SmolLM-135M on a Colab T4 (~12 GB)
4. Have **permissive license** (Apache 2.0 / MIT)
5. Have **clean HuggingFace Transformers integration** (no custom loader code)
6. Have **active maintenance** so the API isn't a moving target

## Options considered

| Option | Pre-trained | Language-aligned | Size | License | Verdict |
|---|---|---|---|---|---|
| **SigLIP-base/16 (Google 2023)** | ✅ ~10B image-text pairs | ✅ Sigmoid contrastive | 86M | Apache 2.0 | **Chosen** |
| CLIP-base (OpenAI 2021) | ✅ ~400M image-text pairs | ✅ Softmax contrastive | 86M | MIT | Outperformed by SigLIP at same params per Zhai et al. 2023 |
| DINOv2-small (Meta 2023) | ✅ Self-supervised | ❌ No language alignment | 22M | Apache 2.0 | Richer pure-visual features but no language grounding — wrong tool for VLA |
| ViT-base/16 ImageNet | ✅ ImageNet-21k classification | ❌ No language alignment | 86M | Apache 2.0 | Too generic; closed-vocabulary signal won't help for novel instructions |
| PaLI-X vision (Google) | ✅ | ✅ | Large | Closed | Not downloadable |
| SigLIP-2 (Google 2024) | ✅ improved | ✅ | 86M base | Apache 2.0 | Strong candidate; **deferred to a later revision** to keep this chapter pinned on the more widely-documented SigLIP-1 |
| EVA-02 (BAAI) | ✅ | ✅ | 86M+ | MIT | Less HF integration, more obscure for MQR readers |

## Why SigLIP wins

- **Sigmoid loss > softmax loss** at the same model size. Zhai et al. 2023 ablations show consistent improvement across batch sizes. The sigmoid loss treats each (image, text) pair independently, avoiding the dependence on giant negative-sample batches that CLIP requires.
- **Language alignment is decisive for VLA**. The reader's policy will train to map "pick the red cube" to motor commands. The vision encoder must already encode "red" and "cube" in vectors close to the language model's embeddings of those words. SigLIP delivers this; DINOv2 doesn't.
- **86M params fits the T4 budget** alongside SmolLM-135M (~270 MB) with room for the fusion transformer activations.
- **First-class HuggingFace Transformers integration** — `AutoModel.from_pretrained("google/siglip-base-patch16-224")` works out of the box.
- **Wide community adoption** — π0, PaliGemma, SmolVLA all use SigLIP. Reader familiarity helps when they read papers later.

## Frozen vs trainable

We freeze SigLIP throughout Chapter 3 (all parameters `requires_grad = False`). Rationale:
- Our dataset (50 episodes from `lerobot/svla_so100_pickplace`) is too small to fine-tune an 86M-param encoder without overfitting
- Production VLAs (RT-2, OpenVLA, π0) freeze the vision encoder at this stage too
- LoRA fine-tuning the encoder is a deliberate Chapter 6 exercise, not a Chapter 3 default

## What we explicitly defer

- **SigLIP-2** (2024 update) — better, but the documentation/community is thinner. Revisit if our v1 build hits a clear bottleneck.
- **Multi-camera fusion** — Chapter 3 uses `image_top` only. Wrist camera enters in Chapter 6.
- **LoRA fine-tuning the vision encoder** — Chapter 6.
- **DINOv2 as a complementary stream** (OpenVLA does this) — too much complexity for the build chapter.

## Trade-offs accepted

- **Slight image-text alignment ≠ instruction following**. SigLIP knows "red cube" means red cube, but it doesn't know what to do about it. That's why we still need the fusion transformer + action head.
- **Frozen means no domain adaptation in Ch 3**. SO-100 scenes are distribution-shifted from SigLIP's pre-training. Some patch attention noise is expected; Chapter 6 addresses it with LoRA.

## What would trigger re-evaluation

- SigLIP-base becomes unavailable on HuggingFace (unlikely)
- A clearly-superior model with the same constraints ships (SigLIP-2 promotion is possible)
- We hit a Track A/B compatibility issue (e.g., model file size grows past Colab disk limit)

## References

- SigLIP paper: Zhai et al. 2023, arXiv:2303.15343
- CLIP paper: Radford et al. 2021, arXiv:2103.00020
- DINOv2: Oquab et al. 2023, arXiv:2304.07193
- OpenVLA: Kim et al. 2024, arXiv:2406.09246
- π0: Black et al. 2024 (Physical Intelligence)
