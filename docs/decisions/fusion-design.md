# Decision: Multimodal Fusion Design for Chapter 3

**Date:** 2026-05-21
**Superseded:** 2026-06 (manuscript v5)
**Status:** Superseded by token-level fusion
**Scope:** Chapter 3 fusion; the backbone composed here is consumed by every subsequent code chapter.

> **Update (manuscript v5).** The shipped design is now **token-level fusion**: the two camera views (392 image tokens) and the state token are spliced with `masked_scatter` into the SmolLM2-135M backbone's own input embeddings, and the pretrained attention does the fusing. Hidden width is the backbone's native **576**, the sequence is `[image (392), language (L), state (1)]`, and the output contract is `[B, 392 + L + 1, 576]`. See `src/ch03/vla_backbone.py` (`VLABackbone`).
>
> The standalone from-scratch fuser described below is now **separate-encoder fusion**: a named alternative and optional exercise 3.4, not on the main path. It lives in `src/ch03/fusion_transformer.py` and is not imported by the backbone. The notes below are retained for that exercise's rationale.

## TL;DR (separate-encoder fusion, now the optional exercise)

A from-scratch causal transformer over a single sequence `[image_patches, lang_tokens, state]`, pre-norm, dropout 0.1. Retained as the named alternative the main path contrasts against.

## Constraints

The fusion mechanism must:
1. Mix vision, language, and state into one unified representation
2. Set up cleanly for Chapter 4's autoregressive action token prediction
3. Be implementable in standard PyTorch (`nn.TransformerEncoder`) without custom CUDA
4. Fit a Colab T4 budget alongside SigLIP + SmolLM
5. Match a production-VLA pattern so readers' mental model transfers when they read papers

## Options considered

| Option | Mechanism | Used by | Verdict |
|---|---|---|---|
| **Concat + causal self-attention** | All modalities in one sequence; standard self-attention; causal mask | RT-2, OpenVLA, GPT-4V | **Chosen** |
| Cross-attention (one-way) | Language queries attend to image keys | Flamingo, BLIP, π0 | More efficient at scale but more complex code; not needed for our scale |
| Cross-attention (bidirectional) | Both directions of cross-attention | BLIP-2 Q-former | Adds Q-former overhead; overkill for build chapter |
| Late fusion (separate towers, concat at end) | Independent encoders, no cross-talk | Older VLA prototypes | Loses cross-modal grounding mid-network |
| MoE / multimodal MoE | Routed expert blocks per modality | Some research VLAs | Out of scope for "from scratch" |

## Why concat-and-self-attend wins

- **Pedagogical simplicity**: one transformer block, one attention pattern. Reader who built an LM in PyTorch already has 80% of the code.
- **Matches RT-2 / OpenVLA**: when readers go on to study production papers, the mental model transfers
- **One attention map to debug**: easier to visualize what's attending to what (we exploit this in §3.4.5)
- **Causal mask sets up Ch 4 cleanly**: Chapter 4's action head appends action tokens to the right of the sequence and trains autoregressive prediction. Causal masking from Ch 3 means no architecture surgery later.

## Hyperparameter rationale

| Parameter | Value | Why |
|---|---|---|
| Hidden dim | 512 | Bridges SigLIP 768 + SmolLM 576; multiple of 64 for 8 attention heads |
| Layers | 6 | Fits T4 comfortably with our sequence length (~210 tokens). Ch 6 explores scaling. |
| Attention heads | 8 | Standard for `d=512`; 64-dim per head matches Llama/GPT conventions |
| Head dim | 64 | Standard |
| Dropout | 0.1 | Standard transformer regularization |
| Mask | Upper-triangular `-inf`, causal | Sets up Ch 4 autoregressive action prediction |
| Layer norm placement | Pre-norm | More stable training; standard for modern transformers |

## Causal vs bidirectional

Considered using bidirectional self-attention (every token sees every other token). Rejected because:
- Chapter 4 needs causal for autoregressive action token prediction
- Re-architecting between Ch 3 and Ch 4 would invalidate Ch 3 checkpoints
- Bidirectional gives no measurable advantage at our scale for this task

Trade-off accepted: image patch token N sees only image patch tokens 0..N-1, not the full image. This is unusual for vision (Vision Transformers normally use bidirectional attention internally). The bet: SigLIP has already done that bidirectional work — its output patch tokens are already fully contextualized within each image. Our fusion transformer's job is **cross-modal mixing**, not **intra-image attention**, and causal works fine for that.

## What we deliberately don't do

- **No cross-attention block**: simpler is better at our scale
- **No modality-specific layers**: every token goes through the same fusion transformer
- **No positional encoding on top of SigLIP's existing positional embeddings**: image patches keep SigLIP's positional info; language tokens keep SmolLM's positional info; the concat works because each modality is internally positionally aware before the fusion transformer sees them
- **No special token at boundary positions** (no `<image>`, `<lang>`, `<state>` markers): modality is implicit in position. Add markers in Chapter 6 if multi-camera fusion needs disambiguation.

## What would trigger re-evaluation

- A measurable failure mode in Chapter 4 (e.g., action token prediction collapses) traceable to insufficient cross-modal mixing
- A future chapter genuinely needing bidirectional attention (e.g., a VQA-only fine-tuning task)
- Sequence length grows past Colab T4 memory (e.g., adding two more cameras + longer instructions)

## References

- RT-2: Brohan et al. 2023, arXiv:2307.15818
- OpenVLA: Kim et al. 2024, arXiv:2406.09246
- Flamingo: Alayrac et al. 2022, arXiv:2204.14198 (cross-attention reference)
- BLIP-2: Li et al. 2023, arXiv:2301.12597 (Q-former reference)
- Attention rollout (for visualization basis): Abnar & Zuidema 2020, arXiv:2005.00928
