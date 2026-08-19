# Architecture log — chapter 3

Cross-chapter decisions tracked here. Update with each significant architectural change. Maintained by `lrm-code-agents/chapter-continuity` agent.

## Update (manuscript v5, 2026-06): unified-embedding fusion

The backbone is now `UnifiedEmbeddingBackbone` (was `VLABackbone` / "deep fusion"). Fusion is **unified-embedding fusion**: image and state tokens are spliced into the SmolLM2-135M backbone's own input embeddings with `masked_scatter`, and the pretrained attention fuses them. There is no separate fusion transformer on the main path.

| Changed | From (v3) | To (v5) |
|---|---|---|
| Backbone class | `VLABackbone` | `UnifiedEmbeddingBackbone` |
| Language backbone | SmolLM-135M (`HuggingFaceTB/SmolLM-135M`) | SmolLM2-135M (`HuggingFaceTB/SmolLM2-135M`) |
| Hidden width | 512 (bridge) | 576 (SmolLM2 native; no language projection) |
| Cameras | `up` only, 196 image tokens | `up` + `side`, 392 image tokens (cam0 then cam1) |
| Token order | `[image (196), lang (L), state (1)]` | `[image (392), lang (L), state (1)]` |
| Fuser | from-scratch `FusionTransformer` | the pretrained language backbone itself |
| Vision projection | inside `VisionEncoder` (768->512) | inside `VisionEncoder` (768->576); the backbone composes that encoder rather than rebuilding the vision path |
| Output contract | `[B, 196 + L + 1, 512]` | `[B, 392 + L + 1, 576]` |
| `fusion_transformer.py` | main path | optional exercise 3.4 (separate-encoder fusion); not imported by the main path |

The two placeholder ids (image, state) index two inert rows grown onto the input embedding table; they are spliced over before the backbone runs, so this is NOT the Chapter 4 vocabulary expansion (`resize_token_embeddings` / `add_tokens` stay Chapter 4's first step, still forbidden in Ch 3 source).

### Update (2026-07): compose, don't duplicate

Two reviewer-found duplications are gone. Both were invisible in the output shape, which is why the shape tests passed either way.

| Was | Now |
|---|---|
| The backbone built its own vision path: a bare frozen SigLIP wrapper (`load_siglip` / `FrozenSiglipFeatures`) plus a second `img_proj` 768->576 linear | The backbone composes `VisionEncoder` unchanged (`self.vision_encoder = VisionEncoder(hidden_dim=576)`); frozen SigLIP, the 768->576 projection, and SigLIP's `[0,1]`->`[-1,1]` pixel normalization all live in one place. `load_siglip` and `FrozenSiglipFeatures` are retired |
| Two embedding tables: the grown `self.embed_tokens` plus the language backbone's original table, both trainable, the original never read because `forward` passes `inputs_embeds` | One table: `self.language_backbone.set_input_embeddings(self.embed_tokens)` hands the grown table back. Deliberately not `resize_token_embeddings`, which would rewrite `config.vocab_size` |

Measured effect on `UnifiedEmbeddingBackbone()`:

| | Before | After |
|---|---|---|
| Trainable params | 163,607,040 | 135,295,488 |
| Frozen params | 92,884,224 | 92,884,224 |
| Total params | 256,491,264 | 228,179,712 |

The 28,311,552 trainable parameters removed are exactly one SmolLM2 embedding table (49,152 x 576). `config.vocab_size` stays 49,152 and the `[B, 392 + L + 1, 576]` contract is unchanged.

### Update (2026-07): real sample, position-level splice test, dtype-safe table growth

Three follow-ups, none of which change the architecture or the Ch 4 contract.

1. **One real Chapter 2 sample ships in the package.** `src/ch03/assets/` holds the two camera views of one `lerobot/svla_so101_pickplace` timestep as lossless PNGs (uint8, so `/255` reproduces the dataloader's float32 frames exactly) plus the z-scored state and the episode's task string. `from ch03 import load_sample` returns `(images [1, 2, 3, 480, 640] float32 in [0,1], state [1, 6] float32, instruction)`. The dataset stores state as float64; `load_sample` casts it. The real task string is `"pink lego brick into the transparent box"` (not the paraphrase earlier drafts used). The notebook now runs every listing on this sample: no `torch.rand`, no dataloader, no video decoder, no network. Measured on it: L = 9, output `[1, 402, 576]`.
2. **The splice is tested position by position.** Every earlier test asserted shapes, counts, and template order, all of which stay correct under a swapped camera, a transposed reshape, or interleaved batch rows. `test_splice_puts_every_vector_at_the_right_position` captures `inputs_embeds` with a forward pre-hook on the language backbone, then checks, for a batch of 2 with four distinct frames, that positions 0..195 are camera 0, 196..391 are camera 1, the next L are the embedding-table rows for the text ids, and the last is that row's state token. Verified to fail on both a camera swap and a batch-interleaving reshape.
3. **`_grow_embeddings` inherits the source table's dtype and device.** `nn.Embedding` defaults to float32; under transformers 5 SmolLM2 loads in bfloat16 and `forward` died with a dtype mismatch. No-op under this repo's `transformers<5` pin (parameter counts and all existing tests unchanged), so readers with a fresh install are covered either way.

### Update (2026-07): backbone renamed to `VLABackbone`, fusion term retired

A rename, nothing else. No behaviour change, no compatibility alias.

| Changed | From | To |
|---|---|---|
| Backbone class | `UnifiedEmbeddingBackbone` | `VLABackbone` |
| Fusion term in prose | "unified-embedding fusion" | "token-level fusion" (Yin et al., 2023, arXiv:2306.13549) |

The book has retired "unified-embedding fusion" as its term in favour of the established literature term. The class was named after the retired term, and "unified embedding" also collides with SigLIP's shared image-text embedding space, which section 3.2 spends pages on. `VLABackbone` makes no taxonomy claim, matches the chapter title, and aligns the class with its own module name `vla_backbone.py` and test file `test_vla_backbone.py`. Done now because Chapter 4 is still in draft and MEAP has not shipped.

The mechanism is untouched: image and state tokens are still spliced into the language backbone's own input embeddings with `masked_scatter` and the pretrained attention still does the fusing. Parameter counts (135,295,488 trainable / 92,884,224 frozen / 228,179,712 total), the `[B, 392 + L + 1, 576]` output contract, and all 52 tests are identical before and after.

**Chapter 4's repo must adopt `VLABackbone`.** No `UnifiedEmbeddingBackbone` alias ships - two names for one class is worse than a clean break, and nothing published depends on the old one.

The dated update sections above are records of their moment and keep the class name they were written with; only live contracts below were renamed.

The rows below predate the v5 update and describe the superseded v3 design; retained for history.

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
| Vision encoder frozen + pinned to `eval()` | Small dataset, inherit pretrain; `train()` override keeps SigLIP in eval so dropout never corrupts the "frozen" features during Ch 4 training | Ch 4-5 (Ch 6 may LoRA-fine-tune) |
| Frozen-vision visualization = **patch self-similarity** (not attention rollout) | Deep-research verdict (2026-06-15): on a CLS-less contrastive encoder, rollout has no CLS row to read and deep-layer token mixing means attention no longer points to input patches; self-similarity (cosine sim of a query patch to all patches, on the raw 768-dim SigLIP features) is crisper, more honest, less code, and transfers to the DINOv2 exercise. No `output_attentions`/eager needed — encoder always SDPA, returns `[B, 196, 576]` | Ch 3 figures 3.3/3.6; viz_similarity.py |
| Camera input = both `observation.images.up` and `observation.images.side` (392 image tokens) | Two views disambiguate depth/occlusion; matches Ch 2 dataset | All later chapters |
| Image preprocessing | Ch 2 ships `[3, 480, 640]` float in `[0, 1]`; `preprocess_image` resizes to `[3, 224, 224]`. SigLIP `[-1,1]` normalization lives inside `VisionEncoder` | All later chapters |
| Language backbone = SmolLM-135M, **trainable** (not frozen) | A small LM benefits from fine-tuning on instructions; Ch 4 trains it end-to-end with the action head. Only the vision encoder is frozen in Ch 3 | Ch 4 trains SmolLM; Ch 6 may LoRA |
| **No vocab expansion in Ch 3** | Vocab expansion + action tokens belong to Ch 4 (action head story) | Ch 4 calls `add_tokens` + `resize_token_embeddings` as its first step |
| Hidden dim = 512 | Bridges SigLIP 768 + SmolLM 576; multiple of 64 for 8 heads | All later chapters |
| Attention heads = 8 (64-dim each) | Standard for d=512 | All later chapters |
| Fusion = concat + causal self-attention (RT-2 / OpenVLA pattern) | Simplest fusion supporting Ch 4's autoregressive generation | All later chapters; see `docs/decisions/fusion-design.md` |
| Fusion layers = 6 | Fits T4 with headroom | Ch 6 may scale |
| Dropout = 0.1 in fusion transformer | Standard regularization | All later chapters |
| Token order = `[image_patches (196), lang_tokens (L), state (1)]` | Causal attention left-to-right; Ch 4 appends action positions at the right | Ch 4 appends 6×K action token slots |
| Sanity check = patch self-similarity (no training) | Frozen SigLIP already groups object regions: a cube-query patch lights up the cube. Replaces VQA training (v1/v2) and the prompted/object-attention rollout (v5) | — |

## Hand-off contracts

### To chapter 4

```python
from ch03 import VLABackbone

backbone = VLABackbone()
tokens = backbone.tokenizer(
    [instruction], return_tensors="pt", padding=True
)
hidden = backbone(
    images, tokens.input_ids, state, tokens.attention_mask
)
# images:     [B, 2, 3, H, W]  two cameras (overhead, side) in [0, 1];
#                              the vision module resizes to 224
# input_ids:  [B, L]           HF text ids only (pad token = eos)
# state:      [B, 6]           SO-101 state (5 joints + gripper), Ch 2 Table 2.2
# hidden:     [B, N, 576]      contextualized hidden states, N = 392 + L + 1
# tokenizer:  native SmolLM2 (49,152 vocab), never expanded
```

> **Naming:** the three public methods are `embed_inputs(...) ->
> (input_embeddings, attention_mask, position_ids)`,
> `contextualize(...) -> contextualized_hidden_states`, and
> `forward(...) -> contextualized_hidden_states`. `input_ids` is HF's
> text-only tokenizer output; the observation prefix (the assembled
> `N = 392 + L + 1` sequence) is built internally. Chapter 4's repo must
> adopt these names.

Chapter 4 consumes this contract unchanged. A one-shot head reads contextualized positions; the autoregressive and parallel heads extend the observation prefix between `embed_inputs` and `contextualize`, incorporating the returned `attention_mask` rather than building their own. No route expands the language vocabulary, so `add_tokens` and `resize_token_embeddings` stay out of both chapters.

### From chapter 2

```python
from ch02 import make_pickplace_dataloader, normalize, denormalize
from ch02.env import make_env  # optional, for eval rollouts

loader, stats = make_pickplace_dataloader(batch_size=32)
batch = next(iter(loader))
# batch["observation.images.up"]:   [B, 3, 480, 640] float32 in [0, 1]
# batch["observation.images.side"]: [B, 3, 480, 640] float32 in [0, 1]  (Ch 3 uses both cameras)
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
- **Image renormalization for SigLIP resolved**: `preprocess.py` resizes to 224 with `mode="bicubic"`, `align_corners=False`, `antialias=True`, and `VisionEncoder` applies SigLIP's 0.5 mean / 0.5 std `[0,1] -> [-1,1]` mapping. No ImageNet statistics. `tests/test_preprocess.py` pins the resize and the single shared implementation.

## 2026-08-09: Concat migration (placeholder IDs and masked_scatter retired)

The fused sequence is now built by direct concatenation of the encoded
streams, fed to SmolLM2 through `inputs_embeds`. Rationale: only the
language stream has vocabulary ids; the placeholder-ID + masked_scatter
splice taught tensor bookkeeping as if it were a fusion concept, and the
Chapter 4 parallel action head needs the pre-Transformer embeddings
anyway. Changes:

- `VLABackbone` exposes two stages: `embed_inputs(...)` -> (input
  embeddings, attention mask, mask-derived position ids), and
  `contextualize(...)` -> hidden states. `forward` chains them.
- No grown embedding table, no `set_input_embeddings`, no reserved ids.
  The table stays native at 49,152 rows.
- Padding is first-class: `tokenizer.pad_token = eos`, and position ids
  count valid slots only, so the state token's rotary position is
  392 + L_valid regardless of batch padding.
- `StateEncoder` returns `[B, 1, 576]` (sequence dim included).
- `VisionEncoder` accepts any input resolution and resizes internally
  via `preprocess_image` (matches manuscript 3.2.4).
- `tests/test_migration_parity.py` pins numerical equivalence between
  the retired splice and the concat construction; `test_guardrails.py`
  bans `masked_scatter` from `src/`.
- BREAKING for Ch 4: `tokenize_instruction` / `build_sequence_ids`
  removed; forward takes HF text `input_ids` + optional text mask.
