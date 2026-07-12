# Chapter 3: Building the VLA Backbone — Structure & Content Plan (v3)

> **Superseded by v5 (shipped code)**: the prose below is a historical plan snapshot. The shipped backbone is `UnifiedEmbeddingBackbone` (not `VLABackbone`), uses `SmolLM2-135M`, hidden width **576** (the backbone's native width, no 512 down-projection), **two cameras** (`up` + `side`) for **392** image tokens, and fuses by splicing image and state tokens into the language backbone's own stream via `masked_scatter` (the pretrained backbone is the fuser; there is no separate fusion transformer on the main path - `fusion_transformer.py` is only the optional separate-encoder exercise). Output contract: `[B, 392 + L + 1, 576]`. Load-bearing facts are corrected inline below; where a section is clearly a historical snapshot, treat the v5 facts here as authoritative.

**Author**: Krishnam Gupta
**Code repo**: `Large-Robotics-Models-From-Scratch/lrm-code-chapter-3`
**Built on**: chapter 2 (`from ch02 import make_pickplace_dataloader, normalize, denormalize`; dataset = `lerobot/svla_so101_pickplace`; sim env = `PickCubeSO100-v1`)
**Hands off to**: chapter 4 (action head + action tokenization + vocab expansion all happen there)
**Last updated**: 2026-06-15 (v6 — frozen-vision visualization switched from attention rollout to **patch self-similarity**, after a co-author flagged rollout looked blurry and a deep-research pass confirmed it. Verdict: on a CLS-less contrastive encoder rollout has no CLS row and deep-layer token mixing breaks it; self-similarity (cosine sim of a query patch to all patches, on raw 768-dim SigLIP features) is crisper, more honest, less code, and transfers to the DINOv2 exercise. Code simplified — no `output_attentions`/eager; encoder always returns `[B, 196, 576]`. Figures 3.3/3.6 are self-sim grids (`viz_similarity.py`). Rollout demoted to a one-line mention. Exercise 3.1 now re-runs self-sim on DINOv2 + introduces PCA-to-RGB (DINOv2's canonical viz); new Exercise 3.4 = patch-to-text grounding with the MaskCLIP value-readout fix. Research report: deep-research wf_90df93b8-73d.)
**v5**: 2026-06-11 (object-tracking replaced prompted-attention; facts verified; +3 exercises; tokenization callout; SmolLM trainable / SigLIP frozen)
**Previously**: 2026-05-27 (v4 — aligned with Ch 2 verified facts: state_dim=6, camera keys `up`/`side`, dataset `svla_so101_pickplace`, image (3, 480, 640), SO-100 sim + SO-101 dataset framing)

## Archetype

**Primary**: Build chapter (the reader writes real code that runs at the end). Each section adds one component to a system that boots at chapter end.

Code is the spine. Prose explains why each piece exists and what to expect at each step.

---

## MQR alignment

Per `mqr/mqr.md`, the reader has:
- Intermediate Python (classes, functions, dicts, lists, pip)
- Basic deep learning (layers, weights, loss, backprop conceptually)
- Basic transformers + attention (will not need re-derivation)
- Basic PyTorch (training loops, tensors)
- No robotics, no compute clusters, no physical robot

**Concepts we introduce in this chapter** (each gets a defined moment + concept box or inline definition):

| Concept | First appears | Why MQR needs it explained |
|---|---|---|
| Pre-trained encoder (frozen) | 3.2.1 | "Frozen as deliberate design" is new |
| ViT patch tokenization | 3.2.3 | "Image as a sentence of patches" not in MQR baseline |
| SigLIP | 3.2.2 | Specific model, sigmoid contrastive loss |
| Attention rollout | 3.2.4 | Visualization technique not in MQR baseline |
| Projection layer (`nn.Linear` between dims) | 3.2.5 (inline) | Briefly named on first use |
| Decoder-style LM (causal attention) | 3.3.1 | MQR knows "transformer" but not generation specifics |
| SmolLM tokenizer (native, no expansion in ch 3) | 3.3.3 | One-paragraph mention; full BPE + vocab expansion treatment moved to ch 4 |
| Cross-attention vs concat-and-self-attend | 3.4.1 | Two ways to mix modalities |
| State encoder MLP (projection + nonlinearity) | 3.4.2 (inline) | Brief note on when projection uses MLP vs single linear |
| Causal mask | 3.4.3 | Why ch 4's action tokens will be left-to-right predictable |
| Prompted attention visualization | 3.4.5 | Inference-time probe; no training, no VQA classifier |

**Deferred to ch 4** (do not introduce in this chapter):
- BPE tokenizer mechanics in depth
- Vocab expansion (`add_tokens`, `resize_token_embeddings`)
- Action tokens (quantization, 256 bins, 1,536 reserved IDs)
- Action-head autoregressive prediction loop
- VQA-as-probing-classifier framing (we may not need it at all if ch 4 demonstrates alignment via action loss)

Reader who already has these from prior reading should skim the concept boxes. Reader who doesn't should rely on them.

**Concepts we explicitly do NOT cover** (deferred or assumed):
- Self-attention math derivation (assumed from MQR)
- Backprop mechanics (assumed)
- Specific autoregressive prediction loop for action tokens (deferred to chapter 4)
- Continuous flow matching (deferred to chapter 5)
- Quantization for inference (deferred to chapter 10)

## Reader tracks (from `wiki/lrm-book/reader-tracks.md`)

This chapter runs end-to-end on **all three tracks**:
- **Track A (laptop CPU)**: forward passes are slow but functional; VQA training step takes ~30 min on CPU
- **Track B (Colab Pro / consumer GPU)**: full pace; VQA training step ~3 min
- **Track C (full hardware)**: same as A or B for chapter 3 — hardware only matters from chapter 9

## Before you start (chapter prerequisite check)

Reader should have:
- Completed chapter 2
- Cached `lerobot/svla_so101_pickplace` locally (one-line load in chapter 2)
- Working `SO100Env` import from chapter 2 repo
- ~5 GB free disk (for SigLIP + SmolLM model weights)
- Optional: GPU with ≥6 GB VRAM (Colab T4 works; chapter still completes on CPU)

---

## Chapter narrative arc

The reader walks in knowing they have a robot in sim and a dataset of teleoperated pick-place episodes. They walk out having built a 0.5B-parameter neural network that takes an image, an instruction, and the robot's joint positions, and produces a fused contextual representation ready for an action head.

Three stories woven together:
1. **Build story**: load a vision encoder → load a language model → fuse them → wire in robot state
2. **Intuition story**: a VLA is a transformer that "speaks" three modalities and will soon "speak" a fourth
3. **Validation story**: VQA pretraining proves the backbone learned alignment before any action head is attached

---

## Chapter Opening

### "This chapter covers" block (4 bullets, ≤45 chars each)

```
- Loading a frozen vision encoder
- Wiring in a small language backbone
- Fusing image, language, and state tokens
- Visualizing language-driven attention
```

### Hook paragraphs (2 paragraphs)

- **Paragraph 1**: Open with the SO-100 from chapter 2. A red cube sits on the table. The instruction is "pick up the red cube." Three streams of information are about to meet inside one neural network: an image, a sentence, and six numbers. Out the other side will come motor commands.
- **Paragraph 2**: Bridge: a VLA is not a new species of model. It is a Transformer that processes a sequence built from three modalities. Each modality contributes tokens. Self-attention does the rest. Reader who has built a small language model in PyTorch already has 80% of the muscle memory needed.

---

## Section 3.1: The VLA paradigm — vision + language + state → action

**Pages**: 3-4
**Purpose**: Frame the chapter. Establish the three-stream mental model. Show what gets built.

**Subsections**:
- 3.1.1 Opening scenario (1 paragraph) — concrete SO-100 + cube + instruction
- 3.1.2 What is a VLA, mechanically (2 paragraphs) — Transformer over a multimodal token sequence
- 3.1.3 The three streams (1-2 paragraphs + table)
- 3.1.4 What this chapter builds (1 paragraph + figure 3.1)

**Figure 3.1**: Three-stream VLA pipeline overview. Inputs (image, instruction, state) → encoders → fusion → contextualized hidden states → dashed placeholder for action head. Recall figure 1.7 from chapter 1; this chapter constructs what that diagram described.

**Intuition signpost** (close section): "If you've built a sequence model that predicts the next word, you're 80% of the way to a model that predicts the next motor command. The remaining 20% is what we add in this chapter and the next."

**Reader state at end**: Knows what they're building, knows the three input streams and one output stream, knows the action head is deferred.

---

## Section 3.2: The eyes — vision encoder

**Pages**: 6-8
**Purpose**: Build the vision component. Establish the pre-trained-and-frozen pattern. Validate the encoder visually.

**Subsections**:
- 3.2.1 Why pre-trained, why frozen (1 page) — honest about what's inherited
- 3.2.2 SigLIP in 2 paragraphs (concept box)
- 3.2.3 Image → patches → tokens (1-2 pages, includes listing 3.1)
- 3.2.4 Visualizing what SigLIP "sees" (1-2 pages, includes listing 3.2 + figure 3.3)
- 3.2.5 Projecting to the common hidden dim (½ page) — justify "why 512" here + name the projection layer concept on first use ("a projection layer is just a single `nn.Linear(768, 512)` that translates SigLIP's native width into our common width")

**Concept boxes** (3, down from 4):
- Pre-trained-and-frozen (3.2.1) — "what we inherit and what we don't"
- ViT patch tokenization (3.2.3) — "your image as a sentence of patches"
- Attention rollout (3.2.4) — "the model's gaze, made visible"

(SigLIP merged into prose in 3.2.2, not a separate concept box.)

**Listings**:
- 3.1: Loading and freezing SigLIP (~30 lines)
- 3.2: Attention rollout visualization (~50 lines)

**Figures**:
- 3.2: Image → patch grid → token sequence (process diagram)
- 3.3: Patch self-similarity grid. One SO-100 frame; query a cube patch and an arm patch; each lights up its object (cube query -> cube, arm query -> arm). Shows the frozen encoder groups object regions with no training. Caption marks the query patches by position, not color.

**Honesty aside** (in 3.2.1): "From scratch" does not mean we re-derive ImageNet. We download SigLIP's weights and inherit ~400M images of pretraining. What we build from scratch is the wiring around it — how patches flow into our fusion transformer, how attention is masked, how the projection to our common width works.

**Intuition signpost** (close section): "The frozen vision encoder is the cheapest superpower we'll have. Every patch position now carries semantic information about what's there — cube, gripper, table — without us having trained anything yet."

**Reader state at end**: Has loaded a pre-trained vision encoder, understands patch tokenization, has visual proof the encoder attends to relevant objects, understands why we freeze it. Has 196 image tokens per frame, each 512-dim.

---

## Section 3.3: The brain — language backbone

**Pages**: 3-4 (down from 5-7 in v2 — vocab expansion moved to ch 4)
**Purpose**: Build the language component using SmolLM's native tokenizer. No vocab expansion, no action tokens — those live in ch 4 where the action head story plays out.

**Subsections**:
- 3.3.1 Why a language model, not just a text encoder (½ page) — causal generation as the goal
- 3.3.2 SmolLM2-135M in 2 paragraphs (inline, no separate concept box)
- 3.3.3 Tokenize the instruction (½ page) — use SmolLM's native tokenizer, no modifications
- 3.3.4 Forward pass at native 576 (1-2 pages, includes listing 3.3 + figure 3.4)

**Concept boxes** (0 in this section — all tokenization theory moves to ch 4):
- (none — keep section lean)

**Listings**:
- 3.3: Loading SmolLM2, tokenizing an instruction, forward pass at native 576 (~50 lines)

**Figures**:
- 3.4: Token flow through SmolLM. Show "pick up the red cube" → BPE token IDs → SmolLM hidden states → projection to 512-dim. No reserved-slot annotations — ch 4 will add that diagram when it expands the vocab.

### Light tokenizer mention (3.3.3, one paragraph)

> SmolLM uses **Byte-Pair Encoding (BPE)** — a subword tokenization scheme that breaks `"pick up the red cube"` into about 5 integer token IDs. We use the tokenizer as-is in this chapter. Chapter 4 will expand SmolLM's vocabulary to make room for action tokens; for now, we just need to encode an instruction and pass it through the language backbone.

### Hand-off note to ch 4 (one sentence at end of 3.3)

> Chapter 4 will modify SmolLM's vocabulary to add 1,536 action token IDs. The change is one call to `tokenizer.add_tokens` plus one to `model.resize_token_embeddings`, but the *use* of those tokens — and the full action prediction loop — belongs with the action head story in chapter 4.

**Intuition signpost** (close section): "You have a language backbone that turns an instruction into 512-dim hidden states. Chapter 4 will teach it new words — words that mean motor commands. For now, it's a small but real LLM, projected to our common width."

**Reader state at end**: Has a working language backbone. Native SmolLM tokenizer encodes the instruction. Output is `[B, L, 576]` per-token hidden states. No vocab modifications, no action tokens. Save/load uses the native SmolLM2 tokenizer.

---

## Section 3.4: Multimodal fusion + state + prompted attention visualization

**Pages**: 6-8 (down from 8-10 in v2 — VQA training replaced by lighter visualization)
**Purpose**: Compose the three streams. Validate the composition with a prompted attention visualization — no training, just inference + rollout to see which image patches light up for different instructions.

**Subsections**:
- 3.4.1 The fusion question (1 page) — cross-attention vs concat-and-self-attend (cite RT-2 for concat choice)
- 3.4.2 The state encoder (1 page, includes listing 3.4) — start here for roboticist on-ramp; add a 2-sentence note on why this projection uses a 2-layer MLP (`Linear → GELU → Linear`) instead of a single Linear: going from 6 dims up to 512 benefits from a nonlinearity in the middle
- 3.4.3 The fusion transformer (2 pages, includes listing 3.5) — causal mask preview
- 3.4.4 Composing the VLABackbone (1-2 pages, includes listing 3.6 + figure 3.5)
- 3.4.5 Visualizing what the backbone sees, given a prompt (1-2 pages, includes listing 3.7 + figure 3.6)

**Concept boxes** (2):
- Cross-attention vs concat-and-self-attend (3.4.1) — "two ways to mix modalities, we pick the simpler one"
- Causal mask + autoregressive preview (3.4.3) — explains the mask AND why it sets up ch 4

(State encoding inlined; prompted attention viz inlined in 3.4.5.)

**Listings**:
- 3.4: StateEncoder (~25 lines)
- 3.5: FusionTransformer (~80 lines, the meatiest listing)
- 3.6: VLABackbone (~80 lines, the integration)
- 3.7: Prompted attention visualization (~30 lines — pass image through backbone with several instructions, run attention rollout per prompt, save overlay plots)

**Figures**:
- 3.5: Multimodal token layout in fusion transformer. Show 196 image positions, L language positions, 1 state position. Indicate causal attention as left-to-right arrows. Caption labels positions by index range, not color. (No dashed action positions in this figure — ch 4 will introduce them.)
- 3.6: Object-tracking self-similarity grid. Three SO-100 frames where the cube spawns in different positions; query the cube patch in each (the cube moves, so the query moves), and the highlight follows the cube. Shows the frozen encoder localizes the object wherever it spawns — the reason we can freeze it. (v6: now patch self-similarity, not rollout. The "language doesn't steer vision yet" point is a one-line prose caveat, not a figure.)

### Hidden dim justification (in 3.4.1 or 3.2.5)

> Why 576? SigLIP outputs 768-dim hidden states; SmolLM2-135M's native width is 576-dim. We want one common width so all streams can sit in the same sequence and be processed by the language backbone. We use the backbone's own 576-dim width: SigLIP and the state encoder each project into 576 (cheap linear layers), and the image/state tokens are spliced straight into the backbone's input-embedding stream. Using the backbone's native width means no down-projection bottleneck and the pretrained backbone does the fusing. Larger production VLAs use 768, 1024, or higher.

### Beefed-up causal mask explanation (3.4.3)

> A standard transformer's self-attention lets every token see every other token. **Causal attention** restricts each token to seeing only tokens that came before it (left-to-right). We use causal here for one reason: chapter 4 will train the model to predict action tokens autoregressively, one at a time, where each action token depends only on the context before it. If we used bidirectional attention now, we'd have to re-architect for Ch 4. By being causal from the start, the action prediction story drops in cleanly.
>
> **Preview of Ch 4** (one paragraph, no code yet): the action head will append 6 action token positions to the right end of the sequence. With causal masking, position `[image+lang+state+0]` predicts the first action token using image+language+state context. Position `[image+lang+state+1]` predicts the second using everything before plus the first action token. And so on, six positions, six joint commands. The model trains via standard cross-entropy loss on the predicted action token IDs.

### Object-tracking visualization (3.4.5) — patch self-similarity across frames

> We've built a `VLABackbone`. Before attaching an action head, it's worth seeing what the frozen vision encoder already does for us. Take three frames where the cube spawns in different positions, query the cube's patch in each, and measure its self-similarity to every other patch. The highlight follows the cube every time. With no training on our part, SigLIP already localizes the object wherever it spawns — that training-free spatial grounding is exactly why we can freeze the vision encoder and build the policy on top of it.
>
> One honest caveat, in a sentence: the *instruction* does not yet steer where the model looks. The vision encoder never reads the language. Wiring language in is what chapter 4's end-to-end training does. We don't draw a figure of that non-effect; we name it and move on.

### Why patch self-similarity (decision rationale, v6 — research-backed)

Lineage: v4 used a prompted-attention figure (could not reproduce on an untrained backbone); v5 switched to object-tracking via **attention rollout**; a co-author then showed rollout looked blurry on SigLIP, and a deep-research pass (wf_90df93b8-73d, 23/25 claims verified 3-0) confirmed why: rollout was designed to read off a **class token**, which SigLIP has none of, and after a dozen layers of token mixing the attention weights no longer point back to input patches. Patch **self-similarity** reads the frozen 768-dim patch features directly — the most honest view of what the encoder represents, the least code, and it transfers unchanged to the DINOv2 exercise. Rollout drops to a one-line mention with the correct rationale (NOT the refuted "identity/residual smoothing" story). See `resources/deviations.md` D1.

**Intuition signpost** (close section): "You have a VLA backbone. It takes an image, a sentence, and six numbers, and produces a sequence of contextualized hidden states. The frozen vision encoder already finds the cube wherever it sits. Chapter 4 attaches an action head to the right end of this output and teaches the instruction to steer it."

### Optional exercises (3.x) — Manning-style, parity with chapter 2

Chapter 2 ships four "Optional Exercise" boxes; chapter 3 includes 2-3:

- **Exercise 3.1 — Swap SigLIP for DINOv2.** Load `facebook/dinov2-base` in place of SigLIP and re-run the same patch self-similarity. DINOv2 is self-supervised and its dense features are crisper (it has no language alignment, which is why the book uses SigLIP). Then try DINOv2's signature visualization: a PCA of the patch features rendered as RGB, with first-PC thresholding for a foreground mask (Oquab et al. 2023). Honest note for the prose: single-image PCA separates foreground from background, not one object from another.
- **Exercise 3.2 — Add the wrist camera.** The Ch 2 batch also carries `observation.images.side`. Encode it as a second image stream and watch the fused sequence grow to `196 + 196 + L + 1`. Previews the multi-camera fusion chapter 6 builds.
- **Exercise 3.3 — Inspect BPE.** Tokenize five instructions with SmolLM's tokenizer and print how each splits into subword IDs. Connects to the tokenization callout in 3.2.3.
- **Exercise 3.4 — Ground a word to patches (on-thesis).** SigLIP is image-*text* aligned, so you can compute cosine similarity between each patch and the text embedding of "red cube" from SigLIP's text tower. Observe it is noisy and poorly localized: contrastive pre-training trains a single global pooled embedding, not dense patch-text correspondence (vanilla CLIP scores only ~3-6% mIoU at 224/14x14). Then apply the training-free MaskCLIP fix (drop the pool head's query/key projections and read each location's value feature) and watch it sharpen (Zhou et al. 2022, MaskCLIP). This is the clearest demonstration of the vision-language grounding that motivates choosing SigLIP for a VLA.

### Tokenization callout (concept box, 3.2.3) — "everything is tokens"

Place right after "your image as a sentence of patches." Routes through the three modalities the book builds, tied to chapter 1's "motion as a modality," with one clause nodding to the broader LRM family. Do NOT illustrate audio (the book doesn't build it, and audio tokenization genuinely differs from image patching — overclaiming risk).

> **Tokenization is universal.** You just turned an image into patch tokens. Text is already tokens (SmolLM's BPE). And in chapter 4, a motor command becomes an action token. One Transformer processes all three because it never knows or cares which modality a token came from — exactly the "motion as a modality" idea from chapter 1. The same trick extends to audio, LIDAR, and other sensors across the broader family of large robotics models, though each modality tokenizes differently.

(If a figure is wanted, show the three modalities the book builds — image patches + text tokens + action tokens → one Transformer — not an audio waveform.)

**Reader state at end**: Has a complete `VLABackbone` class. Has visual evidence (a 3-panel attention grid) that language conditioning shapes which image patches the backbone attends to. Understands the token layout. Knows why causal masking matters for ch 4. Ready to attach an action head.

---

## Section 3.5: Summary + In the wild

**Pages**: 1-2

### Summary bullets (Manning convention: complete sentences, abstract takeaways)

```
- A VLA backbone composes pre-trained vision and language models with a fusion
  transformer that processes a single multimodal token sequence.
- Freezing the vision encoder lets a small dataset benefit from large-scale
  pre-training without overfitting.
- Causal self-attention over a concatenated sequence is the simplest fusion that
  generalizes from text-only LLMs to multimodal models, and sets up the
  autoregressive action prediction that chapter 4 attaches.
- A prompted attention visualization (no training) demonstrates the frozen
  pre-trained components route attention to instruction-relevant image patches
  before any action head is attached — diagnostic evidence the backbone works.
- The SO-100 sim environment, the SO-101 dataset, and the SO-101 hardware share
  the same 6-DOF observation/action interface, so policies transfer across all
  three with calibration handled in chapter 9.
```

### In the wild sidebar: production VLA backbones

| Model | Backbone size | Vision | Language | Notes |
|---|---|---|---|---|
| Yours (after this chapter) | ~0.5B | SigLIP-base (frozen) | SmolLM2-135M | First-principles build |
| OpenVLA | 7B | DINOv2 + SigLIP | Llama-2-7B | Fine-tuned, OXE-trained |
| RT-2-X | 55B | PaLI-X vision | PaLM-X | Largest, Google internal |
| π0 | 3B | SigLIP-large | PaliGemma | Fine-tuned for flow matching |
| OpenpiZero | 3B | SigLIP-large | PaliGemma | Open reimplementation of π0 |

Same anatomy. You scaled differently.

---

## Architectural decisions (locked)

| Decision | Choice | Why |
|---|---|---|
| Vision encoder | SigLIP-base/16 (224) | Image-text aligned, 196 patch tokens per camera |
| Camera input | both `observation.images.up` and `observation.images.side` (two cameras) | 392 image tokens = 2 x 196 |
| Image preprocessing | Resize 480×640 → 224×224 (Ch 2 ships native dataset resolution; Ch 3 resizes for SigLIP) | Match SigLIP input size |
| Language backbone | SmolLM2-135M | Small, OSS, runs on T4, good tokenizer |
| Tokenizer changes in ch 3 | **None** — native SmolLM2 tokenizer | Vocab expansion moved to ch 4 |
| Fusion mechanism | Unified embedding: image and state tokens spliced into the backbone's stream via `masked_scatter` (the pretrained backbone fuses) | No separate fusion module on the main path; `fusion_transformer.py` is the optional separate-encoder exercise |
| Hidden dim | 576 | The language backbone's native width; all streams project to 576 |
| Attention heads | (backbone native) | Provided by the pretrained SmolLM2-135M backbone |
| Fusion transformer layers | n/a on main path | Backbone is the fuser; the 6-layer fusion transformer is the optional exercise only |
| Dropout | n/a on main path | The separate-encoder exercise uses 0.1 |
| State dim | **6** (5 SO-101 joint positions + gripper position; verified per Ch 2 Table 2.2) | SO-100/101 native; matches dataset |
| Action dim (consumed by Ch 4) | 6 (matches dataset action format) | Ch 4 owns the action head |
| Sim env framing | `PickCubeSO100-v1` (ManiSkill3 ships SO-100 in sim) | SO-100 sim + SO-101 dataset + SO-101 hardware all share 6-DOF interface (per Ch 2 callout) |

**Deferred to ch 4** (was previously in ch 3 v2):
- Action tokens, 256 bins per joint, 1,536 reserved vocab IDs
- `tokenizer.add_tokens` + `model.resize_token_embeddings`
- Save/load coupling between tokenizer and model

---

## Hand-off contract to chapter 4

```python
import torch
from ch03 import UnifiedEmbeddingBackbone

backbone = UnifiedEmbeddingBackbone()
text_ids = backbone.tokenize_instruction(instruction)
sequence_ids = torch.tensor(
    [backbone.build_sequence_ids(text_ids)], dtype=torch.long
)  # [1, N]
hidden = backbone(images, sequence_ids, state)
# images:       [B, 2, 3, 224, 224]  two cameras, resized from Ch 2's (3, 480, 640) by ch03.preprocess
# sequence_ids: [B, N] long tensor   image + text + state template rows
# state:        [B, 6]               SO-101 joint positions (z-scored by ch02 collate)
# hidden:       [B, N, 576]          N = 392 + L + 1
# tokenizer:    native SmolLM2 (49,152 vocab) — no expansion in ch 3
```

Ch 3 imports from Ch 2 via:
```python
from ch02 import make_pickplace_dataloader, normalize, denormalize
```
The dataloader yields `(3, 480, 640)` float32 images in `[0, 1]`. Ch 3's `preprocess.py` resizes to `(3, 224, 224)` for SigLIP. The `task` field per batch is a `list[str]` from Ch 2's collate; Ch 3 uses it as the language input.

Chapter 4 owns the action half of the story:
- Expand SmolLM's vocab by 1,536 action token IDs
- Call `model.resize_token_embeddings` to grow the embedding table
- Append action token positions to the rightmost end of the sequence
- Train autoregressive prediction over them with cross-entropy loss
- De-tokenize predicted action token IDs back to continuous motor commands

The clean handoff means a ch 3 checkpoint is the "backbone only" — ch 4's first step is the vocab/embedding expansion.

---

## Code repository layout

```
lrm-code-chapter-3/
├── README.md
├── CLAUDE.md                       # router for lrm-code-agents
├── pyproject.toml
├── ARCHITECTURE_LOG.md             # cross-chapter decisions log
├── .lrm-agents.yml                 # config overrides
├── .claude/
│   ├── CLAUDE.md -> ../lrm-code-agents/CLAUDE.md
│   └── agents/  -> ../lrm-code-agents/agents/
├── models/
│   ├── __init__.py
│   ├── vision_encoder.py           # ~80 LOC
│   ├── language_backbone.py        # ~80 LOC (slimmer — no vocab expansion)
│   ├── state_encoder.py            # ~30 LOC
│   ├── fusion_transformer.py       # ~150 LOC
│   └── vla_backbone.py             # ~100 LOC
├── data/
│   └── preprocess.py               # ~60 LOC
├── viz_attention.py                # ~80 LOC
├── viz_prompted_attention.py       # ~50 LOC — replaces train_vqa.py
└── tests/
    ├── test_vision_encoder.py
    ├── test_language_backbone.py
    ├── test_fusion_transformer.py
    └── test_vla_backbone.py
```

Total reader-facing code: ~270 LOC + tests ~120 LOC.

**Removed from v2 layout**:
- `train_vqa.py` (~200 LOC) — VQA training loop replaced by lighter prompted attention viz
- `tokenize_demo.py` (~40 LOC) — moves to ch 4 repo with the vocab expansion content
- `tests/test_vocab_expansion.py` — moves to ch 4 repo

---

## Citations to verify before final draft

| Claim | Source to cite |
|---|---|
| SigLIP — sigmoid loss for image-text pretraining | Zhai et al. 2023 (arxiv 2303.15343) |
| SmolLM family + 135M variant | HuggingFace blog 2024 |
| RT-2 concat-and-self-attend pattern | Brohan et al. 2023 (arxiv 2307.15818) |
| OpenVLA 7B | Kim et al. 2024 (arxiv 2406.09246) |
| π0 flow matching VLA | Black et al. 2024 (Physical Intelligence) |
| Probing classifier framing for VQA | Conneau et al. 2018 (arxiv 1805.01070); Tenney et al. 2019 (arxiv 1905.06316) |
| 256 action bins convention | RT-2 + OpenVLA both use 256 |
| ViT patch tokenization | Dosovitskiy et al. 2020 (arxiv 2010.11929) |
| BPE tokenization | Sennrich et al. 2016 (arxiv 1508.07909) |
| DINOv2 (briefly mentioned) | Oquab et al. 2023 (arxiv 2304.07193) |

## Facts to verify — VERIFIED 2026-06-11 (live model load + tests)

- SmolLM2-135M hidden dim = **576** ✓ (confirmed loading the model; this native 576 is the common width)
- SmolLM2-135M vocab size = **49,152** ✓ (asserted in `test_language_backbone`)
- SigLIP-base/16 at 224 input = **196 patch tokens, 768-dim, no CLS token** ✓
- SigLIP needs `attn_implementation="eager"` to expose attention maps ✓ (SDPA hides them; we default to SDPA and use eager only for the rollout figure)
- SigLIP-base ≈ 86M params (unchanged; not load-bearing)

## Open questions for team

1. **VQA pretraining: keep or skip?** Costs 2-3 pages but provides validation evidence. **Recommendation: keep**, framed honestly as a teaching tool not industry practice.
2. **6 vs 12 fusion transformer layers?** **Recommendation: 6** for chapter 3-5; chapter 6 explores scaling.
3. **Attention rollout viz library: build from scratch or use `vit-rollout`?** **Recommendation: build from scratch** (~40 lines, fits "from scratch" philosophy).
4. **State encoder: 2-layer MLP or 3-layer?** **Recommendation: 2-layer** for code brevity.
5. **`tokenize_demo.py` as a chapter repo resource?** Not in book listings (would push us over Manning code budget) but lives in the repo so curious readers can experiment with the tokenizer end-to-end. **Recommendation: yes**.
6. **Image preprocessing: do we explain center-crop here or save for ch 9?** Briefly here (½ paragraph in 3.2.3) since it affects every later chapter; full sim-to-real treatment in ch 9.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| SmolLM tokenizer expansion breaks HF auto-config | Early smoke test in `tests/test_vocab_expansion.py` |
| SigLIP attention rollout viz finicky across versions | Pin transformers to a tested version |
| VQA synthetic data feels contrived | Use real frames from `svla_so101_pickplace`; design questions that require vision (not answerable from state alone) |
| 415 LOC across 8 listings might exceed Manning code budget | Compress 3.4 (LanguageBackbone forward) and 3.5 (StateEncoder) into prose; target 6-7 listings |
| Cross-chapter import from Ch 2 repo could break | Pin `lrm-code-chapter-2` to specific git SHA |
| 7 concept boxes still high vs Manning 3-5 ideal | Some are tiny (1 paragraph). Final draft may inline 1-2 more. |

---

## What changed from v1 (the v1 → v2 diff)

Driven by 9-persona panel critique on 2026-05-14:
- **Concept boxes 11 → 7** (Erik): merged SigLIP into prose, merged "encoder vs decoder" into 3.3.1 prose, merged "encoding numeric state" inline, merged "VQA as a probe" into 3.4.5 prose
- **Added MQR alignment section** at the top with concept inventory (MQR Primary, MQR Secondary)
- **Added prerequisite check + Track A/B/C compute note** (Erik, Track A/B readers)
- **Beefed up tokenizer concept box** with BPE basics + embedding table coupling (MQR Primary, MQR Secondary)
- **Beefed up action token concept box** with concrete example + Ch 4 preview (Karpathy, MQR Primary)
- **Added hidden dim 512 justification** (Charlie)
- **Added attention heads, dropout, RT-2 citation** to architectural decisions (Charlie, Grad Student)
- **Added causal mask + autoregressive preview** in 3.4.3 (Karpathy, MQR Primary)
- **Added VQA framing as probing classifier** with citation (Grad Student)
- **Added honesty asides** about "from scratch" boundaries and VQA-as-teaching (Karpathy)
- **Added intuition signposts** at close of every section (MQR Secondary, Buyer)
- **Added `tokenize_demo.py`** as repo resource (open question 5)
- **Added citation list + facts to verify** (Grad Student, Fact-Checker)
- **Added "what changed" section** for review trail

## What changed v2 → v3 (2026-05-18 team feedback)

- **Tokenization moved to chapter 4 entirely**: vocab expansion, `add_tokens`, `resize_token_embeddings`, the 1,536 action token slots, the BPE/embedding-table concept box — all of it now lives with the action head story in chapter 4. Chapter 3 uses SmolLM's native tokenizer unchanged.
  - Section 3.3 shrinks from 5-7 pages to 3-4 pages
  - Listing 3.3 simplifies from "load + expand vocab" (~40 LOC) to "load + tokenize + forward" (~50 LOC)
  - 2 concept boxes removed from ch 3
  - Hand-off contract to ch 4 now explicitly lists vocab expansion as ch 4's first step
- **VQA training loop replaced by prompted attention visualization** in 3.4.5: same pedagogical payoff (showing patches light up for relevant instructions) but with no training, no synthetic VQA dataset, no probing classifier, no accuracy curve. Inference-only diagnostic.
  - Section 3.4.5 shrinks from 2-3 pages to 1-2 pages
  - Listing 3.7 (formerly 3.8) shrinks from ~80 LOC to ~30 LOC
  - Figure 3.6 changes from "VQA accuracy + before/after" to "3-panel attention grid for different instructions"
  - Removes Conneau/Tenney probing-classifier framing (no longer needed)
- **Projection layer mentions added** (3.2.5 inline + 3.4.2 MLP note): light coverage so MQR Secondary knows what `nn.Linear(in, out)` does and when to use an MLP variant. Total cost: ~3 sentences. No concept box.
- **Concept inventory updated**: removed tokenizer + vocab expansion from ch 3 list, added explicit "deferred to ch 4" section.
- **Total chapter shrinkage**: 22-30 pages → ~18-24 pages. Code budget 415 LOC → 270 LOC. Concept boxes 7 → 5 (now hits Manning's 3-5 ideal range).

---

## Cross-references

- **Builds on**: chapter 2's `SO100Env` and `LeRobotDataset.from_pretrained("lerobot/svla_so101_pickplace")`
- **Hands off to**: chapter 4's discrete action head will attach to the right end of the backbone's output
- **Recalls from chapter 1**: figure 1.7 (the Mental Model) — same anatomy, now built
- **Wiki**: see `~/Desktop/wiki/lrm-book/ch3-outline-detailed.md` for the full intuition-and-explainer outline
- **Process**: see `PROCESS.md` at lrm-book repo root for the cross-author workflow
