# Chapter 3: Building the VLA Backbone — Structure & Content Plan (v3)

**Author**: Krishnam Gupta
**Code repo**: `Large-Robotics-Models-From-Scratch/lrm-code-chapter-3`
**Built on**: chapter 2 (`SO100Env`, `lerobot/svla_so100_pickplace`)
**Hands off to**: chapter 4 (action head + action tokenization + vocab expansion all happen there)
**Last updated**: 2026-05-18 (v3 — tokenization moved to ch 4, VQA replaced by prompted attention viz, projection-layer mentions added)

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
- Cached `lerobot/svla_so100_pickplace` locally (one-line load in chapter 2)
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
- Validating alignment with a VQA probe
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

**Pages**: 7-9 (was 6-8; +1 page for the vision-encoder-taxonomy on-ramp added 2026-05-21)
**Purpose**: Build the vision component. Establish a broad mental map of vision encoders first, then narrow to SigLIP, then build.

**Subsections**:
- **3.2.0 What is a vision encoder, and what kinds exist** (1-1.5 pages — NEW intuition on-ramp)
  - What "vision encoder" means generally — a neural network that turns pixels into vectors
  - **Three big families** (concept box):
    | Family | Training signal | Examples | What it's good at |
    |---|---|---|---|
    | Supervised classification | Labeled categories (ImageNet) | ResNet, EfficientNet, ViT-classification | Closed-vocabulary recognition |
    | Self-supervised vision-only | No labels — reconstruct masked patches, contrastive on augmentations | DINOv2, MAE, SimCLR | Rich visual features without labels; great for downstream fine-tuning |
    | Image-text contrastive | Image-caption pairs | CLIP, SigLIP, ALIGN | Vision features aligned with language; zero-shot recognition |
  - **Why this taxonomy matters for VLA**: we need the third family. Robot policies must ground language ("red cube") in image regions. Self-supervised vision (DINOv2) gives richer features but lives in its own vector space, disconnected from text. Supervised classification labels won't help us understand novel instructions. Image-text contrastive encoders bridge the two.
  - **A note on "self-supervised" learning** (1 paragraph): instead of needing human labels, the model invents its own learning signal — predict the next word in a sentence (GPT), or reconstruct masked patches in an image (MAE), or pull matching image-text pairs closer in embedding space (CLIP/SigLIP). Self-supervision is why pre-trained encoders exist at internet scale.
  - **Honest aside**: production VLAs often combine families (OpenVLA uses both DINOv2 *and* SigLIP; π0 uses SigLIP-large alone). We pick one for simplicity.
- 3.2.1 Why pre-trained, why frozen (½ page) — honest about what's inherited (shorter now since taxonomy section covered the "why pre-trained" angle)
- 3.2.2 Why SigLIP specifically for our task (½ page) — narrow from family to model + 4-row comparison table (SigLIP vs CLIP vs DINOv2 vs raw ViT) + the 1-line verdict ("image-text alignment + sigmoid loss outperforms CLIP at same size")
- 3.2.3 Image → patches → tokens (1-2 pages, includes listing 3.1)
- 3.2.4 Visualizing what SigLIP "sees" (1-2 pages, includes listing 3.2 + figure 3.3)
- 3.2.5 Projecting to the common hidden dim (½ page) — justify "why 512" + name the projection layer concept ("a projection layer is just a single `nn.Linear(768, 512)` that translates between widths")

**Concept boxes** (4):
- Three families of vision encoders (3.2.0) — supervised / self-supervised / image-text contrastive
- Pre-trained-and-frozen (3.2.1) — "what we inherit and what we don't"
- ViT patch tokenization (3.2.3) — "your image as a sentence of patches"
- Attention rollout (3.2.4) — "the model's gaze, made visible"

(SigLIP gets a focused half-page narrowing section at 3.2.2, not a separate concept box — it leans on the taxonomy already established.)

**Listings**:
- 3.1: Loading and freezing SigLIP (~30 lines)
- 3.2: Attention rollout visualization (~50 lines)

**Figures**:
- 3.2: Image → patch grid → token sequence (process diagram)
- 3.3: SigLIP attention overlay on three SO-100 frames (cube highlighted, gripper highlighted, table suppressed) — caption notes left, center, right by position not color

**Honesty aside** (in 3.2.1): "From scratch" does not mean we re-derive ImageNet. We download SigLIP's weights and inherit ~400M images of pretraining. What we build from scratch is the wiring around it — how patches flow into our fusion transformer, how attention is masked, how the projection to our common width works.

**Intuition signpost** (close section): "The frozen vision encoder is the cheapest superpower we'll have. Every patch position now carries semantic information about what's there — cube, gripper, table — without us having trained anything yet."

**Reader state at end**: Has loaded a pre-trained vision encoder, understands patch tokenization, has visual proof the encoder attends to relevant objects, understands why we freeze it. Has 196 image tokens per frame, each 512-dim.

---

## Section 3.3: The brain — language backbone

**Pages**: 3-4 (down from 5-7 in v2 — vocab expansion moved to ch 4)
**Purpose**: Build the language component using SmolLM's native tokenizer. No vocab expansion, no action tokens — those live in ch 4 where the action head story plays out.

**Subsections**:
- 3.3.1 Why a language model, not just a text encoder (½ page) — causal generation as the goal
- 3.3.2 SmolLM-135M in 2 paragraphs (inline, no separate concept box)
- 3.3.3 Tokenize the instruction (½ page) — use SmolLM's native tokenizer, no modifications
- 3.3.4 Forward pass + projection to 512 (1-2 pages, includes listing 3.3 + figure 3.4)

**Concept boxes** (0 in this section — all tokenization theory moves to ch 4):
- (none — keep section lean)

**Listings**:
- 3.3: Loading SmolLM, tokenizing an instruction, forward pass, project 576 → 512 (~50 lines)

**Figures**:
- 3.4: Token flow through SmolLM. Show "pick up the red cube" → BPE token IDs → SmolLM hidden states → projection to 512-dim. No reserved-slot annotations — ch 4 will add that diagram when it expands the vocab.

### Light tokenizer mention (3.3.3, one paragraph)

> SmolLM uses **Byte-Pair Encoding (BPE)** — a subword tokenization scheme that breaks `"pick up the red cube"` into about 5 integer token IDs. We use the tokenizer as-is in this chapter. Chapter 4 will expand SmolLM's vocabulary to make room for action tokens; for now, we just need to encode an instruction and pass it through the language backbone.

### Hand-off note to ch 4 (one sentence at end of 3.3)

> Chapter 4 will modify SmolLM's vocabulary to add 1,536 action token IDs. The change is one call to `tokenizer.add_tokens` plus one to `model.resize_token_embeddings`, but the *use* of those tokens — and the full action prediction loop — belongs with the action head story in chapter 4.

**Intuition signpost** (close section): "You have a language backbone that turns an instruction into 512-dim hidden states. Chapter 4 will teach it new words — words that mean motor commands. For now, it's a small but real LLM, projected to our common width."

**Reader state at end**: Has a working language backbone. Native SmolLM tokenizer encodes the instruction. Output is `[B, L, 512]` per-token hidden states. No vocab modifications, no action tokens. Save/load uses the native SmolLM tokenizer.

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
- 3.6: Prompted attention grid. Same SO-100 scene, three different instructions ("pick the red cube", "pick the blue cube", "show the gripper") → three different attention heatmaps. Demonstrates the backbone routes attention to instruction-relevant patches without any training.

### Hidden dim justification (in 3.4.1 or 3.2.5)

> Why 512? SigLIP outputs 768-dim hidden states; SmolLM-135M outputs 576-dim. We want one common width so all streams can sit in the same sequence and be processed by one transformer. 512 is a round power of two slightly below both natives — every stream projects down (cheap linear layers, no information bottleneck for our scale), and we pick a multiple of 64 so 8 attention heads × 64-dim each works cleanly. Larger production VLAs use 768, 1024, or higher. We pick 512 because it fits T4 memory comfortably with our sequence length (~210 tokens).

### Beefed-up causal mask explanation (3.4.3)

> A standard transformer's self-attention lets every token see every other token. **Causal attention** restricts each token to seeing only tokens that came before it (left-to-right). We use causal here for one reason: chapter 4 will train the model to predict action tokens autoregressively, one at a time, where each action token depends only on the context before it. If we used bidirectional attention now, we'd have to re-architect for Ch 4. By being causal from the start, the action prediction story drops in cleanly.
>
> **Preview of Ch 4** (one paragraph, no code yet): the action head will append 6 action token positions to the right end of the sequence. With causal masking, position `[image+lang+state+0]` predicts the first action token using image+language+state context. Position `[image+lang+state+1]` predicts the second using everything before plus the first action token. And so on, six positions, six joint commands. The model trains via standard cross-entropy loss on the predicted action token IDs.

### Prompted attention visualization (3.4.5) — replaces VQA training

> We've built a `VLABackbone`, but we haven't trained anything yet beyond the projection layers' default initialization. Does the backbone actually attend to instruction-relevant regions of the scene? We can answer that with a single forward pass and an attention rollout — no training, no labels, no VQA dataset.
>
> The recipe: take one SO-100 frame from `svla_so100_pickplace`. Pass it through the backbone three times with three different instructions: "pick the red cube", "pick the blue cube", "show the gripper". For each pass, run attention rollout on the SigLIP layers (chapter 3.2.4 already wrote this) and overlay the heatmap on the original frame.
>
> If the backbone is doing its job, the three heatmaps should differ: "red cube" lights up red-cube patches, "blue cube" lights up blue-cube patches, "gripper" lights up the gripper. If they're identical, the language conditioning isn't reaching the vision features — and we have a bug to fix before chapter 4.
>
> Honest framing: this isn't training, it's a **diagnostic**. We're inspecting what the frozen pre-trained components already do when wired together. Production VLAs skip this step because they trust their components. Chapter 3 readers don't yet — and one figure of comparative heatmaps builds that trust cheaply.
>
> If the team wants a richer probe later (a real VQA training stage), chapter 6 has a natural slot for it during multi-task curriculum work. For chapter 3, the visualization is enough.

### Why this replaces VQA pretraining (decision rationale)

The v2 plan included a full VQA training loop (~80 LOC) and 2-3 pages of treatment. Team feedback (2026-05-18) flagged that this was scope creep — readers expect chapter 3 to *build* the backbone, not *train* it. The lighter visualization keeps the validation story (does the backbone work?) without adding a training pipeline that chapters 4-6 will subsume anyway. Net savings: ~50 LOC and ~2 pages.

**Intuition signpost** (close section): "You have a VLA backbone. It takes an image, a sentence, and six numbers, and produces a sequence of contextualized hidden states. Pass the same image with different instructions and you see different patches light up — the conditioning works. Chapter 4 attaches an action head to the right end of this output."

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
- Action tokens reserved in the language vocabulary preserve the contract between
  the backbone and any action head added later.
- A VQA probe validates that vision and language are aligned in the shared space
  before attaching task-specific output heads.
- Causal self-attention over a concatenated sequence is the simplest fusion that
  generalizes from text-only LLMs to multimodal models.
```

### In the wild sidebar: production VLA backbones

| Model | Backbone size | Vision | Language | Notes |
|---|---|---|---|---|
| Yours (after this chapter) | ~0.5B | SigLIP-base (frozen) | SmolLM-135M | First-principles build |
| OpenVLA | 7B | DINOv2 + SigLIP | Llama-2-7B | Fine-tuned, OXE-trained |
| RT-2-X | 55B | PaLI-X vision | PaLM-X | Largest, Google internal |
| π0 | 3B | SigLIP-large | PaliGemma | Fine-tuned for flow matching |
| OpenpiZero | 3B | SigLIP-large | PaliGemma | Open reimplementation of π0 |

Same anatomy. You scaled differently.

---

## Architectural decisions (locked)

| Decision | Choice | Why |
|---|---|---|
| Vision encoder | SigLIP-base/16 (224) | Image-text aligned, 196 patch tokens |
| Camera input | `image_top` only (top-down arm-mounted view) | Single stream for chapter 3; wrist camera added in chapter 6 |
| Image preprocessing | Center-crop 480×640 → resize 224×224 | Match SigLIP input size (relevant for sim-to-real in ch 9) |
| Language backbone | SmolLM-135M | Small, OSS, runs on T4, good tokenizer |
| Tokenizer changes in ch 3 | **None** — native SmolLM tokenizer | Vocab expansion moved to ch 4 |
| Fusion mechanism | Concat + causal self-attention (RT-2 style; Brohan et al. 2023) | Simpler "from scratch" build, matches OpenVLA |
| Hidden dim | 512 | Bridges SigLIP 768 + SmolLM 576; multiple of 64 for clean head dim |
| Attention heads | 8 (64-dim per head) | Standard ratio for d=512 |
| Fusion transformer layers | 6 | Fits T4; chapter 6 explores scaling |
| Dropout | 0.1 in fusion transformer | Standard transformer regularization |
| State dim | 6 (5 arm + 1 gripper) | SO-100 native |

**Deferred to ch 4** (was previously in ch 3 v2):
- Action tokens, 256 bins per joint, 1,536 reserved vocab IDs
- `tokenizer.add_tokens` + `model.resize_token_embeddings`
- Save/load coupling between tokenizer and model

---

## Hand-off contract to chapter 4

```python
from lrm_ch03 import VLABackbone

backbone = VLABackbone(hidden_dim=512)
hidden = backbone(image, instruction, state)
# image:       [B, 3, 224, 224]   top camera, preprocessed
# instruction: List[str]          batch of B instructions
# state:       [B, 6]             SO-100 joint positions
# hidden:      [B, 196 + L + 1, 512]
# tokenizer:   native SmolLM (49,152 vocab) — no expansion in ch 3
```

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

## Facts to verify

- SmolLM-135M hidden dim (576 assumed; verify in HF model card)
- SmolLM-135M vocab size (49,152 assumed; verify)
- SigLIP-base/16 specifically (vs /14) at 224 input size
- SigLIP-base parameter count (~86M assumed; verify)

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
| VQA synthetic data feels contrived | Use real frames from `svla_so100_pickplace`; design questions that require vision (not answerable from state alone) |
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

- **Builds on**: chapter 2's `SO100Env` and `LeRobotDataset.from_pretrained("lerobot/svla_so100_pickplace")`
- **Hands off to**: chapter 4's discrete action head will attach to the right end of the backbone's output
- **Recalls from chapter 1**: figure 1.7 (the Mental Model) — same anatomy, now built
- **Wiki**: see `~/Desktop/wiki/lrm-book/ch3-outline-detailed.md` for the full intuition-and-explainer outline
- **Process**: see `PROCESS.md` at lrm-book repo root for the cross-author workflow
