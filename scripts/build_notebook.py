"""Build notebooks/ch03.ipynb from the chapter's components.

The notebook drives the installed ``ch03`` package (and ``ch02`` for the
dataloader). The full annotated source of each listing lives in
``src/ch03/`` and in the book prose; the notebook constructs and exercises
each component end to end, mirroring the chapter's seven listings.

Run: ``python scripts/build_notebook.py`` (needs nbformat from [dev]).
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md("""
# Chapter 3: Building the VLA Backbone

You walk in with a robot in simulation and a dataset of teleoperated
pick-and-place episodes from Chapter 2. You walk out having built a
vision-language-action backbone: a network that takes two camera views,
an instruction, and the robot's joint state, and produces a fused
sequence of hidden states ready for an action head (Chapter 4).

Fusion here is *token-level fusion*. The two camera views and the state
are encoded to the language backbone's width and concatenated with the
language embeddings into one sequence, which enters SmolLM2 through its
``inputs_embeds`` interface, so the pretrained backbone is the fuser:
there is no separate fusion module on the main path. The output is
``[B, 392 + L + 1, 576]``.

The full annotated source of every listing is in ``src/ch03/`` and in the
book prose. This notebook installs the package, then constructs and runs
each component on a real sample from the Chapter 2 dataset: two camera
views, the instruction, and the joint state of one recorded timestep,
which ship inside the package so nothing here runs on random noise.
""")

code("""
# Colab setup: install the Chapter 3 backbone package. On a local machine
# where it is already installed, this cell is a no-op. The distribution
# name is lrm-ch03 (the import name is ch03).
import sys

ORG = "https://github.com/Large-Robotics-Models-From-Scratch"
if "google.colab" in sys.modules:
    !pip install -q "lrm-ch03 @ git+{ORG}/lrm-code-chapter-3.git"

# Chapter 2's live data pipeline is NOT needed here: one real sample from
# its dataset (both camera views, the state, the instruction) ships inside
# the ch03 package, so this notebook never runs on random noise. Chapter 2
# also does not install alongside Chapter 3 under this repo's pins:
# lerobot 0.5.1 requires huggingface-hub>=1.0,<2.0 while transformers
# <5.0 requires huggingface-hub<1.0. To stream whole episodes yourself,
# relax the transformers pin to allow 5.x (pip then resolves lerobot
# 0.5.1 + transformers 5.x + huggingface-hub 1.x) and have a system
# FFmpeg for torchcodec, or no torchcodec at all so lerobot falls back to
# its bundled pyav decoder:
#   !pip install -q "lrm-ch02[data] @ git+{ORG}/lrm-code-chapter-2.git"
""")

md("""
## 3.1 Load two camera views, an instruction, and a state

Chapter 3 consumes the data contract Chapter 2 froze: batches from
``make_pickplace_dataloader`` carry two camera images,
``observation.images.up`` and ``observation.images.side``, each
``[B, 3, 480, 640]`` in ``[0, 1]``, the 6-dim joint state, the 6-dim
action (Chapter 4 uses that), and ``task``, the natural-language
instruction. Chapter 2's loader needs the ``[data]`` extra and a video
decoder, so one real sample from that dataset travels with this package
instead: ``load_sample`` returns exactly what one dataloader row hands
over, and every listing below runs on it.
""")

code("""
import torch

from ch03 import load_sample

images, state, instruction = load_sample()   # one real Chapter 2 sample
up = images[0, 0]     # [3, 480, 640] in [0,1] observation.images.up
side = images[0, 1]   # [3, 480, 640] in [0,1] observation.images.side

print("images:", tuple(images.shape), "| up:", tuple(up.shape))
print("state:", tuple(state.shape), state.dtype, "| z-scored")
print("instruction:", instruction)
""")

md("""
## 3.2 The eyes: vision encoder

### Listing 3.1 Loading and freezing SigLIP

``VisionEncoder`` wraps a frozen SigLIP-base/16 and projects its 768-dim
patch tokens to the book's common 576-dim width. A 224x224 image at patch
size 16 gives a 14x14 grid, so we get 196 patch tokens per frame. (Full
source: ``src/ch03/vision_encoder.py``.)
""")

code("""
from ch03 import VisionEncoder, preprocess_image

vision_encoder = VisionEncoder().eval()
up_224 = preprocess_image(up)                  # [3,480,640] -> [3,224,224]
patches = vision_encoder(up_224.unsqueeze(0))
print("patch tokens:", tuple(patches.shape))        # [1, 196, 576]
""")

md("""
### Listing 3.2 What the frozen encoder groups (patch self-similarity)

Pick one patch and measure cosine similarity between its frozen SigLIP
feature and every other patch. Querying a brick patch lights up the brick;
querying an arm patch lights up the arm. The frozen encoder already
groups object regions, with no training. The grid is 14x14, and the two
queries below are the brick's cell and the arm's cell in this frame; on
a different frame, pick the cells the brick and the arm land in there.
""")

code("""
from ch03.viz_similarity import similarity_grid

fig_3_3 = similarity_grid(            # Figure 3.3
    vision_encoder,
    up,
    [(11, 4, "brick query"), (5, 7, "arm query")],
)
""")

md("""
## 3.3 The state encoder

### Listing 3.3 The state encoder

A two-layer MLP (``Linear -> GELU -> Linear``) lifts the 6 joint numbers
(five arm joints plus the gripper) into one 576-dim state token that sits
beside the image and language tokens.
""")

code("""
from ch03 import StateEncoder

state_encoder = StateEncoder()
state_token = state_encoder(state)                   # state is [1, 6]
print("state token:", tuple(state_token.shape))      # [1, 576]
""")

md("""
## 3.4 The brain: the language backbone

### Listing 3.4 Looking up SmolLM2 token embeddings

SmolLM2-135M's native tokenizer (49,152 ids, no vocabulary changes here;
that is Chapter 4's first step) turns the instruction into L token ids,
and the embedding table maps each id to a 576-dim vector: a row lookup,
no Transformer layers yet. SmolLM2's native width is already the book's
common width D=576, so the language stream needs no projection. These
embeddings are uncontextualized; contextualization happens once, over
the whole fused sequence, in Section 3.5.
""")

code("""
from transformers import AutoModel, AutoTokenizer

SMOLLM_MODEL = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(SMOLLM_MODEL)
language_backbone = AutoModel.from_pretrained(SMOLLM_MODEL)

input_ids = tokenizer(instruction, return_tensors="pt").input_ids
embed_tokens = language_backbone.get_input_embeddings()
embeddings = embed_tokens(input_ids)
print("token ids:", input_ids[0].tolist())
print("language embeddings:", tuple(embeddings.shape))  # [1, L, 576]
""")

md("""
## 3.5 Multimodal fusion
""")

md("""
### Listing 3.5 Building the multimodal input sequence

The backbone composes what you already built. Its vision path is the
``VisionEncoder`` from listing 3.1, used unchanged, so the frozen SigLIP,
the resize to 224, the 768 to 576 projection, and SigLIP's pixel
normalization stay in one place; alongside it sit the state encoder and
the SmolLM2-135M backbone itself.

``embed_inputs`` builds the fused sequence by direct concatenation:
the 392 visual embeddings (196 for camera 0, then 196 for camera 1),
the L language embeddings looked up from SmolLM2's own table, then the
one state embedding. Only the language stream carries vocabulary ids;
the image and state streams enter as vectors, so nothing about the
tokenizer or the embedding table changes and ``config.vocab_size``
stays 49,152. It returns the input embeddings together with the
attention mask and mask-derived position ids that describe them, laid
out ``[image (392), text (L), state (1)]``. (Full source:
``src/ch03/vla_backbone.py``.)
""")

code("""
from ch03 import VLABackbone

backbone = VLABackbone().eval()

tokens = backbone.tokenizer([instruction], return_tensors="pt",
                            padding=True)
L = tokens.input_ids.shape[1]
with torch.no_grad():
    embeddings, mask, position_ids = backbone.embed_inputs(
        images, tokens.input_ids, state, tokens.attention_mask
    )
print("input embeddings:", tuple(embeddings.shape),
      "= [1, 392 +", L, "+ 1, 576]")
print("state position id:", position_ids[0, -1].item())  # 392 + L
print("vocab size unchanged:",
      backbone.language_backbone.config.vocab_size)   # 49152
""")

md("""
### Listing 3.6 Contextualizing the multimodal sequence

``contextualize`` passes the completed embedding sequence through
SmolLM2 via ``inputs_embeds``; vocabulary ids play no role at this
stage. The output has the same ``[B, 392 + L + 1, 576]`` shape, but
each position's vector has been updated with the context available to
it under the causal attention pattern. The identical shapes hide the
chapter's central point: the sequence structure is preserved while the
information carried by its positions changes. ``forward`` chains
``embed_inputs`` and ``contextualize``; Chapter 4's parallel action
head calls the two stages separately so it can extend the sequence
between them.
""")

code("""
with torch.no_grad():
    hidden = backbone.contextualize(embeddings, mask, position_ids)
print("hidden states:", tuple(hidden.shape))  # same shape, new content
delta = (hidden - embeddings).abs().mean().item()
print(f"mean |change| per element after fusion: {delta:.3f}")
""")

md("""
### Listing 3.7 Following one observation through the complete backbone

Before handing the backbone to Chapter 4, check the contract holds: the
output sequence is ``392 + L + 1`` long and 576 wide, the tokenizer was
never expanded (``config.vocab_size`` is still 49,152, and the padding
token reuses end-of-text), and the vision path is the composed
``VisionEncoder`` rather than a second copy of it. These are the same
invariants the ``tests/`` suite asserts.
""")

code("""
from ch03 import VisionEncoder

B, N, width = hidden.shape
assert N == 392 + L + 1, (N, L)
assert width == 576, width
assert backbone.language_backbone.config.vocab_size == 49152
assert (backbone.tokenizer.pad_token_id
        == backbone.tokenizer.eos_token_id)

# The vision path is section 3.2's encoder, composed, not rebuilt.
assert isinstance(backbone.vision_encoder, VisionEncoder)
assert not hasattr(backbone, "img_proj")

# No placeholder rows, no grown table: the embedding table is native.
assert (backbone.language_backbone.get_input_embeddings()
        .num_embeddings == 49152)

trainable = sum(p.numel() for p in backbone.parameters()
                if p.requires_grad)
frozen = sum(p.numel() for p in backbone.parameters()
             if not p.requires_grad)
assert trainable < 140_000_000, trainable
print("contract OK:", (B, N, width), "| vocab still 49152")
print(f'real sample: "{instruction}" | L = {L}')
print(f"trainable: {trainable:,} | frozen: {frozen:,}")
""")

md("""
### The frozen encoder finds the object from either viewpoint

Query the brick patch in each camera view. The brick sits in a different
part of the frame in each one, and the highlight lands on it both times.
With no training, the frozen vision encoder localizes the object wherever
it appears, which is why we can freeze it and build the policy on top.
(Each panel's query is the brick's grid cell in that view; the brick
moves between views, so the query moves with it. Stream more frames with
Chapter 2's loader to watch the same thing across an episode.)

The instruction does not steer this yet; the vision encoder never reads
the language. Wiring language into where the model looks is what Chapter 4
trains.
""")

code("""
from ch03.viz_similarity import tracking_grid

fig_track = tracking_grid(          # bonus viz (not a chapter figure)
    vision_encoder,
    [up, side],
    queries=[(11, 4), (8, 11)],     # the brick's cell in each view
    labels=["overhead camera", "side camera"],
)
""")

md("""
## (Optional) Exercise 3.3: separate-encoder fusion

The main path lets the pretrained backbone fuse the streams. The optional
``FusionTransformer`` is the named alternative: a from-scratch stack of
pre-norm causal self-attention blocks that you bolt onto the frozen
streams and compare against token-level fusion. It is not on the
main path and is not imported by ``VLABackbone``. (Source:
``src/ch03/fusion_transformer.py``.)
""")

code("""
from ch03.fusion_transformer import FusionTransformer

fusion_transformer = FusionTransformer()              # hidden_dim=576
dummy = torch.rand(1, 392 + L + 1, 576)
print("fused:", tuple(fusion_transformer(dummy).shape))
""")

md("""
## Summary

You built a VLA backbone from pre-trained parts: a frozen SigLIP vision
encoder, a trainable SmolLM2-135M language backbone, and a state encoder,
fused by token-level fusion. The two camera views and the state are
encoded to the backbone's width and concatenated with the language
embeddings into one sequence that SmolLM2 reads through
``inputs_embeds``, so the pretrained backbone is the fuser; there is no
separate fusion module on the main path. The backbone exposes two
stages, ``embed_inputs`` (input embeddings, mask, position ids) and
``contextualize`` (hidden states), and maps two images, an instruction,
and the robot state to ``[B, 392 + L + 1, 576]`` hidden states. Chapter
4 reads those representations, or extends the input sequence before
contextualization, to train the first working policy.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python"},
}

out = Path(__file__).resolve().parents[1] / "notebooks" / "ch03.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print(f"wrote {out} ({len(cells)} cells)")
