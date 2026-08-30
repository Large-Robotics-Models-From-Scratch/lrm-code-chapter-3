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
VLA backbone: a network that takes two camera views, an instruction,
and the robot's joint state, and produces contextualized hidden states
ready for an action head (Chapter 4).

Fusion here happens by *direct concatenation*. The two camera views and
the state are encoded to the language backbone's width and concatenated
with the language input embeddings into one sequence, the observation
prefix, which enters SmolLM2 through its ``inputs_embeds`` interface.
The pretrained language backbone is the fuser: there is no separate
fusion module on the main path. The output is ``[B, 392 + L + 1, 576]``.

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
# the ch03 package, so this notebook never runs on random noise. The public
# Chapter 2 and Chapter 3 packages can be installed together. To stream
# whole episodes yourself, also install Chapter 2 and provide system FFmpeg
# for torchcodec, or omit torchcodec so LeRobot falls back to its bundled
# pyav decoder:
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
patch representations to the book's common 576-dim width. A 224x224 image
at patch size 16 gives a 14x14 grid, so we get 196 visual input
embeddings per frame. The resize to 224 is bicubic with antialiasing, and
it lives in one place, ``preprocess_image``, which the encoder itself
calls for frames that arrive at another size. (Full source:
``src/ch03/vision_encoder.py``.)
""")

code("""
from ch03 import VisionEncoder, preprocess_image

vision_encoder = VisionEncoder().eval()
up_224 = preprocess_image(up)                  # [3,480,640] -> [3,224,224]
patches = vision_encoder(up_224.unsqueeze(0))
print("visual embeddings:", tuple(patches.shape))   # [1, 196, 576]
""")

md("""
### Listing 3.2 What the frozen encoder groups (patch self-similarity)

Pick one patch and measure cosine similarity between its frozen SigLIP
feature and every other patch. Querying a brick patch lights up the brick;
querying an arm patch lights up the arm. The frozen encoder already
groups object regions, with no training. The grid is 14x14, and the three
queries below mark the brick, arm, and tabletop in this frame; on a
different frame, pick the cells those regions land in there.
""")

code("""
from ch03.viz_similarity import similarity_grid

fig_3_3 = similarity_grid(            # Figure 3.3
    vision_encoder,
    up,
    [
        (11, 4, "brick query patch"),
        (5, 6, "arm query patch"),
        (12, 8, "tabletop query patch"),
    ],
    similarity_range=(-0.1, 1.0),
)
""")

md("""
## 3.3 The state encoder

### Listing 3.3 The state encoder

A two-layer MLP (``Linear -> GELU -> Linear``) lifts the 6 joint numbers
(five arm joints plus the gripper) into one 576-dim state embedding,
shape [1, 1, 576] with the sequence dimension included, which occupies
the state position beside the visual and language positions.
""")

code("""
from ch03 import StateEncoder

state_encoder = StateEncoder()
state_embedding = state_encoder(state)               # state is [1, 6]
print("state embedding:", tuple(state_embedding.shape))   # [1, 1, 576]
""")

md("""
## 3.4 The brain: the language backbone

### Listing 3.4 Looking up SmolLM2 input embeddings

SmolLM2-135M's native tokenizer (49,152 ids, and no vocabulary changes
here or in Chapter 4) turns the instruction into L token ids, and the
embedding table maps each id to a 576-dim input embedding: a row lookup,
no Transformer layers yet. SmolLM2's native width is already the book's
common width, d_model = 576, so the language stream needs no projection.
Only language positions carry vocabulary ids. These input embeddings are
uncontextualized; contextualization happens once, over the whole
multimodal sequence, in Section 3.5.
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
print("language input embeddings:", tuple(embeddings.shape))  # [1,L,576]
""")

md("""
## 3.5 Multimodal fusion
""")

md("""
### Listing 3.5 Building the multimodal sequence

The backbone composes what you already built. Its vision path is the
``VisionEncoder`` from listing 3.1, used unchanged, so the frozen SigLIP,
the resize to 224, the 768 to 576 projection, and SigLIP's pixel
normalization stay in one place; alongside it sit the state encoder and
the SmolLM2-135M language backbone itself.

``embed_inputs`` builds the observation prefix by direct concatenation:
the 392 visual input embeddings (196 for the overhead camera, then 196
for the side camera), the L language input embeddings looked up from
SmolLM2's own table, then the one state embedding. Only the language
stream carries vocabulary ids; the visual and state streams enter as
vectors, so nothing about the tokenizer or the embedding table changes
and ``config.vocab_size`` stays 49,152. It returns the input embeddings
together with the attention mask and the mask-derived position ids that
describe them, laid out ``[image (392), text (L), state (1)]``. (Full
source: ``src/ch03/vla_backbone.py``.)
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
### Listing 3.6 Transforming input embeddings into hidden states

``contextualize`` passes the completed embedding sequence through
SmolLM2 via ``inputs_embeds``; vocabulary ids play no role at this
stage. The output has the same ``[B, 392 + L + 1, 576]`` shape, but
each position's vector has been updated with the context available to
it under the causal attention pattern. The identical shapes hide the
chapter's central point: the sequence structure is preserved while the
information carried by its positions changes. ``forward`` chains
``embed_inputs`` and ``contextualize``; Chapter 4's parallel action
head calls the two stages separately so it can extend the observation
prefix between them.
""")

code("""
with torch.no_grad():
    hidden = backbone.contextualize(embeddings, mask, position_ids)
print("hidden states:", tuple(hidden.shape))  # same shape, new content
delta = (hidden - embeddings).abs().mean().item()
print(f"mean |change| per element after contextualizing: {delta:.3f}")
""")

md("""
### Listing 3.7 Running the complete VLA backbone on one observation

The two stages above are the whole backbone, and ``forward`` composes
them, so one call takes the recorded observation to hidden states.
Before handing the backbone to Chapter 4, check the contract holds: the
output sequence is ``392 + L + 1`` long and 576 wide, the tokenizer was
never expanded (``config.vocab_size`` is still 49,152, and the padding
token reuses end-of-text), and the vision path is the composed
``VisionEncoder`` rather than a second copy of it. These are the same
invariants the ``tests/`` suite asserts.
""")

code("""
from ch03 import VisionEncoder

with torch.no_grad():                      # the ordinary forward path
    hidden_states = backbone(
        images, tokens.input_ids, state, tokens.attention_mask
    )
print("camera frames: ", tuple(images.shape))
print("language ids:  ", tuple(tokens.input_ids.shape))
print("robot state:   ", tuple(state.shape))
print("hidden states: ", tuple(hidden_states.shape))

# forward is exactly embed_inputs followed by contextualize.
assert torch.allclose(hidden_states, hidden, atol=1e-5)

B, N, width = hidden_states.shape
assert N == 392 + L + 1, (N, L)
assert width == 576, width
assert backbone.language_backbone.config.vocab_size == 49152
assert (backbone.tokenizer.pad_token_id
        == backbone.tokenizer.eos_token_id)

# The vision path is section 3.2's encoder, composed, not rebuilt.
assert isinstance(backbone.vision_encoder, VisionEncoder)
assert not hasattr(backbone, "img_proj")

# The embedding table stays native: 49,152 rows, no grown copy.
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
## (Optional) Separate-encoder fusion

The main path lets the pretrained language backbone fuse the streams.
The optional ``FusionTransformer`` is the named alternative: a
from-scratch stack of pre-norm causal self-attention blocks that you
bolt onto the frozen streams and compare against the direct
concatenation on the main path. It is not on the main path and is not
imported by ``VLABackbone``. (Source:
``src/ch03/fusion_transformer.py``.)
""")

code("""
from ch03.fusion_transformer import FusionTransformer

fusion_transformer = FusionTransformer()              # hidden_dim=576
dummy = torch.rand(1, 392 + L + 1, 576)
print("hidden states:", tuple(fusion_transformer(dummy).shape))
""")

md("""
## Summary

You built a VLA backbone from pre-trained parts: a frozen SigLIP vision
encoder, a trainable SmolLM2-135M language backbone, and a state
encoder, joined by direct concatenation. The two camera views and the
state are encoded to the language backbone's width and concatenated
with the language input embeddings into one observation prefix that
SmolLM2 reads through ``inputs_embeds``, so the pretrained language
backbone is the fuser; there is no separate fusion module on the main
path. The backbone exposes two stages, ``embed_inputs`` (input
embeddings, attention mask, position ids) and ``contextualize``
(contextualized hidden states), and ``forward`` chains them, mapping
two images, an instruction, and the robot state to
``[B, 392 + L + 1, 576]`` hidden states. Chapter 4 reads those hidden
states, or extends the observation prefix before contextualization, to
train the first working policy.
""")

# Deterministic cell ids. nbformat mints a fresh uuid per cell on every
# run, so without this the notebook shows a diff on every rebuild even
# when nothing changed, and a genuinely stale notebook is invisible in
# the noise. Rebuilding now produces a byte-identical file unless the
# content actually moved.
for index, cell in enumerate(cells):
    cell["id"] = f"ch03-{index:02d}"

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
