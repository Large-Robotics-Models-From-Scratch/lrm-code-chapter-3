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

Fusion here is *unified embedding*. The two camera views and the state
are projected and spliced into the language backbone's own token stream
with ``masked_scatter``, so the pretrained backbone is the fuser: there
is no separate fusion module on the main path. The output is
``[B, 392 + L + 1, 576]``.

The full annotated source of every listing is in ``src/ch03/`` and in the
book prose. This notebook installs the package, then constructs and runs
each component on a real frame from the Chapter 2 dataset.
""")

code("""
# Colab setup: install the Chapter 3 backbone package. On a local machine
# where it is already installed, this cell is a no-op. The distribution
# name is lrm-ch03 (the import name is ch03).
import sys

ORG = "https://github.com/Large-Robotics-Models-From-Scratch"
if "google.colab" in sys.modules:
    !pip install -q "lrm-ch03 @ git+{ORG}/lrm-code-chapter-3.git"

# Chapter 2's data pipeline (real SO-100 frames) is optional here and
# currently cannot install alongside Chapter 3: lerobot 0.5.1 pins
# huggingface-hub>=1.0 while transformers pins <1.0. Until those are
# reconciled the next cell falls back to synthetic frames. To use real
# data once the pins are fixed, add:
#   !pip install -q "lrm-ch02[data] @ git+{ORG}/lrm-code-chapter-2.git"
""")

md("""
## 3.1 Load two camera views, an instruction, and a state

Chapter 3 imports the data contract Chapter 2 froze:
``make_pickplace_dataloader``. Each batch carries two camera images,
``observation.images.up`` and ``observation.images.side``, each
``[B, 3, 480, 640]`` in ``[0, 1]``, the 6-dim joint state, the 6-dim
action (Chapter 4 uses that), and ``task``, the natural-language
instruction. We pull one frame from each camera to drive the rest of the
chapter. If the dataset is not available locally, we fall back to
synthetic frames so the notebook still runs.
""")

code("""
import torch

try:
    from ch02 import make_pickplace_dataloader
    loader, stats = make_pickplace_dataloader(batch_size=8)
    batch = next(iter(loader))
    up = batch["observation.images.up"][0]      # [3, 480, 640] in [0,1]
    side = batch["observation.images.side"][0]   # [3, 480, 640] in [0,1]
    frames = list(batch["observation.images.up"][:3])  # cube in 3 spots
    state = batch["observation.state"][0]        # [6] z-scored
    instruction = batch["task"][0]
except Exception as exc:                       # no [data] extra / no net
    print(f"Using synthetic frames (Chapter 2 data unavailable: {exc})")
    up = torch.rand(3, 480, 640)
    side = torch.rand(3, 480, 640)
    frames = [torch.rand(3, 480, 640) for _ in range(3)]
    state = torch.zeros(6)
    instruction = "pick up the red cube"

print("up:", tuple(up.shape), "| side:", tuple(side.shape))
print("state:", tuple(state.shape), "| instruction:", instruction)
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
feature and every other patch. Querying a cube patch lights up the cube;
querying an arm patch lights up the arm. The frozen encoder already
groups object regions, with no training. (Pick grid positions that land
on the cube and the arm in your frame; the grid is 14x14.)
""")

code("""
from ch03.viz_similarity import similarity_grid

fig_3_3 = similarity_grid(            # Figure 3.3
    vision_encoder,
    up,
    [(11, 6, "cube query"), (5, 7, "arm query")],
)
""")

md("""
## 3.3 The brain: language backbone

### Listing 3.3 Loading SmolLM2 and encoding the instruction

``LanguageBackbone`` uses SmolLM2-135M with its native 49,152-token
tokenizer (no vocabulary changes here; that is Chapter 4's first step).
SmolLM2's native hidden width is 576, the book's common width, so the
language stream needs no projection. Unlike the frozen vision encoder,
SmolLM2 stays trainable so Chapter 4 can fine-tune it.
""")

code("""
from ch03 import LanguageBackbone

language_backbone = LanguageBackbone()
lang_hidden, lang_mask = language_backbone([instruction])
print("language tokens:", tuple(lang_hidden.shape))  # [1, L, 576]
""")

md("""
## 3.4 State, then unified-embedding fusion

### Listing 3.4 The state encoder

A two-layer MLP (``Linear -> GELU -> Linear``) lifts the 6 joint numbers
into one 576-dim token that sits beside the image and language tokens.
""")

code("""
from ch03 import StateEncoder

state_encoder = StateEncoder()
state_token = state_encoder(state.unsqueeze(0))
print("state token:", tuple(state_token.shape))      # [1, 576]
""")

md("""
### Listing 3.5 Composing the UnifiedEmbeddingBackbone

The backbone owns the three projections (a frozen SigLIP, a 768 to 576
image projection, and the state encoder) plus the SmolLM2-135M backbone
itself. It also grows the input embedding table by two inert rows so two
placeholder ids, one for image patches and one for the state, index
validly. Those rows are overwritten by the splice before the backbone
runs, so this is not Chapter 4's vocabulary expansion: the tokenizer is
untouched and ``config.vocab_size`` stays 49,152.

``build_sequence_ids`` templates one row in the order
``[image (392), text (L), state (1)]``: 392 image placeholder ids (196
for camera 0, then 196 for camera 1), the tokenized text, then one state
placeholder id. ``tokenize_instruction`` returns the L native SmolLM2
text ids; note HF hands those back under the key ``input_ids``, and we
assemble them into OUR ``sequence_ids``. (Full source:
``src/ch03/vla_backbone.py``.)
""")

code("""
from ch03 import UnifiedEmbeddingBackbone

backbone = UnifiedEmbeddingBackbone().eval()

text_ids = backbone.tokenize_instruction(instruction)
sequence_ids = backbone.build_sequence_ids(text_ids)
L = len(text_ids)
print("template length:", len(sequence_ids), "= 392 +", L, "+ 1")
print("vocab size unchanged:",
      backbone.language_backbone.config.vocab_size)   # 49152
""")

md("""
### Listing 3.6 The forward pass: the masked_scatter splice

``forward(images, sequence_ids, state)`` encodes the two camera views,
projects them to 576, looks up the template's embeddings, and uses
``masked_scatter`` to drop the 392 image tokens and the 1 state token
into their reserved placeholder slots. The pretrained SmolLM2 attention
then fuses everything in one pass. The output ``[B, 392 + L + 1, 576]``
is the contract Chapter 4 attaches an action head to.
""")

code("""
images = torch.stack([up, side])                     # [2, 3, 480, 640]
images = preprocess_image(images).unsqueeze(0)       # [1, 2, 3, 224, 224]
ids = torch.tensor([sequence_ids])                   # [1, N]
with torch.no_grad():
    hidden = backbone(images, ids, state.unsqueeze(0))
print("backbone output:", tuple(hidden.shape))       # [1, 392 + L + 1, 576]
""")

md("""
### Listing 3.7 Definition of done: verifying the backbone

Before handing the backbone to Chapter 4, check the contract holds: the
output sequence is ``392 + L + 1`` long and 576 wide, the language stream
survives the splice unchanged, and the tokenizer was never expanded
(``config.vocab_size`` is still 49,152). These are the same invariants
the ``tests/`` suite asserts.
""")

code("""
B, S, width = hidden.shape
assert S == 392 + L + 1, (S, L)
assert width == 576, width
assert backbone.language_backbone.config.vocab_size == 49152
print("contract OK:", (B, S, width), "| vocab still 49152")
""")

md("""
### The frozen encoder tracks the object across frames

Query the cube patch in three frames where the cube spawns in different
positions. The highlighted region follows the cube each time. With no
training, the frozen vision encoder localizes the object wherever it is,
which is why we can freeze it and build the policy on top. (Each frame's
query is the cube's grid cell in that frame; the cube moves, so the query
moves with it.)

The instruction does not steer this yet; the vision encoder never reads
the language. Wiring language into where the model looks is what Chapter 4
trains.
""")

code("""
from ch03.viz_similarity import tracking_grid

fig_3_4 = tracking_grid(            # Figure 3.4
    vision_encoder,
    frames,
    queries=[(11, 5), (11, 8), (9, 6)],   # the cube's cell in each frame
    labels=["cube position 1", "cube position 2", "cube position 3"],
)
""")

md("""
## (Optional) Exercise 3.4: separate-encoder fusion

The main path lets the pretrained backbone fuse the streams. The optional
``FusionTransformer`` is the named alternative: a from-scratch stack of
pre-norm causal self-attention blocks that you bolt onto the frozen
streams and compare against unified-embedding fusion. It is not on the
main path and is not imported by ``UnifiedEmbeddingBackbone``. (Source:
``src/ch03/fusion_transformer.py``.)
""")

code("""
from ch03 import FusionTransformer

fusion_transformer = FusionTransformer()              # hidden_dim=576
dummy = torch.rand(1, 392 + L + 1, 576)
print("fused:", tuple(fusion_transformer(dummy).shape))
""")

md("""
## Summary

You built a VLA backbone from pre-trained parts: a frozen SigLIP vision
encoder, a trainable SmolLM2-135M language backbone, and a state encoder,
fused by unified embedding. The two camera views and the state are
spliced into the backbone's own token stream with ``masked_scatter``, so
the pretrained backbone is the fuser; there is no separate fusion module
on the main path. The backbone maps two images, an instruction, and the
robot state to ``[B, 392 + L + 1, 576]`` hidden states. Chapter 4
attaches an action head to the right end of that sequence and trains the
first working policy.
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
