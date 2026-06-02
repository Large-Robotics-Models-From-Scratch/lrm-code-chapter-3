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
vision-language-action backbone: a network that takes an image, an
instruction, and the robot's joint state, and produces a fused sequence
of hidden states ready for an action head (Chapter 4).

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
# reconciled the next cell falls back to a synthetic frame. To use real
# data once the pins are fixed, add:
#   !pip install -q "lrm-ch02[data] @ git+{ORG}/lrm-code-chapter-2.git"
""")

md("""
## 3.1 Load a frame, an instruction, and a state

Chapter 3 imports the data contract Chapter 2 froze:
``make_pickplace_dataloader``. Each batch carries the top camera image
``[B, 3, 480, 640]`` in ``[0, 1]``, the 6-dim joint state, the 6-dim
action (Chapter 4 uses that), and ``task``, the natural-language
instruction. We pull one frame to drive the rest of the chapter. If the
dataset is not available locally, we fall back to a synthetic frame so the
notebook still runs.
""")

code("""
import torch

try:
    from ch02 import make_pickplace_dataloader
    loader, stats = make_pickplace_dataloader(batch_size=8)
    batch = next(iter(loader))
    frame = batch["observation.images.up"][0]      # [3, 480, 640] in [0,1]
    frames = list(batch["observation.images.up"][:3])  # cube in 3 positions
    state = batch["observation.state"][0]          # [6] z-scored
    instruction = batch["task"][0]
except Exception as exc:                       # no [data] extra / no net
    print(f"Using synthetic frames (Chapter 2 data unavailable: {exc})")
    frame = torch.rand(3, 480, 640)
    frames = [torch.rand(3, 480, 640) for _ in range(3)]
    state = torch.zeros(6)
    instruction = "pick up the red cube"

print("frame:", tuple(frame.shape), "| state:", tuple(state.shape))
print("instruction:", instruction)
""")

md("""
## 3.2 The eyes: vision encoder

### Listing 3.1 Loading and freezing SigLIP

``VisionEncoder`` wraps a frozen SigLIP-base/16 and projects its 768-dim
patch tokens to the book's common 512-dim width. A 224x224 image at patch
size 16 gives a 14x14 grid, so we get 196 patch tokens. (Full source:
``src/ch03/vision_encoder.py``.)
""")

code("""
from ch03 import VisionEncoder, preprocess_image

vision_encoder = VisionEncoder().eval()
image_224 = preprocess_image(frame)            # [3,480,640] -> [3,224,224]
patches = vision_encoder(image_224.unsqueeze(0))
print("patch tokens:", tuple(patches.shape))        # [1, 196, 512]
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
    frame,
    [(11, 6, "cube query"), (5, 7, "arm query")],
)
""")

md("""
## 3.3 The brain: language backbone

### Listing 3.3 Loading SmolLM and encoding the instruction

``LanguageBackbone`` uses SmolLM-135M with its native 49,152-token
tokenizer (no vocabulary changes here; that is Chapter 4's first step) and
projects the 576-dim hidden states to 512. Unlike the frozen vision
encoder, SmolLM stays trainable so Chapter 4 can fine-tune it.
""")

code("""
from ch03 import LanguageBackbone

language_backbone = LanguageBackbone()
lang_hidden, lang_mask = language_backbone([instruction])
print("language tokens:", tuple(lang_hidden.shape))  # [1, L, 512]
""")

md("""
## 3.4 Multimodal fusion + state

### Listing 3.4 The state encoder

A two-layer MLP (``Linear -> GELU -> Linear``) lifts the 6 joint numbers
into one 512-dim token that sits beside the image and language tokens.
""")

code("""
from ch03 import StateEncoder

state_encoder = StateEncoder()
state_token = state_encoder(state.unsqueeze(0))
print("state token:", tuple(state_token.shape))      # [1, 1, 512]
""")

md("""
### Listing 3.5 The fusion transformer

Six pre-norm self-attention blocks with causal masking. Causal here is
deliberate: Chapter 4 appends action tokens to the right end of the
sequence and predicts them left to right, so a causal mask now means no
architecture surgery later.
""")

code("""
from ch03 import FusionTransformer

fusion_transformer = FusionTransformer()
dummy = torch.rand(1, 203, 512)                      # 196 + L + state
print("fused:", tuple(fusion_transformer(dummy).shape))
""")

md("""
### Listing 3.6 Composing the VLABackbone

The backbone wires the three streams into one sequence
``[image (196), language (L), state (1)]`` and runs the fusion
transformer. The output ``[B, 196 + L + 1, 512]`` is the contract
Chapter 4 attaches an action head to.
""")

code("""
from ch03 import VLABackbone

backbone = VLABackbone(hidden_dim=512).eval()
with torch.no_grad():
    hidden = backbone(frame.unsqueeze(0), [instruction], state.unsqueeze(0))
print("backbone output:", tuple(hidden.shape))       # [1, 196 + L + 1, 512]
""")

md("""
### Listing 3.7 The frozen encoder tracks the object across frames

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

fig_3_6 = tracking_grid(            # Figure 3.6
    vision_encoder,
    frames,
    queries=[(11, 5), (11, 8), (9, 6)],   # the cube's cell in each frame
    labels=["cube position 1", "cube position 2", "cube position 3"],
)
""")

md("""
## Summary

You built a VLA backbone from pre-trained parts: a frozen SigLIP vision
encoder, a trainable SmolLM language backbone, a state encoder, and a
causal fusion transformer. It maps an image, an instruction, and the robot
state to ``[B, 196 + L + 1, 512]`` hidden states. Chapter 4 attaches an
action head to the right end of that sequence and trains the first working
policy.
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
