"""Migration parity: the concat backbone equals the retired splice.

Chapter 3 originally built the fused sequence by growing SmolLM2's
embedding table with two placeholder rows (image id 49152, state id
49153), templating an integer row ``[image x392, text, state]``, and
overwriting the placeholder rows with ``masked_scatter``. The chapter
now concatenates the encoded streams directly and feeds the backbone
through ``inputs_embeds``.

The two constructions are mathematically the same sequence, so for an
unpadded input the retired path and the current ``forward`` must
produce the same hidden states. This test keeps a frozen, minimal copy
of the old path (built from the SAME live components, so weights match
by construction) and pins that equivalence permanently. If it ever
fails, the migration changed the model, not just the teaching.
"""

import pytest
import torch
import torch.nn as nn

from ch03 import VLABackbone
from ch03.vla_backbone import IMAGE_TOKENS, SMOLLM_VOCAB, SMOLLM_WIDTH

IMAGE_ID = SMOLLM_VOCAB
STATE_ID = SMOLLM_VOCAB + 1


def _legacy_splice_forward(
    backbone: VLABackbone,
    images: torch.Tensor,
    text_ids: list[int],
    state: torch.Tensor,
) -> torch.Tensor:
    """The retired placeholder-ID + masked_scatter path, verbatim.

    Reuses the live backbone's encoders and language model so the
    weights are identical; only the sequence construction differs.
    """
    B = images.shape[0]
    base = backbone.language_backbone.get_input_embeddings()

    # Grow a throwaway table two rows past the vocab, as the old
    # __init__ did. The extra rows are inert: the splice overwrites
    # them before the Transformer runs.
    grown = nn.Embedding(
        STATE_ID + 1,
        base.embedding_dim,
        dtype=base.weight.dtype,
        device=base.weight.device,
    )
    with torch.no_grad():
        grown.weight[:SMOLLM_VOCAB] = base.weight

    sequence_ids = torch.tensor(
        [[IMAGE_ID] * IMAGE_TOKENS + text_ids + [STATE_ID]] * B
    )

    patches = backbone.vision_encoder(images.flatten(0, 1))
    img = patches.reshape(B, -1, SMOLLM_WIDTH)
    emb = grown(sequence_ids)
    img = img.to(emb.dtype)
    # The old StateEncoder returned [B, 576]; the current one returns
    # [B, 1, 576]. masked_scatter copies element-wise in row-major
    # order, so both layouts fill the single state row identically.
    state_tok = backbone.state_encoder(state).to(emb.dtype)
    img_mask = (sequence_ids == IMAGE_ID).unsqueeze(-1)
    st_mask = (sequence_ids == STATE_ID).unsqueeze(-1)
    emb = emb.masked_scatter(img_mask, img)
    emb = emb.masked_scatter(st_mask, state_tok)
    return backbone.language_backbone(
        inputs_embeds=emb
    ).last_hidden_state


@pytest.mark.integration
def test_concat_equals_retired_splice():
    backbone = VLABackbone().eval()
    torch.manual_seed(7)
    images = torch.rand(2, 2, 3, 224, 224)
    text_ids = [101, 102, 103, 104, 105]
    state = torch.rand(2, 6)
    input_ids = torch.tensor([text_ids] * 2)

    with torch.no_grad():
        legacy = _legacy_splice_forward(backbone, images, text_ids, state)
        current = backbone(images, input_ids, state)

    assert legacy.shape == current.shape
    # The legacy path passed neither attention_mask nor position_ids;
    # the current path passes all-ones and 0..N-1, which are the same
    # semantics through a different Hugging Face code path, hence a
    # tight tolerance rather than bitwise equality.
    torch.testing.assert_close(current, legacy, atol=1e-5, rtol=1e-5)
