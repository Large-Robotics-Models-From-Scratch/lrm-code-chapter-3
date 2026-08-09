"""Tests for VLABackbone (concat design, two-stage interface).

Marked integration (downloads SigLIP and SmolLM2).
"""

import pytest
import torch

from ch03 import VLABackbone
from ch03.vision_encoder import NUM_PATCHES, VisionEncoder
from ch03.vla_backbone import IMAGE_TOKENS, SMOLLM_VOCAB, SMOLLM_WIDTH

# Frozen SigLIP (~92.9M) plus SmolLM2-135M and two small projections.
# One embedding table is ~28.3M rows, so a duplicated table would push
# trainable past this ceiling.
MAX_TRAINABLE_PARAMS = 140_000_000


@pytest.fixture(scope="module")
def backbone():
    return VLABackbone().eval()


def _two_camera_batch(batch_size: int) -> torch.Tensor:
    """[B, 2, 3, 224, 224] random frames in [0, 1] (two camera views)."""
    return torch.rand(batch_size, 2, 3, 224, 224)


def _text_ids(ids: list[int], batch_size: int) -> torch.Tensor:
    return torch.tensor([ids] * batch_size, dtype=torch.long)


@pytest.mark.integration
def test_ch4_contract_shape(backbone, dummy_state):
    # Hidden states are [B, 392 + L + 1, 576]; Chapter 4 reads this.
    input_ids = _text_ids([101, 102, 103, 104], 2)
    L = input_ids.shape[1]
    with torch.no_grad():
        hidden = backbone(_two_camera_batch(2), input_ids, dummy_state)
    assert hidden.shape == (2, IMAGE_TOKENS + L + 1, SMOLLM_WIDTH)


@pytest.mark.integration
def test_two_stage_interface_matches_forward(backbone, dummy_state):
    # forward must be exactly embed_inputs -> contextualize, so heads
    # that call the stages separately see the same numbers.
    input_ids = _text_ids([101, 102, 103], 2)
    images = _two_camera_batch(2)
    with torch.no_grad():
        emb, mask, pos = backbone.embed_inputs(
            images, input_ids, dummy_state
        )
        staged = backbone.contextualize(emb, mask, pos)
        direct = backbone(images, input_ids, dummy_state)
    assert torch.equal(staged, direct)


@pytest.mark.integration
def test_embed_inputs_layout(backbone):
    # Shapes and counts can be right while the wrong vector sits in a
    # slot: a transposed reshape, a swapped camera, or interleaved
    # batch rows would keep the contract shape and silently train a
    # broken policy. Check the assembled sequence position by
    # position, with a batch of 2 and four visually distinct frames.
    torch.manual_seed(0)
    images = torch.rand(2, 2, 3, 224, 224)
    ids = [101, 102, 103, 104, 105]
    L = len(ids)
    input_ids = _text_ids(ids, 2)
    state = torch.rand(2, 6)

    with torch.no_grad():
        emb, mask, pos = backbone.embed_inputs(images, input_ids, state)

    assert emb.shape == (2, IMAGE_TOKENS + L + 1, SMOLLM_WIDTH)
    assert mask.shape == (2, IMAGE_TOKENS + L + 1)
    assert bool((mask == 1).all())  # nothing padded here

    table = backbone.language_backbone.get_input_embeddings()
    assert emb.dtype == table.weight.dtype
    text_rows = table(torch.tensor(ids))
    with torch.no_grad():
        for b in range(2):
            cam0 = backbone.vision_encoder(images[b, 0].unsqueeze(0))[0]
            cam1 = backbone.vision_encoder(images[b, 1].unsqueeze(0))[0]
            # Distinct frames give distinct tokens, so none of the
            # comparisons below can pass by coincidence.
            assert not torch.allclose(cam0, cam1, atol=1e-3)
            # Camera 0 (overhead) occupies positions 0..195, camera 1
            # (side) 196..391 - batched through the encoder in one
            # call, hence the float32 reduction-order tolerance.
            torch.testing.assert_close(
                emb[b, :NUM_PATCHES], cam0, atol=1e-4, rtol=1e-4
            )
            torch.testing.assert_close(
                emb[b, NUM_PATCHES:IMAGE_TOKENS],
                cam1,
                atol=1e-4,
                rtol=1e-4,
            )
            # The text block is a plain table lookup: exact.
            assert torch.equal(
                emb[b, IMAGE_TOKENS : IMAGE_TOKENS + L], text_rows
            )
            # The last position is this row's state token.
            state_token = backbone.state_encoder(
                state[b].unsqueeze(0)
            )[0, 0]
            torch.testing.assert_close(
                emb[b, -1], state_token, atol=1e-5, rtol=1e-5
            )


@pytest.mark.integration
def test_padded_batch_positions(backbone):
    # Padding sits between the text block and the state embedding, so
    # position ids must count valid slots only: every row's state
    # token gets rotary position 392 + L_valid, however much padding
    # the batch carries.
    images = _two_camera_batch(2)
    state = torch.rand(2, 6)
    pad = backbone.tokenizer.pad_token_id
    input_ids = torch.tensor(
        [[101, 102, 103, pad, pad], [101, 102, 103, 104, 105]]
    )
    text_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])

    with torch.no_grad():
        emb, mask, pos = backbone.embed_inputs(
            images, input_ids, state, text_mask
        )

    # Mask: images and state are always valid; text follows text_mask.
    assert bool((mask[:, :IMAGE_TOKENS] == 1).all())
    assert bool((mask[:, -1] == 1).all())
    assert torch.equal(mask[:, IMAGE_TOKENS:-1], text_mask)

    # Image positions are 0..391 in both rows.
    expected_img = torch.arange(IMAGE_TOKENS)
    assert torch.equal(pos[0, :IMAGE_TOKENS], expected_img)
    assert torch.equal(pos[1, :IMAGE_TOKENS], expected_img)

    # State position is 392 + L_valid per row, not the physical index.
    assert pos[0, -1].item() == IMAGE_TOKENS + 3
    assert pos[1, -1].item() == IMAGE_TOKENS + 5
    # Padded slots are masked out and pinned to position 0.
    assert bool((pos[0, IMAGE_TOKENS + 3 : -1] == 0).all())


@pytest.mark.integration
def test_padded_vs_unpadded_equivalence(backbone):
    # The proof the mask and position ids work: the same instruction
    # with and without trailing pad columns must give the same
    # contextualized state representation.
    torch.manual_seed(1)
    images = _two_camera_batch(1)
    state = torch.rand(1, 6)
    ids = [101, 102, 103, 104]
    pad = backbone.tokenizer.pad_token_id

    unpadded_ids = torch.tensor([ids])
    padded_ids = torch.tensor([ids + [pad, pad]])
    padded_mask = torch.tensor([[1, 1, 1, 1, 0, 0]])

    with torch.no_grad():
        clean = backbone(images, unpadded_ids, state)
        padded = backbone(images, padded_ids, state, padded_mask)

    # The state token is physically last in both layouts.
    torch.testing.assert_close(
        padded[0, -1], clean[0, -1], atol=1e-4, rtol=1e-4
    )
    # And the real text positions match too.
    torch.testing.assert_close(
        padded[0, IMAGE_TOKENS : IMAGE_TOKENS + 4],
        clean[0, IMAGE_TOKENS : IMAGE_TOKENS + 4],
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.integration
def test_raw_resolution_accepted(backbone, dummy_state):
    # Chapter 2 ships [B, 2, 3, 480, 640]; the vision module resizes
    # internally, so raw frames flow through the whole backbone.
    images = torch.rand(2, 2, 3, 480, 640)
    input_ids = _text_ids([101, 102, 103], 2)
    with torch.no_grad():
        hidden = backbone(images, input_ids, dummy_state)
    assert hidden.shape == (2, IMAGE_TOKENS + 3 + 1, SMOLLM_WIDTH)


@pytest.mark.integration
def test_malformed_images_rejected(backbone, dummy_state):
    # A single-camera batch would silently produce a 196-token image
    # block; embed_inputs must reject it instead.
    input_ids = _text_ids([101, 102], 2)
    one_camera = torch.rand(2, 1, 3, 224, 224)
    with pytest.raises(ValueError, match="images must be"):
        backbone.embed_inputs(one_camera, input_ids, dummy_state)


@pytest.mark.integration
def test_no_vocab_expansion(backbone):
    # Chapter 3 never touches the vocabulary: no placeholder rows, no
    # grown table, no resize. Chapter 4 owns vocabulary expansion.
    assert backbone.language_backbone.config.vocab_size == SMOLLM_VOCAB
    table = backbone.language_backbone.get_input_embeddings()
    assert table.num_embeddings == SMOLLM_VOCAB


@pytest.mark.integration
def test_pad_token_reuses_eos(backbone):
    # SmolLM2 ships no pad token; the backbone reuses end-of-text.
    # Reusing an existing id keeps the vocabulary untouched.
    assert backbone.tokenizer.pad_token_id is not None
    assert (
        backbone.tokenizer.pad_token_id == backbone.tokenizer.eos_token_id
    )


@pytest.mark.integration
def test_vision_path_is_the_composed_vision_encoder(backbone):
    # The backbone composes section 3.2's VisionEncoder; it does not
    # rebuild the vision path. The 768 to 576 projection lives inside
    # that encoder, so there is no second img_proj on the backbone.
    assert isinstance(backbone.vision_encoder, VisionEncoder)
    assert not hasattr(backbone, "img_proj")


@pytest.mark.integration
def test_composed_vision_path_keeps_pixel_normalization(backbone):
    # SigLIP wants [-1, 1]; the buffers that map [0, 1] into it must
    # still be on the composed path, and the projection must still be
    # inside it (output width 576, not SigLIP's raw 768).
    encoder = backbone.vision_encoder
    assert encoder.pixel_mean.item() == 0.5
    assert encoder.pixel_std.item() == 0.5

    frames = torch.rand(2, 3, 224, 224)  # [0, 1], as Chapter 2 ships
    with torch.no_grad():
        composed = encoder(frames)
    assert composed.shape == (2, NUM_PATCHES, SMOLLM_WIDTH)

    # Same weights in a standalone VisionEncoder give the same
    # numbers, which pins the whole path, not just the width.
    standalone = VisionEncoder(hidden_dim=SMOLLM_WIDTH).eval()
    standalone.load_state_dict(encoder.state_dict())
    with torch.no_grad():
        reference = standalone(frames)
    assert torch.equal(composed, reference)

    # And normalization is load-bearing: skipping it changes output.
    with torch.no_grad():
        unnormalized = encoder.project(
            encoder.siglip(pixel_values=frames).last_hidden_state
        )
    assert not torch.allclose(composed, unnormalized, atol=1e-4)


@pytest.mark.integration
def test_trainable_parameter_budget(backbone):
    # A duplicated embedding table would add ~28.3M trainable params.
    trainable = sum(
        p.numel() for p in backbone.parameters() if p.requires_grad
    )
    assert trainable < MAX_TRAINABLE_PARAMS, trainable


@pytest.mark.integration
def test_frozen_vision_stays_frozen(backbone):
    # Composing VisionEncoder must not thaw SigLIP.
    siglip = backbone.vision_encoder.siglip
    assert all(not p.requires_grad for p in siglip.parameters())
    backbone.train()
    assert not siglip.training
    backbone.eval()
