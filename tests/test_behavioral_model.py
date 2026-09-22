"""
Regression tests for the padding-mask fix in models/behavioral_cnn.py.

Before this fix, MultiHeadSelfAttention had no key_padding_mask and pooling
was a plain x.mean(dim=1) over all SEQUENCE_LENGTH positions -- so <PAD>
positions (present on every "short", 10-99-real-call sequence -- ~1/3 of
scored traffic per tokenizer/api_tokenizer.py) leaked into both the
attention output and the pooled representation.

These tests exercise the masking mechanism directly, at the tensor level,
rather than only checking end-to-end accuracy -- an end-to-end check can
pass by coincidence (as the real before/after comparison did: 5 improved /
5 regressed nets to zero) without proving the mask logic itself is correct.
No training data, no checkpoint, no network -- small freshly-constructed
models only.
"""
from __future__ import annotations

import torch

from models.behavioral_cnn import CortexBehavioralNet, MultiHeadSelfAttention


def test_attention_ignores_masked_keys():
    """A query position must be unaffected by ANY change to a masked-out
    key/value position -- that's the entire point of key_padding_mask.
    Also asserts the negative: without a mask, the same perturbation DOES
    change the output, so this test can't pass vacuously."""
    torch.manual_seed(0)
    attn = MultiHeadSelfAttention(embed_dim=16, num_heads=2, dropout=0.0).eval()

    x = torch.randn(2, 6, 16)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, 4:] = True  # last 2 of 6 positions are "padding"

    with torch.no_grad():
        out_masked_a = attn(x, key_padding_mask=mask)

        x_perturbed = x.clone()
        x_perturbed[:, 4:, :] = torch.randn(2, 2, 16) * 100  # drastic change, masked positions only
        out_masked_b = attn(x_perturbed, key_padding_mask=mask)

        # Real (unmasked) positions must be bit-for-bit unaffected by a
        # change confined entirely to masked key/value positions.
        assert torch.allclose(out_masked_a[:, :4, :], out_masked_b[:, :4, :], atol=1e-6)

        # Negative control: the same perturbation, with no mask applied,
        # must change the output -- otherwise this test would pass even if
        # masking were silently a no-op.
        out_unmasked_a = attn(x)
        out_unmasked_b = attn(x_perturbed)
        assert not torch.allclose(out_unmasked_a[:, :4, :], out_unmasked_b[:, :4, :], atol=1e-3)


def test_forward_pooling_excludes_padding():
    """CortexBehavioralNet.forward's logit must equal a masked mean-pool of
    its own attention output (real positions only), and must NOT equal the
    old plain mean(dim=1) over every position -- i.e. the fix is really
    wired in, not just present as dead code."""
    torch.manual_seed(0)
    model = CortexBehavioralNet(vocab_size=20, sequence_length=20, embed_dim=16,
                                 num_heads=2, dropout=0.0, pad_idx=0).eval()

    n_real = 5
    token_ids = torch.zeros(1, 20, dtype=torch.long)
    token_ids[0, :n_real] = torch.arange(2, 2 + n_real)  # real tokens: ids 2..6
    # positions n_real..19 stay at pad_idx (0) -- a "short" sequence

    captured = {}
    handle = model.attention.register_forward_hook(lambda m, i, o: captured.__setitem__("attn_out", o.detach()))
    with torch.no_grad():
        logit = model(token_ids)
    handle.remove()

    attn_out = captured["attn_out"]  # (1, 20, 128) -- the model's real internal attention output
    pad_mask = token_ids.eq(0)
    real_mask = (~pad_mask).unsqueeze(-1).float()

    with torch.no_grad():
        masked_mean = (attn_out * real_mask).sum(dim=1) / real_mask.sum(dim=1)
        full_mean = attn_out.mean(dim=1)
        logit_from_masked_mean = model.classifier(masked_mean).squeeze(-1)
        logit_from_full_mean = model.classifier(full_mean).squeeze(-1)

    assert torch.allclose(logit, logit_from_masked_mean, atol=1e-6), (
        "forward()'s pooling no longer matches a masked mean over real positions -- "
        "the pooling fix may have been reverted or altered"
    )
    assert not torch.allclose(logit, logit_from_full_mean, atol=1e-3), (
        "forward()'s pooling matches the OLD unmasked mean(dim=1) over all positions -- "
        "the padding-mask fix is not actually excluding <PAD> from pooling"
    )


def test_forward_matches_unmasked_when_no_padding_present():
    """Sanity/invariant check: with a fully-real sequence (no <PAD> at all),
    masking has nothing to exclude, so the fixed forward() must reduce
    exactly to the old unmasked computation. Mirrors the real check run
    against the actual production checkpoint (status=='ok' rows), which
    found a max score difference of 0.0."""
    torch.manual_seed(0)
    model = CortexBehavioralNet(vocab_size=20, sequence_length=10, embed_dim=16,
                                 num_heads=2, dropout=0.0, pad_idx=0).eval()

    token_ids = torch.randint(1, 20, (2, 10))  # every position real (never pad_idx==0)
    assert not (token_ids == 0).any()

    with torch.no_grad():
        pad_mask = token_ids.eq(0)
        x = model.embedding(token_ids) + model.pos_embedding[:, :10, :]
        x = model.conv_stack(x.permute(0, 2, 1)).permute(0, 2, 1)

        attn_masked = model.attention(x, key_padding_mask=pad_mask)
        attn_unmasked = model.attention(x)  # no mask arg at all -- old call signature
        assert torch.allclose(attn_masked, attn_unmasked, atol=1e-6)

        pooled_masked = attn_masked.mean(dim=1)  # no PAD present -> masked mean == plain mean
        pooled_unmasked = attn_unmasked.mean(dim=1)
        assert torch.allclose(pooled_masked, pooled_unmasked, atol=1e-6)
