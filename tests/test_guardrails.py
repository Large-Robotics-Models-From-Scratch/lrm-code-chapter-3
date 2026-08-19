"""Source-scan guardrails (pure, no model download).

Enforces the Chapter 3 / Chapter 4 boundary: vocab expansion and action
tokens belong to Chapter 4. None of the Chapter 3 source may call them.
"""

from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "ch03"
# add_tokens / resize_token_embeddings: vocabulary expansion belongs
# to Chapter 4. masked_scatter: the retired splice construction; the
# chapter now concatenates streams and feeds inputs_embeds, and the
# only sanctioned copy of the old path lives in
# tests/test_migration_parity.py.
FORBIDDEN = ["add_tokens", "resize_token_embeddings", "masked_scatter"]


@pytest.mark.parametrize("forbidden", FORBIDDEN)
def test_no_vocab_expansion_in_source(forbidden):
    offenders = [
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if forbidden in path.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        f"'{forbidden}' belongs to Chapter 4, found in {offenders}"
    )
