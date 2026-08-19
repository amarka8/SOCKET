"""Regression test for the empty-BLOCK_N-chunk NaN in the sparse-decode online softmax.

THE BUG (pre-existing, fixed in 6ee1d2f): _fwd_kernel_sparse_decode_stage1 walks the sparse
list in BLOCK_N-wide chunks. When EVERY slot of a chunk is padding (-1) or past cur_seq_len,
att_value is all -inf; if max_logic is also still -inf (no valid token seen yet) then
new_max_logic is -inf and both

    exp_logic   = exp(att_value - new_max_logic)
    logic_scale = exp(max_logic  - new_max_logic)

evaluate exp(-inf - -inf) = exp(nan) = nan, poisoning acc/sum_exp for the whole block. Stage2
merges the NaN partial and it reaches the logits.

It never fired in the original benchmark because torch.topk returned the heavy list in
descending-score order, which placed the -1 dedup holes where they never filled a chunk.
Selecting the SAME SET in a different order (topk sorted=False) makes a fully-padded chunk
likely, and the model then emits ALL-NaN logits (greedy output collapses to token 0).

The test drives it directly by placing a fully-padded chunk in the list. Run with
SOCKET_NAN_GUARD=0 to compile the guard out and reproduce the ORIGINAL behaviour, which is
what makes this a genuine regression test: it FAILS (NaN) on the pre-fix code path and
PASSES on the fixed one.

  pytest tests/test_empty_chunk_nan.py -v                     # guard on  -> passes
  SOCKET_NAN_GUARD=0 pytest tests/test_empty_chunk_nan.py -v  # guard off -> test_bug_reproduces
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

B, HQ, HKV, D = 1, 4, 2, 128
BLOCK_SEQ, BLOCK_N = 256, 16
SEQ_LEN = 4096


def _build(pad_chunk: bool):
    """Sparse list whose SECOND BLOCK_N chunk is entirely padding when pad_chunk=True."""
    width = BLOCK_SEQ
    lst = torch.full((B, HQ, width), -1, dtype=torch.int32, device="cuda")
    for h in range(HQ):
        idx = torch.arange(width, device="cuda", dtype=torch.int32) % SEQ_LEN
        lst[0, h] = idx
        if pad_chunk:
            # blank exactly one BLOCK_N-wide chunk -> that chunk is all -inf in the kernel
            lst[0, h, BLOCK_N:2 * BLOCK_N] = -1
    return lst


def _run(lst):
    from kernels.sparse import sparse_attention_fwd
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn((B, HQ, D), generator=g, device="cuda", dtype=torch.bfloat16)
    k = torch.randn((B, HKV, SEQ_LEN, D), generator=g, device="cuda", dtype=torch.bfloat16)
    v = torch.randn((B, HKV, SEQ_LEN, D), generator=g, device="cuda", dtype=torch.bfloat16)
    ln = torch.full((B, HQ), lst.shape[-1], device="cuda", dtype=torch.int32)
    return sparse_attention_fwd(q, k, v, lst, ln, block_seq=BLOCK_SEQ)


def _reference(lst):
    """Dense reference over exactly the non-padding slots (dedup: each token once)."""
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn((B, HQ, D), generator=g, device="cuda", dtype=torch.bfloat16)
    k = torch.randn((B, HKV, SEQ_LEN, D), generator=g, device="cuda", dtype=torch.bfloat16)
    v = torch.randn((B, HKV, SEQ_LEN, D), generator=g, device="cuda", dtype=torch.bfloat16)
    rep = HQ // HKV
    out = torch.zeros((B, HQ, D), device="cuda", dtype=torch.float32)
    for h in range(HQ):
        sel = lst[0, h]
        sel = sel[sel >= 0].long()
        kk = k[0, h // rep, sel].float()
        vv = v[0, h // rep, sel].float()
        att = (q[0, h].float() @ kk.T) / (D ** 0.5)
        out[0, h] = torch.softmax(att, dim=-1) @ vv
    return out


def test_no_padding_is_finite():
    """Control: with no fully-padded chunk the kernel is finite either way."""
    out = _run(_build(pad_chunk=False))
    assert torch.isfinite(out).all(), "kernel produced non-finite output on a dense list"


@pytest.mark.skipif(os.environ.get("SOCKET_NAN_GUARD", "1") != "1",
                    reason="guard disabled; see test_bug_reproduces")
def test_fully_padded_chunk_is_finite_and_correct():
    """THE REGRESSION. A fully padded BLOCK_N chunk must not poison the block."""
    lst = _build(pad_chunk=True)
    out = _run(lst)
    assert torch.isfinite(out).all(), (
        "empty-BLOCK_N-chunk NaN regression: a fully padded chunk produced non-finite "
        "output (exp(-inf - -inf) = nan in the online softmax)")
    ref = _reference(lst)
    err = (out.float() - ref).abs().max().item()
    assert err < 2e-2, f"padded-chunk output disagrees with the dense reference: {err}"


@pytest.mark.skipif(os.environ.get("SOCKET_NAN_GUARD", "1") == "1",
                    reason="only meaningful with SOCKET_NAN_GUARD=0")
def test_bug_reproduces_without_guard():
    """With the guard compiled out we must see the ORIGINAL NaN -- proves the test bites."""
    out = _run(_build(pad_chunk=True))
    assert not torch.isfinite(out).all(), (
        "expected the pre-fix NaN with SOCKET_NAN_GUARD=0; if this passes, the test no "
        "longer reproduces the bug it is guarding against")
