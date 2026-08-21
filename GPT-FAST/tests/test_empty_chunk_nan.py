"""Regression test for the empty-chunk NaN in the sparse-decode online softmax.

THE BUG. _fwd_kernel_sparse_decode_stage1 walks each BLOCK_SEQ partition of the sparse list
in BLOCK_N-wide chunks, resetting the running max to -inf at the start of every partition.
When EVERY slot of a chunk is padding (-1) or past cur_seq_len, att_value is all -inf; if the
running max is also still -inf then new_max_logic is -inf and BOTH

    exp_logic   = exp(att_value - new_max_logic)
    logic_scale = exp(max_logic  - new_max_logic)

evaluate exp(-inf - -inf) = exp(nan) = nan, poisoning acc/sum_exp for the whole partition.
A fully padded partition additionally reaches the store with sum_exp == 0, i.e. 0/0 = nan and
log(0) = -inf, and stage2 merges the poisoned partial straight into the output.

The list produced by build_sparse_list_decode contains -1 holes (unfilled cache columns and
heavy tokens deduplicated against sink/window), so this is reachable on real inputs: it
produced ALL-NaN logits and a greedy stream collapsing to token 0.

The test drives it directly with a fully padded LEADING partition, which is independent of
BLOCK_N and exercises all four guards (stage1 softmax, stage1 store, stage2 merge, stage2
store). To see the pre-fix behaviour, revert the guard commit and re-run: this test then
fails with non-finite output.

Standalone runner (pytest cannot import in this cluster's module stack: its anyio dependency
pulls in ssl, which fails with an OPENSSL_3.3.0 mismatch):

    python GPT-FAST/tests/test_empty_chunk_nan.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

B, HQ, HKV, D = 1, 4, 2, 128
BLOCK_SEQ = 256
SEQ_LEN = 4096


def _build(pad_first_block: bool):
    """Two BLOCK_SEQ partitions; optionally make the FIRST one entirely padding.

    Position matters: stage1 resets max_logic = -inf at the start of every BLOCK_SEQ
    partition, so exp(-inf - -inf) = nan only arises while that partition's running max is
    still -inf. A padded chunk that FOLLOWS a valid token in the same partition is harmless,
    which is why the bug is positional (and why descending-score list order hid it).
    """
    width = 2 * BLOCK_SEQ
    lst = torch.arange(width, device="cuda", dtype=torch.int32) % SEQ_LEN
    lst = lst.view(1, 1, width).expand(B, HQ, width).contiguous()
    if pad_first_block:
        lst[:, :, :BLOCK_SEQ] = -1
    return lst


def _inputs():
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn((B, HQ, D), generator=g, device="cuda", dtype=torch.bfloat16)
    k = torch.randn((B, HKV, SEQ_LEN, D), generator=g, device="cuda", dtype=torch.bfloat16)
    v = torch.randn((B, HKV, SEQ_LEN, D), generator=g, device="cuda", dtype=torch.bfloat16)
    return q, k, v


def _run(lst):
    from kernels.sparse import sparse_attention_fwd
    q, k, v = _inputs()
    ln = torch.full((B, HQ), lst.shape[-1], device="cuda", dtype=torch.int32)
    return sparse_attention_fwd(q, k, v, lst, ln, block_seq=BLOCK_SEQ)


def _reference(lst):
    """Dense softmax over exactly the non-padding slots."""
    q, k, v = _inputs()
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
    """Control: with no padding the kernel is finite either way."""
    out = _run(_build(pad_first_block=False))
    assert torch.isfinite(out).all(), "non-finite output on a fully dense list"


def test_fully_padded_partition_is_finite_and_correct():
    """THE REGRESSION. A fully padded leading partition must not poison the output."""
    lst = _build(pad_first_block=True)
    out = _run(lst)
    n = int((~torch.isfinite(out)).sum())
    assert n == 0, (
        f"empty-chunk NaN regression: {n}/{out.numel()} non-finite outputs from a fully "
        f"padded leading partition (exp(-inf - -inf) = nan in the online softmax)")
    err = (out.float() - _reference(lst)).abs().max().item()
    assert err < 2e-2, f"padded-partition output disagrees with the dense reference: {err}"


def _main():
    if not torch.cuda.is_available():
        print("SKIP (no CUDA)")
        return 0
    rc = 0
    for fn in (test_no_padding_is_finite, test_fully_padded_partition_is_finite_and_correct):
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            rc = 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print("OK" if rc == 0 else "FAILURES")
    return rc


if __name__ == "__main__":
    raise SystemExit(_main())
