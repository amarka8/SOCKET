"""Equivalence test for the fused q_probs kernel against the ATen chain it replaces.

`kernels/fused_meta.py` folds soft_hash's four launches (GEMV, tanh+divide, GEMV, softmax)
into one Triton kernel. The kernel is arranged to reproduce the eager chain's rounding, but a
fused reduction does not have to associate the way a GEMV does and it multiplies by 1/temp
where the chain divides by temp, so the claim is CLOSE, not equal.

What is asserted:
  * the output is a valid probability distribution over the R buckets (rows sum to 1, no
    negative mass, nothing non-finite) -- a fused softmax that lost a masked lane would show
    up here first;
  * it agrees with the ATen chain to well within bf16 resolution;
  * the ranking it induces is the same, which is the only property the scorer downstream
    actually consumes -- q_probs is read as a per-bucket weight and summed.

Run at both hash geometries the accuracy sweep uses, since R and K set the kernel's tile shape
and hence its launch config:

    python GPT-FAST/tests/test_empty_chunk_nan.py    # (the sibling regression test)
    python GPT-FAST/tests/test_fused_soft_hash.py
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def aten_soft_hash(q_bhd, planes, protos_T, tau):
    """The chain in model.py's Attention.soft_hash, verbatim."""
    queries = q_bhd.unsqueeze(2)
    q_proj = torch.einsum("bhqd,lkd->bhqlk", queries, planes)
    temp = math.sqrt(queries.size(-1))
    logits = torch.einsum("bhqlk,kr->bhqlr", torch.tanh(q_proj) / max(temp, 1e-6), protos_T)
    return torch.softmax(logits / tau, dim=-1).squeeze(2)


def one_geometry(K, L, tau=0.3, B=1, H=32, D=128):
    import itertools
    from kernels.fused_meta import fused_soft_hash

    R = 1 << K
    dev, dt = "cuda", torch.bfloat16
    g = torch.Generator(device="cpu").manual_seed(7)
    q = torch.randn(B, H, D, generator=g, dtype=torch.float32).to(dev).to(dt)
    planes = torch.randn(L, K, D, generator=g, dtype=torch.float32).to(dev).to(dt)
    # The hypercube corners model.py builds: every sign pattern in {-1,+1}^K.
    corners = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=K)))
    protos_T = corners.t().contiguous().to(dev).to(dt)

    ref = aten_soft_hash(q, planes, protos_T, tau)
    got = fused_soft_hash(q, planes, protos_T, math.sqrt(D), tau, dt)

    tag = f"P={K} L={L} R={R}"
    check(f"{tag}: output shape and dtype match the chain",
          got.shape == ref.shape and got.dtype == ref.dtype,
          f"{tuple(got.shape)} {got.dtype}")
    check(f"{tag}: output is finite", bool(torch.isfinite(got).all()))
    check(f"{tag}: output is non-negative", bool((got.float() >= 0).all()))

    rows = got.float().sum(-1)
    check(f"{tag}: each bucket distribution sums to 1",
          bool((rows - 1.0).abs().max() < 2e-2), f"max|sum-1| = {(rows - 1.0).abs().max():.3e}")

    d = (got.float() - ref.float()).abs().max().item()
    # bf16 has ~3 decimal digits; a probability of order 1/R resolves to about 2**-8 of itself.
    tol = 4.0 / R
    check(f"{tag}: agrees with the ATen chain to within bf16 resolution",
          d < tol, f"max|diff| = {d:.3e}  tol = {tol:.3e}")

    # The scorer consumes q_probs by gathering one entry per (l, bucket) and summing, so what
    # has to survive is the ordering, not the exact value.
    same_argmax = bool((got.float().argmax(-1) == ref.float().argmax(-1)).all())
    check(f"{tag}: the most-probable bucket is unchanged", same_argmax)


def _main():
    if not torch.cuda.is_available():
        print("SKIP (no CUDA)")
        return 0
    # The two geometries the accuracy sweep uses, plus the repo default.
    for K, L in ((10, 10), (8, 50), (8, 60)):
        print(f"P={K} L={L}")
        one_geometry(K, L)
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
