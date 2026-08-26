"""One Triton kernel for the decode-time q_probs build, replacing a four-kernel ATen chain.

WHAT IT REPLACES.  `Attention.soft_hash` in model.py computes, per (b, h, l):

    proj[k]  = bf16( sum_d q[d] * planes[l,k,d] )          a bf16 GEMV, fp32 accumulation
    t[k]     = bf16( bf16(tanh(fp32(proj[k]))) / temp )    torch.tanh, then a scalar divide
    logit[r] = bf16( sum_k t[k] * protos_T[k,r] )          a second bf16 GEMV
    out[r]   = bf16( softmax_r( fp32(logit[r]) / tau ) )

as four separate launches over tensors of a few hundred kilobytes.  At those sizes the work
is launch latency rather than data movement, so folding the chain into a single kernel with
two block-wide reductions is worth more than anything done inside it.

NOT BITWISE.  Everything here is arranged to reproduce the eager chain's rounding -- every
intermediate is round-tripped through bf16 at the point the ATen chain would materialize a
bf16 tensor -- but a fused reduction is not obliged to associate the same way a GEMV does, so
expect agreement to about one bf16 ulp rather than exactly.  For calibration, Inductor's own
compiled version of the same chain is not bitwise equal to eager either.  ROUND_TRIP=1 (the default) matches EAGER; Inductor
keeps the (logit / tau) intermediate in fp32 across the fused divide and softmax, which is
ROUND_TRIP=0.

ON BY DEFAULT.  SOCKET_FUSED_SOFTHASH=0 restores the ATen chain for A/B comparison.

WHAT IS DELIBERATELY NOT HERE.  An earlier version of this file also fused the per-step KV
metadata (the new token's bucket code and value norm).  That kernel put one program on each
(batch, kv-head) -- eight CTAs on a 132-SM GPU -- and ran L*K sequentially dependent
block-wide reductions inside it.  It is not ported.
"""
import os

import torch
import triton
import triton.language as tl

try:
    from torch.library import triton_op, wrap_triton
    _HAS_TRITON_OP = True
except Exception:  # pragma: no cover - torch < 2.6
    _HAS_TRITON_OP = False

    def wrap_triton(k):  # type: ignore
        return k

# `tanh` and `exp` must come from the SAME source Inductor uses for the baseline kernels, or
# the fused result is not comparable with it: Inductor emits libdevice calls.
#
# Import the FUNCTIONS unqualified, never the module.  Inductor rebuilds a user-defined Triton
# kernel's source in a compile subprocess and only re-emits an import for a referenced global
# when `symbol.__module__` starts with "triton".  A module object has no `__module__`, so
# `libdevice.tanh(x)` inside the kernel raises NameError under torch.compile while working
# fine in eager.  A function does have one.
try:
    from triton.language.extra.libdevice import tanh as _libdev_tanh
    from triton.language.extra.libdevice import exp as _libdev_exp
except Exception:  # pragma: no cover - older triton layout
    from triton.language.extra.cuda.libdevice import tanh as _libdev_tanh
    from triton.language.extra.cuda.libdevice import exp as _libdev_exp


def _sh_warps(K: int, R: int) -> int:
    """num_warps for the q_probs kernel.

    The optimum tracks the SIZE of the [BLOCK_K, BLOCK_R] tile rather than L: the wide tile a
    large R produces has enough parallelism for four warps, while a narrow one is fastest on a
    single warp and loses badly as warps are added.
    """
    bk = triton.next_power_of_2(K)
    br = triton.next_power_of_2(R)
    return 4 if bk * br >= 8192 else 1


# ROUND_TRIP is a tl.constexpr fed from a module-level literal, not from the environment: a
# constexpr read from os.environ inside a traced region is baked at trace time and is not part
# of Inductor's FX-graph cache key, so two runs sharing a cache and differing only in it can
# silently execute the same kernel. See the same argument in sparse.py.
_ROUND_TRIP = True


@triton.jit
def _fwd_kernel_soft_hash_qprobs(
    Q, Planes, ProtosT, Out, inv_temp, tau,
    H: tl.constexpr, D: tl.constexpr, L: tl.constexpr, K: tl.constexpr, R: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_R: tl.constexpr,
    ROUND_TRIP: tl.constexpr,
):
    l = tl.program_id(0)
    bh = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    offs_r = tl.arange(0, BLOCK_R)
    mask_r = offs_r < R

    # First GEMV: q against this table's K planes, as one block-wide reduction.
    q = tl.load(Q + bh * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    pl = tl.load(Planes + (l * K + offs_k[:, None]) * D + offs_d[None, :],
                 mask=mask_k[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    pr = tl.sum(q[None, :] * pl, axis=1)
    pr = pr.to(tl.bfloat16).to(tl.float32)
    th = _libdev_tanh(pr).to(tl.bfloat16).to(tl.float32)
    th = (th * inv_temp).to(tl.bfloat16).to(tl.float32)
    th = tl.where(mask_k, th, 0.0)

    # Second GEMV: the tanh'd projections against the hypercube corners.
    pt = tl.load(ProtosT + offs_k[:, None] * R + offs_r[None, :],
                 mask=mask_k[:, None] & mask_r[None, :], other=0.0).to(tl.float32)
    acc = tl.sum(th[:, None] * pt, axis=0)

    # Softmax over the R bucket logits, with the masked lanes pushed to -inf so they neither
    # win the max nor contribute to the sum.
    logit = acc.to(tl.bfloat16).to(tl.float32)
    x = logit / tau
    if ROUND_TRIP:
        x = x.to(tl.bfloat16).to(tl.float32)
    x = tl.where(mask_r, x, float("-inf"))
    m = tl.max(x, axis=0)
    e = _libdev_exp(x - m)
    e = tl.where(mask_r, e, 0.0)
    s = tl.sum(e, axis=0)
    tl.store(Out + (bh * L + l) * R + offs_r, (e / s).to(Out.dtype.element_ty), mask=mask_r)


def _soft_hash_qprobs_impl(
    q_bhd: torch.Tensor,       # [B,H,D]
    planes: torch.Tensor,      # [L,K,D]
    protos_T: torch.Tensor,    # [K,R]
    out: torch.Tensor,         # [B,H,L,R] (written)
    inv_temp: float, tau: float,
) -> None:
    B, H, D = q_bhd.shape
    L, K, _ = planes.shape
    R = protos_T.shape[1]
    wrap_triton(_fwd_kernel_soft_hash_qprobs)[(L, B * H)](
        q_bhd, planes, protos_T, out, inv_temp, tau,
        H=H, D=D, L=L, K=K, R=R,
        BLOCK_D=triton.next_power_of_2(D),
        BLOCK_K=triton.next_power_of_2(K),
        BLOCK_R=triton.next_power_of_2(R),
        ROUND_TRIP=_ROUND_TRIP,
        num_warps=_sh_warps(K, R), num_stages=1,
    )


if _HAS_TRITON_OP:
    soft_hash_qprobs_op = triton_op(
        "socket::soft_hash_qprobs", _soft_hash_qprobs_impl, mutates_args={"out"})
else:  # pragma: no cover - torch < 2.6
    soft_hash_qprobs_op = torch.no_grad()(_soft_hash_qprobs_impl)


@torch.no_grad()
def fused_soft_hash(q_bhd, planes, protos_T, temp, tau, out_dtype):
    """Decode-only replacement for Attention.soft_hash. Returns q_probs [B,H,L,R]."""
    B, H, _ = q_bhd.shape
    L = planes.shape[0]
    R = protos_T.shape[1]
    out = torch.empty((B, H, L, R), device=q_bhd.device, dtype=out_dtype)
    soft_hash_qprobs_op(q_bhd, planes, protos_T, out, 1.0 / max(temp, 1e-6), float(tau))
    return out
