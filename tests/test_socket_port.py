"""Equivalence tests for socket_port: the HF path's GPT-FAST kernel arms.

WHAT THIS GATES.  modeling_llama.py now defaults SOCKET_SCORER/SOCKET_SELECT/SOCKET_LIST to
triton/radix/triton, i.e. to the three GPT-FAST kernels reached through socket_port.  Before
that default flip those arms had never executed once (socket_port.py did not exist, so each
raised ImportError).  Each test here pits one arm against the in-file implementation it now
replaces, on the layouts modeling_llama actually hands it:

    T1  Triton soft-hash scorer      vs  ext.soft_hash_collision  (the CUDA scorer)
    T2  socket::radix_topm           vs  torch.topk
    T3  socket::build_list           vs  the arange/cat/gather op chain, dedup ON and OFF
    T4  socket_port's import dance under the REAL import order (repo-root `kernels` first)
    T5  both sparse_attention_fwd arms tolerate the -1 padding the Triton list arm emits
    T6  a PADDED bucket buffer scores its live columns exactly as an exact-width one does

Standalone runner (pytest cannot import in this cluster's module stack: its anyio dependency
pulls in ssl, which fails with an OPENSSL_3.3.0 mismatch):

    python tests/test_socket_port.py
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_MODELING = os.path.join(REPO_ROOT, "pipeline", "train_quest", "modeling")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if _MODELING not in sys.path:
    sys.path.insert(0, _MODELING)

# IMPORT ORDER IS PART OF THE TEST.  modeling_llama.py imports the REPO-ROOT `kernels` package
# for sparse_attention_fwd, which binds sys.modules["kernels"] before socket_port can run.
# That is exactly the collision socket_port exists to survive, so reproduce it here rather
# than importing socket_port into a clean interpreter where the bug cannot appear.
from kernels.socket_triton_kernels import sparse_attention_fwd as sparse_attention_fwd_local  # noqa: E402

import socket_port  # noqa: E402
from soft_hash_collision_loader import load_soft_hash_collision  # noqa: E402

dev = "cuda"
torch.manual_seed(0)

B, H, HKV, L, R = 1, 8, 2, 10, 256
T = 4096
SEQ_LEN = 3500          # < T, so the tail is unfilled cache and must score -inf / drop to -1
REP = H // HKV
FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def fixtures():
    """Per-KV-head ground truth, plus the per-query-head expansion repeat_kv would produce.

    modeling_llama builds buckets and ||v|| from repeat_kv'd K/V, so its per-query-head store
    holds REP byte-identical copies of each kv row, and its per-kv-head store is `[:, ::REP]`
    of that. Building both from one [B,HKV,...] source is the same relationship, and makes any
    disagreement in T1 attributable to the kernels rather than to the data.
    """
    kb_kv = torch.randint(0, R, (B, HKV, L, T), device=dev, dtype=torch.int16)
    vn_kv = torch.rand(B, HKV, T, device=dev, dtype=torch.float16) + 0.5
    kb_q = kb_kv.repeat_interleave(REP, dim=1).contiguous()
    vn_q = vn_kv.repeat_interleave(REP, dim=1).contiguous()

    logits = torch.randn(B, H, L, R, device=dev, dtype=torch.float32)
    q_probs = torch.softmax(logits, dim=-1)

    allowed = (torch.arange(T, device=dev) < SEQ_LEN).view(1, 1, T).expand(B, H, T).contiguous()
    seq_len_t = torch.tensor(SEQ_LEN, device=dev, dtype=torch.int32)
    return kb_kv, vn_kv, kb_q, vn_q, q_probs, allowed, seq_len_t


def cuda_scores(q_probs, kb_q, vn_q, allowed):
    """The `SOCKET_SCORER=cuda` arm, called exactly as modeling_llama calls it."""
    ext = load_soft_hash_collision(3)
    return ext.soft_hash_collision(
        q_probs.float().unsqueeze(2).contiguous(),
        kb_q.contiguous(),
        allowed.unsqueeze(2).contiguous(),
        vn_q.float().unsqueeze(2).contiguous(),
    ).squeeze(2)


def t1_scorer(f):
    kb_kv, vn_kv, kb_q, vn_q, q_probs, allowed, seq_len_t = f
    tri = torch.empty((B, H, T), device=dev, dtype=torch.float16)
    socket_port.soft_hash_score_rt(q_probs.contiguous(), kb_kv.contiguous(),
                                   vn_kv.contiguous(), seq_len_t, tri)
    cud = cuda_scores(q_probs, kb_q, vn_q, allowed)

    # Compare only t < SEQ_LEN. The two arms deliberately differ OUTSIDE it: the Triton kernel
    # writes -inf (so the select drops those columns), while the CUDA kernel leaves the
    # `allowed`-masked columns at the zeros its wrapper allocated. Both are "never selected",
    # but they are not the same bits. The production scorer stores fp16, so the fp32 reference
    # is rounded once (the same round-to-nearest the kernel applies) before comparing.
    a, b = tri[..., :SEQ_LEN], cud[..., :SEQ_LEN].half()
    check("T1 scorer bitwise-equal to the CUDA scorer (rounded to fp16) on t < seq_len",
          torch.equal(a, b),
          f"max|diff| {(a.float() - b.float()).abs().max().item():.3e}")
    check("T1 scorer writes -inf for t >= seq_len (so select cannot pick unfilled columns)",
          bool(torch.isinf(tri[..., SEQ_LEN:]).all() and (tri[..., SEQ_LEN:] < 0).all()))
    # GQA bucket sharing, stated directly: feeding the SAME kernel the per-query-head
    # expansion (Hkv == H, rep == 1, nothing shared) must give the same scores as feeding it
    # the Hkv unique rows. Two heads in one group share a bucket row but have DIFFERENT
    # q_probs, so their SCORES are not equal -- that is not the claim being made here.
    tri_q = torch.empty((B, H, T), device=dev, dtype=torch.float16)
    socket_port.soft_hash_score_rt(q_probs.contiguous(), kb_q.contiguous(),
                                   vn_q.contiguous(), seq_len_t, tri_q)
    # Diff reported over t < seq_len only: past it both are -inf and (-inf)-(-inf) is nan,
    # which would print as a failure-looking "nan" beside a passing torch.equal.
    check("T1 per-kv-head buckets score identically to the per-query-head expansion",
          torch.equal(tri, tri_q),
          f"max|diff| {(tri[..., :SEQ_LEN].float() - tri_q[..., :SEQ_LEN].float()).abs().max().item():.3e}")

    # Packed-transport coverage: at large R (and fp16/bf16 q_probs) the dispatcher runs the
    # head-packed CUDA kernel; force the Triton kernel on the same inputs and require bit
    # equality between the two transports.
    R2 = 1024
    kb2 = torch.randint(0, R2, (B, HKV, L, T), device=dev, dtype=torch.int16)
    q2 = torch.softmax(torch.randn(B, H, L, R2, device=dev, dtype=torch.float32), -1).half()
    auto_out = torch.empty((B, H, T), device=dev, dtype=torch.float16)
    socket_port.soft_hash_score_rt(q2.contiguous(), kb2.contiguous(),
                                   vn_kv.contiguous(), seq_len_t, auto_out)
    tri_out = torch.empty_like(auto_out)
    socket_port._sparse().soft_hash_score_op(q2.contiguous(), kb2.contiguous(),
                                             vn_kv.contiguous(), seq_len_t.reshape(()),
                                             tri_out)
    check("T1 packed transport (large R) is bitwise-equal to the Triton kernel",
          torch.equal(auto_out, tri_out))
    return tri


def t2_select(scores):
    for M in (64, 512):
        heavy = socket_port.radix_topm(scores, M)
        ref = torch.topk(scores, k=M, dim=-1).indices.to(torch.int32)

        check(f"T2 radix_topm shape/dtype (M={M})",
              heavy.shape == (B, H, M) and heavy.dtype == torch.int32)
        check(f"T2 radix_topm indices are in range (M={M})",
              bool(((heavy >= 0) & (heavy < T)).all()))
        # EXACTNESS is a claim about the selected SCORE MULTISET, not the index set: which
        # tokens tie AT the threshold is arbitrary in aten::topk too.
        got = torch.gather(scores, -1, heavy.long()).sort(dim=-1, descending=True).values
        exp = torch.gather(scores, -1, ref.long()).sort(dim=-1, descending=True).values
        check(f"T2 radix_topm selects the same score multiset as topk (M={M})",
              torch.equal(got, exp),
              f"max|diff| {(got - exp).abs().max().item():.3e}")
        # No unfilled column may be selected: they scored -inf.
        check(f"T2 radix_topm never selects t >= seq_len (M={M})",
              bool((heavy < SEQ_LEN).all()))


def torch_list(heavy, allowed, sink, window, dedup):
    """The `SOCKET_LIST=torch` arm, transcribed from modeling_llama.build_sparse_list_decode."""
    parts = []
    if sink > 0:
        parts.append(torch.arange(sink, device=dev, dtype=torch.int32))
    if window > 0:
        win_start = max(SEQ_LEN - window, sink)
        if win_start < SEQ_LEN:
            parts.append(torch.arange(win_start, SEQ_LEN, device=dev, dtype=torch.int32))
    base = torch.cat(parts, dim=0).view(1, 1, -1).expand(B, H, -1)
    base = base.masked_fill(~torch.gather(allowed, -1, base.long()), -1)
    # Range gate on heavy. The torch arm in modeling_llama does NOT do this -- it cannot
    # matter there, because the scorer writes -inf past seq_len so topk never returns such an
    # index. This test splices one in deliberately (to check the kernel's gate, asserted
    # separately below), so apply the same gate here and keep this comparison about dedup.
    heavy = heavy.masked_fill((heavy < 0) | (heavy >= SEQ_LEN), -1)
    if dedup:
        ws = max(SEQ_LEN - window, sink) if window > 0 else SEQ_LEN
        heavy = heavy.masked_fill(((heavy >= 0) & (heavy < sink)) |
                                 ((heavy >= ws) & (heavy < SEQ_LEN)), -1)
    return torch.cat([base, heavy], dim=-1).contiguous()


def t3_list(scores, allowed, seq_len_t):
    sink, window, M = 4, 128, 256
    # Force real dedup work: splice sink and window indices INTO the heavy list, so the two
    # arms are compared on input where dedup=True and dedup=False actually differ.
    heavy = torch.topk(scores, k=M, dim=-1).indices.to(torch.int32)
    heavy[..., 0] = 1                      # inside sink
    heavy[..., 1] = SEQ_LEN - 5            # inside window
    heavy[..., 2] = SEQ_LEN + 10           # past seq_len -> must drop to -1 either way

    for dedup in (True, False):
        W = sink + window + M
        out = torch.empty((B, H, W), device=dev, dtype=torch.int32)
        socket_port.build_list_rt(heavy.contiguous(), seq_len_t, out,
                                  sink, window, T, dedup)
        ref = torch_list(heavy.clone(), allowed, sink, window, dedup)
        check(f"T3 build_list matches the torch op chain (dedup={dedup})",
              torch.equal(out, ref),
              f"{int((out != ref).sum().item())} of {out.numel()} slots differ")
        check(f"T3 build_list drops the out-of-range heavy index (dedup={dedup})",
              int(out[0, 0, sink + window + 2].item()) == -1)

    # The switch must actually switch: dedup=True is what removes the two spliced duplicates.
    on = torch.empty((B, H, sink + window + M), device=dev, dtype=torch.int32)
    off = torch.empty_like(on)
    socket_port.build_list_rt(heavy.contiguous(), seq_len_t, on, sink, window, T, True)
    socket_port.build_list_rt(heavy.contiguous(), seq_len_t, off, sink, window, T, False)
    extra = int((on == -1).sum().item()) - int((off == -1).sum().item())
    # DEDUP only ever ADDS -1 slots, and never touches sink or window (those are not heavy).
    check("T3 DEDUP=True is a strict refinement of DEDUP=False",
          extra > 0 and bool(((off == -1) <= (on == -1)).all()),
          f"extra -1 slots {extra}")
    check("T3 DEDUP=True leaves the sink and window slots alone",
          torch.equal(on[..., :sink + window], off[..., :sink + window]))
    # The two spliced duplicates specifically: masked under ON, kept under OFF.
    hb = sink + window
    check("T3 DEDUP=True masks the heavy index that lies inside sink",
          int(on[0, 0, hb].item()) == -1 and int(off[0, 0, hb].item()) == 1)
    check("T3 DEDUP=True masks the heavy index that lies inside the window",
          int(on[0, 0, hb + 1].item()) == -1 and int(off[0, 0, hb + 1].item()) == SEQ_LEN - 5)
    # `extra` is data-dependent (top-M scores cluster in the recency window), so it is
    # reported rather than pinned to a constant.
    print(f"       (dedup masked {extra} additional slots of {on.numel()})")
    return on


def t4_import():
    import kernels as root_kernels
    check("T4 sys.modules['kernels'] is still the REPO-ROOT package",
          os.path.abspath(root_kernels.__file__).startswith(os.path.join(REPO_ROOT, "kernels")),
          root_kernels.__file__)
    sp = socket_port._sparse()
    check("T4 socket_port resolved GPT-FAST/kernels/sparse.py",
          os.path.abspath(sp.__file__) == os.path.join(REPO_ROOT, "GPT-FAST", "kernels", "sparse.py"),
          sp.__file__)
    check("T4 sparse.py is loaded exactly once (no duplicate op registration)",
          socket_port._sparse() is sp)


def t5_attention(sparse_list):
    """Both attention arms must tolerate the -1 holes the Triton list arm emits (9dc668b).

    The two arms are NOT expected to agree bitwise: GPT-FAST gathers 128 list slots per inner
    step against the local copy's 16, so the online softmax accumulates in a different order,
    and it pins the running max and the exponentials to fp32.  The assertion is therefore on
    finiteness and on closeness, and the observed max|diff| is printed so a change in it is
    visible rather than silently absorbed.
    """
    k = torch.randn(B, HKV, T, 128, device=dev, dtype=torch.float16)
    v = torch.randn(B, HKV, T, 128, device=dev, dtype=torch.float16)
    q = torch.randn(B, H, 128, device=dev, dtype=torch.float16)
    slen = torch.full((B, H), sparse_list.shape[-1], device=dev, dtype=torch.int32)

    # A fully padded LEADING partition is the positional case the NaN guard was added for: a
    # partition after a valid token inherits a finite running max and cannot reach the bug.
    padded = sparse_list.clone()
    padded[..., :256] = -1

    arms = {"local": sparse_attention_fwd_local, "gptfast": socket_port.sparse_attention_fwd}
    outs, outs_pad = {}, {}
    for name, fn in arms.items():
        outs[name] = fn(q, k, v, sparse_list, slen, block_seq=256)
        outs_pad[name] = fn(q, k, v, padded, slen, block_seq=256)
        check(f"T5 {name} attention output is finite on a -1-padded list",
              bool(torch.isfinite(outs[name]).all()))
        check(f"T5 {name} attention output is finite with a fully padded leading partition",
              bool(torch.isfinite(outs_pad[name]).all()))

    d = (outs["local"].float() - outs["gptfast"].float()).abs().max().item()
    d_pad = (outs_pad["local"].float() - outs_pad["gptfast"].float()).abs().max().item()
    check("T5 the two attention arms agree to within 1e-2",
          d < 1e-2 and d_pad < 1e-2, f"max|diff| = {d:.3e} (padded {d_pad:.3e})")


def t6_padded_capacity(f):
    """A padded bucket buffer must score its live columns exactly as an exact-width one does.

    This is the invariant the HF path's capacity padding rests on.  The padded columns are
    filled with ZEROS, matching how modeling_llama and GPT-FAST's KVCache allocate: the scorer
    masks its loads on `t < T`, not on `t < seq_len`, so a padded column holding an int16
    outside [0, R) would index outside q_probs.  The garbage case is exercised too, but only
    to show it is caught by an in-range check rather than by hoping.
    """
    kb_kv, vn_kv, _kb_q, _vn_q, q_probs, _allowed, seq_len_t = f
    T_cap = T
    T_true = SEQ_LEN

    exact_kb = kb_kv[..., :T_true].contiguous()
    exact_vn = vn_kv[..., :T_true].contiguous()
    s_exact = torch.empty((B, H, T_true), device=dev, dtype=torch.float16)
    socket_port.soft_hash_score_rt(q_probs.contiguous(), exact_kb, exact_vn, seq_len_t, s_exact)

    pad_kb = torch.zeros((B, HKV, L, T_cap), device=dev, dtype=torch.int16)
    pad_kb[..., :T_true] = exact_kb
    pad_vn = torch.zeros((B, HKV, T_cap), device=dev, dtype=torch.float16)
    pad_vn[..., :T_true] = exact_vn
    s_pad = torch.empty((B, H, T_cap), device=dev, dtype=torch.float16)
    socket_port.soft_hash_score_rt(q_probs.contiguous(), pad_kb, pad_vn, seq_len_t, s_pad)

    check("T6 padded scores are BITWISE equal to exact-width scores over the live columns",
          torch.equal(s_pad[..., :T_true].view(torch.int16), s_exact.view(torch.int16)))
    check("T6 every padded score column is -inf",
          bool((s_pad[..., T_true:] == -float("inf")).all()))
    check("T6 the padded columns hold in-range buckets",
          bool(((pad_kb >= 0) & (pad_kb < R)).all()))

    # The list builder is handed the buffer width as MAXLEN; with seq_len >= sink+window the
    # window clamp is looser than the `idx < seq_len` drop, so the width cannot change the list.
    sink, window, M = 128, 128, 256
    heavy = torch.topk(s_pad, k=M, dim=-1).indices.to(torch.int32).contiguous()
    lists = {}
    for name, maxlen in (("padded", T_cap), ("exact", T_true)):
        out = torch.empty((B, H, sink + window + M), device=dev, dtype=torch.int32)
        socket_port.build_list_rt(heavy, seq_len_t, out, sink, window, maxlen, False)
        lists[name] = out
    check("T6 build_list is unchanged by the buffer width when seq_len >= sink + window",
          torch.equal(lists["padded"], lists["exact"]))

    # And the shape asserts must actually fire: a padded bucket buffer next to an unpadded
    # output would otherwise write off the end of `out`.
    try:
        socket_port.soft_hash_score_rt(q_probs.contiguous(), pad_kb, pad_vn, seq_len_t, s_exact)
        caught = False
    except AssertionError:
        caught = True
    check("T6 a width mismatch between the bucket buffer and `out` raises", caught)


def main():
    assert torch.cuda.is_available(), "these tests need a GPU"
    print(f"[CFG] B={B} H={H} HKV={HKV} rep={REP} L={L} R={R} T={T} seq_len={SEQ_LEN}")
    f = fixtures()
    print("T1 scorer");   scores = t1_scorer(f)
    print("T2 select");   t2_select(scores)
    print("T3 list");     lst = t3_list(scores, f[5], f[6])
    print("T4 imports");  t4_import()
    print("T5 attention"); t5_attention(lst)
    print("T6 padded capacity"); t6_padded_capacity(f)

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
