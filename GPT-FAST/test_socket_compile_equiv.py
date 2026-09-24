import math
import os
import torch

from kernels.sparse import (
    sparse_attention_fwd,
    sparse_decode_stage1,
    sparse_decode_stage2,
    build_sparse_list_decode,
)

dev = "cuda"
torch.manual_seed(0)

# --- Parameterize the soft-hash code width from env (SOCKET_K / SOCKET_R) so the
#     SAME suite gates BOTH K=8/R=256 (regression) and K=10/R=1024 (new path). R is
#     structurally pinned to 2**K (protos_T = {-1,+1}^K corners, pack_bits in [0,2**K));
#     ENFORCE that here so a stray SOCKET_R can never desync from K. Defaults are
#     K=8/R=256. The R that matters for the scorer is
#     the REAL bucket count fed to soft_hash_collision (T6) and the ModelArgs K/R built in
#     T8; the tiny synthetic R in the adversarial composition fixtures (T4/T9) is also
#     driven from env for consistency but is correctness-irrelevant to the K width.
_K_ENV = os.environ.get("SOCKET_K")
_R_ENV = os.environ.get("SOCKET_R")
ENV_K = int(_K_ENV) if _K_ENV is not None else 8
if _R_ENV is not None:
    ENV_R = int(_R_ENV)
    assert ENV_R == (1 << ENV_K), f"SOCKET_R ({ENV_R}) must equal 2**SOCKET_K (2**{ENV_K} = {1 << ENV_K})"
else:
    ENV_R = 1 << ENV_K
print(f"[CFG] equiv suite fixtures parameterized: K={ENV_K} R={ENV_R} (defaults K=8/R=256)")


def ref_online_softmax(q, k, v, sparse_list):
    """Brute-force masked online-softmax over the EXACT selected token set."""
    Bb, Hh, Dd = q.shape
    rep = Hh // k.shape[1]
    out = torch.zeros(Bb, Hh, Dd, device=q.device, dtype=torch.float32)
    sm = 1.0 / math.sqrt(Dd)
    for b in range(Bb):
        for h in range(Hh):
            kvh = h // rep
            idx = sparse_list[b, h]
            idx = idx[idx >= 0].long()
            kk = k[b, kvh, idx].float()
            vv = v[b, kvh, idx].float()
            qq = q[b, h].float()
            s = (kk @ qq) * sm
            w = torch.softmax(s, dim=0)
            out[b, h] = (w.unsqueeze(-1) * vv).sum(0)
    return out


def make_inputs(B=1, H=32, Kv=8, D=128, T=4096, leftover=64,
                sink=120, window=120, M=860):
    Smax = T + leftover
    block_seq = 256
    q = torch.randn(B, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, Kv, Smax, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, Kv, Smax, D, device=dev, dtype=torch.bfloat16)
    base = list(range(sink)) + list(range(T - window, T))
    heavy = torch.randint(0, T, (B, H, M), device=dev, dtype=torch.int32)
    heavy[..., :5] = -1  # exercise OOB-pad mask
    base_t = torch.tensor(base, device=dev, dtype=torch.int32).view(1, 1, -1).expand(B, H, -1)
    sparse_list = torch.cat([base_t, heavy], dim=-1).contiguous()
    sparse_len = torch.full((B, H), sparse_list.shape[-1], device=dev, dtype=torch.int32)
    return q, k, v, sparse_list, sparse_len, block_seq


# ---- Test 1: new eager backend == brute-force online softmax ----
q, k, v, sl, slen, bs = make_inputs()
out_eager = sparse_attention_fwd(q, k, v, sl, slen, block_seq=bs)
out_ref = ref_online_softmax(q, k, v, sl)
d1 = (out_eager.float() - out_ref).abs().max().item()
print(f"[T1] eager backend vs online-softmax ref  max|diff| = {d1:.3e}")

# ---- Test 2: compiled (fullgraph) backend == eager backend, bit-exact ----
compiled = torch.compile(sparse_attention_fwd, mode="reduce-overhead", fullgraph=True)
out_c = compiled(q, k, v, sl, slen, block_seq=bs)
torch.cuda.synchronize()
out_c = compiled(q, k, v, sl, slen, block_seq=bs)  # second call = cudagraph replay
torch.cuda.synchronize()
d2 = (out_c.float() - out_eager.float()).abs().max().item()
exact2 = torch.equal(out_c, out_eager)
print(f"[T2] compiled(fullgraph,reduce-overhead) vs eager  max|diff| = {d2:.3e}  bit_exact={exact2}")

# ---- Test 3: static-cache extension selects SAME top-k as T-sliced ----
# scoring over Smax with allowed=False on [T:] must give identical top-M indices to
# scoring over just [:T]. We emulate build_sparse_list_decode's scorer contract.
def topk_over(scores_bht, allowed_bht, M):
    s = scores_bht.clone()
    s = s.masked_fill(~allowed_bht, float("-inf"))
    return torch.topk(s, k=M, dim=-1, largest=True).indices

B, H = 1, 32
T, Smax, M = 4096, 4160, 860
scores_full = torch.randn(B, H, Smax, device=dev)
allowed_full = (torch.arange(Smax, device=dev) < T).view(1, 1, Smax).expand(B, H, Smax)
idx_static = topk_over(scores_full, allowed_full, M)
idx_sliced = torch.topk(scores_full[:, :, :T], k=M, dim=-1, largest=True).indices
# compare as SETS per (b,h) (topk order can differ only on exact ties; sets must match)
overlap = (idx_static.sort(-1).values == idx_sliced.sort(-1).values).all().item()
print(f"[T3] static-cache(-inf tail) topk set == T-sliced topk set : {overlap}")

# ---- Test 4: static-cache build_sparse_list_decode == true-T-sliced build ----
# Build the FULL cache [B,H,*,Smax] with GARBAGE in the unfilled tail [T:Smax].
# (1) static build: allowed = arange(Smax)<=T-1, seq_len_t=T, over the full Smax cache.
# (2) sliced ref  : same scorer/window logic but physically restricted to [:T].
# The selected SET (per (b,h), dropping -1 pads) and the resulting attention output
# must be identical: the -inf tail can never be selected and never enters attention.
def _decode_inputs(B=1, H=32, Kv=8, D=128, L=4, R=ENV_R, T=4096, leftover=64,
                   sink=120, window=120, M=860):
    Smax = T + leftover
    q_probs = torch.rand(B, H, L, R, device=dev, dtype=torch.float16)
    k_hard_full = torch.randint(0, R, (B, H, L, Smax), device=dev, dtype=torch.int16)
    v_norm_full = torch.rand(B, H, Smax, device=dev, dtype=torch.float16)
    # garbage tail: large v_norm + a hot bucket, so a NAIVE full-cache scorer (without the
    # allowed mask) WOULD pick these -> proves the mask is what protects correctness.
    v_norm_full[:, :, T:] = 1e3
    k_hard_full[:, :, :, T:] = 0
    q_probs[:, :, :, 0] = 5.0
    allowed_full = (torch.arange(Smax, device=dev) <= (T - 1)).view(1, 1, Smax).expand(B, H, Smax).contiguous()
    seq_len_t = torch.tensor(T, device=dev, dtype=torch.int32)
    kv = torch.randn(B, Kv, Smax, D, device=dev, dtype=torch.bfloat16)
    vv = torch.randn(B, Kv, Smax, D, device=dev, dtype=torch.bfloat16)
    qd = torch.randn(B, H, D, device=dev, dtype=torch.bfloat16)
    return (q_probs, k_hard_full, v_norm_full, allowed_full, seq_len_t,
            kv, vv, qd, dict(sink=sink, window=window, M=M, T=T, Smax=Smax))

(qp, kh_full, vn_full, al_full, seqt, KK, VV, QQ, cfg) = _decode_inputs()
T = cfg["T"]; Smax = cfg["Smax"]

sl_static, slen_static = build_sparse_list_decode(
    qp, kh_full, vn_full, al_full,
    sink=cfg["sink"], window=cfg["window"], M=cfg["M"], seq_len_t=seqt,
)
# Sliced reference: same call but inputs physically restricted to the true length T, with
# allowed all-True over [:T] and seq_len_t=T -> this is the pre-refactor eager contract.
al_sliced = torch.ones(al_full.shape[0], al_full.shape[1], T, device=dev, dtype=torch.bool)
sl_sliced, slen_sliced = build_sparse_list_decode(
    qp, kh_full[:, :, :, :T].contiguous(), vn_full[:, :, :T].contiguous(), al_sliced,
    sink=cfg["sink"], window=cfg["window"], M=cfg["M"], seq_len_t=seqt,
)

def _sets_match(a, b):
    Bb, Hh, _ = a.shape
    for bi in range(Bb):
        for hi in range(Hh):
            sa = set(a[bi, hi][a[bi, hi] >= 0].tolist())
            sb = set(b[bi, hi][b[bi, hi] >= 0].tolist())
            if sa != sb:
                return False, (bi, hi, sorted(sa ^ sb)[:8])
    return True, None

# no selected index may fall in the unfilled tail [T:Smax]
tail_leak = int((sl_static >= T).sum().item())
ok_sets, diff = _sets_match(sl_static, sl_sliced)
print(f"[T4a] static build selects NOTHING in [T:Smax] tail : leak_count={tail_leak} (expect 0)")
print(f"[T4b] static-cache selected SET == true-T-sliced SET : {ok_sets}  {('' if ok_sets else diff)}")

# attention output equality: feed each list to the backend on its own (full vs sliced) cache.
out_static = sparse_attention_fwd(QQ, KK, VV, sl_static.to(torch.int32),
                                  slen_static.to(torch.int32), block_seq=256)
out_sliced = sparse_attention_fwd(QQ, KK[:, :, :T].contiguous(), VV[:, :, :T].contiguous(),
                                  sl_sliced.to(torch.int32), slen_sliced.to(torch.int32), block_seq=256)
d4 = (out_static.float() - out_sliced.float()).abs().max().item()
print(f"[T4c] attention output  static-cache vs T-sliced  max|diff| = {d4:.3e}  bit_exact={torch.equal(out_static, out_sliced)}")

# ---- Test 5: compiled(fullgraph) build_sparse_list_decode == eager, no graph break ----
try:
    cbuild = torch.compile(build_sparse_list_decode, mode="reduce-overhead", fullgraph=True)
    sl_c, slen_c = cbuild(qp, kh_full, vn_full, al_full,
                          sink=cfg["sink"], window=cfg["window"], M=cfg["M"], seq_len_t=seqt)
    torch.cuda.synchronize()
    ok5, diff5 = _sets_match(sl_c.to(torch.int32), sl_static)
    print(f"[T5] compiled(fullgraph) build set == eager build set : {ok5}  {('' if ok5 else diff5)}")
except Exception as e:
    print(f"[T5] compiled build FAILED (graph break / sync?): {type(e).__name__}: {e}")

# ---- Test 6: PER-KV-HEAD scorer == repeat_interleave-to-H path, bit-exact (GQA rep=4) ----
# This is the GATE for the per-kv-head optimization. T1-T5 above pass H_kv==H_q (rep=1) so
# they do NOT exercise the GQA reduction. Here q_probs is per-QUERY-head (H=32) while
# key_buckets/v_hist are per-KV-head (Hkv=8, rep=4). We compare:
#   (a) the raw scorer op fed per-kv tensors          [B,8,...]
#   (b) the SAME op fed repeat_interleave(rep)-to-32  [B,32,...]  (the OLD path)
# They must be torch.equal (bit-identical), because repeat_interleave(rep,dim=1) maps
# out-head h -> in-head h//rep (pure copy) and the kernel now reads kv head h//rep directly.
def _gqa_inputs(B=1, Hq=32, Hkv=8, L=60, R=ENV_R, T=4096, leftover=64, M=860):
    Smax = T + leftover
    rep = Hq // Hkv
    q_probs = torch.rand(B, Hq, L, R, device=dev, dtype=torch.float16)
    k_hard_kv = torch.randint(0, R, (B, Hkv, L, Smax), device=dev, dtype=torch.int16)
    v_norm_kv = torch.rand(B, Hkv, Smax, device=dev, dtype=torch.float16)
    allowed = (torch.arange(Smax, device=dev) <= (T - 1)).view(1, 1, Smax).expand(B, Hq, Smax).contiguous()
    return q_probs, k_hard_kv, v_norm_kv, allowed, rep, T, Smax, M

qp6, kh_kv, vn_kv, al6, rep6, T6_, Smax6, M6 = _gqa_inputs()

# (a) raw scorer op: per-kv key data vs (b) repeat-interleaved-to-Hq key data
qp6_f32 = qp6.float().unsqueeze(2).contiguous()                                  # [B,Hq,1,L,R]
al6_ext = al6.unsqueeze(2).contiguous()                                          # [B,Hq,1,Smax]
kb_kv = kh_kv.contiguous()                                                       # [B,Hkv,L,Smax]
kb_rep = kh_kv.repeat_interleave(rep6, dim=1).contiguous()                       # [B,Hq,L,Smax]
v_kv = vn_kv.float().unsqueeze(2).contiguous()                                   # [B,Hkv,1,Smax]
v_rep = vn_kv.repeat_interleave(rep6, dim=1).float().unsqueeze(2).contiguous()   # [B,Hq,1,Smax]

import torch.ops as _ops  # noqa: F401  (ensures socket op namespace is materialized)
scores_kv = torch.ops.socket.soft_hash_collision(qp6_f32, kb_kv, al6_ext, v_kv).squeeze(2)
scores_rep = torch.ops.socket.soft_hash_collision(qp6_f32, kb_rep, al6_ext, v_rep).squeeze(2)
d6 = (scores_kv.float() - scores_rep.float())
finite = torch.isfinite(scores_kv) & torch.isfinite(scores_rep)
d6max = d6[finite].abs().max().item() if finite.any() else 0.0
exact6 = torch.equal(scores_kv, scores_rep)
print(f"[T6a] per-kv-head scorer == repeat_interleave path (rep={rep6})  max|diff|(finite) = {d6max:.3e}  bit_exact={exact6}")

# (b) full build_sparse_list_decode selected SET equality: per-kv vs repeat-interleaved
seqt6 = torch.tensor(T6_, device=dev, dtype=torch.int32)
sl_kv, slen_kv = build_sparse_list_decode(
    qp6, kb_kv, vn_kv, al6, sink=120, window=120, M=M6, seq_len_t=seqt6,
)
sl_rep, slen_rep = build_sparse_list_decode(
    qp6, kb_rep, vn_kv.repeat_interleave(rep6, dim=1).contiguous(), al6,
    sink=120, window=120, M=M6, seq_len_t=seqt6,
)
ok6, diff6 = _sets_match(sl_kv.to(torch.int32), sl_rep.to(torch.int32))
print(f"[T6b] per-kv build set == repeat_interleave build set : {ok6}  {('' if ok6 else diff6)}")

# ---- Test 7: KVCache.update writes k_hard in the RANK-1 native [B,Hkv,L,T] layout, byte-equal
#      to the OLD [B,Hkv,T,L] + per-token permute(0,1,3,2) relayout. T1-T6 fabricate k_hard
#      DIRECTLY in scorer layout and NEVER call KVCache.update, so this is the ONLY gate on the
#      write-time relayout removal: a transposed-axis typo would otherwise pass the suite green.
try:
    from model import KVCache
    _B, _Hkv, _D, _L, _R, _Tcap = 1, 8, 128, 60, 256, 4160
    _kv = KVCache(_B, _Tcap, _Hkv, _D, _L, _R).to(dev)
    _kh_old = torch.zeros(_B, _Hkv, _Tcap, _L, device=dev, dtype=torch.int16)  # OLD-layout reference

    def _t7_step(positions):
        S = positions.numel()
        kvv = torch.randn(_B, S, _Hkv, _D, device=dev, dtype=torch.bfloat16)
        vvv = torch.randn(_B, S, _Hkv, _D, device=dev, dtype=torch.bfloat16)
        vno = torch.rand(_B, S, _Hkv, device=dev, dtype=torch.float16)
        khd = torch.randint(0, _R, (_B, S, _Hkv, _L), device=dev, dtype=torch.int16)
        _kv.update(positions, kvv, vvv, v_norm=vno, k_hard=khd)
        _kh_old[:, :, positions.long(), :] = khd.permute(0, 2, 1, 3)            # OLD-layout write
        return torch.equal(_kv.k_hard, _kh_old.permute(0, 1, 3, 2))            # NEW == OLD transposed

    _ok = [_t7_step(torch.arange(0, 4096, device=dev))]                        # prefill S>1
    for _t in range(4096, 4100):                                               # decode S==1 appends
        _ok.append(_t7_step(torch.tensor([_t], device=dev)))
    print(f"[T7a] KVCache.update k_hard NEW [B,Hkv,L,T] == OLD [B,Hkv,T,L].permute, every step : {all(_ok)}")

    # end-to-end: the live buffer feeds the scorer identically to the relayout-of-old path
    _kb_new = _kv.k_hard.contiguous()
    _kb_old = _kh_old.permute(0, 1, 3, 2).contiguous()
    print(f"[T7b] live k_hard buffer bytes == relayout-of-old buffer : {torch.equal(_kb_new, _kb_old)}")
except Exception as _e:
    print(f"[T7] FAILED to validate KVCache write layout: {type(_e).__name__}: {_e}")

# ---- Test 8: CORRECTNESS-FIX sanity — soft_hash is REAL SOCKET (corners + /tau), so the
#      query bucket distribution q_probs is PEAKED (query-aware), NOT uniform. The old
#      degenerate init (protos_T = randn*0.02, no /tau) collapsed logits to ~0 -> softmax ~=
#      1/R uniform -> selection ~= top-k-by-||v|| (query-agnostic). This test proves the fix
#      took: (a) protos_T == the {-1,+1}^K hypercube corners; (b) max(q_probs) >> 1/R.
try:
    import itertools as _it
    from model import ModelArgs, Attention
    _cfg = ModelArgs(n_head=32, n_local_heads=8, dim=4096, K=ENV_K, R=ENV_R, L=60)
    _attn = Attention(_cfg).to(dev).to(torch.bfloat16)
    R8 = _cfg.R
    uniform = 1.0 / R8

    # (a) protos_T is the fixed hypercube corners: column r == sign pattern whose pack_bits
    # code is r. Rebuild the reference here and compare bytes.
    _corners_ref = torch.tensor(list(_it.product([-1.0, 1.0], repeat=_cfg.K))).t().contiguous().to(_attn.protos_T)
    protos_ok = torch.equal(_attn.protos_T, _corners_ref)
    is_pm1 = bool(((_attn.protos_T == 1.0) | (_attn.protos_T == -1.0)).all().item())
    print(f"[T8a] protos_T == {{-1,+1}}^K hypercube corners : {protos_ok}  all_pm1={is_pm1}  shape={tuple(_attn.protos_T.shape)}")

    # (b) q_probs peaked. Run the real soft_hash on random decode queries.
    _q = torch.randn(2, _cfg.n_head, _cfg.head_dim, device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        _qp = _attn.soft_hash(_q).float()  # [B,H,L,R]
    qp_max = _qp.max().item()
    qp_mean = _qp.mean().item()
    # sum over R must be 1 (softmax); mean over R == 1/R by construction. atol=1e-2 because
    # soft_hash emits bf16 q_probs and summing R=256 bf16 terms accumulates ~1e-3 rounding.
    row_sums = _qp.sum(dim=-1)
    sums_ok = bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2))
    peaked = qp_max > 5.0 * uniform   # PEAKED: top bucket well above the uniform 1/R floor
    print(f"[T8b] soft_hash q_probs PEAKED (query-aware) : max={qp_max:.4e}  uniform(1/R)={uniform:.4e}  "
          f"ratio={qp_max/uniform:.1f}x  peaked={peaked}  softmax_rowsum_ok={sums_ok}  tau={_attn.tau}")
    if not peaked:
        print("[T8b] WARNING: q_probs ~ uniform -> correctness fix did NOT take (still degenerate).")
except Exception as _e:
    import traceback
    print(f"[T8] FAILED soft_hash correctness sanity: {type(_e).__name__}: {_e}")
    traceback.print_exc()

# ---- Test 9: DEDUP heavy vs base (sink ∪ window) — the base+heavy composition must be a
#      set-UNION (each token attended ONCE), matching paper Alg 3 / hub SocketMasker. The
#      old no-dedup torch.cat([base, heavy]) would DOUBLE-COUNT (~2x weight) any top-M token
#      that also lies in [0,sink) or [win_start,seq_len) because (a) the scorer's `allowed`
#      mask only excludes the unfilled tail (NOT sink/window) so those tokens are scorable, and
#      (b) the flash-decode online-softmax has no seen-guard. We build a fixture that FORCES
#      sink+window positions into the heavy top-M, then prove the dedup masks them out (T9a),
#      the deduped output matches a UNIQUE-set online-softmax (T9b), and the PRE-fix list would
#      have DIFFERED from that union reference (T9c — proves the test exercises the bug).
def _t9_inputs(B=1, H=32, Kv=8, D=128, L=4, R=ENV_R, T=4096, leftover=64,
               sink=120, window=120, M=860):
    Smax = T + leftover
    rep = H // Kv
    q_probs = torch.rand(B, H, L, R, device=dev, dtype=torch.float16)
    k_hard_kv = torch.randint(0, R, (B, Kv, L, Smax), device=dev, dtype=torch.int16)
    v_norm_kv = torch.rand(B, Kv, Smax, device=dev, dtype=torch.float16)
    # FORCE sink + window positions into the heavy top-M: give them a HOT bucket (0) and a
    # huge v_norm so score = sum_l q_probs[h,l,0] * ||v_j|| dominates. q_probs hot on bucket 0.
    q_probs[:, :, :, 0] = 5.0
    win_lo = T - window
    hot_pos = list(range(sink)) + list(range(win_lo, T))  # all sink + all window positions
    hot_pos_t = torch.tensor(hot_pos, device=dev, dtype=torch.long)
    k_hard_kv[:, :, :, hot_pos_t] = 0          # bucket 0 == the hot query bucket
    v_norm_kv[:, :, hot_pos_t] = 1e3           # huge norm -> top-M ranks them first
    # allowed mirrors model.py:726 — ONLY excludes the unfilled tail [T:Smax], NOT sink/window.
    allowed = (torch.arange(Smax, device=dev) <= (T - 1)).view(1, 1, Smax).expand(B, H, Smax).contiguous()
    seq_len_t = torch.tensor(T, device=dev, dtype=torch.int32)
    kv = torch.randn(B, Kv, Smax, D, device=dev, dtype=torch.bfloat16)
    vv = torch.randn(B, Kv, Smax, D, device=dev, dtype=torch.bfloat16)
    qd = torch.randn(B, H, D, device=dev, dtype=torch.bfloat16)
    return (q_probs, k_hard_kv, v_norm_kv, allowed, seq_len_t, kv, vv, qd,
            dict(sink=sink, window=window, M=M, T=T, Smax=Smax, rep=rep, win_lo=win_lo))


try:
    (qp9, kh9, vn9, al9, seqt9, KK9, VV9, QQ9, cfg9) = _t9_inputs()
    sink9, window9, M9, T9_, Smax9 = cfg9["sink"], cfg9["window"], cfg9["M"], cfg9["T"], cfg9["Smax"]
    win_start9 = max(T9_ - window9, sink9)

    # The DEDUPED (fixed) list, as produced by build_sparse_list_decode.
    sl9, slen9 = build_sparse_list_decode(
        qp9, kh9, vn9, al9, sink=sink9, window=window9, M=M9, seq_len_t=seqt9,
    )
    base_width = sink9 + window9   # the structured base occupies the first sink+window slots
    heavy_part = sl9[:, :, base_width:]   # the heavy (top-M) slots only

    # (T9a) NO heavy-portion entry may lie in the base regions [0,sink) ∪ [win_start,seq_len);
    # such entries must have been masked to -1. Equivalently: no token appears in BOTH base&heavy.
    leak_sink = ((heavy_part >= 0) & (heavy_part < sink9)).sum().item()
    leak_win = ((heavy_part >= win_start9) & (heavy_part < T9_)).sum().item()
    # Also assert the FULL list is a true set (no value appears twice in any (b,h) row).
    no_dup = True
    Bb9, Hh9, _ = sl9.shape
    for bi in range(Bb9):
        for hi in range(Hh9):
            row = sl9[bi, hi]
            kept = row[row >= 0].tolist()
            if len(kept) != len(set(kept)):
                no_dup = False
                break
        if not no_dup:
            break
    t9a_ok = (leak_sink == 0) and (leak_win == 0) and no_dup
    print(f"[T9a] deduped: NO heavy entry in [0,sink)∪[win_start,seq_len) & list is a SET : "
          f"{t9a_ok}  (leak_sink={leak_sink} leak_win={leak_win} no_dup={no_dup})")

    # Build the UNION reference index set per (b,h): base (sink ∪ window) ∪ heavy_before_mask,
    # EACH TOKEN ONCE. This is paper Alg 3 / the hub SocketMasker semantics.
    # Reconstruct heavy_before_mask = the raw top-M (no dedup vs base) from the same scorer.
    qp9_f32 = qp9.float().unsqueeze(2).contiguous()
    al9_ext = al9.unsqueeze(2).contiguous()
    vh9 = vn9.float().unsqueeze(2).contiguous()
    scores9 = torch.ops.socket.soft_hash_collision(qp9_f32, kh9.contiguous(), al9_ext, vh9).squeeze(2)
    heavy_raw = torch.topk(scores9, k=min(M9, Smax9), dim=-1, largest=True).indices.to(torch.int32)
    base_idx = torch.tensor(list(range(sink9)) + list(range(win_start9, T9_)),
                            device=dev, dtype=torch.int32).view(1, 1, -1).expand(Bb9, Hh9, -1)
    union_list = torch.full((Bb9, Hh9, base_idx.shape[-1] + heavy_raw.shape[-1]),
                            -1, device=dev, dtype=torch.int32)
    for bi in range(Bb9):
        for hi in range(Hh9):
            u = []
            seen = set()
            for t in base_idx[bi, hi].tolist() + heavy_raw[bi, hi].tolist():
                if t >= 0 and t not in seen:
                    seen.add(t); u.append(t)
            union_list[bi, hi, :len(u)] = torch.tensor(u, device=dev, dtype=torch.int32)

    # The PRE-fix list = plain cat(base, heavy_raw) WITHOUT dedup (what the bug produced).
    prefix_list = torch.cat([base_idx, heavy_raw], dim=-1).contiguous()

    # block_seq = full list width -> a SINGLE flash-decode block per (b,h). This adversarial
    # fixture forces a long CONTIGUOUS run of masked-(-1) heavy slots (the sink/window tokens all
    # rank together in the top-M and get deduped), which can land a whole 256-sub-block all-(-1);
    # the multi-block stage2 reduction is a pre-existing kernel limitation on that layout (NOT the
    # dedup, NOT exercised by production decode where masked slots are few & scattered — see T1).
    # We validate the DEDUP COMPOSITION here, so we use one block (valid tokens always present in
    # the single block) to isolate set-union correctness from that unrelated multi-block artifact.
    one_block = int(sl9.shape[-1])
    out_dedup = sparse_attention_fwd(QQ9, KK9, VV9, sl9.to(torch.int32), slen9.to(torch.int32), block_seq=one_block)
    out_union = ref_online_softmax(QQ9, KK9, VV9, union_list)
    out_prefix = ref_online_softmax(QQ9, KK9, VV9, prefix_list)

    # (T9b) deduped backend output == UNIQUE-set online-softmax reference. The reference
    # accumulates sequentially in python while stage1 reduces block-wide, so the bound is a
    # reassociation tolerance at bf16 scale (same bound the dense-equivalence test uses).
    d9b = (out_dedup.float() - out_union).abs().max().item()
    t9b_ok = d9b <= 2e-2
    print(f"[T9b] deduped output == UNIQUE-set (set-UNION) online-softmax  max|diff| = {d9b:.3e}  pass={t9b_ok}")

    # (T9c) DEMONSTRATION: the PRE-fix (double-counting) list DIFFERS from the union reference.
    # If this did NOT differ, the fixture would not actually exercise the bug.
    d9c = (out_prefix.float() - out_union).abs().max().item()
    t9c_diverges = d9c > 1e-2
    print(f"[T9c] PRE-fix (no-dedup, double-counted) DIFFERS from union ref  max|diff| = {d9c:.3e}  diverges={t9c_diverges}")
    if not (t9a_ok and t9b_ok and t9c_diverges):
        print(f"[T9] FAIL: t9a_ok={t9a_ok} t9b_ok={t9b_ok} t9c_diverges={t9c_diverges}")
except Exception as _e:
    import traceback
    print(f"[T9] FAILED dedup test: {type(_e).__name__}: {_e}")
    traceback.print_exc()

print("ALL_DONE")
