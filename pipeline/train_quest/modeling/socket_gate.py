"""Equivalence gates for the GPT-FAST kernel port, run INSIDE a live LongBench forward pass.

Everything here runs on tensors the model actually produced at decode step k of a real
LongBench prompt -- never on torch.randn stand-ins.  Synthetic tensors have already produced
a wrong conclusion (a scorer microbenchmark), and for the SELECT gate they
would be actively misleading: whether the top-M boundary score is TIED depends entirely on the
real score distribution.

Gates, in the order the task asks for them:

  G1 SCORER   The GPT-FAST Triton scorer (GQA-shared) vs the eval path's load_inline CUDA
              scorer `ext.soft_hash_collision`, BITWISE over the live columns.  Also the same
              kernel on the PADDED buffer vs on an exact-width copy, which is what proves the
              capacity padding is inert, plus a direct check that every padded column really
              did come back -inf.

  G2 SELECT   The selected SCORE MULTISET from the radix threshold select must equal
              torch.topk's, exactly.  Index sets may legitimately differ, but ONLY on tokens
              whose score equals the threshold, so the tie count at the boundary is reported:
              with a live soft-hash it should be ~0.  Many ties would itself be evidence the
              hash has gone inert.

  G3 LIVENESS protos_T in {-1,+1}; planes std ~1; q_probs deviation from uniform 1/R; and the
              decisive one -- the top-256 overlap between two query heads that SHARE a KV head
              (identical buckets, identical ||v||, so any divergence is the QUERY talking).
              Overlap trending toward 100% means the query has stopped mattering.

  G4 FINITE   No NaN / no +-inf in the attention output, and no -1 index escaping into an
              out-of-bounds gather (the two bugs guarded in 9dc668b).

Set SOCKET_GATE=1 to run them; SOCKET_GATE_LAYERS / SOCKET_GATE_STEPS bound the cost.  The
gate is OFF for scoring runs -- it forces the per-query-head bucket layout the optimized arm
exists to avoid, which at a padded capacity is four times the store.
"""

import os
import torch


def _fmt(x):
    return "nan" if x != x else f"{x:.6g}"


class GateState:
    """Per-process accumulator so the gate can print ONE summary at the end."""

    def __init__(self):
        self.scorer_max_abs_diff_cuda = 0.0
        self.scorer_max_abs_diff_gf = 0.0
        self.scorer_exact_bits_cuda = True
        self.scorer_exact_bits_gf = True
        self.scorer_arrays = 0
        self.scorer_elems = 0
        self.select_calls = 0
        self.select_multiset_ok = 0
        self.select_index_identical = 0
        self.select_ties_total = 0
        self.select_max_ties = 0
        self.overlap_sum = 0.0
        self.overlap_n = 0
        self.list_calls = 0
        self.list_identical = 0
        self.nonfinite_logits = 0
        self.neg_index_slots = 0
        self.notes = []

    def report(self):
        lines = ["", "=" * 78, "[SOCKET-GATE] summary", "=" * 78]
        lines.append(
            f"G1 scorer  arrays={self.scorer_arrays} elems={self.scorer_elems}  "
            f"BITWISE-vs-CUDA={self.scorer_exact_bits_cuda} max|diff|={_fmt(self.scorer_max_abs_diff_cuda)}  "
            f"BITWISE-padded-vs-exact-width={self.scorer_exact_bits_gf} "
            f"max|diff|={_fmt(self.scorer_max_abs_diff_gf)}"
        )
        lines.append(
            f"G2 select  calls={self.select_calls} multiset_equal={self.select_multiset_ok} "
            f"index_identical={self.select_index_identical} "
            f"boundary_ties total={self.select_ties_total} max_per_call={self.select_max_ties}"
        )
        ov = (self.overlap_sum / self.overlap_n * 100.0) if self.overlap_n else float("nan")
        lines.append(f"G3 head-overlap (top-256, two query heads sharing a KV head): {_fmt(ov)}%"
                     f"  over {self.overlap_n} pairs   [~30% = LIVE, ~100% = DEAD]")
        lines.append(
            f"G4 finite  nonfinite_logit_rows={self.nonfinite_logits}  "
            f"negative_index_slots_seen={self.neg_index_slots} (must be tolerated, not crash)"
        )
        if self.list_calls:
            lines.append(f"G5 list    calls={self.list_calls} identical_to_torch={self.list_identical}")
        for n in self.notes:
            lines.append("   note: " + n)
        lines.append("=" * 78)
        print("\n".join(lines), flush=True)


_STATE = None


def state():
    global _STATE
    if _STATE is None:
        _STATE = GateState()
    return _STATE


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16 if t.dtype == torch.float16 else torch.int32)


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor):
    """Exact bit comparison of two float tensors of the same dtype, -inf included.

    `a == b` is False for NaN and True for -inf == -inf, so comparing the raw integer views
    is the only way to say "bitwise".  Returns (all_equal, max_abs_diff_over_finite).
    """
    assert a.dtype == b.dtype, (a.dtype, b.dtype)
    ai = _bits(a)
    bi = _bits(b)
    eq = bool(torch.equal(ai, bi))
    finite = torch.isfinite(a) & torch.isfinite(b)
    if finite.any():
        d = (a[finite] - b[finite]).abs().max().item()
    else:
        d = 0.0
    return eq, d


@torch.no_grad()
def gate_scorer(tag, q_probs, kb_kv, vn_kv, kb_q, vn_q, allowed_bht, seq_len_t, T_true, port):
    """G1. The GPT-FAST scorer, on a PADDED buffer, vs the eval path's CUDA scorer, bitwise.

    kb_kv/vn_kv are PER-KV-HEAD ([B,Hkv,L,T] / [B,Hkv,T]); kb_q/vn_q are the PER-QUERY-HEAD
    tensors the eval path's CUDA extension demands ([B,H,L,T] / [B,H,T]).  T here is the
    BUFFER width, which the eval path pads past the live length T_true.

    Two things have to hold, and they are separate claims:

      1. The Triton scorer agrees with the CUDA one over the live columns.  This is the
         cross-implementation check against the older reference scorer.
      2. Running it on a padded buffer changes nothing over the live columns.  That is the
         claim the padding rests on: the -inf write past seq_len is what makes the extra
         columns inert, and if it ever stopped doing so the top-M select would start choosing
         padding.  Checked by re-running the SAME kernel on the exact-width slice.

    Both comparisons are over [..., :T_true] -- past that the two disagree by construction
    (the padded run writes -inf, the exact-width run has no such column at all).
    Returns (padded Triton scores, CUDA scores), which the caller reuses so the gate costs
    extra scorer runs rather than extra model steps.
    """
    st = state()
    B, H, L, R = q_probs.shape
    T = kb_kv.shape[-1]
    dev = q_probs.device

    qp = q_probs.contiguous()
    scores_tri = torch.empty((B, H, T), device=dev, dtype=port.scores_dtype())
    port.soft_hash_score_rt(qp, kb_kv.contiguous(), vn_kv.contiguous(), seq_len_t, scores_tri)

    # --- reference 1: the eval path's own load_inline CUDA scorer, called exactly as
    # build_sparse_list_decode calls it (fp32 q_probs, [B,H,1,L,R]; int16 [B,H,L,T] buckets;
    # bool [B,H,1,T] allowed; fp32 [B,H,1,T] v).
    from .modeling_llama import _get_soft_hash_ext
    ext = _get_soft_hash_ext()
    scores_cuda = ext.soft_hash_collision(
        qp.float().unsqueeze(2).contiguous(),
        kb_q.to(torch.int16).contiguous(),
        allowed_bht.unsqueeze(2).contiguous(),
        vn_q.float().unsqueeze(2).contiguous(),
    ).squeeze(2)
    # The CUDA reference computes fp32; cast to the stored dtype (a no-op when it is fp32,
    # the kernel's own single rounding when it is fp16) before the bit comparison.
    eq_c, d_c = _bitwise_equal(scores_tri[..., :T_true],
                               scores_cuda[..., :T_true].to(scores_tri.dtype))

    # --- reference 2: the same kernel on an EXACT-width copy of the same data.
    kb_x = kb_kv[..., :T_true].contiguous()
    vn_x = vn_kv[..., :T_true].contiguous()
    scores_exact = torch.empty((B, H, T_true), device=dev, dtype=port.scores_dtype())
    port.soft_hash_score_rt(qp, kb_x, vn_x, seq_len_t, scores_exact)
    eq_g, d_g = _bitwise_equal(scores_tri[..., :T_true], scores_exact)

    st.scorer_arrays += 1
    st.scorer_elems += scores_tri[..., :T_true].numel()
    st.scorer_exact_bits_cuda = st.scorer_exact_bits_cuda and eq_c
    st.scorer_exact_bits_gf = st.scorer_exact_bits_gf and eq_g
    st.scorer_max_abs_diff_cuda = max(st.scorer_max_abs_diff_cuda, d_c)
    st.scorer_max_abs_diff_gf = max(st.scorer_max_abs_diff_gf, d_g)
    # A padded column that is not -inf means the mask stopped working: report it directly
    # rather than waiting for it to show up as a strange top-M choice.
    if T > T_true:
        bad_pad = int(torch.isfinite(scores_tri[..., T_true:]).sum().item())
        if bad_pad:
            st.notes.append(f"{tag}: {bad_pad} padded score columns are FINITE (expected -inf)")
    else:
        bad_pad = 0
    print(f"[SOCKET-GATE] G1 {tag} T={T} T_true={T_true} bitwise_vs_cuda={eq_c} "
          f"maxdiff={_fmt(d_c)} bitwise_padded_vs_exact={eq_g} maxdiff={_fmt(d_g)} "
          f"finite_padding_cols={bad_pad}", flush=True)
    return scores_tri, scores_cuda


@torch.no_grad()
def gate_select(tag, scores, M, port):
    """G2. radix threshold select vs torch.topk: SCORE MULTISET equality + tie count."""
    st = state()
    idx_topk = torch.topk(scores, k=M, dim=-1, largest=True).indices
    idx_radix = port.radix_topm(scores, M).long()

    s_topk = torch.gather(scores, -1, idx_topk)
    s_radix = torch.gather(scores, -1, idx_radix)
    # Multiset equality: sort both selected score rows descending and compare BITWISE.
    ms_ok = torch.equal(
        _bits(torch.sort(s_topk, dim=-1, descending=True).values),
        _bits(torch.sort(s_radix, dim=-1, descending=True).values),
    )
    idx_same = torch.equal(torch.sort(idx_topk, dim=-1).values,
                           torch.sort(idx_radix, dim=-1).values)

    # Boundary ties: how many tokens in the WHOLE row carry exactly the threshold score.
    # Anything above 1 per row means topk and radix could legitimately keep different tokens.
    thresh = s_topk.min(dim=-1, keepdim=True).values            # [B,H,1]
    ties = ((scores == thresh) & torch.isfinite(scores)).sum(-1)  # [B,H]
    ties_excess = (ties - 1).clamp(min=0)
    n_ties = int(ties_excess.sum().item())
    max_ties = int(ties_excess.max().item())

    st.select_calls += 1
    st.select_multiset_ok += int(ms_ok)
    st.select_index_identical += int(idx_same)
    st.select_ties_total += n_ties
    st.select_max_ties = max(st.select_max_ties, max_ties)
    print(f"[SOCKET-GATE] G2 {tag} M={M} multiset_equal={ms_ok} index_set_equal={idx_same} "
          f"boundary_ties(excess)={n_ties} max_per_row={max_ties}", flush=True)
    return idx_radix


@torch.no_grad()
def gate_head_overlap(tag, scores, rep, k=256):
    """G3. Two query heads sharing a KV head see IDENTICAL buckets and identical ||v||.

    Their scores differ only through q_probs, so the overlap of their top-k selections is a
    direct read of how much the QUERY influences selection.  A jump toward 100% means the
    soft-hash has gone inert and no accuracy number means anything.
    """
    st = state()
    B, H, T = scores.shape
    kk = min(k, T)
    idx = torch.topk(scores, kk, dim=-1).indices  # [B,H,kk]
    tot, n = 0.0, 0
    for b in range(B):
        for h0 in range(0, H, rep):
            if h0 + 1 >= H:
                continue
            a = set(idx[b, h0].tolist())
            c = set(idx[b, h0 + 1].tolist())
            tot += len(a & c) / kk
            n += 1
    if n:
        st.overlap_sum += tot
        st.overlap_n += n
        print(f"[SOCKET-GATE] G3 {tag} top-{kk} within-GQA-group overlap = "
              f"{tot / n * 100:.1f}%  ({n} pairs)", flush=True)


@torch.no_grad()
def gate_liveness(tag, planes, protos_T, q_probs):
    st = state()
    uniq = sorted(set(protos_T.float().unique().tolist()))
    plane_std = float(planes.float().std().item())
    R = q_probs.shape[-1]
    dev_unif = float((q_probs.float() - 1.0 / R).abs().mean().item()) * R
    ok = (set(uniq) == {-1.0, 1.0}) and plane_std > 0.5 and dev_unif > 0.05
    msg = (f"protos_T uniq={uniq} planes_std={plane_std:.4f} "
           f"q_probs mean|dev| from 1/R = {dev_unif * 100:.1f}%  -> "
           f"{'LIVE' if ok else 'DEAD'}")
    st.notes.append(f"{tag} liveness: {msg}")
    print(f"[SOCKET-GATE] G3 {tag} {msg}", flush=True)


@torch.no_grad()
def gate_list(tag, list_new, list_ref):
    st = state()
    same = torch.equal(list_new, list_ref)
    st.list_calls += 1
    st.list_identical += int(same)
    if not same:
        d = (list_new != list_ref).sum().item()
        st.notes.append(f"{tag} list mismatch in {d} slots")
    print(f"[SOCKET-GATE] G5 {tag} triton_list == torch_list : {same}", flush=True)


@torch.no_grad()
def gate_finite(tag, out, sparse_list):
    st = state()
    bad = int((~torch.isfinite(out.float())).sum().item())
    neg = int((sparse_list < 0).sum().item())
    st.nonfinite_logits += bad
    st.neg_index_slots += neg
    print(f"[SOCKET-GATE] G4 {tag} nonfinite_out={bad} neg_index_slots={neg} "
          f"out_absmax={out.float().abs().max().item():.4g}", flush=True)


@torch.no_grad()
def dump_capture(path, **tensors):
    """Persist REAL tensors so the offline test can re-run the gates without a model load."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in tensors.items()}, path)
    print(f"[SOCKET-GATE] captured -> {path}", flush=True)
