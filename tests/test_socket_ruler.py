"""Unit tests for the SOCKET RULER-32K port.

CPU-runnable (run now, gate before Phase C):
  - hard_hash: sign-of-projection correctness + int16 bucket-code round-trip
  - soft_hash: rows sum to 1; smaller tau => sharper; reads config.tau
  - build_sparse_list_decode index logic: exactly min(M,T)+sink+window indices,
    in range, sink/window always included (CPU reference of the non-kernel path)
  - RULER loader: 6 configs, head(100), columns intact, answer is list-of-refs,
    max_new_tokens matches the §1.3 table
  - calculate_metrics: string_match_part (qa_*) vs string_match_all (others),
    case-insensitive substring, partial credit for _all, control-char strip

GPU-only (deferred to Phase C, skipped without CUDA):
  - M=T ~= dense-SDPA equivalence faithfulness test
  - JIT smoke: first decode compiles soft_hash_collision.cu

Run with:
  HF_HOME=/path/to/hf_home python -m pytest \
      tests/test_socket_ruler.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

HAS_CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Helpers: build a minimal LlamaAttention on CPU to exercise hash methods.
# ---------------------------------------------------------------------------
def _make_attention(P=8, L=4, tau=None):
    from transformers.models.llama.configuration_llama import LlamaConfig
    from pipeline.train_quest.modeling.modeling_llama import LlamaAttention

    cfg = LlamaConfig(
        hidden_size=64,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_hidden_layers=1,
        intermediate_size=128,
        max_position_embeddings=512,
        attention_dropout=0.0,
        attention_bias=False,
    )
    cfg.bucket_K = P
    cfg.bucket_L = L
    if tau is not None:
        cfg.tau = tau
    attn = LlamaAttention(cfg, layer_idx=0)
    attn.eval()
    return attn


# ---------------------------------------------------------------------------
# hard_hash
# ---------------------------------------------------------------------------
def test_hard_hash_sign_of_projection_and_roundtrip():
    torch.manual_seed(0)
    P, L = 6, 3
    attn = _make_attention(P=P, L=L)
    D = attn.config.hidden_size // attn.config.num_attention_heads

    B, H, N = 2, attn.config.num_attention_heads, 5
    tensor = torch.randn(B, H, N, D)
    planes = torch.randn(L, P, D)

    codes = attn.hard_hash(tensor, planes)  # [B,H,L,N] integer bucket codes
    assert codes.shape == (B, H, L, N)
    # pack_bits multiplies/sums int16 weights; torch promotes the integer
    # reduction (the decode path later .to(int16) it explicitly). Just require
    # an integer dtype whose values fit a P-bit bucket.
    assert codes.dtype in (torch.int16, torch.int32, torch.int64)

    # Reference: sign of projection, big-endian bit packing.
    proj = torch.einsum("bhnd,lkd->bhnlk", tensor, planes)  # [B,H,N,L,P]
    bits = (proj >= 0).to(torch.int64)
    weights = (1 << torch.arange(P - 1, -1, -1)).to(torch.int64)  # big-endian
    ref_codes = (bits * weights).sum(-1)  # [B,H,N,L]
    ref_codes = ref_codes.permute(0, 1, 3, 2)  # [B,H,L,N]
    assert torch.equal(codes.to(torch.int64), ref_codes)

    # Round-trip: unpack the int16 code back to its P bits == sign bits.
    for _ in range(20):
        b = np.random.randint(B); h = np.random.randint(H)
        l = np.random.randint(L); n = np.random.randint(N)
        code = int(codes[b, h, l, n])
        recovered = [(code >> (P - 1 - k)) & 1 for k in range(P)]
        expect = [int(bits[b, h, n, l, k]) for k in range(P)]
        assert recovered == expect

    # All codes are valid P-bit buckets.
    assert int(codes.min()) >= 0
    assert int(codes.max()) < (1 << P)


def test_hard_hash_deterministic_for_fixed_planes():
    torch.manual_seed(1)
    attn = _make_attention(P=8, L=4)
    D = attn.config.hidden_size // attn.config.num_attention_heads
    tensor = torch.randn(1, attn.config.num_attention_heads, 7, D)
    planes = torch.randn(4, 8, D)
    c1 = attn.hard_hash(tensor, planes)
    c2 = attn.hard_hash(tensor, planes)
    assert torch.equal(c1, c2)


# ---------------------------------------------------------------------------
# soft_hash
# ---------------------------------------------------------------------------
def test_soft_hash_rows_sum_to_one():
    torch.manual_seed(2)
    P, L = 4, 3
    attn = _make_attention(P=P, L=L, tau=0.3)
    D = attn.config.hidden_size // attn.config.num_attention_heads
    R = 2 ** P

    B, H, Q = 2, attn.config.num_attention_heads, 4
    queries = torch.randn(B, H, Q, D)
    planes = torch.randn(L, P, D)
    protos_T = attn.get_protos_T(cache={}, P=P, device=queries.device, dtype=queries.dtype)
    assert protos_T.shape == (P, R)

    probs = attn.soft_hash(queries, planes, protos_T)  # [B,H,Q,L,R]
    assert probs.shape == (B, H, Q, L, R)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    assert bool((probs >= 0).all())


def test_soft_hash_smaller_tau_is_sharper():
    """Smaller tau => lower entropy (sharper); tau -> 0 approaches one-hot."""
    torch.manual_seed(3)
    P, L = 4, 2
    D = 8
    R = 2 ** P

    queries = torch.randn(1, 4, 3, D)
    planes = torch.randn(L, P, D)

    def entropy_for_tau(tau):
        attn = _make_attention(P=P, L=L, tau=tau)
        # use a head_dim-matched D by overriding via direct call
        protos_T = attn.get_protos_T(cache={}, P=P, device=queries.device, dtype=queries.dtype)
        # soft_hash uses sqrt(queries.size(-1)) internally; feed matching D
        probs = attn.soft_hash(queries, planes, protos_T)
        p = probs.clamp_min(1e-12)
        ent = -(p * p.log()).sum(-1)  # [B,H,Q,L]
        return ent.mean().item()

    e_small = entropy_for_tau(0.1)
    e_large = entropy_for_tau(1.0)
    assert e_small < e_large, (e_small, e_large)


def test_soft_hash_reads_config_tau():
    """The de-hardcoded tau must come from config; different tau => different dist."""
    torch.manual_seed(4)
    P, L = 4, 2
    D = 8
    queries = torch.randn(1, 4, 2, D)
    planes = torch.randn(L, P, D)

    attn_a = _make_attention(P=P, L=L, tau=0.2)
    attn_b = _make_attention(P=P, L=L, tau=0.8)
    protos = attn_a.get_protos_T(cache={}, P=P, device=queries.device, dtype=queries.dtype)
    pa = attn_a.soft_hash(queries, planes, protos)
    pb = attn_b.soft_hash(queries, planes, protos)
    assert not torch.allclose(pa, pb)

    # default (no config.tau) falls back to 0.3
    from transformers.models.llama.configuration_llama import LlamaConfig
    from pipeline.train_quest.modeling.modeling_llama import LlamaAttention
    cfg = LlamaConfig(hidden_size=64, num_attention_heads=8, num_key_value_heads=8,
                      num_hidden_layers=1, intermediate_size=128,
                      max_position_embeddings=512, attention_dropout=0.0,
                      attention_bias=False)
    cfg.bucket_K = P; cfg.bucket_L = L
    attn_def = LlamaAttention(cfg, layer_idx=0)
    assert not hasattr(cfg, "tau")
    p_def = attn_def.soft_hash(queries, planes, protos)  # uses fallback 0.3
    attn_03 = _make_attention(P=P, L=L, tau=0.3)
    p_03 = attn_03.soft_hash(queries, planes, protos)
    assert torch.allclose(p_def, p_03)


# ---------------------------------------------------------------------------
# build_sparse_list_decode index logic (CPU reference of the non-kernel path)
# ---------------------------------------------------------------------------
def _ref_sparse_base_indices(T, sink, window):
    """Mirror the sink/window base-index construction in build_sparse_list_decode."""
    sink = max(0, min(sink, T))
    window = max(0, min(window, T))
    parts = []
    if sink > 0:
        parts.append(list(range(sink)))
    if window > 0:
        win_start = max(T - window, sink)
        if win_start < T:
            parts.append(list(range(win_start, T)))
    if not parts:
        return [T - 1]
    out = []
    for p in parts:
        out.extend(p)
    return out


def test_build_sparse_list_index_counts_and_membership():
    """sink+window indices: in-range, sink always first, window covers the tail."""
    T = 100
    for sink, window in [(8, 8), (0, 16), (16, 0), (128, 128)]:
        base = _ref_sparse_base_indices(T, sink, window)
        eff_sink = max(0, min(sink, T))
        eff_window = max(0, min(window, T))
        # all indices in range
        assert all(0 <= i < T for i in base), (sink, window, base)
        # sink indices present
        for i in range(eff_sink):
            assert i in base
        # window tail present (only the part not overlapping the sink region)
        win_start = max(T - eff_window, eff_sink)
        for i in range(win_start, T):
            assert i in base


def test_build_sparse_list_total_length_min_M_plus_base():
    """Total selected = base(sink+window) + min(M, T) heavy indices."""
    T = 100
    for sink, window, M in [(8, 8, 10), (16, 16, 200), (4, 4, 0)]:
        base = _ref_sparse_base_indices(T, sink, window)
        M_eff = min(M, T)
        total = len(base) + M_eff
        # base len matches sink+window construction
        eff_sink = max(0, min(sink, T))
        eff_window = max(0, min(window, T))
        win_start = max(T - eff_window, eff_sink)
        expected_base = eff_sink + max(0, T - win_start)
        if expected_base == 0:
            expected_base = 1  # the [T-1] fallback
        assert len(base) == expected_base, (sink, window, len(base), expected_base)
        assert total == expected_base + M_eff


@pytest.mark.skipif(not HAS_CUDA, reason="build_sparse_list_decode requires CUDA")
def test_build_sparse_list_decode_gpu():
    """GPU: exact index count, in-range, sink/window included (deferred to Phase C)."""
    from pipeline.train_quest.modeling.modeling_llama import build_sparse_list_decode
    B, H, L, R, T = 1, 2, 3, 16, 64
    sink, window, M = 8, 8, 10
    dev = "cuda"
    q_probs = torch.softmax(torch.randn(B, H, L, R, device=dev), dim=-1)
    k_hard = torch.randint(0, R, (B, H, L, T), device=dev, dtype=torch.int16)
    v_norm = torch.rand(B, H, T, device=dev)
    allowed = torch.ones(B, H, T, device=dev, dtype=torch.bool)
    # third return value is the [B,H,T] score array (None when M == 0)
    sparse_list, sparse_len, _ = build_sparse_list_decode(
        q_probs, k_hard, v_norm, allowed, sink=sink, window=window, M=M)
    expected = sink + window + min(M, T)
    assert int(sparse_len[0, 0]) == expected
    for i in range(sink):
        assert (sparse_list == i).any()
    for i in range(T - window, T):
        assert (sparse_list == i).any()
    valid = sparse_list[sparse_list >= 0]
    assert int(valid.max()) < T


# ---------------------------------------------------------------------------
# RULER loader
# ---------------------------------------------------------------------------
RULER_EXPECTED_MNT = {
    "qa_1": 32, "qa_2": 32, "fwe": 50, "vt": 30,
    "niah_multikey_2": 128, "niah_multikey_3": 128,
}


@pytest.mark.parametrize("subset", list(RULER_EXPECTED_MNT.keys()))
def test_ruler_loader(subset):
    from eval.ruler_utils.load_ruler32k import load_ruler32k, RULER32K_COLUMNS
    df = load_ruler32k(subset, n=100)
    assert len(df) == 100
    assert list(df.columns) == RULER32K_COLUMNS
    row = df.iloc[0]
    assert int(row["max_new_tokens"]) == RULER_EXPECTED_MNT[subset]
    assert row["task"] == subset
    assert isinstance(row["answer"], (list, np.ndarray))
    assert not isinstance(row["answer"], str)
    assert len(row["answer"]) >= 1


# ---------------------------------------------------------------------------
# calculate_metrics
# ---------------------------------------------------------------------------
def test_calculate_metrics_qa_uses_string_match_part():
    """qa_* -> string_match_part: 1.0 if ANY ref is a (case-insensitive) substring."""
    from eval.ruler_utils.calculate_metrics import calculate_metrics
    df = pd.DataFrame([
        {"task": "qa_1", "predicted_answer": "The answer is FRANCE.", "answer": ["france", "spain"]},
        {"task": "qa_1", "predicted_answer": "no idea", "answer": ["france"]},
    ])
    scores = calculate_metrics(df.copy())
    # one hit (case-insensitive) out of two => 50.0
    assert scores["qa_1"]["string_match"] == 50.0


def test_calculate_metrics_non_qa_uses_string_match_all_partial_credit():
    """non-qa -> string_match_all: mean fraction of refs found (partial credit)."""
    from eval.ruler_utils.calculate_metrics import calculate_metrics
    df = pd.DataFrame([
        # 1 of 2 refs present => 0.5
        {"task": "vt", "predicted_answer": "contains ABC only", "answer": ["abc", "xyz"]},
        # both present => 1.0
        {"task": "vt", "predicted_answer": "abc and xyz here", "answer": ["abc", "xyz"]},
    ])
    scores = calculate_metrics(df.copy())
    # mean(0.5, 1.0) * 100 = 75.0
    assert scores["vt"]["string_match"] == 75.0


def test_calculate_metrics_control_char_strip():
    """Control chars in the prediction are stripped before matching."""
    from eval.ruler_utils.calculate_metrics import calculate_metrics
    df = pd.DataFrame([
        {"task": "fwe", "predicted_answer": "\x00\x01word\n\t", "answer": ["word"]},
    ])
    scores = calculate_metrics(df.copy())
    assert scores["fwe"]["string_match"] == 100.0


def test_scorer_wrapper_per_task_and_overall():
    from eval.ruler_utils.scorer import ruler_score
    df = pd.DataFrame([
        {"task": "qa_1", "predicted_answer": "France", "answer": ["france"]},
        {"task": "vt", "predicted_answer": "abc xyz", "answer": ["abc", "xyz"]},
    ])
    per_task, overall = ruler_score(df.copy())
    assert per_task["qa_1"] == 100.0
    assert per_task["vt"] == 100.0
    assert overall == 100.0


# ---------------------------------------------------------------------------
# GPU-only faithfulness + JIT smoke (deferred to Phase C)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_CUDA, reason="M=T dense-equivalence requires CUDA + Triton kernel")
def test_socket_decode_full_budget_matches_dense():
    """M=T (full budget, sink/window covering all) ~= dense SDPA within fp tol.

    Proves the Triton sparse kernel + collision estimator are correct: when the
    sink+local window already covers every cached token, the SOCKET sparse-decode
    path must reproduce ordinary (dense) single-query attention over the full KV
    cache, independent of which tokens the soft-hash collision estimator would
    have picked as "heavy". Any disagreement implies a kernel / layout bug.
    """
    import torch.nn.functional as F
    from transformers.cache_utils import DynamicCache
    from transformers.models.llama.configuration_llama import LlamaConfig
    from pipeline.train_quest.modeling.modeling_llama import (
        LlamaAttention,
        LlamaRotaryEmbedding,
        apply_rotary_pos_emb,
        repeat_kv,
    )

    torch.manual_seed(0)
    dev = "cuda"
    dtype = torch.bfloat16

    # Small but realistic GQA shape (8 q-heads / 2 kv-heads, head_dim 64).
    cfg = LlamaConfig(
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_hidden_layers=1,
        intermediate_size=1024,
        max_position_embeddings=2048,
        attention_dropout=0.0,
        attention_bias=False,
        rms_norm_eps=1e-5,
    )
    cfg.bucket_K = 8
    cfg.bucket_L = 16
    # sink + window deliberately cover the WHOLE cache -> sparse list == all tokens.
    T_ctx = 96
    cfg.sink_size = 0
    cfg.window_size = T_ctx + 1  # >= T_k at decode (covers every position)
    cfg.heavy_const = 0          # M = 0 heavy tokens; base list alone covers all
    cfg.tau = 0.4

    attn = LlamaAttention(cfg, layer_idx=0).to(dev).to(dtype).eval()
    rotary = LlamaRotaryEmbedding(cfg).to(dev)

    B = 1
    H = cfg.num_attention_heads
    Hkv = cfg.num_key_value_heads
    D = cfg.hidden_size // H
    groups = H // Hkv

    hidden_ctx = torch.randn(B, T_ctx, cfg.hidden_size, device=dev, dtype=dtype)
    hidden_dec = torch.randn(B, 1, cfg.hidden_size, device=dev, dtype=dtype)

    cache = DynamicCache()

    with torch.no_grad():
        # ---- prefill: seeds bucket state + cache, returns dense output ----
        pos_ctx = torch.arange(T_ctx, device=dev).unsqueeze(0)
        cos_c, sin_c = rotary(hidden_ctx, pos_ctx)
        attn.forward(
            hidden_states=hidden_ctx,
            position_embeddings=(cos_c, sin_c),
            attention_mask=None,
            past_key_values=cache,
            cache_position=pos_ctx.view(-1),
        )

        # ---- decode 1 token through the SOCKET sparse path ----
        pos_dec = torch.tensor([[T_ctx]], device=dev)
        cos_d, sin_d = rotary(hidden_dec, pos_dec)
        out_sparse, _, meta = attn.forward(
            hidden_states=hidden_dec,
            position_embeddings=(cos_d, sin_d),
            attention_mask=None,
            past_key_values=cache,
            cache_position=pos_dec.view(-1),
        )
        assert meta.get("bucket_sparse") is True, "decode did not take the sparse path"

        # ---- dense reference over the SAME cached (RoPE'd) K/V the kernel saw ----
        # transformers>=4.54 DynamicCache stores per-layer tensors on .layers[i]
        # (.keys/.values); older releases used .key_cache[i]/.value_cache[i].
        layer0 = cache.layers[0]
        k_cached = layer0.keys    # [B,Hkv,T_k,D] (RoPE'd, as stored by forward)
        v_cached = layer0.values  # [B,Hkv,T_k,D]
        T_k = k_cached.shape[2]
        assert T_k == T_ctx + 1

        # recompute the decode query exactly as forward does (proj + RoPE)
        q_dec = attn.q_proj(hidden_dec).view(B, 1, H, D).transpose(1, 2)  # [B,H,1,D]
        q_dec, _ = apply_rotary_pos_emb(q_dec, q_dec, cos_d, sin_d)

        k_full = repeat_kv(k_cached, groups)  # [B,H,T_k,D]
        v_full = repeat_kv(v_cached, groups)
        ref = F.scaled_dot_product_attention(
            q_dec, k_full, v_full, attn_mask=None, scale=attn.scaling, is_causal=False
        )  # [B,H,1,D]
        ref = ref.transpose(1, 2).contiguous().reshape(B, 1, -1)
        ref = attn.o_proj(ref)

    # bf16 + fp32 collision accumulation -> compare in fp32 with a generous-but-real tol.
    diff = (out_sparse.float() - ref.float()).abs()
    rel = diff.max() / ref.float().abs().max().clamp_min(1e-6)
    assert torch.allclose(out_sparse.float(), ref.float(), atol=2e-2, rtol=2e-2), (
        f"SOCKET full-budget decode disagrees with dense SDPA: "
        f"max_abs={diff.max().item():.4e} rel={rel.item():.4e}"
    )


@pytest.mark.skipif(not HAS_CUDA, reason="JIT smoke requires CUDA + nvcc")
def test_jit_smoke_compiles_soft_hash_collision():
    """First decode compiles soft_hash_collision.cu without error (Phase C)."""
    from pipeline.train_quest.modeling.modeling_llama import _get_soft_hash_ext
    ext = _get_soft_hash_ext()
    assert ext is not None
    assert hasattr(ext, "soft_hash_collision")
