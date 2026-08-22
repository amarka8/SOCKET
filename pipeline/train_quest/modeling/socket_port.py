"""Bridge that lets the HF eval path call the GPT-FAST decode kernels.

WHY THIS FILE EXISTS AT ALL.  The throughput numbers this project quotes for SOCKET at long
context were measured in GPT-FAST, using three kernels that live in
`GPT-FAST/kernels/sparse.py`:

    the Triton soft-hash SCORER      socket::soft_hash_score   (GQA-group bucket sharing)
    the exact radix top-M SELECT     socket::radix_topm        (kernels/radix_select.cu)
    the one-kernel LIST assembly     socket::build_list

`modeling_llama.py` carries its own, older implementation of the same three steps (the
load_inline CUDA scorer, torch.topk, and an arange/cat/gather op chain).  Its SOCKET_SCORER /
SOCKET_SELECT / SOCKET_LIST switches select between the two, so that the HF eval numbers and
the GPT-FAST timing numbers are produced by the SAME kernels rather than by two independent
reimplementations that have to be kept in agreement by hand.  This module is what those
switches call.

WHY IT IS NOT A PLAIN IMPORT.  There are two DIFFERENT top-level packages named `kernels` in
this repo:

    <repo>/kernels/           block_sparse_flash_attention.py, socket_triton_kernels.py
    <repo>/GPT-FAST/kernels/  sparse.py, radix_select*, soft_hash_collision*

`modeling_llama.py` already imports the first one (`from kernels.socket_triton_kernels import
sparse_attention_fwd`), so by the time anything here runs, `sys.modules["kernels"]` is the
REPO-ROOT package.  Putting `GPT-FAST` on sys.path afterwards does nothing: the name is
already bound, and `import kernels.sparse` would fail.  Whichever `kernels` won sys.path
first also silently decided which `soft_hash_collision_loader` got compiled, which is how an
earlier attempt at this port went wrong.

So: load each GPT-FAST module by EXPLICIT FILE PATH via importlib, under private module names
that cannot collide.  `sparse.py` lazily does `from kernels.radix_select_loader import ...`
and `from kernels.soft_hash_collision_loader import ...` inside function bodies, so those two
submodule names are pre-seeded into sys.modules from the GPT-FAST directory before sparse.py
can ask for them.  Both are seeded only if absent, so we never displace a module some other
importer already owns.

WHAT IS EXPORTED.  Thin pass-throughs, deliberately adding no arithmetic of their own -- the
point of the port is that the HF path executes the same code, so anything computed here
instead of there would defeat it.

    soft_hash_score_rt(q_probs, key_buckets, v_norm, seq_len_t, out) -> None   (writes `out`)
    radix_topm(scores, M) -> [B,H,M] int32
    build_list_rt(heavy, seq_len_t, out, sink, window, maxlen, dedup) -> None  (writes `out`)
"""
import importlib.util
import os
import sys

import torch

# <repo>/GPT-FAST/kernels
_GPT_FAST_KERNELS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "GPT-FAST", "kernels")
)


def _load_by_path(module_name: str, filename: str):
    """Import <_GPT_FAST_KERNELS>/<filename> as `module_name`, bypassing sys.path entirely."""
    path = os.path.join(_GPT_FAST_KERNELS, filename)
    if not os.path.isfile(path):
        raise ImportError(
            f"socket_port: cannot find {path}. The HF path's Triton scorer / radix select / "
            f"Triton list-assembly arms reuse the GPT-FAST kernels, so GPT-FAST/kernels must "
            f"be present in the same checkout."
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec so a module that imports itself (or is re-entered) resolves.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_SPARSE = None


def _sparse():
    """GPT-FAST/kernels/sparse.py, loaded once per process."""
    global _SPARSE
    if _SPARSE is not None:
        return _SPARSE

    # Seed the two submodule names sparse.py asks for by their PACKAGE-QUALIFIED names, so its
    # lazy `from kernels.<x> import ...` hits sys.modules and never consults sys.path (where
    # `kernels` is the repo-root package and has neither of these).  Only if absent: if some
    # other importer already owns the name, its module is the one that must be used.
    for sub, fname in (
        ("radix_select_loader", "radix_select_loader.py"),
        ("soft_hash_collision_loader", "soft_hash_collision_loader.py"),
    ):
        qualified = f"kernels.{sub}"
        if qualified not in sys.modules:
            _load_by_path(qualified, fname)

    # If sparse.py is ALREADY imported under its own package name -- e.g. this process is a
    # GPT-FAST run that also pulled in the HF modeling file -- reuse that module object.
    # Loading the file a second time under a second name would re-execute its
    # torch.library.custom_op / triton_op registrations for "socket::radix_topm" etc., which
    # raises on the duplicate name.
    for existing in ("kernels.sparse", "sparse"):
        mod = sys.modules.get(existing)
        if mod is not None and getattr(mod, "__file__", None) and \
                os.path.abspath(mod.__file__) == os.path.join(_GPT_FAST_KERNELS, "sparse.py"):
            _SPARSE = mod
            return _SPARSE

    _SPARSE = _load_by_path("_socket_gptfast_sparse", "sparse.py")
    return _SPARSE


# ---------------------------------------------------------------------------
# SCORER.  socket::soft_hash_score, the MUTATING triton_op form (it writes `out` rather than
# allocating).  sparse.py also registers a non-mutating `soft_hash_score_alloc` around the
# same kernel; the mutating form is the one GPT-FAST's own decode path uses, so it is the one
# used here.
#
# LAYOUT.  key_buckets/v_norm are PER-KV-HEAD ([B,Hkv,L,T] / [B,Hkv,T]) while q_probs and out
# are PER-QUERY-HEAD ([B,H,L,R] / [B,H,T]).  The kernel recovers a query head's bucket row as
# h // (H // Hkv), which is exactly the mapping repeat_kv applies -- see the LAYOUT CONTRACT
# note on build_sparse_list_decode in modeling_llama.py.  Hkv is read off key_buckets, so any
# GQA group size works; passing per-query-head buckets (Hkv == H, rep == 1) is also valid and
# simply shares nothing.
# ---------------------------------------------------------------------------
def soft_hash_score_rt(q_probs, key_buckets, v_norm, seq_len_t, out):
    """Score [B,H,T] into `out` (fp32). seq_len_t is an int32 scalar tensor on device."""
    return _sparse().soft_hash_score_op(
        q_probs, key_buckets, v_norm, seq_len_t.reshape(()), out
    )


# ---------------------------------------------------------------------------
# SELECT.  socket::radix_topm -- an EXACT top-M index select (same score multiset as
# aten::topk; which tokens tie AT the threshold is arbitrary in topk too).  The tuning knobs
# (SOCKET_RS_*) are read inside sparse.py, so they apply identically to both paths.
# ---------------------------------------------------------------------------
def radix_topm(scores, M):
    """scores [B,H,T] fp32 (contiguous) -> [B,H,M] int32 indices."""
    _sparse()  # importing sparse.py is what registers the operator
    return torch.ops.socket.radix_topm(scores.contiguous(), int(M))


# ---------------------------------------------------------------------------
# LIST ASSEMBLY.  socket::build_list -- emits sink | window | heavy directly, gating every
# slot on `0 <= idx < seq_len` (which is precisely what the `allowed` mask encodes, so no
# allowed tensor is needed or read).
#
# `dedup` is threaded through rather than hardwired because the HF path must be able to
# reproduce BOTH of its arms: its torch op chain concatenates base and heavy without
# deduplicating unless SOCKET_DEDUP=1, and a switch that silently deduplicated would make the
# two arms disagree for reasons unrelated to the kernels being compared.
# ---------------------------------------------------------------------------
def build_list_rt(heavy, seq_len_t, out, sink, window, maxlen, dedup=True):
    """Write the [B,H,W] int32 index list into `out`. W must be sink + window + M."""
    return _sparse().build_list_op(
        heavy, seq_len_t.reshape(()), out,
        int(sink), int(window), int(maxlen), bool(dedup),
    )
