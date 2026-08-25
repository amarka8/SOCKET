"""End-to-end A/B of modeling_llama's two arms through build_sparse_list_decode itself.

test_socket_port.py checks each kernel against its replacement in isolation. This checks the
FUNCTION modeling_llama actually calls -- including the piece isolation cannot cover: the
bucket LAYOUT CONTRACT. The arms need different stores ([B,H,L,T] for the CUDA scorer,
[B,Hkv,L,T] for the Triton one), `_BUCKET_LAYOUT` decides which, and prefill slices
accordingly. A transpose or an off-by-rep there produces plausible-looking garbage rather than
an error, so the two arms are run on the SAME underlying data and their selections compared.

The arm switches are read once at import time, so the two arms cannot coexist in one process:
the parent re-execs itself twice with different env and compares the two dumps.

    python tests/test_modeling_llama_arms.py
"""
import os
import subprocess
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_MODELING = os.path.join(REPO_ROOT, "pipeline", "train_quest", "modeling")

B, H, HKV, L, R = 1, 8, 2, 10, 256
T = 4096
SEQ_LEN = 3500
SINK, WINDOW, M = 4, 128, 256
REP = H // HKV


def fixtures(dev):
    """Identical in both children: seeded, and generated in a fixed op order."""
    g = torch.Generator(device="cpu").manual_seed(1234)
    # The store is a CAPACITY: columns past SEQ_LEN are padding and must be zero, exactly as
    # modeling_llama's prefill allocates them. Randomising them would be an out-of-bounds read
    # in the scorer, which masks its loads on the buffer width rather than on the live length.
    kb_kv = torch.zeros((B, HKV, L, T), dtype=torch.int16)
    kb_kv[..., :SEQ_LEN] = torch.randint(0, R, (B, HKV, L, SEQ_LEN), generator=g, dtype=torch.int16)
    kb_kv = kb_kv.to(dev)
    vn_kv = torch.zeros((B, HKV, T), dtype=torch.float32)
    vn_kv[..., :SEQ_LEN] = torch.rand(B, HKV, SEQ_LEN, generator=g, dtype=torch.float32) + 0.5
    vn_kv = vn_kv.half().to(dev)
    q_probs = torch.softmax(torch.randn(B, H, L, R, generator=g, dtype=torch.float32), -1).to(dev)

    # What prefill stores: buckets/||v|| built from repeat_kv'd K/V, i.e. REP identical copies.
    kb_q = kb_kv.repeat_interleave(REP, dim=1).contiguous()
    vn_q = vn_kv.repeat_interleave(REP, dim=1).contiguous()
    allowed = (torch.arange(T, device=dev) < SEQ_LEN).view(1, 1, T).expand(B, H, T).contiguous()
    seq_len_t = torch.tensor(SEQ_LEN, device=dev, dtype=torch.int32)
    return kb_kv, vn_kv, kb_q, vn_q, q_probs, allowed, seq_len_t


def child(out_path):
    sys.path.insert(0, REPO_ROOT)
    sys.path.insert(0, _MODELING)
    import modeling_llama as ml

    dev = "cuda"
    kb_kv, vn_kv, kb_q, vn_q, q_probs, allowed, seq_len_t = fixtures(dev)

    # Reproduce prefill's layout decision rather than hardcoding one: if _BUCKET_LAYOUT and the
    # scorer arm ever drift apart, this test must fail rather than paper over it by picking the
    # layout the scorer happens to want.
    if ml._BUCKET_LAYOUT == "kv":
        kb, vn = kb_kv, vn_kv
    else:
        kb, vn = kb_q, vn_q

    lst, slen, scores = ml.build_sparse_list_decode(
        q_probs, kb, vn, allowed,
        sink=SINK, window=WINDOW, M=M, seq_len_t=seq_len_t, T_true=SEQ_LEN,
    )

    # Run the selected list through both attention arms on the same K/V, so the comparison
    # covers the fourth stage as well as the three the list builder owns.
    gk = torch.Generator(device="cpu").manual_seed(99)
    k = torch.randn(B, HKV, T, 128, generator=gk, dtype=torch.float32).half().to(dev)
    v = torch.randn(B, HKV, T, 128, generator=gk, dtype=torch.float32).half().to(dev)
    q = torch.randn(B, H, 128, generator=gk, dtype=torch.float32).half().to(dev)
    attn = ml._port().sparse_attention_fwd if ml._ARM_ATTN == "gptfast" \
        else ml.sparse_attention_fwd_local
    out = attn(q, k, v, lst.to(torch.int32), slen.to(torch.int32), block_seq=256)

    torch.save({
        "arm": (ml._ARM_SCORER, ml._ARM_SELECT, ml._ARM_LIST, ml._ARM_DEDUP),
        "attn": ml._ARM_ATTN,
        "layout": ml._BUCKET_LAYOUT,
        "list": lst.cpu(), "len": slen.cpu(),
        "scores": None if scores is None else scores.cpu(),
        "out": out.float().cpu(),
    }, out_path)
    print(f"[child] arm={ml._ARM_SCORER}/{ml._ARM_SELECT}/{ml._ARM_LIST}/{ml._ARM_ATTN} "
          f"layout={ml._BUCKET_LAYOUT} list={tuple(lst.shape)} "
          f"live={int((lst >= 0).sum().item())}", flush=True)
    return 0


def run_child(tag, env_extra, out_path):
    env = dict(os.environ)
    env.update(env_extra)
    env["SOCKET_ARM_CHILD"] = out_path
    print(f"--- spawning {tag}: {env_extra}", flush=True)
    r = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env,
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stdout.write(r.stderr[-4000:])
    return r.returncode


def main():
    scratch = os.environ.get("SOCKET_ARM_SCRATCH", "/tmp")
    old_p = os.path.join(scratch, "arm_old.pt")
    new_p = os.path.join(scratch, "arm_new.pt")

    # The old in-file implementations, and the GPT-FAST kernels that are now the default.
    # SOCKET_DEDUP is pinned to 0 in both: it changes WHICH tokens are kept, so leaving it
    # free would confound a kernel comparison with a policy difference.
    rc = run_child("legacy (cuda/topk/torch/local)",
                   {"SOCKET_SCORER": "cuda", "SOCKET_SELECT": "topk",
                    "SOCKET_LIST": "torch", "SOCKET_ATTN": "local",
                    "SOCKET_DEDUP": "0"}, old_p)
    rc |= run_child("default (triton/radix/triton/gptfast)",
                    {"SOCKET_SCORER": "triton", "SOCKET_SELECT": "radix",
                     "SOCKET_LIST": "triton", "SOCKET_ATTN": "gptfast",
                     "SOCKET_DEDUP": "0"}, new_p)
    if rc != 0:
        print("FAILED: a child process did not complete")
        return 1

    old, new = torch.load(old_p, weights_only=False), torch.load(new_p, weights_only=False)
    fails = []

    def check(name, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    check("E1 the arms really did take different code paths",
          old["arm"][:3] == ("cuda", "topk", "torch") and old["attn"] == "local"
          and new["arm"][:3] == ("triton", "radix", "triton") and new["attn"] == "gptfast")
    check("E2 the bucket layout tracked the scorer arm",
          old["layout"] == "q" and new["layout"] == "kv",
          f"legacy={old['layout']} default={new['layout']}")

    so, sn = old["scores"], new["scores"]
    check("E3 scores bitwise-equal on t < seq_len (the layout contract holds on real shapes)",
          torch.equal(so[..., :SEQ_LEN].float(), sn[..., :SEQ_LEN].float()),
          f"max|diff| {(so[..., :SEQ_LEN].float() - sn[..., :SEQ_LEN].float()).abs().max().item():.3e}")

    lo, ln = old["list"], new["list"]
    check("E4 same list width", lo.shape == ln.shape, f"{tuple(lo.shape)} vs {tuple(ln.shape)}")
    check("E5 same sparse_len", torch.equal(old["len"], new["len"]))

    # Compare the SELECTED SETS, not the slot order: torch.topk returns indices sorted by
    # descending score while radix_topm emits in grid order, so element-wise equality of the
    # heavy region is not a property either implementation claims.
    same = True
    for b in range(lo.shape[0]):
        for h in range(lo.shape[1]):
            a = set(lo[b, h][lo[b, h] >= 0].tolist())
            c = set(ln[b, h][ln[b, h] >= 0].tolist())
            if a != c:
                same = False
                print(f"       (b={b},h={h}) legacy-only={sorted(a - c)[:8]} "
                      f"default-only={sorted(c - a)[:8]}")
    check("E6 the two arms select the SAME token set", same)

    check("E7 nothing past seq_len was selected", bool((ln[ln >= 0] < SEQ_LEN).all()))

    # E8 covers the fourth stage. The two attention kernels differ in inner gather width and in
    # where they pin fp32, so they are close rather than equal; the tolerance is what separates
    # "a different reduction order" from "a different answer".
    d = (old["out"] - new["out"]).abs().max().item()
    rel = d / max(new["out"].abs().max().item(), 1e-6)
    check("E8 the two attention arms agree to within 1e-2 relative",
          rel < 1e-2, f"max|diff| = {d:.3e}  relative = {rel:.3e}")

    print()
    if fails:
        print(f"FAILED ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    _child = os.environ.get("SOCKET_ARM_CHILD")
    raise SystemExit(child(_child) if _child else main())
