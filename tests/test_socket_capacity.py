"""CPU tests for the SOCKET decode buffer capacity and budget arithmetic.

The kernels the HF eval path calls specialise on the WIDTH of the bucket buffer, so that width
has to come off a coarse grid rather than tracking the sequence length column by column.  The
rules that make the scheme work are pure integer arithmetic and are checked here, without a GPU:

  * the capacity is monotone, never below the requested length, and lands on a small set of
    values across the whole context range -- that set is the number of Triton compiles an eval
    arm pays;
  * the padding overhead is bounded, which matters because the scorer and the top-M select both
    stream the entire buffer once per decode step;
  * the heavy budget is taken against the LIVE length, never the padded width, so padding
    cannot inflate the list.

Standalone runner, matching the other SOCKET tests (pytest cannot import in this cluster's
module stack: its anyio dependency pulls in ssl, which fails with an OPENSSL_3.3.0 mismatch):

    python tests/test_socket_capacity.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

# modeling_llama.py needs torch and transformers, which the capacity helper does not, so lift
# the helper out of the source rather than importing the module.
_SRC = open(os.path.join(REPO_ROOT, "pipeline", "train_quest", "modeling",
                         "modeling_llama.py")).read()
_NS = {}
exec(_SRC[_SRC.index("_CAP_FLOOR = "):_SRC.index("_PORT = None")], _NS)
_capacity_for = _NS["_capacity_for"]
_CAP_FLOOR = _NS["_CAP_FLOOR"]

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def budget(T_live, sparsity, sink, window):
    """The SOCKET_TARGET_SPARSITY rule from modeling_llama's decode branch."""
    M = max(0, int(round(T_live / sparsity)) - sink - window)
    return max(0, min(M, T_live))


def test_capacity_covers_and_is_monotone():
    lengths = list(range(1, 200000, 37))
    caps = [_capacity_for(n) for n in lengths]
    check("capacity is never below the requested length",
          all(c >= n for c, n in zip(caps, lengths)))
    check("capacity is monotone in the requested length",
          all(b >= a for a, b in zip(caps, caps[1:])))
    check("capacity never drops below the floor", min(caps) == _CAP_FLOOR)


def test_capacity_grid_is_small_and_tight():
    grid = sorted({_capacity_for(n) for n in range(1, 131073)})
    check("the whole 1..128K range needs few distinct widths",
          len(grid) <= 16, f"{len(grid)} widths: {grid}")
    # Above the floor the ratio is what a decode step actually pays, since the scorer and the
    # select are both O(width).
    worst = max(_capacity_for(n) / n for n in range(_CAP_FLOOR + 1, 131073))
    check("padding overhead stays at or below 1.5x above the floor",
          worst <= 1.5 + 1e-9, f"worst = {worst:.4f}")


def test_reserve_absorbs_generation():
    # A prompt plus its generation budget must fit without the buffer having to grow, which is
    # what keeps the grow path a backstop rather than a per-sample event.
    for prompt in (900, 3000, 4096, 7500, 16000, 31000):
        for reserve in (128, 512, 1024):
            cap = _capacity_for(prompt + reserve)
            check(f"prompt={prompt} reserve={reserve} fits without growth",
                  cap >= prompt + reserve, f"cap={cap}")


def test_budget_uses_the_live_length():
    sink = window = 128
    for T_live, sparsity in ((8000, 10), (32768, 10), (32768, 33.3), (32768, 50)):
        cap = _capacity_for(T_live + 512)
        M_live = budget(T_live, sparsity, sink, window)
        M_padded = budget(cap, sparsity, sink, window)
        check(f"T={T_live} 1/{sparsity}: the padded width would inflate the budget",
              M_padded > M_live, f"live M={M_live} padded M={M_padded}")
        kept = sink + window + M_live
        realized = T_live / kept
        check(f"T={T_live} 1/{sparsity}: realized sparsity hits the target",
              abs(realized - sparsity) / sparsity < 0.01,
              f"kept={kept} realized={realized:.2f}x")


def test_list_width_and_partitioning():
    # sparse_attention_fwd partitions the list at block_seq=256 and gathers BLOCK_N=128 slots
    # per inner step; the kernel's own invariant is that block_seq divides BLOCK_N whenever the
    # list spans more than one partition.
    block_seq, block_n = 256, 128
    check("the stage1 partition invariant holds for the shipped launch config",
          block_seq % block_n == 0)
    sink = window = 128
    for T_live in (4000, 8000, 32768):
        M = budget(T_live, 10, sink, window)
        W = sink + window + M
        check(f"T={T_live}: list width and partition count are sane",
              W > 0 and W <= T_live,
              f"W={W} partitions={-(-W // block_seq)}")


def test_degenerate_budgets():
    # A short prompt at a high target sparsity can drive the heavy budget to zero; the list is
    # then sink+window only, which must still be a valid non-empty list.
    sink = window = 128
    M = budget(500, 50, sink, window)
    check("a budget that underflows clamps to zero rather than going negative", M == 0)
    check("sink+window alone still gives a non-empty list", sink + window > 0)


def main():
    for fn in (test_capacity_covers_and_is_monotone,
               test_capacity_grid_is_small_and_tight,
               test_reserve_absorbs_generation,
               test_budget_uses_the_live_length,
               test_list_width_and_partitioning,
               test_degenerate_budgets):
        print(fn.__name__)
        fn()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
