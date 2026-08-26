"""Tabulate the LongBench kernel-parity sweep written by run_lb_kernels.sbatch.

Three tables, because the sweep answers three different questions and mixing them invites
reading an equivalence check as a result:

  1. SCORES        every arm against the dense ceiling.
  2. EQUIVALENCE   legacy vs gptfast at a FIXED hash geometry. The two are different
                   implementations of the same computation, so these should agree; a large
                   gap is a bug, not a finding. Selection ties break differently between
                   torch.topk and the radix select, and the two attention kernels accumulate
                   the online softmax in a different order, so exact equality is not expected.
  3. GEOMETRY      the new kernels at each (P, L). This is the actual question: how the
                   hash geometry moves the scores.

usage:  python scripts/socket_accuracy/collect.py [out_dir]
"""
import json
import os
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "/scratch/sj157/socket_speed/kfix/out"

ARMS = ["dense", "legacy_P8L50", "gptfast_P8L50",
        "legacy_P10L10", "gptfast_P10L10", "gptfast_P10L60"]
LABEL = {
    "dense": "dense (no SOCKET)",
    "legacy_P8L50": "P8/L50   legacy kernels",
    "gptfast_P8L50": "P8/L50   GPT-FAST kernels",
    "legacy_P10L10": "P10/L10  legacy kernels",
    "gptfast_P10L10": "P10/L10  GPT-FAST kernels",
    "gptfast_P10L60": "P10/L60  GPT-FAST kernels",
}
DS = ["hotpotqa", "samsum", "qasper", "multifieldqa_en"]
PAIRS = [("legacy_P8L50", "gptfast_P8L50", "P8/L50"),
         ("legacy_P10L10", "gptfast_P10L10", "P10/L10")]


def load():
    scores = {}
    for arm in ARMS:
        for ds in DS:
            path = os.path.join(OUT, f"{arm}_{ds}", "raw_results.json")
            if not os.path.exists(path):
                continue
            try:
                r = json.load(open(path))
            except Exception as exc:
                print(f"  (unreadable {path}: {exc})")
                continue
            n = r.get("total", 0)
            scores[(arm, ds)] = (r["total_score"] / n * 100 if n else None, n)
    return scores


def cell(v):
    return f"{v[0]:.2f}" if v and v[0] is not None else "--"


def main():
    S = load()
    have = sum(1 for a in ARMS for d in DS if (a, d) in S)
    counts = {v[1] for v in S.values()}
    print(f"{OUT}\n{have}/{len(ARMS) * len(DS)} cells present"
          f"{'' if len(counts) <= 1 else '   WARNING: arms saw DIFFERENT sample counts ' + str(sorted(counts))}")
    if have and len(counts) == 1:
        print(f"n = {counts.pop()} samples per cell")
    print()

    print("1. SCORES (higher is better)")
    print(f"{'arm':30}" + "".join(f"{d[:14]:>16}" for d in DS))
    for arm in ARMS:
        print(f"{LABEL[arm]:30}" + "".join(f"{cell(S.get((arm, d))):>16}" for d in DS))
    print()
    print("   gap vs dense (negative = worse than dense)")
    for arm in ARMS[1:]:
        row = f"   {LABEL[arm]:27}"
        for d in DS:
            v, dv = S.get((arm, d)), S.get(("dense", d))
            row += f"{(f'{v[0] - dv[0]:+.2f}' if v and dv and None not in (v[0], dv[0]) else '--'):>16}"
        print(row)
    print()

    print("2. EQUIVALENCE -- legacy vs GPT-FAST kernels at a fixed geometry")
    print("   (these are two implementations of the same computation; a large gap is a bug)")
    worst = 0.0
    for old_arm, new_arm, tag in PAIRS:
        row = f"   {tag:27}"
        for d in DS:
            o, n = S.get((old_arm, d)), S.get((new_arm, d))
            if o and n and None not in (o[0], n[0]):
                delta = n[0] - o[0]
                worst = max(worst, abs(delta))
                row += f"{delta:+16.2f}"
            else:
                row += f"{'--':>16}"
        print(row)
    if worst:
        print(f"   largest |delta| = {worst:.2f}")
    print()

    print("3. GEOMETRY -- the GPT-FAST kernels at each (P, L)")
    for arm in ("gptfast_P10L10", "gptfast_P8L50", "gptfast_P10L60"):
        print(f"   {LABEL[arm]:27}" + "".join(f"{cell(S.get((arm, d))):>16}" for d in DS))
    print("   delta vs P10/L60 (the shipped RULER geometry)")
    for arm in ("gptfast_P10L10", "gptfast_P8L50"):
        row = f"   {LABEL[arm]:27}"
        for d in DS:
            v, ref = S.get((arm, d)), S.get(("gptfast_P10L60", d))
            row += f"{(f'{v[0] - ref[0]:+.2f}' if v and ref and None not in (v[0], ref[0]) else '--'):>16}"
        print(row)


if __name__ == "__main__":
    main()
