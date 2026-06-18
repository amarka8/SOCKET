# SOCKET_orig Integration Plan: RULER-32K port + SmallWorld→SOCKET rename + test plan

Status: SCOPING (read-only). This document is the implementation spec; no code has been changed.
Repo: `/scratch/sj157/SOCKET_orig` (clean `main`, working tree clean, efficient SOCKET decode merged).
Paper: `/scratch/sj157/socket_paper.txt`.

The repo's REAL accuracy path is the HF-model + soft-LSH masker in
`pipeline/train_quest/run.py` → custom `modeling/modeling_llama.py` → manual decode in `CE.py`,
scored by `eval/longbench_utils/eval_long_bench.py:scorer`. The GPT-FAST kernels are
throughput-only and are NOT used for accuracy. All accuracy work goes through the train_quest path.

---

## 0. Verified facts (checked live against the repo this pass)

- `run.py:149` gates the masker on the literal string: `use_smallworld = pipeline_params.get("method") == "smallworld"`; `run.py:157` `if use_smallworld and is_llama_instruct:`. `topk_*` attrs set here are DEAD (modeling never reads them).
- `modeling_llama.py:638` `return F.softmax(logits/0.3, dim=-1)` — **tau is hardcoded 0.3**.
- SOCKET knobs are read by getattr-with-default only, set nowhere in config:
  `:443 bucket_K=getattr(config,"bucket_K",8)` (=paper P; default 8, paper wants 10),
  `:444 bucket_L=getattr(config,"bucket_L",60)` (=paper L; matches),
  `:805 sink=getattr(config,"sink_size",20)`, `:806 window=getattr(config,"window_size",20)`,
  `:807 M_cfg=getattr(config,"heavy_const",getattr(config,"heavy_size",0.1))` (float=ratio).
- Ground-truth load `run.py:100-106`: only `if eval_params['dataset'] in LONGBENCH_DATASET` → `longbench_eval.load_data` (= `load_from_disk('dataset/longbench/{ds}')`), reads `example['answers']` + `ds[0]["all_classes"]`. RULER tasks are NOT in this set → no GT, scorer KeyErrors. **First blocker.**
- Scorer `run.py:409-413`: `longbench_eval.scorer(dataset, [answer], [ground_truth[idx]], all_classes)` via `dataset2metric`. RULER needs substring metrics, not qa_f1/rouge. **Second blocker.**
- Prompt build (`dataset.py`): `get_dataset`/`load_longctx_dataset` create a `prompt` column via `dataset_config['instruction'].format(**ex)`; `get_val_dataset:217` tokenizes `sample["prompt"]` with chat template (system="You are a useful assistant.", `apply_chat_template(..., add_generation_prompt=True)`), middle-truncates to `max_model_len`. RULER prompt = `context+question+answer_prefix` and must NOT be re-wrapped in a longbench instruction. **Third blocker.**
- `CE.generate` (CE.py:115-210): batch_size==1, greedy argmax only, carries `past_key_values` (DynamicCache w/ bucket_states) across steps → step 0 dense prefill seeds state, steps 1+ sparse decode. `do_sample/temperature/top_p` ignored. max_new_tokens is a single eval_params value today.
- Dataset on hub `xAlg-AI/att-hub-ruler-32k`: 13 configs, 200 rows each, columns EXACTLY `['context','question','answer_prefix','answer','task','max_new_tokens']`; `answer` is a numpy ndarray of ref strings; config name == split name == task name. All 6 requested tasks present at 32768-token budget.
- `calculate_metrics.py` (hub): `string_match_part` for `task.split("_")[0]=="qa"` (max-over-refs substring, 1 if ANY ref in pred), `string_match_all` otherwise (mean fraction of refs that are substrings of pred). Control-char strip `[\x00-\x1f]`. Self-contained (no rouge/jieba).
- SmallWorld footprint dirs confirmed: `scripts/small_world_inference/`, `config/pipeline_config/SmallWorld/` (14 JSONs / 4 model subdirs), `longbench/SmallWorld/` (result artifacts). Refs in `run.py:149,157` + `REPRODUCE.txt`. GPT-FAST already uses `socket`/`SOCKET` naming (coherent target).
- Dead scaffold: `method:"fafo"` (no Python branch → falls to dense), `method:"LearnHash"`/`"train"` shell vars → nonexistent config dirs, `train_mode:"train_k"` not in `set_masker_mode` allowed set {joint, only_selector, inference_only}. All already broken.

---

## 1. RULER-32K PORT

Strategy: REIMPLEMENT against SOCKET's own train_quest runner (do NOT vendor sparse-attention-hub's
`base.py`/`executor.py`/`ModelAdapterHF` — SOCKET lacks the Request/RequestResponse/ModelAdapter contract
and the hub framework is not installed). Reuse SOCKET's dataset→get_val_dataset→CE.generate→scorer flow;
add a RULER branch at each of the 3 blockers. Keep the hub's data contract + scoring VERBATIM so numbers
stay paper-comparable.

### 1.1 Files to ADD (all under `/scratch/sj157/SOCKET_orig`)

1. `eval/ruler_utils/__init__.py` — empty package marker.
2. `eval/ruler_utils/calculate_metrics.py` — **COPY VERBATIM** from
   `/scratch/sj157/sparse-attention-hub/benchmark/ruler32k/calculate_metrics.py`
   (`string_match_part`, `string_match_all`, `calculate_metrics(df)`; control-char strip). Do not
   modify; this is what makes scores match the paper/hub. Note `answer` stays a list/ndarray of refs —
   never `str()` it or take `[0]`.
3. `eval/ruler_utils/load_ruler32k.py` — loader. For each requested subset:
   `load_dataset("xAlg-AI/att-hub-ruler-32k", subset, split=subset).to_pandas().head(N)` (N=100 per task,
   capped PER-SUBSET not on a concatenated df). Returns a df/Dataset keeping columns
   `context, question, answer_prefix, answer, task, max_new_tokens`. Compute node has internet (memory).
   Optional: `save_to_disk('dataset/ruler32k/<task>')` for offline reuse, but in-memory is simpler.
4. `eval/ruler_utils/scorer.py` (thin) — `ruler_scorer(task, pred, refs)` wrapping calculate_metrics
   semantics for the single-sample call site, OR (preferred) accumulate predictions and score the whole
   df once via `calculate_metrics` at end of the eval loop. Returns per-task `string_match` + overall mean.
5. `config/eval_config/ruler32k/{qa_1,qa_2,fwe,vt,niah_multikey_2,niah_multikey_3}.json` — 6 eval configs.
   Each: `eval_params.dataset = "<task>"`, a new `eval_params.benchmark = "ruler32k"` flag (so dispatch
   knows to take the RULER branch), and `eval_metrics = ["ruler_string_match"]`. `max_new_tokens` and
   `answer_prefix` come from the DATASET per row, NOT from config (do not hardcode). Mirror the
   `management.sub_dir` block from `config/eval_config/longbench/qasper.json`.

### 1.2 Files to EDIT (RULER branches at the 3 blockers + hyperparam threading)

A. `eval/longbench_utils/constants.py` — add `RULER_DATASET = ["qa_1","qa_2","fwe","vt","niah_multikey_2","niah_multikey_3"]` (or all 13). Keep separate from `LONGBENCH_DATASET`.

B. `pipeline/train_quest/dataset.py` — RULER branch in `get_dataset` (and skip `get_val_dataset`'s
   longbench-only middle-truncate concern by feeding a pre-built prompt):
   - When `dataset in RULER_DATASET`: load via `load_ruler32k`, build per-row
     `prompt = context + question + answer_prefix` (NO `instruction.format`; RULER prompts are
     self-contained), `answer = row["answer"]` (keep list), `idx = i`, and carry
     `max_new_tokens = row["max_new_tokens"]` and `task = row["task"]` into the sample dict so the runner
     can read per-row max_new_tokens.
   - `get_val_dataset` already chat-templates `sample["prompt"]` (system + user + add_generation_prompt);
     RULER reuses this unchanged (query is in the prompt → SOCKET soft-hashes the live decode query →
     query-aware retrieval works). Confirm the RULER `dataset` name is NOT in the raw-tokenize exception
     list `["trec","triviaqa","samsum","lsht","lcc","repobench-p"]` (it isn't) so it gets the chat path.

C. `pipeline/train_quest/run.py`:
   - GT load (after line 106): add `elif eval_params['dataset'] in RULER_DATASET:` → build
     `ground_truth = [row_answer_list for each row]` (already a list per row), `all_classes = None`,
     and stash the per-row `task` + `max_new_tokens` alongside.
   - eval loop (~line 380-413): for RULER, set `max_new_tokens = per_row max_new_tokens` (override the
     single eval_params value); after generating, append `predicted_answer` to a results df rather than
     calling `longbench_eval.scorer`. After the loop, score once via
     `eval.ruler_utils.calculate_metrics.calculate_metrics(results_df)` and report per-task
     `string_match` + overall mean. (Single-pass scoring matches the hub exactly and avoids a per-call
     metric shim.)
   - SOCKET hyperparam threading in the `use_smallworld`(→`use_socket`) branch: set on `llama_config`
     from `pipeline_params`: `bucket_K` (P), `bucket_L` (L), `sink_size`, `window_size`, `heavy_const`
     (ratio), and a NEW `tau` attr.

D. `pipeline/train_quest/modeling/modeling_llama.py:638` — de-hardcode tau:
   `return F.softmax(logits / float(getattr(self.config, "tau", 0.3)), dim=-1)`.

E. Config that drives the run (pipeline_config; see §2 for the renamed SOCKET dir): add
   `bucket_K:10, bucket_L:60, sink_size:128, window_size:128, heavy_const:0.0422, tau:0.4` to the SOCKET
   inference pipeline config used for RULER. (20× @ 32768: keep ~1/20 of tokens; heavy_size ≈
   `1/20 - 256/32768 ≈ 0.0422`, with sink128+window128 the 256-token correction; total kept ≈ 5%.)
   Verify achieved sparsity in logs, not just the knob.

### 1.3 The 6 task subsets and their scoring metric

| subset | category | metric | max_new_tokens (dataset) | paper 20× target |
|---|---|---|---|---|
| qa_1 | qa | string_match_part | 32 | 82 |
| qa_2 | qa | string_match_part | 32 | 52 (inherently low; not a bug) |
| fwe | fwe | string_match_all | 50 | 86.0 |
| vt | vt | string_match_all | 30 | 91.4 |
| niah_multikey_2 (nm2) | niah | string_match_all | 128 | 93 |
| niah_multikey_3 (nm3) | niah | string_match_all | 128 | 92 |

### 1.4 Run command

Same entry point as longbench, pointing at a ruler eval config + a SOCKET pipeline config with K/L/tau/sparsity set:
```
python pipeline/train_quest/main.py \
  --pipeline_config_dir config/pipeline_config/SOCKET/Llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct-inference-only.json \
  --eval_config_dir   config/eval_config/ruler32k/niah_multikey_2.json \
  --output_folder_dir longbench/SOCKET/Llama-3.1-8B-Instruct/ruler32k_nm2 \
  --exp_desc ruler32k_nm2_20x_K10_L60
```
(repeat per task; or a small launcher loop over the 6 configs). batch_size==1, greedy. Requires a valid
`hf_access_token` in `config/access_tokens.py` (currently empty → gated Llama 401s) and CUDA + triton +
nvcc (first decode JIT-compiles `soft_hash_collision.cu`).

---

## 2. SMALLWORLD → SOCKET RENAME/REMOVE

Verdict: coherence RENAME of live pieces + REMOVE of the dead training scaffold. "SmallWorld" is SOCKET's
former internal codename; the `method=="smallworld"` gate IS the SOCKET masker activation. The single
highest-risk coupling: rename the config `method` string WITHOUT updating `run.py:149` → masker silently
disables → dense attention, no error. Always move these two together.

### 2.1 Per-item plan (sequenced so the repo never breaks; see §4 for ordering)

1. `git mv scripts/small_world_inference scripts/socket_inference`
2. `git mv config/pipeline_config/SmallWorld config/pipeline_config/SOCKET`
3. `git mv longbench/SmallWorld longbench/SOCKET` (stale result artifacts incl. qasper 87.86/200 — rename is lossless; prefer over delete).
4. `run.py:149` `use_smallworld = ... == "smallworld"` → `use_socket = ... == "socket"`; `run.py:157` `if use_smallworld and is_llama_instruct:` → `if use_socket and is_llama_instruct:` (rename the var at both sites).
5. The 3 LIVE inference configs (`*-inference-only.json` ×2, `*-inference-32hadamard.json`): `"method":"smallworld"` → `"method":"socket"`.
6. The 2 hadamard configs (`*-inference-256hadamard.json`, `*-inference-512hadamard.json`): `"method":"fafo"` → `"method":"socket"`, keep `"train_mode":"inference_only"`. This FIXES them (currently fall through to dense).
7. All 4 inference `.sh` in `scripts/socket_inference/`: set `method="SOCKET"` (the `${method}` var feeds both `config/pipeline_config/${method}/...` AND `--output_folder_dir`). Verify interpolated path resolves after the dir rename.
8. `REPRODUCE.txt:3,9` — update path to `scripts/socket_inference/.../inference.sh` and `method="SOCKET"`.
9. REMOVE dead scaffold: `git rm -r scripts/train/` + the 9 training configs (`config/pipeline_config/SOCKET/*/{*-joint.json,*-selector.json,*-traink.json}`). They use the unwired `fafo` method, point at nonexistent `LearnHash`/`train` config dirs, and use the invalid `train_k` mode. (If retention is wanted for future learned-masker work, isolate under `config/pipeline_config/SOCKET/_train_scaffold/` and mark clearly — out of scope for a coherence pass.)

### 2.2 Coupling rules (must change together)
- config-internal `"method"` lowercase ↔ `run.py` gate string (load-bearing for masker activation).
- shell `method=` var ↔ config DIR name + output dir (path-only, cosmetic for correctness).
- Convention: dir = `SOCKET`, shell `method="SOCKET"`, config-internal `"method":"socket"` (lowercase, matches existing `baseline`/`fafo` lowercase). Preserve the existing no-task-segment layout `pipeline_config/${method}/${model}/...` for inference configs (baseline has an extra task segment; don't homogenize unless rewriting the .sh path templates).
- The `is_llama_instruct` allowlist (`run.py:150-155`) is orthogonal but note a non-Llama model silently runs dense even after rename.

### 2.3 Verification after rename
- `grep -rIn -iE 'small.?world|learnhash|fafo' /scratch/sj157/SOCKET_orig` → ZERO hits (except intentional historical notes).
- `ls` each inference `.sh`'s interpolated `--pipeline_config_dir` to confirm it resolves.
- Smoke eval qasper (the recorded 87.86 task) on `Llama-3.1-8B-Instruct-inference-only.json`: confirm `run.py:157` SOCKET branch is hit (temp print / assert `set_masker_mode` called) and a non-dense plausible score is produced — proves the rename did not silently disable the masker.

---

## 3. TEST PLAN

### 3.1 RULER runs (6 tasks × 100 samples, 20× sparsity, K=10/L=60, τ=0.4)
Model Llama-3.1-8B-Instruct, ctx 32768, Setup-B (dense context, sparse question+decode, all layers),
masker stack Sink(128)+Local(128)+SOCKET(K=10,L=60,τ=0.4,heavy≈0.0422), seed 42 (note repo `set_seed(41)`
at run.py — align to 42 to match paper). 100 rows/task via `.head(100)`. Optional τ sweep {0.3,0.4,0.5}.

### 3.2 LongBench runs (3-4 tasks, 20×, K=10/L=60) — LABEL as a config DELTA
Paper uses P=8 for LongBench (not 10) and reports only 10×/33× (no 20× point). So this is interpolated, not
a faithful repro; expect AVG just under 48.8. Tasks already on disk: pick QASPER, MultiFieldQA-en, HotpotQA,
GovReport (+ PassageRetrieval as a ~99-100 sanity check). Reuse the EXISTING longbench path
(`get_dataset` longbench branch → scorer) with the SOCKET masker config — no new code beyond §2 rename.

### 3.3 SOCKET correctness unit tests (separate from accuracy; write under `tests/`)
- `hard_hash`: sign-of-projection correctness; int16 bucket-code packing round-trips; deterministic per plane set.
- `soft_hash`: rows sum to 1 (valid softmax over hypercube corners); monotonic in q·plane alignment; τ effect (smaller τ → sharper → approaches hard-LSH one-hot as τ→0); reads `config.tau` after de-hardcode.
- `build_sparse_list_decode`: returns exactly `min(M,T) + sink + window` indices; no out-of-range indices; respects allowed mask; sink+window always included.
- **Faithfulness equivalence**: SOCKET decode with M=T (full budget, sink/window large enough to cover all) ≈ dense sdpa output within fp tolerance — proves the Triton sparse kernel + collision estimator are correct.
- JIT smoke: first decode compiles `soft_hash_collision.cu` (nvcc + writable TRITON_CACHE_DIR) without error.
- RULER loader test: 6 configs load, 200→head(100) rows, columns intact, `answer` is list-of-refs, `max_new_tokens` matches table.
- calculate_metrics test: `string_match_part` (qa) vs `string_match_all` (others) on synthetic preds/refs (case-insensitive substring; partial credit for `_all`); control-char strip.

### 3.4 Accuracy sanity vs paper (close, not exact)
Paper Table 1 RULER-HARD-32K, Llama-3.1-8B-Instruct, **20× (P=10/L=60)**:

| task | nm2 | nm3 | vt | fwe | qa1 | qa2 | avg(6) |
|---|---|---|---|---|---|---|---|
| SOCKET 20× | 93 | 92 | 91.4 | 86.0 | 82 | 52 | **82.7** |

τ-ablation @20× (Table 6, 5-task avg, no fwe): τ=0.4 → nm2=96 qa1=78 vt=96.4 nm3=94 qa2=52, avg 83.28
(best); τ=0.3 avg 82.80; τ=0.5 avg 82.16. In-table baselines @20×: Quest avg 83.8, PQcache 83.1.
Dense upper bound (RULER-16K Table 10): vt 97.4 / qa1 80.5 / qa2 51.5 / fwe 93.17 / nm2 99.5 / nm3 100.

"Close not exact" bar is correct because: greedy decode, single seed, 100-sample subset, τ=0.3 default vs
0.4 best, merged-kernel numeric drift. qa_2 ≈50-54 is intrinsic, NOT a failure.

LongBench bracket (paper, AVG): SOCKET 10× = 48.8, 33× = 47.83; per-task 10× e.g. QAS=46.7, MFQA=54.54,
HPQA=54.72, GOV=35.41, Retrieval=100. A 20× run should land between, slightly under 48.8.

---

## 4. ORDERING (sequential vs parallelizable)

**Phase A — RENAME (do first, atomically; touches run.py + configs that the RULER work also edits):**
Steps 2.1.1→2.1.9 in order. The 3 dir `git mv`s (1-3) can be done together. Then run.py edit (4) +
config `method` edits (5,6) MUST land together (coupling rule). Then .sh (7), REPRODUCE.txt (8), removals
(9). Verify §2.3 before proceeding. **Rationale: §1 edits run.py:100-106/380-413 and adds the SOCKET
hyperparam threading inside the same `use_smallworld→use_socket` branch — doing the rename first avoids a
second pass over run.py and prevents the silent-dense regression while RULER code is in flight.**

**Phase B — RULER PORT (after rename):**
- B1 (parallel, independent files): add `eval/ruler_utils/` (calculate_metrics copy, loader, scorer),
  `config/eval_config/ruler32k/*.json`, `tests/` unit tests. No mutation conflict.
- B2 (sequential, same files): edit `constants.py` (add RULER_DATASET) → `dataset.py` (RULER branch) →
  `run.py` (GT branch + eval-loop branch + hyperparam threading) → `modeling_llama.py:638` (de-hardcode
  tau). These touch shared files and depend on RULER_DATASET existing; do in this order.

**Phase C — TESTS / RUNS (after A+B):**
- C1 unit tests (3.3) — run first; gate the accuracy runs on them passing (esp. M=T equivalence).
- C2 the 6 RULER runs (3.1) and the 3-4 LongBench runs (3.2) are mutually independent → parallelizable
  across GPUs once code is frozen. Cap each task to 100 rows independently.
- C3 compare to §3.4 table.

**Cannot parallelize:** any two edits to the same file (run.py, dataset.py, constants.py,
modeling_llama.py) — serialize within Phase B2. The rename (Phase A) must fully precede Phase B (both edit
run.py).

---

## 5. BIGGEST RISKS

1. **Silent masker disable on rename** (highest): renaming config `"method"` without updating `run.py:149/157` → dense attention, no error, bogus "SOCKET" numbers. Mitigate: change together; verify `set_masker_mode` is called in the qasper smoke test.
2. **SOCKET knobs not configurable today**: bucket_K default 8 (paper 10), τ hardcoded 0.3, heavy/sink/window default 0.1/20/20 (~10×, not 20× / not sink128/window128). Out-of-the-box numbers will NOT match the paper. Must thread K/L/τ/sink/window/heavy from config + de-hardcode τ (§1.2 C/D/E). Verify achieved sparsity in logs.
3. **RULER not recognized → KeyError**: not in LONGBENCH_DATASET, no GT/all_classes, scorer crashes. Mitigate: RULER_DATASET set + GT branch + single-pass calculate_metrics (§1.2 A/C).
4. **Wrong scorer**: using dataset2metric (qa_f1/rouge) on RULER gives garbage. Must use copied string_match (part for qa_*, all otherwise).
5. **Prompt corruption**: re-wrapping the self-contained RULER prompt in a longbench `instruction.format` template breaks it; conversely answer_prefix MUST be fed (load-bearing for niah/vt/fwe) but NOT scored. Build `context+question+answer_prefix`, chat-template once.
6. **Per-task max_new_tokens**: a single global value truncates niah (128) or over-generates vt (30)/qa (32), shifting substring scores. Honor per-row dataset value.
7. **Output-stripping mismatch**: special/chat tokens left in decoded output inflate/deflate substring match. Strip special tokens (CE.generate already decodes; ensure special-token strip akin to `_strip_special_tokens`).
8. **`answer` is numpy ndarray**: keep list-of-refs semantics; never `str()`/`[0]`.
9. **Per-subset cap, not combined**: load `.head(100)` per task; capping a concatenated df biases coverage.
10. **Env**: empty `hf_access_token` (gated Llama 401), missing nvcc/empty TRITON_CACHE_DIR (first-decode JIT fails at runtime), batch_size==1 greedy-only (do_sample ignored on SOCKET path).
11. **LongBench config delta**: paper P=8 + only 10×/33×; the 20×/K=10 LongBench run is interpolated, not a repro — label it.
12. **Seed**: repo `set_seed(41)`, paper uses 42 — align to 42 for comparison; single seed → several-to-tens of points RULER variance per task (expected).
