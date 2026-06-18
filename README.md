# SOCKET: SOft Collision Kernel EsTimator for Sparse Attention

SOCKET is a **data-agnostic soft-LSH sparse-attention** method for long-context LLM
inference. Traditional Locality-Sensitive Hashing (LSH) produces *binary* collision
signals — a key either lands in the query's bucket or it does not — which is too coarse
to rank keys reliably. SOCKET replaces the hard bucket match with a **softmax over
hypercube-bucket logits**: each query distributes probability mass across all buckets of
each hash table, turning LSH into a graded, query-aware **scoring kernel**. Aggregating
this soft collision evidence across tables yields a stable ranking that selects the
top-*k* keys per query using only compact bucket codes (≈600 bits/token) plus a per-key
value norm — no full key/value reads. SOCKET matches or surpasses prior sparse-attention
methods on long-context benchmarks and, with a custom CUDA scoring kernel plus a
Flash-Decode Triton backend, reaches up to ~1.5× the decode throughput of FlashAttention.

See the paper: *SOCKET: SOft Collision Kernel EsTimator for Sparse Attention*
(arXiv:2602.06283).

This repository has **two independent parts**:

- **(a) Accuracy / evaluation path** — an HF (`transformers`) Llama model with a soft-LSH
  masker (`pipeline/train_quest/`, `modeling/modeling_llama.py`), evaluated on RULER-32K
  and LongBench. This is the path that produces accuracy numbers.
- **(b) Throughput path** — a `GPT-FAST` fork (`GPT-FAST/`) with compiled, CUDA-graph
  decode kernels (a custom CUDA soft-hash scorer + Triton sparse-decode kernels) used only
  to measure decode throughput vs. FlashAttention-2/3. It is **not** used for accuracy.

---

## 1. What's in the repo

| Path | What it is |
|---|---|
| `pipeline/train_quest/main.py` | Accuracy entry point (parses configs, runs eval). |
| `pipeline/train_quest/run.py` | Eval driver: model load, dataset dispatch, decode, scoring. |
| `pipeline/train_quest/modeling/modeling_llama.py` | SOCKET soft-LSH masker (hard/soft hash, top-k selection). |
| `pipeline/train_quest/modeling/soft_hash_collision.cu` | CUDA soft-hash scorer (accuracy path). |
| `config/pipeline_config/SOCKET/<model>/` | Pipeline configs (model + SOCKET knobs). |
| `config/eval_config/{ruler32k,longbench}/<task>.json` | Per-task eval configs. |
| `eval/ruler_utils/` | RULER-32K loader + scoring (`load_ruler32k.py`, `calculate_metrics.py`, `scorer.py`). |
| `eval/longbench_utils/` | LongBench scoring (`eval_long_bench.py`, `dataset2metric`). |
| `GPT-FAST/generate.py`, `GPT-FAST/model.py` | Throughput benchmark + sparse/dense decode model. |
| `GPT-FAST/kernels/` | CUDA soft-hash scorer (`soft_hash_collision.cu` + loader) and Triton sparse kernels (`sparse.py`). |
| `tests/test_socket_ruler.py` | Masker / loader / scoring / dense-equivalence tests. |
| `GPT-FAST/test_socket_compile_equiv.py` | Kernel compile/optimization equivalence script. |
| `RULER_RESULTS.md` | Recorded RULER-32K + LongBench accuracy campaign. |

---

## 2. Environment setup

The two paths use **separate Python environments** (different torch/CUDA wheels). Both are
Python 3.13 venvs; CUDA toolkit `12.9.1` is loaded as a module.

### Accuracy path (e.g. `swa_env`)

- Python 3.13, `torch 2.8.0+cu128`, `transformers==4.57.6`, `triton 3.4.0`, `datasets`.
- Install from `requirements.txt` (it pins `transformers==4.57.6`; `torch`/CUDA wheels are
  left to you, hence the commented `# torch` / `# nvidia-*` lines).

```bash
module load GCCcore/13.3.0 CUDA/12.9.1
python -m venv swa_env && source swa_env/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or your CUDA wheel
pip install -r requirements.txt
```

### Throughput path (`GPT-FAST/`, e.g. `prism_env`)

- Python 3.13, `torch 2.11.0+cu129` (newer torch, needed for `torch.compile` +
  CUDA-graph decode), plus a FlashAttention build (`flash_attn` for FA2 and/or
  `flash_attn_interface` for FA3) for the dense baselines.

```bash
module load GCCcore/13.3.0 CUDA/12.9.1
source prism_env/bin/activate
export TORCH_CUDA_ARCH_LIST="9.0"     # Hopper / H200
```

### Shared prerequisites

- **CUDA `12.9.1`** with a working `nvcc` (the soft-hash kernel is JIT-built at runtime).
- **`HF_HOME`** pointed at a cache that already holds `meta-llama/Llama-3.1-8B-Instruct`
  (the campaign used `HF_HOME=/scratch/sj157/hf_home`). The model is gated; either rely on
  cached weights or put a token in `config/access_tokens.py`:

  ```python
  hf_access_token = "hf_..."   # main.py only calls login() when this is non-empty
  ```

  An empty token (the default) is fine — `main.py` skips `login()` and falls back to
  cached HF credentials / local weights.
- **First decode JIT-compiles the CUDA soft-hash kernel** (`soft_hash_collision.cu`) via
  `torch.utils.cpp_extension.load_inline`. This needs `nvcc` on `PATH` and a **writable
  `TORCH_EXTENSIONS_DIR`** (for the compiled `.so`); the Triton sparse kernels likewise
  need a writable **`TRITON_CACHE_DIR`**. Set both before running:

  ```bash
  export TRITON_CACHE_DIR=/scratch/$USER/.cache/triton
  export TORCH_EXTENSIONS_DIR=/scratch/$USER/.cache/torch_ext
  mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
  ```

---

## 3. Testing accuracy (model + a specific dataset)

The accuracy entry point is `pipeline/train_quest/main.py`. It takes:

- `--pipeline_config_dir` — pipeline config (model name + SOCKET knobs).
- `--eval_config_dir` — eval config (which dataset/task + scoring metric).
- `--output_folder_dir` — where results are written.
- `--exp_desc` — a cosmetic run label.

`run.py` reads `LOCAL_RANK`/`RANK` from the environment, so set them when launching with
plain `python` (single GPU):

```bash
export RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=27501
```

> The repository's original launcher, `scripts/socket_inference/Llama-3.1-8B-Instruct/inference.sh`,
> uses `deepspeed --master_port 27501 pipeline/train_quest/main.py ...` (which sets those
> vars for you). The recorded accuracy campaign in `RULER_RESULTS.md` ran the plain
> `python pipeline/train_quest/main.py` form below.

### RULER-32K

Six tasks are wired: `qa_1`, `qa_2`, `fwe`, `vt`, `niah_multikey_2`, `niah_multikey_3`
(configs in `config/eval_config/ruler32k/`).

```bash
cd /scratch/sj157/SOCKET_orig
HF_HOME=/scratch/sj157/hf_home python pipeline/train_quest/main.py \
  --pipeline_config_dir config/pipeline_config/SOCKET/Llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct-inference-ruler.json \
  --eval_config_dir     config/eval_config/ruler32k/niah_multikey_2.json \
  --output_folder_dir   runs/ruler32k/Llama-3.1-8B-Instruct/niah_multikey_2 \
  --exp_desc            ruler32k_nm2_20x_K10_L60
```

Repeat for each of the six tasks (swap the `--eval_config_dir`). The RULER dataset
(`xAlg-AI/att-hub-ruler-32k`) is loaded by `eval/ruler_utils/load_ruler32k.py` (200 rows
per task, capped to the first 100 via `.head()`; override with the `RULER_NUM_SAMPLES`
env var). Scoring is `eval/ruler_utils/calculate_metrics.py`:
`string_match_part` for `qa_*` tasks (1.0 if any reference is a substring of the
prediction), `string_match_all` otherwise (mean fraction of references found).

### LongBench

Same entry point, pointing at a `config/eval_config/longbench/<task>.json`:

```bash
HF_HOME=/scratch/sj157/hf_home python pipeline/train_quest/main.py \
  --pipeline_config_dir config/pipeline_config/SOCKET/Llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct-inference-ruler.json \
  --eval_config_dir     config/eval_config/longbench/qasper.json \
  --output_folder_dir   runs/longbench/Llama-3.1-8B-Instruct/qasper \
  --exp_desc            longbench_qasper_20x
```

LongBench tasks read from on-disk splits under `dataset/longbench/` and are scored by
`eval/longbench_utils/eval_long_bench.py` via `dataset2metric` (e.g. `qasper → qa_f1`,
`gov_report → rouge`, `passage_retrieval_en → retrieval`). Available task configs are in
`config/eval_config/longbench/`.

### SOCKET knobs (in the pipeline config)

The masker reads these from the pipeline config (`modeling_llama.py`); the shipped
`Llama-3.1-8B-Instruct-inference-ruler.json` sets them for the paper's 20× RULER point:

| Key | Paper symbol | Meaning | Default if unset | Ruler config |
|---|---|---|---|---|
| `bucket_K` | **P** | bits / SimHash planes per table (R = 2^P buckets) | 8 | **10** |
| `bucket_L` | **L** | number of hash tables | 60 | 60 |
| `sink_size` | — | always-kept sink (prefix) tokens | 20 | 128 |
| `window_size` | — | always-kept local-window (recent) tokens | 20 | 128 |
| `heavy_const` | sparsity | **fraction of context kept** as soft-LSH heavy tokens (float ⇒ ratio, int ⇒ absolute count) | 0.1 | 0.0422 |
| `tau` | τ | softmax temperature over bucket logits (smaller ⇒ sharper ⇒ → hard LSH) | 0.3 | 0.4 |

Total tokens kept per query = `sink + window + round(heavy_const × T)`. With
`sink=window=128` and `heavy_const=0.0422` at 32K context this is ~5% kept (20×
compression, confirmed in `RULER_RESULTS.md`).

To change:

- **Model** — set `pipeline_params.model_name` (the SOCKET masker activates only for the
  `method == "socket"` + instruct-Llama allow-list; otherwise it runs dense). Configs for
  `Llama-3.2-1B-Instruct` are also provided.
- **Dataset / task** — point `--eval_config_dir` at a different task config; the
  `eval_params.dataset` + `benchmark` fields select the loader/scorer branch.
- **Sparsity** — change `heavy_const` (and optionally `sink_size`/`window_size`). Verify
  the *achieved* sparsity from the runtime log (`frac_kept`), not just the knob.

Outputs land under `--output_folder_dir`: `raw_results.json` (scores) and
`output_config.json` (full fused config incl. per-task results), plus `input_config/` and
`exp.log`.

---

## 4. Testing throughput (GPT-FAST decode benchmark)

`GPT-FAST/generate.py` measures **decode-only** tokens/sec. The SOCKET sparse path is
selected with `--decode_type sparse`; the dense FlashAttention baselines with
`--decode_type dense`. Use `--compile` for the real (CUDA-graph) numbers.

### Knobs are environment variables

SOCKET config is passed via env vars read by `GPT-FAST/model.py`:

| Env var | Controls | Default |
|---|---|---|
| `SOCKET_K` | P (bits per table); R is derived as `2**K` | 8 |
| `SOCKET_R` | R (= number of hypercube corners); pinned to `2**SOCKET_K` | 256 |
| `SOCKET_L` | L (number of hash tables) | 60 |
| `SOCKET_HEAVY_CONST` | the per-query heavy **token budget** (the "HEAVY" count, an absolute int) | 860 |
| `SOCKET_DECODE_WARMUP` | untimed decode steps run **before** the timed window (excludes JIT/autotune + CUDA-graph recording so only steady-state is timed) | 0 |
| `SOCKET_FORCE_FA2` | `1` forces the FA2 dense backend even when FA3 is importable | 0 |
| `USE_FLASHATTN3` | `1` uses the flash path (FA2 or FA3, whichever imported); `0` is the SDPA control | 1 |
| `SOCKET_REQUIRE_BACKEND` | `{fa2,fa3,flash,sdpa}` — asserts the kernel that actually ran matches, and turns the SDPA fallback into a hard error (refuses to report a mislabeled backend) | (none) |

> Note: FA3 is preferred at import (`flash_attn_interface`); `SOCKET_FORCE_FA2=1`
> short-circuits to FA2 (`flash_attn`). `USE_FLASHATTN3` only gates flash-vs-SDPA; it does
> **not** by itself select FA3.

### Sparsity / HEAVY convention

Context length **N** is set entirely by the prompt token count (via `--prompt_file`); there
is no context-length flag. The benchmark harness picks the heavy budget per (N, ratio) as

```
HEAVY = round(N / ratio) − 240        (240 = sink_size + window_size = 120 + 120)
```

so the total kept set is `HEAVY + 240` and the realized sparsity is
`prompt_len / (HEAVY + 240)`. This convention lives in the harness
(`socket_bench/launch_matrix.sh`, `socket_bench/agg_matrix_K8K10.py`); the runtime itself
just reads the absolute `SOCKET_HEAVY_CONST`.

### Concrete example

```bash
cd /scratch/sj157/SOCKET_orig/GPT-FAST
export SOCKET_L=60 SOCKET_DECODE_WARMUP=8 TORCH_CUDA_ARCH_LIST="9.0"

# SOCKET sparse (e.g. N≈72K, 40× sparsity → HEAVY = round(72000/40) − 240 = 1560)
SOCKET_K=8 SOCKET_R=256 SOCKET_HEAVY_CONST=1560 \
python generate.py \
  --checkpoint_path /path/to/checkpoints/meta-llama/llama-3.1-8b/model.pth \
  --prompt_file /scratch/sj157/socket_bench/prompts/sw_72000.txt \
  --batch_size 1 --top_k 1 --temperature 1.0 \
  --decode_type sparse --compile --num_samples 5 --max_new_tokens 50

# Dense FA2 baseline (same prompt)
SOCKET_FORCE_FA2=1 USE_FLASHATTN3=1 SOCKET_REQUIRE_BACKEND=fa2 \
python generate.py ...same flags... --decode_type dense --compile --num_samples 5 --max_new_tokens 50

# Dense FA3 baseline
USE_FLASHATTN3=1 SOCKET_REQUIRE_BACKEND=fa3 \
python generate.py ...same flags... --decode_type dense --compile --num_samples 5 --max_new_tokens 50
```

The checkpoint is the standard gpt-fast `model.pth` (the config is inferred from the parent
directory name, e.g. `llama-3.1-8b`); the tokenizer is `tokenizer.model` next to it.

**Reading the output:** look for the line

```
Decode-only: <sec> sec, <tok/s> tokens/sec
```

(it excludes prefill **and** the `SOCKET_DECODE_WARMUP` steps). The
`[ATTN-BACKEND] required=... effective=... ran=...` line (stderr) confirms which kernel ran,
and `[PROMPT-LEN] prompt_len=...` reports the realized context length.

### The `socket_bench` harness

`/scratch/sj157/socket_bench/` runs the full SOCKET-vs-FA2-vs-FA3 matrix on **H200**:

- `matrix_K8K10.sbatch` — one SLURM job per context (`N`, `H333`, `H35`, `H40`), running
  the FA2 and FA3 dense baselines (measured once per context, since dense is
  config-independent) plus **six** SOCKET cells: `{K=8/R=256, K=10/R=1024} × {33.3×, 35×,
  40×}`. It loads `CUDA/12.9.1`, activates `prism_env`, sets fresh per-job
  `TRITON_CACHE_DIR`/`TORCH_EXTENSIONS_DIR`, `SOCKET_L=60`, `SOCKET_DECODE_WARMUP=8`.
- `launch_matrix.sh` — submits the 4 context jobs (N = 18K/36K/72K/140K) with the
  precomputed HEAVY triples.
- `orig_correct_40x.sbatch` — a single (N, HEAVY) SOCKET cell + FA2/FA3 baselines.
- `agg_matrix_K8K10.py` — parses the SLURM logs (`logs/*.out` for throughput / `[PROMPT-LEN]`,
  `logs/*.err` for `[ATTN-BACKEND]`), takes the **median** of the warm decode-only samples,
  and emits `RESULTS_matrix_K8K10_L60.md` (six throughput tables, columns: `ctx | SOCKET
  tok/s | FA2 tok/s | FA3 tok/s | SOCKET/FA2 | SOCKET/FA3 | prompt_len | backend | NaN?`)
  plus a ratio plot. SOCKET reaches ~1.3–1.4× FA2/FA3 at 140K context in those tables.

---

## 5. Tests

- `tests/test_socket_ruler.py` — pytest suite for the **accuracy path**. CPU tests cover the
  hard hash (sign-of-projection, int16 bucket-code round-trip, determinism), the soft hash
  (rows sum to 1, smaller τ is sharper, **τ is read from config** with a 0.3 fallback), the
  sparse-list builder (counts = `sink + window + min(M, T)`, valid indices), the RULER
  loader (6 tasks, 100 rows, per-task `max_new_tokens`), and `calculate_metrics`
  (`string_match_part` vs `string_match_all`, control-char strip). Three GPU-gated tests
  (auto-skip without CUDA) cover the GPU sparse-list builder, a **M=T dense-equivalence**
  check (SOCKET sparse decode ≈ dense SDPA within fp tolerance, proving the kernels are
  numerically correct), and the JIT-compile smoke test for `soft_hash_collision.cu`.

  ```bash
  cd /scratch/sj157/SOCKET_orig
  HF_HOME=/scratch/sj157/hf_home python -m pytest tests/test_socket_ruler.py -v
  ```

- `GPT-FAST/test_socket_compile_equiv.py` — a standalone CUDA script (not pytest) verifying
  the **throughput kernels'** internal equivalences: compiled vs. eager sparse attention
  (bit-exact under CUDA graphs), the per-kv-head scorer vs. the `repeat_interleave` path,
  static-cache top-k selection, and heavy/base dedup. It prints `[T1]…[T9]` and `ALL_DONE`.

  ```bash
  cd /scratch/sj157/SOCKET_orig/GPT-FAST
  python test_socket_compile_equiv.py                            # K=8 / R=256
  SOCKET_K=10 SOCKET_R=1024 python test_socket_compile_equiv.py  # K=10 / R=1024
  ```
