# Shared environment for the SOCKET accuracy jobs. Sourced by gates.sbatch and
# run_lb_kernels.sbatch; every machine-specific path is an override with a default, so the
# scripts stay readable as a record of how a run was configured.
#
# Set SOCKET_ENV_DIR / SOCKET_HF_HOME / SOCKET_MODEL_PATH before submitting to point at a
# different venv, HF cache or weight tree. The default venv holds torch 2.8 + transformers
# 4.57: modeling_llama.py is a fork of that transformers release, so the major version is a
# hard requirement rather than a preference.
set -uo pipefail

module purge
module load GCCcore/14.3.0
module load Python/3.13.5
module load CUDA/12.9.1

# XALT preloads a libcrypto older than the OpenSSL module's libssl, so `datasets` (which needs
# ssl) fails to import unless it is dropped.
unset LD_PRELOAD
export XALT_EXECUTABLE_TRACKING=no

export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/$USER/.cache}"
export TORCH_HOME="${TORCH_HOME:-/scratch/$USER/torch_home}"

# Without this nvcc builds the soft-hash and radix extensions for every architecture the
# visible cards report, which is minutes of each job spent compiling code it will not run.
# 9.0 is Hopper (H200), the only GPU these jobs are submitted to.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

# The soft-hash CUDA extension is JIT-built and the Triton kernels are JIT-compiled, so both
# caches must be writable. They are per-job: a shared Inductor/Triton cache can serve a kernel
# compiled under different constexpr values, which would make an A/B compare a kernel with
# itself.
export TORCH_EXTENSIONS_DIR="/scratch/$USER/torch_ext/socket_${SLURM_JOB_ID:-nojob}"
export TRITON_CACHE_DIR="/scratch/$USER/triton_cache/socket_${SLURM_JOB_ID:-nojob}"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"

# One cache serves everything: the model weights, the RULER-32K shards, and the tokenizer.
# Offline, so a job can never silently reach the network mid-eval and stall behind a rate limit
# or a revision that moved -- the weights must already be local.
export HF_HOME="${SOCKET_HF_HOME:-/scratch/sj157/hf_home}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# The configs name the canonical hub repo, which is what a normal cache resolves. Set
# SOCKET_MODEL_PATH to a directory to load from a local weight tree instead; run.py reads it
# and overrides the config's model_name, so no tracked file has to carry a machine path.
if [ -n "${SOCKET_MODEL_PATH:-}" ]; then export SOCKET_MODEL_PATH; fi

# run.py reads the distributed rank unconditionally, even for a single-GPU eval.
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(( 20000 + ${SLURM_JOB_ID:-0} % 20000 ))

SOCKET_ENV_DIR="${SOCKET_ENV_DIR:-/scratch/sj157/socket_env}"
if [ ! -f "$SOCKET_ENV_DIR/bin/activate" ]; then
  # Fail here rather than letting every step below die with "No module named torch", which
  # reads like a code problem instead of a missing interpreter.
  echo "FATAL: no venv at $SOCKET_ENV_DIR (bin/activate missing). Set SOCKET_ENV_DIR." >&2
  exit 78
fi
source "$SOCKET_ENV_DIR/bin/activate"
# Report what is actually importable. The kernel gates need only torch + triton; the model
# gates additionally need transformers, and SOCKET_HAS_TRANSFORMERS lets a job skip those
# rather than fail, so a torch-only environment still validates everything it can.
python - <<'PYCHK' || exit 78
import sys
import torch, triton
try:
    import transformers
    tf = transformers.__version__
except Exception as exc:
    tf = f"UNAVAILABLE ({type(exc).__name__})"
print(f"[ENV] python={sys.version.split()[0]} torch={torch.__version__} "
      f"triton={triton.__version__} transformers={tf} cuda={torch.cuda.is_available()}")
PYCHK
# modeling_llama.py is a fork of transformers 4.57's Llama implementation, so the model-level
# gates need a 4.x transformers specifically -- a 5.x import succeeds and then fails somewhere
# less legible. The kernel gates do not import it at all.
if python -c "import sys, transformers; sys.exit(0 if transformers.__version__.startswith('4.') else 1)" 2>/dev/null; then
  export SOCKET_HAS_TRANSFORMERS=1
else
  export SOCKET_HAS_TRANSFORMERS=0
fi
