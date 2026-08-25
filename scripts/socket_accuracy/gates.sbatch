#!/bin/bash
# Kernel-equivalence gates for the GPT-FAST port, everything that needs a GPU but not a model.
#
# Runs, in order:
#   tests/test_socket_capacity.py        capacity + budget arithmetic (CPU, cheap, runs first
#                                        so an arithmetic slip fails before any GPU work)
#   tests/test_socket_port.py            each ported kernel vs the implementation it replaces,
#                                        plus the padded-capacity equivalence
#   tests/test_modeling_llama_arms.py    the whole selection pipeline, legacy arm vs default
#   tests/test_socket_ruler.py           the repo's own masker suite
#   GPT-FAST/tests/test_empty_chunk_nan.py   the empty-partition NaN regression
#   GPT-FAST/test_socket_compile_equiv.py    the throughput path's own T1..T9 gates, at both
#                                        hash geometries the accuracy sweep uses
#
# Exit code is non-zero if any of them fails.
#
#SBATCH --account=as143
#SBATCH --partition=commons
#SBATCH --ntasks=1 --cpus-per-task=8 --threads-per-core=1 --mem-per-cpu=16GB
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00
#SBATCH --export=ALL
#SBATCH --job-name=kfix_gates
#SBATCH --output=/scratch/sj157/socket_speed/kfix/logs/%x_%j.out
#SBATCH --error=/scratch/sj157/socket_speed/kfix/logs/%x_%j.err
REPO="${SOCKET_REPO:-/scratch/sj157/SOCKET_orig}"
source "$REPO/scripts/socket_accuracy/env.sh"
cd "$REPO" || exit 1

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
# git is not on the compute nodes' default PATH; the commit is worth recording when it is.
command -v git >/dev/null && git -C "$REPO" rev-parse HEAD || echo "(git unavailable on this node)"
echo

RC=0
run () {  # <label> <command...>
  echo "########## $1  $(date -Is)"
  shift
  "$@"
  local rc=$?
  echo "---- exit=$rc  $(date -Is)"
  [ $rc -ne 0 ] && RC=1
  echo
  return 0
}

# Gates that need only torch + triton.
run "capacity arithmetic (CPU)"   python tests/test_socket_capacity.py
run "socket_port kernels"         python tests/test_socket_port.py
run "empty-chunk NaN guard"       python GPT-FAST/tests/test_empty_chunk_nan.py
run "empty-chunk NaN reproduces without the guard" \
    env SOCKET_NAN_GUARD=0 python GPT-FAST/tests/test_empty_chunk_nan.py
run "fused soft-hash q_probs kernel" \
    env SOCKET_FUSED_SOFTHASH=1 python GPT-FAST/tests/test_fused_soft_hash.py

# Gates that additionally need transformers, because they import modeling_llama.
if [ "${SOCKET_HAS_TRANSFORMERS:-0}" = "1" ]; then
  run "modeling_llama arms"       env SOCKET_ARM_SCRATCH="$TRITON_CACHE_DIR" python tests/test_modeling_llama_arms.py
  run "socket ruler suite"        python -m pytest tests/test_socket_ruler.py -v --no-header -p no:cacheprovider
else
  echo "########## SKIPPED (no transformers in this environment): modeling_llama arms, socket ruler suite"
  echo
fi

# The throughput path's own gates, at the two hash geometries the accuracy sweep uses. They
# import from GPT-FAST/kernels, so they must run with GPT-FAST as the working directory.
( cd GPT-FAST && SOCKET_K=8  SOCKET_R=256  python test_socket_compile_equiv.py ) \
  && echo "---- compile_equiv K=8 ok" || { echo "---- compile_equiv K=8 FAILED"; RC=1; }
( cd GPT-FAST && SOCKET_K=10 SOCKET_R=1024 python test_socket_compile_equiv.py ) \
  && echo "---- compile_equiv K=10 ok" || { echo "---- compile_equiv K=10 FAILED"; RC=1; }

echo "GATES_DONE rc=$RC $(date -Is)"
exit $RC
