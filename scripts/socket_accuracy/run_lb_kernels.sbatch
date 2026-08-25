#!/bin/bash
# LongBench accuracy for the GPT-FAST kernel port: one job per dataset, every arm inside that
# job so all of them run on the same node, on the same samples, back to back.
#
# THE QUESTION THIS ANSWERS. The SOCKET throughput numbers were measured in GPT-FAST with the
# Triton scorer + radix select + Triton list assembly + the stage1/stage2 flash decode, at
# (P=10, L=10) and (P=8, L=50). The accuracy numbers were measured here with a different
# implementation of the same four stages, at a different (P, L). So no published speed number
# and accuracy number described the same system. These arms close that.
#
#   dense           F.scaled_dot_product_attention, no SOCKET            -- the ceiling
#   legacy_P8L50    cuda scorer + torch.topk + torch list + local attn   -- the old kernels
#   legacy_P10L10   same, at the other hash geometry
#   gptfast_P8L50   triton + radix + triton list + gptfast attn          -- THE DELIVERABLE
#   gptfast_P10L10  same, at the faster hash geometry
#   gptfast_P10L60  the new kernels at the P/L the shipped RULER config uses
#
# legacy vs gptfast at a fixed (P, L) is the EQUIVALENCE check: the two implementations are
# meant to compute the same thing, so a large gap there is a bug and not a result. gptfast at
# different (P, L) is the actual finding: whether the geometry that decodes fastest also
# scores.
#
# SOCKET_DEDUP stays 0 in every arm. It changes WHICH tokens are attended rather than which
# kernel computes them, so letting it vary would confound a kernel comparison with a policy one.
#
# usage:  DSLIST="hotpotqa samsum qasper multifieldqa_en" sbatch run_lb_kernels.sbatch
#         DSLIST=qasper ARMS="dense gptfast_P10L10" LIMIT=10 sbatch run_lb_kernels.sbatch
#
#SBATCH --account=as143
#SBATCH --partition=commons
#SBATCH --ntasks=1 --cpus-per-task=8 --threads-per-core=1 --mem-per-cpu=32GB
#SBATCH --gres=gpu:h200:1
#SBATCH --time=11:00:00
#SBATCH --export=ALL
#SBATCH --job-name=kfix_lb
#SBATCH --output=/scratch/sj157/socket_speed/kfix/logs/%x_%j.out
#SBATCH --error=/scratch/sj157/socket_speed/kfix/logs/%x_%j.err
REPO="${SOCKET_REPO:-/scratch/sj157/SOCKET_orig}"
source "$REPO/scripts/socket_accuracy/env.sh"
cd "$REPO" || exit 1

if [ "${SOCKET_HAS_TRANSFORMERS:-0}" != "1" ]; then
  echo "FATAL: the accuracy path needs transformers; none is importable in $SOCKET_ENV_DIR" >&2
  exit 78
fi

nvidia-smi --query-gpu=name --format=csv,noheader
command -v git >/dev/null && git -C "$REPO" rev-parse HEAD || echo "(git unavailable on this node)"

OUT="${SOCKET_OUT:-/scratch/sj157/socket_speed/kfix/out}"
mkdir -p "$OUT"

# Bound the eval so the whole matrix fits in one job. shuffle=False, so every arm sees the
# same samples -- that identity is what makes the arms comparable at this sample count.
export SOCKET_LB_LIMIT="${LIMIT:-30}"
export SOCKET_SMOKE_LOG=1
export SOCKET_TARGET_SPARSITY="${SPARSITY:-10}"

SOCKET_CFG=config/pipeline_config/SOCKET/Llama-3.1-8B-Instruct
DENSE_CFG=config/pipeline_config/baseline/longbench/Llama-3.1-8B-Instruct/lb-dense.json

# Sets the four arm switches and the global CFG. It must NOT be called through command
# substitution: $(arm_setup ...) would run it in a subshell and throw every export away, so
# each SOCKET arm would silently run with the DEFAULT kernels and the legacy-vs-gptfast
# comparison would be a kernel compared against itself.
arm_setup () {
  unset SOCKET_SCORER SOCKET_SELECT SOCKET_LIST SOCKET_ATTN SOCKET_DEDUP
  local legacy="SOCKET_SCORER=cuda SOCKET_SELECT=topk SOCKET_LIST=torch SOCKET_ATTN=local"
  local gptfast="SOCKET_SCORER=triton SOCKET_SELECT=radix SOCKET_LIST=triton SOCKET_ATTN=gptfast"
  case "$1" in
    dense)          CFG=$DENSE_CFG; return 0 ;;
    legacy_P8L50)   export $legacy;  CFG=$SOCKET_CFG/lb-P8L50.json  ;;
    legacy_P10L10)  export $legacy;  CFG=$SOCKET_CFG/lb-P10L10.json ;;
    gptfast_P8L50)  export $gptfast; CFG=$SOCKET_CFG/lb-P8L50.json  ;;
    gptfast_P10L10) export $gptfast; CFG=$SOCKET_CFG/lb-P10L10.json ;;
    gptfast_P10L60) export $gptfast; CFG=$SOCKET_CFG/lb-P10L60.json ;;
    *) echo "UNKNOWN ARM $1" >&2; return 1 ;;
  esac
  export SOCKET_DEDUP=0
}

run_one () {  # <arm> <dataset>
  local arm=$1 ds=$2 cfg out
  CFG=""
  arm_setup "$arm" || return 1     # NOT $(...) -- see the note on arm_setup
  cfg="$CFG"
  if [ "$arm" = "dense" ]; then unset SOCKET_TARGET_SPARSITY
  else export SOCKET_TARGET_SPARSITY="${SPARSITY:-10}"; fi
  out="$OUT/${arm}_${ds}"
  mkdir -p "$out"
  echo "########## ARM=$arm DS=$ds cfg=$(basename $cfg) \
scorer=${SOCKET_SCORER:-na} select=${SOCKET_SELECT:-na} list=${SOCKET_LIST:-na} \
attn=${SOCKET_ATTN:-na} dedup=${SOCKET_DEDUP:-na} sparsity=${SOCKET_TARGET_SPARSITY:-dense} $(date -Is)"
  python pipeline/train_quest/main.py \
    --exp_desc "kfix_${arm}_${ds}" \
    --pipeline_config_dir "$cfg" \
    --eval_config_dir "config/eval_config/longbench/${ds}.json" \
    --train_config_dir config/train_config/train/LongAlpaca-12k.json \
    --output_folder_dir "$out" \
    --deepspeed_config config/deepspeed_config/zero3_inference.json \
    2>&1 | grep -E "SOCKET-SMOKE|LB-LIMIT|SOCKET-MODEL|Error|Traceback|error" | tail -20
  echo "---- exit=${PIPESTATUS[0]} arm=$arm ds=$ds $(date -Is)"
  echo
}

for DS in ${DSLIST:?set DSLIST, e.g. DSLIST="hotpotqa samsum qasper multifieldqa_en"}; do
  for ARM in ${ARMS:-dense legacy_P8L50 legacy_P10L10 gptfast_P8L50 gptfast_P10L10 gptfast_P10L60}; do
    run_one "$ARM" "$DS"
  done
done
echo "LB_ALL_DONE $(date -Is)"
