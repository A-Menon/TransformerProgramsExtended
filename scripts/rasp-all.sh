#!/usr/bin/env bash
# Runs every combination of hyperparam variant × module config
set -euo pipefail

# ───────────── global knobs (identical for every task) ─────────────
SEED=0
DATASET_SIZE=20000
BATCH=512
EPOCHS=250

LR=2e-2           # the paper’s LR
CLIP=1.0          # grad‑norm clip
GUMBEL_SAMPLES=4
TAU_INIT=3.0
TAU_END=0.1       # cools far enough for discrete convergence
TAU_SCHED=geomspace

TASKS=(double_hist hist reverse sort most_freq)

# hyperparam variants: name → vocab‐scale, max‐scale
declare -A V_SCALE=( [baseline]=1 [longlen]=1 [bigvocab]=2 )
declare -A M_SCALE=( [baseline]=1 [longlen]=2 [bigvocab]=2 )

# module configs: name → flags
declare -A MODULE_FLAGS=(
  [none]=""
  [prefix]="--use_prefix_counts"
  [expert]="--use_experts"
  [both]="--use_prefix_counts --use_experts"
)

for DATASET in "${TASKS[@]}"; do
  case "$DATASET" in
    hist)         VOCAB=8  MAX=8  L=1 H=4 M=2 ;;
    double_hist)  VOCAB=8  MAX=8  L=3 H=4 M=2 ;;
    reverse)      VOCAB=8  MAX=8  L=3 H=8 M=2 ;;
    sort)         VOCAB=8  MAX=8  L=3 H=8 M=4 ;;
    most_freq)    VOCAB=8  MAX=8  L=4 H=8 M=4 ;;
  esac

  N_HEADS_CAT=$((H/2)); N_HEADS_NUM=$((H/2))
  N_CAT_MLPS=$((M/2));  N_NUM_MLPS=$((M/2))
  ATT_TYPE="both"
  MLP_TYPE="--mlp_type cat"

  case "$DATASET" in
    hist)
      ATT_TYPE="num"
      N_HEADS_CAT=0; N_HEADS_NUM=$H
      ;;
    double_hist)
      ATT_TYPE="both"
      N_HEADS_CAT=2;  N_HEADS_NUM=2
      N_CAT_MLPS=1;   N_NUM_MLPS=1
      MLP_TYPE="--mlp_type mix"
      ;;
  esac

  echo -e "\n=== Dataset: $DATASET (L=$L, H=$H, M=$M) ==="

  for VARIANT in baseline longlen bigvocab; do
    vc=${V_SCALE[$VARIANT]}
    mc=${M_SCALE[$VARIANT]}
    VOCAB_VAR=$((VOCAB * vc))
    MAX_VAR=$((MAX * mc))

    echo "--- Variant: $VARIANT (vocab=${VOCAB_VAR}, maxlen=${MAX_VAR}) ---"

    for MODULE in none prefix expert both; do
      # ---- skip some ----
      # if []; then
      #   echo "---- Skipping $DATASET baseline with module $MODULE ----"
      #   continue
      # fi

      FLAGS="${MODULE_FLAGS[$MODULE]}"
      if [[ "$DATASET" =~ ^(hist|double_hist)$ ]]; then
        FLAGS="$FLAGS --count_only"
      fi
      OUTDIR="output/rasp/${DATASET}/${VARIANT}/modules_${MODULE}/k${VOCAB_VAR}_len${MAX_VAR}_L${L}_H${H}_M${M}/s${SEED}"

      echo "---- Modules: $MODULE → $OUTDIR ----"
      python src/run.py \
        --dataset "$DATASET" \
        --vocab_size "$VOCAB_VAR" \
        --dataset_size "$DATASET_SIZE" \
        --min_length 1 --max_length "$MAX_VAR" \
        --batch_size "$BATCH" --n_epochs "$EPOCHS" \
        --lr "$LR" --max_grad_norm "$CLIP" \
        --sample_fn gumbel_soft --gumbel_samples "$GUMBEL_SAMPLES" \
        --tau_init "$TAU_INIT" --tau_end "$TAU_END" --tau_schedule "$TAU_SCHED" \
        --n_vars_cat 1 --n_vars_num 1 --d_var "$MAX_VAR" \
        --n_layers "$L" \
        --n_heads_cat "$N_HEADS_CAT" --n_heads_num "$N_HEADS_NUM" \
        --n_cat_mlps "$N_CAT_MLPS" --n_num_mlps "$N_NUM_MLPS" \
        --attention_type "$ATT_TYPE" \
        --rel_pos_bias fixed --one_hot_embed --dropout 0.0 \
        --mlp_vars_in 2 --d_mlp 64 \
        $MLP_TYPE \
        --selector_width 0 \
        --seed "$SEED" --unique 1 \
        --save --save_code \
        $FLAGS \
        --output_dir "$OUTDIR"
    done
  done
done