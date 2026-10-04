#!/usr/bin/env bash
# Policy grid: 3 conditions x 3 seeds at 60k steps (clean s0 already done).
# Run with: setsid nohup bash scripts/run_grid_train.sh > data/grid_train.log 2>&1 &
set -u
: "${HF_TOKEN:?Set HF_TOKEN before running (never commit tokens)}"
export HF_TOKEN
cd /home/raihan/Desktop/APPS/research-learn/demodoctor

run_one() { # condition, seed, dataset_repo, dataset_root
  local cond=$1 seed=$2 repo=$3 root=$4
  local outdir="data/policy_${cond}_s${seed}"
  rm -rf "$outdir"
  .venv/bin/lerobot-train \
    --dataset.repo_id="$repo" \
    ${root:+--dataset.root="$root"} \
    --policy.type=act \
    --policy.repo_id="raihan-js/demodoctor-act-${cond}-s${seed}" \
    --batch_size=32 --steps=60000 \
    --output_dir="$outdir" --job_name="pusht_act_${cond}_s${seed}" \
    --seed="$seed" --save_checkpoint_to_hub=false
}

# clean s1, s2 (s0 done)
run_one clean 1 lerobot/pusht ""
run_one clean 2 lerobot/pusht ""
# corrupted s0, s1, s2
run_one corrupted 0 raihan-js/demodoctor-pusht-corrupted data/pusht_corrupted/corrupted
run_one corrupted 1 raihan-js/demodoctor-pusht-corrupted data/pusht_corrupted/corrupted
run_one corrupted 2 raihan-js/demodoctor-pusht-corrupted data/pusht_corrupted/corrupted
# cleaned s0, s1, s2
run_one cleaned 0 raihan-js/demodoctor-pusht-cleaned data/pusht_cleaned/cleaned
run_one cleaned 1 raihan-js/demodoctor-pusht-cleaned data/pusht_cleaned/cleaned
run_one cleaned 2 raihan-js/demodoctor-pusht-cleaned data/pusht_cleaned/cleaned
echo GRID_DONE
