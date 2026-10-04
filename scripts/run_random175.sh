#!/usr/bin/env bash
# Size-matched control: 3 seeds on random-175. Answers whether the cleaning
# deficit is data SIZE (random175 ~= cleaned) or quality (random175 ~= corrupted).
# Run with: setsid nohup bash scripts/run_random175.sh > data/grid_random175.log 2>&1 &
set -u
: "${HF_TOKEN:?Set HF_TOKEN before running (never commit tokens)}"
export HF_TOKEN
cd /home/raihan/Desktop/APPS/research-learn/demodoctor

for seed in 0 1 2; do
  outdir="data/policy_random175_s${seed}"
  rm -rf "$outdir"
  .venv/bin/lerobot-train \
    --dataset.repo_id=raihan-js/demodoctor-pusht-random175 \
    --dataset.root=data/pusht_random175/random175 \
    --policy.type=act \
    --policy.repo_id="raihan-js/demodoctor-act-random175-s${seed}" \
    --batch_size=32 --steps=60000 \
    --output_dir="$outdir" --job_name="pusht_act_random175_s${seed}" \
    --seed="$seed" --save_checkpoint_to_hub=false
done
echo GRID_DONE
