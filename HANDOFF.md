> **SUPERSEDED 2026-10-05.** The control grid finished and the experiment is complete (a null result). See `README.md` for results and `results/` for the numbers. The status below is historical.

# HANDOFF — DemoDoctor session state (2026-10-04)

Read this first if you are picking up this project in a new AI session.
Everything below was true at handoff time. Verify process aliveness before acting.

## What this project is

DemoDoctor: find bad robot demonstrations automatically, measure what they cost
a policy. PushT + ACT + 5 fault types + label-free detectors + policy grid.
Full context: `AGENTS.md` in this repo.

## Results so far (verified)

Detector F1 vs manifest (206 episodes, 30% corrupted):
- stall 0.867 | jitter 0.687 | lag 0.119 (null) | truncation ~0 (null) | dup null

9-run policy grid, 60k steps, 50 rollouts each (success / max_reward):
- clean:      s0 2.0/0.468 | s1 0.0/0.406 | s2 4.0/0.405
- corrupted:  s0 2.0/0.412 | s1 4.0/0.442 | s2 0.0/0.373
- cleaned:    s0 PENDING    | s1 0.0/0.377 | s2 0.0/0.349

Stats: clean-vs-corrupted max_reward p=0.581 (no effect); corrupted-vs-cleaned
p=0.078 (cleaning trends WORSE). Open question: quality or quantity?
The control grid below decides it.

## Currently running (background, survives terminal close)

Size-matched control: random-175 subset of corrupted set × 3 seeds (s0/s1/s2),
60k steps each, ~2h per run, ~6h total. Started 2026-10-04.
- Script: `scripts/run_random175.sh` (uses $HF_TOKEN from env — NEVER commit a token)
- Log: `data/grid_random175.log`
- Checkpoints: `data/policy_random175_s{0,1,2}/`
- Check aliveness: `ps aux | grep '[l]erobot-train'` and tail the log.
- If dead with no traceback: likely machine reboot — restart the script
  (it has no resume; each run restarts from scratch, ~2h each).

## Next steps in order

1. Wait for control grid to finish (all 3 runs).
2. Eval all 3: `scripts/run_grid_eval.py` only covers clean/corrupted/cleaned —
   eval random175 manually:
   `.venv/bin/lerobot-eval --policy.path=data/policy_random175_s{SEED}/checkpoints/last/pretrained_model --env.type=pusht --eval.n_episodes=50 --eval.batch_size=1`
3. Final table + verdict:
   - random175 ≈ cleaned → deficit is SIZE (quantity beats quality) → real finding.
   - random175 ≈ corrupted → quality irrelevant here → pure null, publish honestly.
4. Release (only if verdict warrants full treatment, else honest slim version):
   - `devto_article.md` (draft exists, results TBD)
   - HF dataset (needs HF_TOKEN env; user provides it — check chat history)
   - GitHub repo (create `raihan-js/demodoctor` if missing — secret-scan blocked once
     before because a token was hardcoded; `git grep` for tokens before pushing)
   - Resume: `/home/raihan/Desktop/APPS/research-learn/Raihan-Sikder-Resume.pdf`
     (+ copy to `/home/raihan/Desktop/APPS/career/`) — currently at 9 projects.
   - Portfolio: `/tmp/portfolio` clone of raihan-js.github.io (may be stale — re-clone).
   - Parent AGENTS.md: `/home/raihan/Desktop/APPS/research-learn/AGENTS.md`.

## Environment facts (do not re-derive)

- venv: `demodoctor/.venv`, torch 2.11.0+cu130, lerobot **0.6.1** (CLI uses
  `--save_checkpoint_to_hub=false`; pods had 0.4.4 with different flags).
- Local video backend: torchcodec (works). Pods needed `--video-backend pyav`.
- Eval MUST use `--eval.batch_size=1` (async workers lose gym_pusht registration).
- RunPod: balance NEGATIVE, all pods dead. Local-only until user tops up.
- Do NOT kill pid 4653 (system uvicorn, outside this project).
- Do NOT commit tokens. GitHub push protection blocks secrets.
- Datasets on HF: `raihan-js/demodoctor-pusht-corrupted`,
  `raihan-js/demodoctor-pusht-cleaned` (private). Local copies under `data/`.

## Key files

- `src/demodoctor/faults.py` + `detectors.py` (37 tests green)
- `scripts/build_corrupted.py`, `build_cleaned.py`, `build_random_subset.py`
- `scripts/score_detectors.py`, `run_grid_train.sh`, `run_grid_eval.py`
- `data/results/detector_scores.json`, `data/results/policy_grid.json` (partial)
