# AGENTS.md

Stable guidance for future Codex sessions in this repository.

## Project Purpose

This repository investigates SFT-style teacher-forced token behavior versus RL/generation-style token behavior on GSM8K and related probes. The main workflows extract token logits, compute token-level statistics such as entropy/SOP/AOP, run single-token update interventions, and analyze forgetting on MMLU probe items.

## Repository Shape

- Main experiment config: `configs/train_basic.yaml`.
- Main SFT logit extraction script: `get_logits_sft.py`.
- Main generation/RL-style logit extraction script: `get_logits_grpo.py`.
- Single-token intervention and forgetting prototype: `get_one_step_scores.py`.
- Shared utilities live under `utils/`.
- Batch/cluster launch scripts live under `bashes/`.
- Generated artifacts usually go under `results/`, `logs/`, `out/`, `figures/`, or `wandb/`.

## Coding Conventions

- Prefer small, localized patches that match the existing script-oriented style.
- Keep configuration in YAML when adding run parameters.
- Reuse existing utilities in `utils/` before adding new helpers.
- Use explicit tensor shapes in comments when they clarify model/logit alignment.
- Preserve the current causal-LM convention: token at position `pos` is predicted from logits at `pos - 1`.
- Avoid broad refactors unless explicitly requested.

## Cluster and Safety Rules

- Do not run long jobs on `oat0`.
- Do not run `srun` or `sbatch` unless the user explicitly asks.
- Do not download large models or datasets unless the user explicitly approves.
- Do not launch GPU/model-loading jobs just to inspect code.
- Do not run scripts that update conda environments, sync remote files, or use SSH/rsync unless explicitly approved.
- Treat tokens and credentials in shell scripts as sensitive. Do not print or propagate them unnecessarily.

## Commands Requiring Explicit Approval

Ask before running:

- `srun ...`
- `sbatch ...`
- `conda-env update ...`
- model or dataset downloads from Hugging Face or other registries
- remote commands such as `ssh ...` or `rsync ...`
- any long training, generation, extraction, or probe-evaluation run
- destructive git or filesystem operations

## Review Expectations

- Prefer minimal patches.
- After editing, show `git diff`.
- Do not revert user changes unless explicitly instructed.
- Keep generated outputs and large artifacts out of source changes unless the user asks for them.
