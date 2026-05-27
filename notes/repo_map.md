# Repository Map

This repository is a compact, script-heavy project for comparing SFT-style token behavior with RL/generation-style behavior. The main domain is GSM8K, with MMLU probe data used by the single-token forgetting prototype.

## Main Directories

- `configs/`: YAML experiment configuration. Currently `train_basic.yaml`.
- `utils/`: shared data processing, tokenization, metric, and analysis helpers.
- `bashes/`: SLURM-oriented launch scripts for logit extraction and one-step score runs.
- `notes/`: Codex-facing notes and task planning.
- `results/`: experiment outputs, including previous logit/statistic runs.
- `logs/`: SLURM stdout/stderr logs.
- `figures/`: generated or saved plots.
- `out/`: small resolved config/output artifacts.
- `wandb/`: W&B/debug logs.
- `.ipynb_checkpoints/`: notebook checkpoints.

## Main Scripts

- `get_logits_sft.py`: loads GSM8K, builds supervised token labels, optionally trains with `SFTTrainer`, and extracts teacher-forced logits over valid answer tokens. Saves `sft_logits.pt`.
- `get_logits_grpo.py`: performs generation using `model.generate` in greedy or sampling mode, collects generation scores as RL-style logits, decodes responses, checks GSM8K answers, and saves `rl_greedy_logits.pt` or `rl_tmp{temperature}_logits.pt`.
- `get_one_step_scores.py`: prototype for single-token interventions. It evaluates MMLU probes before/after one-token SFT updates and writes summary/domain forgetting metrics.
- `analyze_logits.py`: loads saved logit tensors, computes entropy/probability/rank statistics, and writes plots.
- `train.py`: currently empty.
- `backup_train_sft.py`, `backup_train_grpo.py`: backup/older versions of training or extraction scripts.
- `llama.py`: small standalone script.

## Configs

- `configs/train_basic.yaml` defines:
  - project name: `sft_vs_rl`
  - dataset: `openai/gsm8k`, config `main`
  - default model: `microsoft/Phi-3-mini-4k-instruct`
  - model max length: `256`
  - training hyperparameters such as learning rate, epochs, batch size, logging, and save steps

The main scripts accept CLI overrides for model name, learning rate, epochs, batch size, save/logging steps, sample counts, and generation parameters.

## Data Loading Path

GSM8K path:

1. Scripts call `datasets.load_dataset("openai/gsm8k", "main", split="train")`, usually through config values.
2. `utils/gsm8k_tokenizer.py` provides `GSM8KTokenizerProcessor`.
3. Each GSM8K example is converted into:
   - `input_ids`
   - `attention_mask`
   - `labels`
   - `input_ids_rl`
   - `attention_mask_rl`
   - `ground_truth`
4. SFT uses the full prompt plus answer with answer tokens unmasked in `labels`.
5. RL/generation uses `input_ids_rl` and `attention_mask_rl`, which contain the prompt ending at `Answer:\n`.

MMLU probe path:

1. `utils/prob_mmlu_gen.py` loads `cais/mmlu` subjects.
2. It formats multiple-choice prompts with answer letters.
3. It tokenizes prompt plus target answer for teacher-forced log-prob evaluation.
4. `get_one_step_scores.py` groups probes by domain and computes before/after log-prob deltas.

## Model, Loss, and Evaluation Flow

SFT logit extraction:

1. Load tokenizer and `AutoModelForCausalLM`.
2. Process GSM8K examples with `GSM8KTokenizerProcessor`.
3. For each sample, run the model with `input_ids`, `attention_mask`, and `labels`.
4. Collect logits only where labels are not `-100` and attention mask is active.
5. Save logits, labels, decoded responses, and model metadata.

Generation/RL-style extraction:

1. Load tokenizer and `AutoModelForCausalLM`.
2. Process GSM8K examples.
3. Generate from `input_ids_rl`.
4. Use returned generation `scores` as per-step logits.
5. Decode generated answer, compare extracted solution to `ground_truth`, and save correctness metadata.

Single-token intervention prototype:

1. Evaluate MMLU probe log-probs on the base model.
2. Clone the base model state.
3. For each supervised GSM8K token position, restore the base model state.
4. Compute `sop`, `aop_l2`, `aop_kl`, and entropy via `utils/tracking_metrics.py`.
5. Apply one single-token SFT update.
6. Re-evaluate probe log-probs.
7. Save forgetting summaries by probe domain.

Important alignment convention:

- For causal LM scoring, token at position `pos` is predicted by logits at `pos - 1`.

## Output, Checkpoint, and Logging Conventions

- SFT extraction saves `sft_logits.pt`.
- Generation extraction saves files such as `rl_greedy_logits.pt` and `rl_tmp1.0_logits.pt`.
- One-step intervention saves:
  - `updates_summary.parquet`
  - `updates_by_domain.parquet`
  - optional CSV copies
  - config JSON/YAML metadata
- Scripts commonly save resolved configs to the output directory.
- SLURM scripts write logs to `logs/%x-%j.out` and `logs/%x-%j.err`.
- Batch scripts stage larger files under `/scratch-ssd/$USER/SFT_vs_RL/results/...` and sync selected metadata back under `$HOME/SFT_vs_RL/results/...`.

## Batch Scripts

- `bashes/get_logits_sft.sh`: SLURM script that loops over models and runs SFT plus greedy/sample generation logit extraction.
- `bashes/get_one_step_scores.sh`: SLURM script for single-token MMLU forgetting runs.
- `bashes/_install_env_int.sh`: conda environment update/activation helper.

These scripts use `srun`, may update conda environments, may load large models/datasets, and may run SSH/rsync. Do not run them without explicit approval.

## Lightweight Dry-Run Commands

No truly lightweight dry-run command was found. The available sample-count flags reduce work but still load models and datasets, so they should not be treated as cheap:

- `python get_logits_sft.py --config ./configs/train_basic.yaml --model_name <model> --output_dir <dir> --do_extract_logits --extract_n_samples 1`
- `python get_logits_grpo.py --config ./configs/train_basic.yaml --model_name <model> --output_dir <dir> --mode greedy --max_new_tokens 8 --n_samples 1`
- `python get_one_step_scores.py --config ./configs/train_basic.yaml --model_name <model> --output_dir <dir> --extract_n_samples 1 --prob_n_samples 1`

Use these only with explicit approval if they would download/load models or run on GPU/cluster resources.
