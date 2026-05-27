# Codex workplan for SFT_vs_RL

Global rules:
- Do not run long jobs on oat0.
- Do not run srun/sbatch unless explicitly asked.
- Do not download large models/datasets.
- Prefer minimal patches.
- Always show git diff after editing.


Session summary 2026-05-18:
- Added HH alignment pipeline stages: `debug_hidden_states.py`, `utils/hidden_records.py`, `utils/hh_metrics.py`, `utils/hh_sampling.py`, `run_hh_analysis.py`, and `validate_hh_outputs.py`.
- `run_hh_analysis.py` extracts token hidden records, samples pairs, computes raw and centered dot/cosine metrics, saves CSV/JSON/JSONL outputs, and supports debug token alignment, pair samples, and tensor dumps.
- Added deterministic dry tests in `utils/hh_metrics.py` and `utils/hh_sampling.py`; both passed.
- Added experiment script `bashes/hh_GSM8K_try.sh` for GSM8K first 100 examples, layers `0,4,8,12,20,-1`, about 10k cross-sample pairs.
- Fixed cluster script default to allow HF downloads on cluster and use scratch HF caches; no HF token is hard-coded.
- Current next step: rerun `bashes/hh_GSM8K_try.sh` on cluster, then validate with `validate_hh_outputs.py` and inspect debug JSONL/PT files.

Session summary 2026-05-20:
- Goal: extend HH alignment beyond GSM8K to mixed dataset experiments and make raw selected examples identical across model runs.
- Files inspected: `run_hh_analysis.py`, `utils/hh_dataset_adapters.py`, `utils/hidden_records.py`, `utils/hh_sampling.py`, `bashes/hh_dataset_combos.sh`, `utils/gsm8k_tokenizer.py`, `notes/codex_worklog.md`.
- Files modified: `run_hh_analysis.py`, `utils/hh_dataset_adapters.py`, `utils/hidden_records.py`, `utils/hh_sampling.py`, `bashes/hh_dataset_combos.sh`, `notes/codex_worklog.md`.
- Commands run: `git commit -m "Add HH alignment dataset adapters"`, AST/import checks for HH files, `python utils/hh_dataset_adapters.py`, `python utils/hh_sampling.py`, `bash -n bashes/hh_dataset_combos.sh`, and read-only `git diff`/status commands.
- Current status: mixed adapters support `gsm8k_mmlu`, `gsm8k_dolly`, `dolly_mmlu`, and `dolly_gsm8k`; cross-dataset pair filtering is available; raw examples are sampled before tokenization and recorded in `run_config.json`; Dolly `prompt_text` indentation bug is fixed.
- Recommended next step: rerun the failed Dolly/MMLU or Dolly/GSM8K cluster job and compare `selected_raw_examples` across model output directories before interpreting HH metrics.
