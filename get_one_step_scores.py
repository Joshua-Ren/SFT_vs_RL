from pathlib import Path
import json
import pandas as pd
from utils.tracking_metrics import compute_off_policy_scores

import os
import json
import yaml
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from utils.config_read import load_config
from utils.gsm8k_tokenizer import GSM8KTokenizerProcessor
from utils.prob_mmlu_gen import build_mmlu_probe_dataset, tokenize_probe_dataset, load_processed_dataset

# =========================
# 1. Results saver
# =========================
class UpdateResultWriter:
    def __init__(self, domains):
        self.domains = domains
        self.summary_rows = []   # each elements is for one (s_u, y_u) pair's influence on all probing data tokens
        self.domain_rows = []    # 

    def add_update(
        self,
        update_id,
        algo,
        sample_id,
        token_pos,
        sop,
        entropy,
        aop_l2,
        aop_kl,
        f_avg_by_domain,
        f_ratio_by_domain,
        extra=None,
    ):
        """
        Args:
            update_id: str/int, index of the update pair (s_u, y_u)
            algo: "sft", "rl_temp1", ...
            sample_id: id in dataset
            token_pos: token id in a sequence
            sop, entropy, aop_l2, aop_kl: float, defined matrics in the paper
            f_avg_by_domain: dict, e.g. {"math": 0.1, "history": 0.03, ...}
            f_ratio_by_domain: dict, same keys as above
            extra: dict, optional
        """
        extra = extra or {}

        # ---- summary row ----
        row = {
            "update_id": update_id,
            "algo": algo,
            "sample_id": sample_id,
            "token_pos": token_pos,
            "sop": sop,
            "entropy":entropy,
            "aop_l2": aop_l2,
            "aop_kl": aop_kl,
            "f_avg_mean": self._safe_mean(f_avg_by_domain),
            "f_ratio_mean": self._safe_mean(f_ratio_by_domain),
        }

        # unrow to detailed table for each domain
        for d in self.domains:
            row[f"f_avg_{d}"] = f_avg_by_domain.get(d, None)
            row[f"f_ratio_{d}"] = f_ratio_by_domain.get(d, None)

        row.update(extra)
        self.summary_rows.append(row)

        # ---- long table rows ----
        for d in self.domains:
            self.domain_rows.append({
                "update_id": update_id,
                "algo": algo,
                "sample_id": sample_id,
                "token_pos": token_pos,
                "domain": d,
                "sop": sop,
                "entropy":entropy,
                "aop_l2": aop_l2,
                "aop_kl": aop_kl,
                "f_avg": f_avg_by_domain.get(d, None),
                "f_ratio": f_ratio_by_domain.get(d, None),
                **extra,
            })

    def to_dataframes(self):
        df_summary = pd.DataFrame(self.summary_rows)
        df_domain = pd.DataFrame(self.domain_rows)
        return df_summary, df_domain

    def save(self, out_dir, config=None, save_csv=False):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        df_summary, df_domain = self.to_dataframes()

        df_summary.to_parquet(out_dir / "updates_summary.parquet", index=False)
        df_domain.to_parquet(out_dir / "updates_by_domain.parquet", index=False)

        if save_csv:
            df_summary.to_csv(out_dir / "updates_summary.csv", index=False)
            df_domain.to_csv(out_dir / "updates_by_domain.csv", index=False)

        if config is not None:
            with open(out_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"Saved to: {out_dir}")

    @staticmethod
    def _safe_mean(x):
        vals = [v for v in x.values() if v is not None]
        return sum(vals) / len(vals) if len(vals) > 0 else None

def compute_forgetting_scores(update_pair, probe_by_domain):
    """
    Return two dics:
        f_avg_by_domain  : {"math": ..., "history": ..., ...}
        f_ratio_by_domain: {"math": ..., "history": ..., ...}
    """
    f_avg_by_domain = {}
    f_ratio_by_domain = {}

    for domain, probe_items in probe_by_domain.items():
        # For example:
        # deltas = [delta_t(update_pair, probe) for probe in probe_items]
        # f_avg_by_domain[domain] = -sum(deltas) / len(deltas)
        # f_ratio_by_domain[domain] = sum(d < -tau for d in deltas) / len(deltas)

        f_avg_by_domain[domain] = 0.0
        f_ratio_by_domain[domain] = 0.0

    return f_avg_by_domain, f_ratio_by_domain

# =========================
# 2. Probe evaluation helpers
# =========================
@torch.no_grad()
def get_probe_logprob(model, probe_item):
    """
    Compute log p(target_text | prompt_text) by teacher forcing.

    probe_item fields:
        input_ids: full prompt+target ids
        attention_mask
        target_start: first target token position in full sequence
        target_end: end position (exclusive)
    """
    model.eval()

    device = model.device
    input_ids = torch.tensor(probe_item["input_ids"]).unsqueeze(0).to(device)         # [1, L]
    attention_mask = torch.tensor(probe_item["attention_mask"]).unsqueeze(0).to(device) # [1, L]

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False,)
    logits = outputs.logits  # [1, L, V]

    log_probs = F.log_softmax(logits, dim=-1)

    target_start = probe_item["target_start"]
    target_end = probe_item["target_end"]
    target_ids = probe_item["target_token_ids"]  # [T]

    # causal LM: token at position t is predicted by logits at t-1
    token_logps = []
    for pos in range(target_start, target_end):
        target_token_id = input_ids[0, pos]
        token_logp = log_probs[0, pos - 1, target_token_id]
        token_logps.append(token_logp)

    total_logp = torch.stack(token_logps).sum().item()
    avg_logp = torch.stack(token_logps).mean().item()

    return {
        "probe_id": probe_item["probe_id"],
        "domain": probe_item["domain"],
        "total_logp": total_logp,
        "avg_logp": avg_logp,
    }


@torch.no_grad()
def eval_probe_dataset(model, tokenized_items):
    """
    Evaluate all probe items once.
    Now it is one-by-one, very inefficient but simple. 
    Can be optimized later by batching items from the same domain together.
    Returns:
        results: list[dict]
        result_by_id: dict[probe_id] -> dict
    """
    
    results = []
    for item in tokenized_items:
        results.append(get_probe_logprob(model, item))

    result_by_id = {x["probe_id"]: x for x in results}
    return results, result_by_id

# =========================
# 3. Delta -> forgetting metrics
# =========================
def compute_forgetting_scores(before_by_id, after_by_id, probe_by_domain, tau=0.1):
    """
    Delta is defined as:
        delta = logp_after - logp_before

    Forgetting metrics:
        F_avg   = average of (-delta)
        F_ratio = fraction of probe items with delta < -tau
    """
    f_avg_by_domain = {}
    f_ratio_by_domain = {}

    for domain, probe_items in probe_by_domain.items():
        deltas = []

        for item in probe_items:
            probe_id = item["probe_id"]
            delta = after_by_id[probe_id]["total_logp"] - before_by_id[probe_id]["total_logp"]
            deltas.append(delta)

        if len(deltas) == 0:
            f_avg_by_domain[domain] = None
            f_ratio_by_domain[domain] = None
            continue

        # Larger f_avg_by_domain means more serious forget
        f_avg_by_domain[domain] = -sum(deltas) / len(deltas)
        f_ratio_by_domain[domain] = sum(d < -tau for d in deltas) / len(deltas)

    return f_avg_by_domain, f_ratio_by_domain

# =========================
# 4. Model state save / restore
# =========================
def clone_model_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def restore_model_state(model, state_dict):
    model.load_state_dict(state_dict)

# =========================
# 5. Single-token SFT update
# =========================
def apply_single_token_update(model, optimizer, input_ids, attention_mask, labels, pos):
    """
    Only update one supervised token at position `pos`.
    Assumes:
        - labels already follow HF causal LM convention
        - no manual label shift needed
    """
    model.train()

    single_labels = torch.full_like(labels, -100)
    single_labels[0, pos] = labels[0, pos]

    optimizer.zero_grad()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=single_labels,
        output_hidden_states=False,
        output_attentions=False,
    )

    loss = outputs.loss
    loss.backward()
    optimizer.step()

    return loss.item()

# =========================
# 6. Main evaluation loop
# =========================
def run_single_token_intervention(
    model,
    optimizer,
    processed_dataset,
    tokenized_items,
    tokenized_by_domain,
    writer,
    max_samples=None,
    tau=0.1,
    algo="sft",
):
    """
    Args:
        model: current model theta_t
        optimizer: optimizer for one-step update
        processed_dataset: SFT-style dataset with input_ids / attention_mask / labels
        tokenized_items: all probe items
        tokenized_by_domain: domain -> list of probe items
        writer: UpdateResultWriter
        compute_sop, compute_aop_l2, compute_aop_kl:
            user-defined functions taking (model, sample, pos) -> float
    """

    # import pdb
    # breakpoint()

    # evaluate probe set once on base model
    _, before_by_id = eval_probe_dataset(model, tokenized_items)

    # save clean base state once
    base_state = clone_model_state(model)

    n_samples = len(processed_dataset) if max_samples is None else min(len(processed_dataset), max_samples)

    for sample_id in tqdm(range(n_samples)):
        sample = processed_dataset[sample_id]

        input_ids = sample["input_ids"].unsqueeze(0).to(next(model.parameters()).device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(next(model.parameters()).device)
        labels = sample["labels"].unsqueeze(0).to(next(model.parameters()).device)

        valid_mask = ((labels != -100) & (attention_mask == 1))
        valid_positions = torch.nonzero(valid_mask[0], as_tuple=False).squeeze(-1).tolist()

        for pos in tqdm(valid_positions):
            # always restore to the same theta_t
            restore_model_state(model, base_state)

            sop, aop_l2, aop_kl, entropy = compute_off_policy_scores(model, sample, pos, gamma=0.5)

            loss_val = apply_single_token_update(
                model=model,
                optimizer=optimizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pos=pos,
            )

            # evaluate probe after one-step update
            _, after_by_id = eval_probe_dataset(model, tokenized_items)

            f_avg_by_domain, f_ratio_by_domain = compute_forgetting_scores(
                before_by_id=before_by_id,
                after_by_id=after_by_id,
                probe_by_domain=tokenized_by_domain,
                tau=tau,
            )

            writer.add_update(
                update_id=f"{sample_id}_{pos}",
                algo=algo,
                sample_id=sample_id,
                token_pos=pos,
                sop=sop,
                entropy=entropy,
                aop_l2=aop_l2,
                aop_kl=aop_kl,
                f_avg_by_domain=f_avg_by_domain,
                f_ratio_by_domain=f_ratio_by_domain,
                extra={
                    "loss": loss_val,
                    "target_token_id": int(labels[0, pos].item()),
                },
            )

    # restore clean model at the end
    restore_model_state(model, base_state)

# =========================
# 7. main
# =========================
def parse_args():
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument("--config", type=str, default="./configs/train_basic.yaml")
    parser.add_argument("--output_dir", type=str, required=True)

    # common overrides
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)

    parser.add_argument("--model_name",type=str, default=None)

    # Train & evaluations
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--extract_n_samples", type=int, default=50)
    parser.add_argument("--prob_n_samples", type=int, default=50)

    return parser.parse_args()


def apply_overrides(config, args):
    if args.lr is not None:
        config["train"]["learning_rate"] = args.lr
    if args.epochs is not None:
        config["train"]["num_train_epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["per_device_train_batch_size"] = args.batch_size
    if args.save_steps is not None:
        config["train"]["save_steps"] = args.save_steps
    if args.logging_steps is not None:
        config["train"]["logging_steps"] = args.logging_steps   
    if args.model_name is not None:
        config["model"]["name"] = args.model_name
    return config


def save_yaml(obj, path):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def main():
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # save resolved config
    save_yaml(config, output_dir / "resolved_config.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name_or_path = config["model"]["name"]

    # ---- load model / tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    ).to(device)

    # ---- optimizer ----
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
    )

    # ---- training dataset
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    processor = GSM8KTokenizerProcessor(model_name_or_path=model_name_or_path,max_length=512)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    processed_dataset = processor.process_dataset(dataset)

    # ---- probing dataset ----
    subjects = [
        "abstract_algebra",
        "college_mathematics",
        "high_school_physics",
        "computer_security",
        "international_law",
        # "moral_disputes",
        # "high_school_us_history",
        # "clinical_knowledge",
    ]
    tokenized_items, tokenized_by_domain = load_processed_dataset(tokenizer, subjects, n_per_subject=args.prob_n_samples)

    # ---- result writer ----
    writer = UpdateResultWriter(domains=list(tokenized_by_domain.keys()))

    # ---- run experiment ----
    run_single_token_intervention(
        model=model,
        optimizer=optimizer,
        processed_dataset=processed_dataset,
        tokenized_items=tokenized_items,
        tokenized_by_domain=tokenized_by_domain,
        writer=writer,
        max_samples=args.extract_n_samples,   # first run a tiny sanity check
        tau=0.1,
        algo="sft",
    )

    # ---- save results ----
    config = {
        "model_name_or_path": model_name_or_path,
        "lr": args.lr,
        "probe_subjects": subjects,
        "n_probe_per_subject": args.prob_n_samples,
        "tau": 0.1,
        "algo": "sft",
        "max_samples": args.extract_n_samples,
    }

    writer.save(
        out_dir=output_dir,
        config=config,
        save_csv=True,
    )


if __name__ == "__main__":
    main()
    # ===================================
    # Developing zone
    # ===================================
    # Step 1: Prepare the base model or checkpoint, i.e., theta_t
    # Step 2: Prepare the training data (x, y), maybe decompose them into (s_u, y_u),
    #         or just use different masks
    # Step 3: Prepare the probing dataset, e.g., MMLU. Specify the domain for each example
    # Step 4: Main loop, for each (s_u, y_u)
        # 4.0 load the model params to M_t, then evaluate to get all p(M_t)
        # 4.1 calculate SoP, AoP_l2, AoP_kl
        # 4.2 update the model using (s_u, y_u)
        # 4.3 evaluate to get M_{t+1}, compare p(M_{t+1}) and p(M_t)
        # 4.4 restore the model to M_t
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_name = "Qwen/Qwen3.5-0.8B"

    #     # --------- Training dataset
    # dataset = load_dataset("openai/gsm8k", "main", split="train")
    # processor = GSM8KTokenizerProcessor(model_name_or_path=model_name,max_length=512)
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    # processed_dataset = processor.process_dataset(dataset)
    #     # -------- Probing dataset
    # subjects = [
    #     "abstract_algebra",
    #     "college_mathematics",
    #     "high_school_physics",
    #     "computer_security",
    #     "international_law",
    #     "moral_disputes",
    #     "high_school_us_history",
    #     "clinical_knowledge",
    # ]

    #     # -------- Training part
    # optimizer = torch.optim.SGD(model.parameters(),lr=1e-3,weight_decay=0.0)

    # n_samples = 2
    # base_state = {k: v.detach().clone() for k, v in model.state_dict().items()} # Our M_t
    # p_prob_before = eval_probe_logp(model)  # Go through the probing dataset, get all token probs, we might use it for every M_{t+1}
    # for i in tqdm(range(n_samples)):
    #     sample = processed_dataset[i]

    #     input_ids = sample["input_ids"].unsqueeze(0).to(device)          # [1, L]
    #     attention_mask = sample["attention_mask"].unsqueeze(0).to(device) # [1, L]
    #     labels = sample["labels"].unsqueeze(0).to(device)                 # [1, L]

    #     # Supervision position
    #     valid_mask = ((labels != -100) & (attention_mask == 1))  # [1, L]

    #     # Find all supervision token positions
    #     valid_positions = torch.nonzero(valid_mask[0], as_tuple=False).squeeze(-1)  # [num_valid]

    #     for pos in valid_positions.tolist():
    #         # ----- Update for each (s_u, y_u)
    #         model.load_state_dict(base_state) # Reset model to M_t
    #         model.train()
    #         # only keep current label, otherwise set to -100
    #         single_labels = torch.full_like(labels, -100)
    #         single_labels[0, pos] = labels[0, pos]

    #         outputs = model(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             labels=single_labels,
    #             output_hidden_states=False,
    #             output_attentions=False,
    #         )

    #         loss = outputs.loss

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()    

    #         # ---------- Now we have M_{t+1}
    #         p_prob_after = eval_probe_logp(model)
