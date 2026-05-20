import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from utils.config_read import load_config
from utils.hh_dataset_adapters import make_hh_dataset_adapter, parse_csv_arg
from utils.hidden_records import extract_token_hidden_records
from utils.hh_metrics import (
    compute_hidden_pair_metrics,
    compute_pair_metrics_for_layer,
    pair_metrics_to_rows,
    summarize_pair_metrics,
)
from utils.hh_sampling import sample_pair_indices_from_metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run offline HH hidden-state alignment analysis."
    )
    parser.add_argument("--config", type=str, default="./configs/train_basic.yaml")
    parser.add_argument("--output_dir", type=str, default="./out/hh_analysis_debug")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train[:2]")
    parser.add_argument(
        "--hh_dataset_adapter",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "mmlu", "dolly"],
        help="Dataset adapter used to build supervised token labels for HH analysis.",
    )
    parser.add_argument(
        "--mmlu_subjects",
        type=str,
        default=None,
        help="Comma-separated MMLU subject list. Defaults to the built-in small subject set.",
    )
    parser.add_argument("--max_samples", type=int, default=2)
    parser.add_argument("--layers", type=str, default="-1")
    parser.add_argument("--max_pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--pair_scope",
        type=str,
        default="all",
        choices=["all", "same", "cross"],
        help="Whether to sample all, same-sample, or cross-sample record pairs.",
    )
    parser.add_argument("--include_self", action="store_true")
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--debug_token_alignment", action="store_true")
    parser.add_argument("--save_pair_samples", action="store_true")
    parser.add_argument("--save_debug_tensors", action="store_true")
    parser.add_argument(
        "--allow_download",
        action="store_true",
        help="Allow Hugging Face model/dataset downloads. Defaults to local cache only.",
    )
    return parser.parse_args()


def pair_scope_to_same_sample(pair_scope):
    if pair_scope == "all":
        return None
    if pair_scope == "same":
        return True
    if pair_scope == "cross":
        return False
    raise ValueError(f"Unknown pair_scope: {pair_scope}")


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metadata_value(metadata, index, key):
    value = metadata[index].get(key)
    if isinstance(value, torch.Tensor):
        return value.item() if value.dim() == 0 else value.tolist()
    return value


def attach_pair_metadata(metric_rows, metadata):
    rows = []
    metadata_keys = [
        "sample_index",
        "example_id",
        "sequence_index",
        "hidden_pos",
        "label_pos",
        "input_token_id",
        "label_token_id",
        "input_token_str",
        "label_token_str",
        "hidden_attention_mask",
        "label_attention_mask",
        "domain",
    ]
    for row in metric_rows:
        out = dict(row)
        left_index = row["left_index"]
        right_index = row["right_index"]
        for key in metadata_keys:
            out[f"left_{key}"] = metadata_value(metadata, left_index, key)
            out[f"right_{key}"] = metadata_value(metadata, right_index, key)
        rows.append(out)
    return rows


def add_metric_variant(rows, variant):
    out = []
    for row in rows:
        updated = dict(row)
        updated["metric_variant"] = variant
        out.append(updated)
    return out


def compute_centered_pair_metrics(hidden_by_layer, pair_indices):
    centered = {}
    for layer_idx, hidden in hidden_by_layer.items():
        hidden_cpu = hidden.detach().cpu().float()
        hidden_centered = hidden_cpu - hidden_cpu.mean(dim=0, keepdim=True)
        centered[layer_idx] = compute_pair_metrics_for_layer(hidden_centered, pair_indices)
    return centered


def make_token_alignment_rows(metadata, max_rows=20):
    rows = []
    for row in metadata[:max_rows]:
        rows.append(
            {
                "example_id": row.get("example_id", row.get("sample_index")),
                "hidden_pos": row["hidden_pos"],
                "target_pos": row["label_pos"],
                "input_token_at_hidden_pos": row["input_token_str"],
                "target_token": row["label_token_str"],
                "label_value": row["label_token_id"],
                "hidden_attention_mask": row.get("hidden_attention_mask"),
                "target_attention_mask": row.get("label_attention_mask"),
                "position_policy": "causal_lm_next_token: hidden_pos = target_pos - 1",
            }
        )
    return rows


def make_pair_sample_rows(metadata, pair_indices, pair_scope, pair_metrics, max_rows=20):
    pairs = pair_indices[:max_rows].detach().cpu()
    final_layer = max(pair_metrics.keys()) if pair_metrics else None
    rows = []

    for pair_offset, (left, right) in enumerate(pairs.tolist()):
        left_row = metadata[left]
        right_row = metadata[right]
        out = {
            "pair_type": pair_scope,
            "i": left,
            "j": right,
            "token_i": left_row.get("label_token_str"),
            "token_j": right_row.get("label_token_str"),
            "token_id_i": left_row.get("label_token_id"),
            "token_id_j": right_row.get("label_token_id"),
            "example_id_i": left_row.get("example_id", left_row.get("sample_index")),
            "example_id_j": right_row.get("example_id", right_row.get("sample_index")),
            "pos_i": left_row.get("label_pos"),
            "pos_j": right_row.get("label_pos"),
            "domain_i": left_row.get("domain"),
            "domain_j": right_row.get("domain"),
        }
        if final_layer is not None:
            out["raw_cosine_layer_minus_1"] = float(pair_metrics[final_layer]["cosine"][pair_offset].item())
            out["raw_cosine_layer_index"] = final_layer
        rows.append(out)

    return rows


def save_debug_tensors(output_dir, hidden_by_layer, pair_indices):
    layer_idx = max(hidden_by_layer.keys())
    torch.save(hidden_by_layer[layer_idx], output_dir / f"debug_hidden_layer_{layer_idx}.pt")
    torch.save(pair_indices.detach().cpu(), output_dir / "debug_pair_indices.pt")


def load_dataset_and_tokenizer(
    config,
    model_name,
    max_length,
    local_files_only,
    adapter_name,
    mmlu_subjects=None,
):
    try:
        adapter = make_hh_dataset_adapter(
            adapter_name=adapter_name,
            model_name_or_path=model_name,
            max_length=max_length,
            config=config,
            local_files_only=local_files_only,
            mmlu_subjects=mmlu_subjects,
        )
        dataset = adapter.load_dataset(
            split=config["runtime_dataset_split"],
            local_files_only=local_files_only,
        )
        return dataset, adapter
    except OSError as exc:
        mode = "local cache only" if local_files_only else "download allowed"
        print(
            "\nFailed to load the dataset or tokenizer from Hugging Face "
            f"({mode}).\n"
            "By default this runner avoids network downloads. If this model "
            "and dataset are not already cached, rerun with `--allow_download` "
            "on a machine where downloads are approved, or pass `--model_name` "
            "for a cached/local model path.\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def load_model(model_name, local_files_only, device):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
        ).to(device)
        model.eval()
        return model
    except OSError as exc:
        mode = "local cache only" if local_files_only else "download allowed"
        print(
            "\nFailed to load the model from Hugging Face "
            f"({mode}).\n"
            "By default this runner avoids network downloads. If this model "
            "is not already cached, rerun with `--allow_download` on a machine "
            "where downloads are approved, or pass `--model_name` for a cached/local model path.\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def collect_hidden_records(model, tokenizer, processed_dataset, max_samples, layers, device):
    all_metadata = []
    hidden_chunks_by_layer = {}
    n_samples = min(max_samples, len(processed_dataset))

    for sample_index in range(n_samples):
        sample = processed_dataset[sample_index]
        batch = {
            "input_ids": sample["input_ids"].unsqueeze(0).to(device),
            "attention_mask": sample["attention_mask"].unsqueeze(0).to(device),
            "labels": sample["labels"].unsqueeze(0).to(device),
            "sample_index": torch.tensor([sample_index], device=device),
        }
        if "example_id" in sample:
            batch["example_id"] = [sample["example_id"]]
        if "domain" in sample:
            batch["domain"] = [sample["domain"]]
        metadata, hidden_by_layer = extract_token_hidden_records(
            model=model,
            batch=batch,
            tokenizer=tokenizer,
            layers=layers,
        )
        all_metadata.extend(metadata)
        for layer_idx, hidden in hidden_by_layer.items():
            hidden_chunks_by_layer.setdefault(layer_idx, []).append(hidden)

    hidden_by_layer = {
        layer_idx: torch.cat(chunks, dim=0) if chunks else torch.empty(0, 0)
        for layer_idx, chunks in hidden_chunks_by_layer.items()
    }
    return all_metadata, hidden_by_layer


def main():
    args = parse_args()
    config = load_config(args.config)
    config["runtime_dataset_split"] = args.dataset_split
    adapter_name = args.hh_dataset_adapter or config.get("hh_dataset_adapter", "gsm8k")
    mmlu_subjects = parse_csv_arg(args.mmlu_subjects)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model_name or config["model"]["name"]
    max_length = config["model"]["max_length"]
    local_files_only = not args.allow_download
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset, processor = load_dataset_and_tokenizer(
        config=config,
        model_name=model_name,
        max_length=max_length,
        local_files_only=local_files_only,
        adapter_name=adapter_name,
        mmlu_subjects=mmlu_subjects,
    )
    processed_dataset = processor.process_dataset(dataset)
    if not processed_dataset:
        raise ValueError("No processed samples were produced.")

    model = load_model(
        model_name=model_name,
        local_files_only=local_files_only,
        device=device,
    )

    metadata, hidden_by_layer = collect_hidden_records(
        model=model,
        tokenizer=processor.tokenizer,
        processed_dataset=processed_dataset,
        max_samples=args.max_samples,
        layers=args.layers,
        device=device,
    )
    if not metadata:
        raise ValueError("No hidden records were collected.")

    pair_indices = sample_pair_indices_from_metadata(
        metadata=metadata,
        max_pairs=args.max_pairs,
        seed=args.seed,
        include_self=args.include_self,
        directed=args.directed,
        same_sample=pair_scope_to_same_sample(args.pair_scope),
    )
    pair_metrics = compute_hidden_pair_metrics(hidden_by_layer, pair_indices)
    centered_pair_metrics = compute_centered_pair_metrics(hidden_by_layer, pair_indices)
    summaries = {
        "raw": summarize_pair_metrics(pair_metrics),
        "centered": summarize_pair_metrics(centered_pair_metrics),
    }
    metric_rows = attach_pair_metadata(
        add_metric_variant(pair_metrics_to_rows(pair_metrics), "raw")
        + add_metric_variant(pair_metrics_to_rows(centered_pair_metrics), "centered"),
        metadata=metadata,
    )

    run_config = {
        "model_name": model_name,
        "hh_dataset_adapter": adapter_name,
        "mmlu_subjects": mmlu_subjects,
        "dataset": config["dataset"],
        "dataset_split": args.dataset_split,
        "max_samples": args.max_samples,
        "layers": args.layers,
        "max_pairs": args.max_pairs,
        "seed": args.seed,
        "pair_scope": args.pair_scope,
        "include_self": args.include_self,
        "directed": args.directed,
        "local_files_only": local_files_only,
        "num_records": len(metadata),
        "num_pairs": int(pair_indices.shape[0]),
        "selected_layers": sorted(hidden_by_layer.keys()),
    }

    write_json(output_dir / "run_config.json", run_config)
    write_json(output_dir / "summary.json", summaries)
    write_jsonl(output_dir / "metadata.jsonl", metadata)
    write_csv(output_dir / "pair_metrics.csv", metric_rows)

    if args.debug_token_alignment:
        token_alignment_rows = make_token_alignment_rows(metadata, max_rows=20)
        write_jsonl(output_dir / "debug_token_alignment.jsonl", token_alignment_rows)
        print("Debug token alignment rows:")
        for row in token_alignment_rows:
            print(row)

    if args.save_pair_samples:
        write_jsonl(
            output_dir / "debug_pair_samples.jsonl",
            make_pair_sample_rows(
                metadata=metadata,
                pair_indices=pair_indices,
                pair_scope=args.pair_scope,
                pair_metrics=pair_metrics,
                max_rows=20,
            ),
        )

    if args.save_debug_tensors:
        save_debug_tensors(output_dir, hidden_by_layer, pair_indices)

    print(f"Saved HH analysis outputs to {output_dir}")
    print(f"num_records: {len(metadata)}")
    print(f"num_pairs: {int(pair_indices.shape[0])}")
    print(f"selected_layers: {sorted(hidden_by_layer.keys())}")


if __name__ == "__main__":
    main()
