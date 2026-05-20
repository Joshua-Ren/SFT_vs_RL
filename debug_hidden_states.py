import argparse
import sys

import torch
from datasets import DownloadConfig, load_dataset
from transformers import AutoModelForCausalLM

from utils.config_read import load_config
from utils.gsm8k_tokenizer import GSM8KTokenizerProcessor
from utils.hidden_records import extract_token_hidden_records, print_hidden_record_debug


def parse_args():
    parser = argparse.ArgumentParser(
        description="Debug hidden-state extraction and causal position alignment."
    )
    parser.add_argument("--config", type=str, default="./configs/train_basic.yaml")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train[:1]")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--max_positions", type=int, default=8)
    parser.add_argument("--layers", type=str, default="-1")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--allow_download",
        action="store_true",
        help="Allow Hugging Face model/dataset downloads. Defaults to local cache only.",
    )
    return parser.parse_args()


def decode_token(tokenizer, token_id):
    text = tokenizer.decode([int(token_id)])
    return repr(text)


def print_alignment_table(tokenizer, input_ids, labels, attention_mask, max_positions):
    valid_mask = ((labels != -100) & (attention_mask == 1)).bool()
    label_positions = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1).tolist()
    rows = []

    for label_pos in label_positions:
        if label_pos == 0:
            continue
        hidden_pos = label_pos - 1
        rows.append(
            {
                "hidden_pos": hidden_pos,
                "input_token": decode_token(tokenizer, input_ids[hidden_pos]),
                "label_pos": label_pos,
                "label_token": decode_token(tokenizer, labels[label_pos]),
            }
        )
        if len(rows) >= max_positions:
            break

    print("\nPosition convention:")
    print("  hidden state at position p predicts the token at position p + 1.")
    print("  selected_hidden_pos = label_pos - 1 for supervised labels.")

    if not rows:
        print("\nNo supervised label positions found for this sample.")
        return

    print("\nSanity table:")
    print(f"{'hidden_pos':>10}  {'input_token':>18}  {'label_pos':>9}  {'label_token':>18}")
    for row in rows:
        print(
            f"{row['hidden_pos']:>10}  "
            f"{row['input_token'][:18]:>18}  "
            f"{row['label_pos']:>9}  "
            f"{row['label_token'][:18]:>18}"
        )


def main():
    args = parse_args()
    config = load_config(args.config)

    model_name = args.model_name or config["model"]["name"]
    max_length = config["model"]["max_length"]
    local_files_only = not args.allow_download

    try:
        download_config = DownloadConfig(local_files_only=local_files_only)
        dataset = load_dataset(
            config["dataset"]["name"],
            config["dataset"]["config"],
            split=args.dataset_split,
            download_config=download_config,
        )

        processor = GSM8KTokenizerProcessor(
            model_name_or_path=model_name,
            max_length=max_length,
            local_files_only=local_files_only,
        )
        tokenizer = processor.tokenizer
    except OSError as exc:
        mode = "local cache only" if local_files_only else "download allowed"
        print(
            "\nFailed to load the dataset or tokenizer from Hugging Face "
            f"({mode}).\n"
            "By default this debug script avoids network downloads. If this "
            "model and dataset are not already cached, either run with "
            "`--allow_download` on a machine where downloads are approved, or "
            "pass `--model_name` for a cached/local model path.\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    processed_dataset = processor.process_dataset(dataset)
    if not processed_dataset:
        raise ValueError("No processed samples were produced.")
    if args.sample_index < 0 or args.sample_index >= len(processed_dataset):
        raise IndexError(
            f"sample_index={args.sample_index} is out of range for {len(processed_dataset)} samples."
        )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
        ).to(device)
    except OSError as exc:
        mode = "local cache only" if local_files_only else "download allowed"
        print(
            "\nFailed to load the model from Hugging Face "
            f"({mode}).\n"
            "By default this debug script avoids network downloads. If this "
            "model is not already cached, either run with `--allow_download` "
            "on a machine where downloads are approved, or pass `--model_name` "
            "for a cached/local model path.\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    model.eval()

    sample = processed_dataset[args.sample_index]
    batch = {
        "input_ids": sample["input_ids"].unsqueeze(0).to(device),
        "attention_mask": sample["attention_mask"].unsqueeze(0).to(device),
        "labels": sample["labels"].unsqueeze(0).to(device),
        "sample_index": torch.tensor([args.sample_index], device=device),
    }

    metadata, hidden_by_layer = extract_token_hidden_records(
        model=model,
        batch=batch,
        tokenizer=tokenizer,
        layers=args.layers,
    )

    print(f"model_name: {model_name}")
    print(f"dataset_split: {args.dataset_split}")
    print(f"sample_index: {args.sample_index}")
    print(f"input_ids shape: {tuple(batch['input_ids'].shape)}")
    print(f"attention_mask shape: {tuple(batch['attention_mask'].shape)}")
    print(f"labels shape: {tuple(batch['labels'].shape)}")

    print_alignment_table(
        tokenizer=tokenizer,
        input_ids=batch["input_ids"][0].cpu(),
        labels=batch["labels"][0].cpu(),
        attention_mask=batch["attention_mask"][0].cpu(),
        max_positions=args.max_positions,
    )
    print_hidden_record_debug(metadata, hidden_by_layer, max_metadata=10)


if __name__ == "__main__":
    main()
