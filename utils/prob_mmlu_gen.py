from datasets import load_dataset
from collections import defaultdict
import random


def format_mmlu_prompt(question, choices):
    return (
        "The following is a multiple choice question.\n\n"
        f"Question: {question}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n\n"
        "Answer:"
    )


def check_answer_tokens(tokenizer):
    for x in [" A", " B", " C", " D"]:
        ids = tokenizer.encode(x, add_special_tokens=False)
        print(f"{repr(x)} -> ids={ids}, n_tokens={len(ids)}")


def build_mmlu_probe_dataset(
    tokenizer,
    subjects,
    n_per_subject=100,
    split="test",
    seed=42,
):
    """
    Returns:
        probe_items: list[dict]
        probe_by_domain: dict[str, list[dict]]
    """
    rng = random.Random(seed)
    probe_items = []

    for subject in subjects:
        ds = load_dataset("cais/mmlu", subject, split=split)

        indices = list(range(len(ds)))
        rng.shuffle(indices)
        indices = indices[:n_per_subject]

        for local_id, idx in enumerate(indices):
            ex = ds[idx]

            question = ex["question"]
            choices = ex["choices"]
            answer_idx = ex["answer"]

            answer_letter = ["A", "B", "C", "D"][answer_idx]
            prompt_text = format_mmlu_prompt(question, choices)
            target_text = " " + answer_letter

            probe_items.append({
                "probe_id": f"{subject}_{split}_{idx}",
                "domain": subject,
                "source_subject": subject,
                "source_split": split,
                "source_idx": idx,
                "question": question,
                "choices": choices,
                "answer_idx": answer_idx,
                "answer_letter": answer_letter,
                "prompt_text": prompt_text,
                "target_text": target_text,
            })

    probe_by_domain = defaultdict(list)
    for item in probe_items:
        probe_by_domain[item["domain"]].append(item)

    return probe_items, dict(probe_by_domain)


def tokenize_probe_item(tokenizer, item, device=None):
    """
    Tokenize one probe item for teacher-forcing log-prob evaluation.
    We concatenate:
        prompt_text + target_text
    and later only evaluate target_text tokens.
    """
    full_text = item["prompt_text"] + item["target_text"]

    prompt_ids = tokenizer.encode(item["prompt_text"], add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    target_ids = full_ids[len(prompt_ids):]

    out = {
        "probe_id": item["probe_id"],
        "domain": item["domain"],
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "target_token_ids": target_ids,
        "target_start": len(prompt_ids),
        "target_end": len(full_ids),
        "prompt_text": item["prompt_text"],
        "target_text": item["target_text"],
    }

    if device is not None:
        import torch
        out["input_ids"] = torch.tensor(out["input_ids"], dtype=torch.long, device=device)
        out["attention_mask"] = torch.tensor(out["attention_mask"], dtype=torch.long, device=device)
        out["target_token_ids"] = torch.tensor(out["target_token_ids"], dtype=torch.long, device=device)

    return out


def tokenize_probe_dataset(tokenizer, probe_items, device=None):
    tokenized_items = [tokenize_probe_item(tokenizer, x, device=device) for x in probe_items]

    probe_by_domain = defaultdict(list)
    for item in tokenized_items:
        probe_by_domain[item["domain"]].append(item)

    return tokenized_items, dict(probe_by_domain)


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)

    subjects = [
        "abstract_algebra",
        "college_mathematics",
        "high_school_physics",
        "computer_security",
        "international_law",
        "moral_disputes",
        "high_school_us_history",
        "clinical_knowledge",
    ]

    print("Checking answer tokenization:")
    check_answer_tokens(tokenizer)

    probe_items, probe_by_domain = build_mmlu_probe_dataset(
        tokenizer=tokenizer,
        subjects=subjects,
        n_per_subject=100,
        split="test",
        seed=42,
    )

    print(f"Total probe items: {len(probe_items)}")
    for k, v in probe_by_domain.items():
        print(k, len(v))

    tokenized_items, tokenized_by_domain = tokenize_probe_dataset(
        tokenizer=tokenizer,
        probe_items=probe_items,
        device=None,
    )

    print("\nExample item:")
    print(tokenized_items[0]["probe_id"])
    print(tokenized_items[0]["domain"])
    print(tokenized_items[0]["target_text"])
    print(tokenized_items[0]["target_token_ids"])
    import pdb
    breakpoint()

def load_processed_dataset(tokenizer, subjects, n_per_subject=100):
    probe_items, _ = build_mmlu_probe_dataset(
        tokenizer=tokenizer,
        subjects=subjects,
        n_per_subject=n_per_subject,
        split="test",
        seed=42,
    )
    tokenized_items, tokenized_by_domain = tokenize_probe_dataset(
        tokenizer=tokenizer,
        probe_items=probe_items,
        device=None,
    )
    return tokenized_items, tokenized_by_domain
