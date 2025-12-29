import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, List
from utils.utils import extract_solution

class GSM8KTokenizerProcessor:
    def __init__(self, model_name_or_path: str, max_length: int = 512):
        """
        Args:
            model_name_or_path: 
            max_length: 
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="right",
            truncation_side="right",
            trust_remote_code=True,
            #use_fast=False 
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        self.answer_prefix = "Answer:\n"

    def _find_answer_start_token_idx(self, input_ids: List[int], prefix_tokens: List[int]) -> int:
        """
        Args:
            input_ids: 
            prefix_tokens:
        Returns:
        """
        prefix_len = len(prefix_tokens)
        for i in range(len(input_ids) - prefix_len + 1):
            if input_ids[i:i+prefix_len] == prefix_tokens:
                return i + prefix_len
            current_token_seq = input_ids[i:i+prefix_len]
            current_text = self.tokenizer.decode(current_token_seq)
            if current_text==self.answer_prefix:
                return i + prefix_len
        # import pdb
        # breakpoint()

        return -1

    def process_single_example(self, example: Dict) -> Dict:
        """
        Args:
            example:
        Returns:
            
        """
        # 1. input text
        question = example["question"].strip()
        answer = example["answer"].strip()
        instruction = 'Please answer the following question and give answer starting with "####".'
        full_text = f"{instruction}\nQuestion: {question}\n{self.answer_prefix}{answer}"
        rl_text = f"{instruction}\nQuestion: {question}\n{self.answer_prefix}"

        # 2. basic tokenize / SFT and RL
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        encoding_rl = self.tokenizer(
            rl_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        input_ids_rl = encoding_rl["input_ids"].squeeze(0)
        attention_mask_rl = encoding_rl["attention_mask"].squeeze(0)

        # 3. label mask
        prefix_encoding = self.tokenizer(
            self.answer_prefix,
            add_special_tokens=True,#False,
            return_tensors="pt"
        )
        prefix_tokens = prefix_encoding["input_ids"].squeeze(0).tolist()
        
        labels = torch.full_like(input_ids, fill_value=-100)
        
        input_ids_list = input_ids.tolist()
        answer_start_idx = self._find_answer_start_token_idx(input_ids_list, prefix_tokens)   
        if answer_start_idx != -1 and answer_start_idx < self.max_length:
            labels[answer_start_idx:] = input_ids[answer_start_idx:]
        # import pdb
        # breakpoint()        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_ids_rl": input_ids_rl,
            "attention_mask_rl": attention_mask_rl,
            "ground_truth": extract_solution(answer),
        }

    def process_dataset(self, dataset) -> List[Dict]:
        processed_data = []
        for example in dataset:
            try:
                processed = self.process_single_example(example)
                processed_data.append(processed)
            except Exception as e:
                print(f"worng case: {e}, example: {example}")
                continue
        return processed_data

if __name__ == "__main__":
    dataset = load_dataset("openai/gsm8k", "main", split="train[:10]")  
    processor = GSM8KTokenizerProcessor(
        model_name_or_path="Qwen/Qwen2.5-0.5B",
        max_length=512
    )
    
    processed_dataset = processor.process_dataset(dataset)

    sample = processed_dataset[0]
    print("===============")
    print(f"Example: \n{processor.tokenizer.decode(sample['input_ids'][:sample['attention_mask'].sum()], skip_special_tokens=True)}")
    N = torch.sum(sample['labels'] == -100).item()
    for i in range(N-5, N+5):
        text = processor.tokenizer.decode(sample['input_ids'][i])
        print(f"ID: {i} TXT: {text} IDS: {sample['input_ids'][i]} LAB: {sample['labels'][i]}")

