from trl import SFTTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from utils.config_read import *
from utils.utils import *
from utils.gsm8k_tokenizer import GSM8KTokenizerProcessor
import torch


# def tokenize_function(examples):
#     return tokenizer(examples["text"], truncation=True, max_length=config["tokenizer"]["max_length"], padding="max_length")

config = load_config("./configs/train_basic.yaml")
dataset = load_dataset(config['dataset']['name'],config['dataset']['config'], split="train")
processor = GSM8KTokenizerProcessor(
    model_name_or_path=config['model']['name'],
    max_length=config['model']['max_length']
)
processed_dataset = processor.process_dataset(dataset)

# ================= Get forward pass, calculate stats of the logp

model = AutoModelForCausalLM.from_pretrained(config['model']['name'])
sample = processed_dataset[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  
inputs = {
    "input_ids": sample["input_ids"].unsqueeze(0).to(device),  # 形状：[1, max_length]
    "attention_mask": sample["attention_mask"].unsqueeze(0).to(device),  # 形状：[1, max_length]
    "labels": sample["labels"].unsqueeze(0).to(device)  # 训练时传，推理时可省略
}

model.train()
with torch.no_grad(): 
    outputs = model(**inputs, output_hidden_states=False, output_attentions=False)
    loss = outputs.loss  
    logits = outputs.logits  
    print(f"训练模式 - 损失值: {loss.item():.4f}")
    print(f"Logits形状: {logits.shape}")










# # ----- Get logits
# i=0

# with torch.no_grad():
#     outputs = model(**inputs, output_hidden_states=False, output_attentions=False)
#     all_logits = outputs.logits  # [batch_size, seq_len, vocab_size]

import pdb
breakpoint()