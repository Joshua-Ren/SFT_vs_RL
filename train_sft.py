from trl import SFTTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from utils.config_read import *
from utils.utils import *
from utils.gsm8k_tokenizer import GSM8KTokenizerProcessor
import torch
from tqdm import tqdm

config = load_config("./configs/train_basic.yaml")
 
MODEL = config['model']['name'].split('/')[1]
SCRATCH_PATH = f'/scratch/joshua52/sft_rl_temp/{MODEL}/'

# def tokenize_function(examples):
#     return tokenizer(examples["text"], truncation=True, max_length=config["tokenizer"]["max_length"], padding="max_length")

dataset = load_dataset(config['dataset']['name'],config['dataset']['config'], split="train")
processor = GSM8KTokenizerProcessor(
    model_name_or_path=config['model']['name'],
    max_length=config['model']['max_length']
)
processed_dataset = processor.process_dataset(dataset)

# ================= Get forward pass, calculate stats of the logp

model = AutoModelForCausalLM.from_pretrained(config['model']['name'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  
model.eval()
with torch.no_grad():
    batch_valid_logits = []
    batch_valid_labels = []
    for i in tqdm(range(50)):
        sample = processed_dataset[i]
        inputs = {
            "input_ids": sample["input_ids"].unsqueeze(0).to(device),  # 形状：[1, max_length]
            "attention_mask": sample["attention_mask"].unsqueeze(0).to(device),  # 形状：[1, max_length]
            "labels": sample["labels"].unsqueeze(0).to(device)  # 训练时传，推理时可省略
        }

        outputs = model(**inputs, output_hidden_states=False, output_attentions=False)
        loss = outputs.loss
        logits = outputs.logits     #[B, L, V]

        B, L, V = logits.shape
        labels = inputs["labels"]   #[B, L]
        valid_mask = ((labels!=-100)*inputs["attention_mask"]).bool()
        for bb in range(logits.shape[0]):
            batch_mask = valid_mask[bb]
            batch_logits = logits[:, batch_mask, :].squeeze(0)
            batch_valid_logits.append(batch_logits.cpu())

            batch_label = labels[bb, batch_mask]
            batch_valid_labels.append(batch_label.cpu())

import os
if not os.path.exists(SCRATCH_PATH):
    os.makedirs(SCRATCH_PATH)

torch.save({
    'sft_logits': batch_valid_logits,
    'sft_labels': batch_valid_labels
}, f'{SCRATCH_PATH}sft_logits.pt')
# # ----- Get logits
# i=0

# with torch.no_grad():
#     outputs = model(**inputs, output_hidden_states=False, output_attentions=False)
#     all_logits = outputs.logits  # [batch_size, seq_len, vocab_size]

# import pdb
# breakpoint()