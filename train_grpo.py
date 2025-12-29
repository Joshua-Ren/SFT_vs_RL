from trl import SFTTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from utils.config_read import *
from utils.utils import *
from utils.gsm8k_tokenizer import GSM8KTokenizerProcessor
import torch
from tqdm import tqdm
import gc

config = load_config("./configs/train_basic.yaml")
TEMP = 1
SAMPLE_list = [True, False]
model = AutoModelForCausalLM.from_pretrained(config['model']['name'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  
model.eval()

for i in range(2):
    SAMPLE = SAMPLE_list[i]
    if not SAMPLE:
        EXP_NAME = "rl_greedy"
    else:
        EXP_NAME = f"rl_tmp{TEMP}"

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

    with torch.no_grad():
        batch_valid_logits = []
        batch_valid_labels = []
        for i in tqdm(range(50)):
            sample = processed_dataset[i]
            # ----- For RL
            inputs = {
                "input_ids": sample["input_ids_rl"].unsqueeze(0).to(device),  # 形状：[1, max_length]
                "attention_mask": sample["attention_mask_rl"].unsqueeze(0).to(device),  # 形状：[1, max_length]
            }
            prompt_len = inputs["input_ids"].shape[1]

            tokenizer = processor.tokenizer
            with torch.no_grad():
                generate_outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,  # 限制生成的answer长度
                    do_sample=SAMPLE,  # 贪心生成（on-policy常用，也可设为True开启采样）
                    temperature=TEMP,
                    top_p=1.0,
                    output_scores=True,  # 必须开启，才能获取生成过程的logits
                    return_dict_in_generate=True,  # 必须开启，返回结构化结果
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
            )
            full_generated_ids = generate_outputs.sequences[0]  # 完整生成的token_ids [full_len]
            generated_text = tokenizer.decode(full_generated_ids, skip_special_tokens=True)
            prefix_tokens = tokenizer(processor.answer_prefix,add_special_tokens=False,return_tensors="pt").input_ids[0].tolist()
            response_start_idx = prompt_len  # Usually OK, if not, use the following code to get response_start_idx
            # answer_start_idx = processor._find_answer_start_token_idx(full_generated_ids.tolist(), prefix_tokens)
            # if response_start_idx == -1:
            #     response_start_idx = prompt_len

            response_logits = torch.cat(generate_outputs.scores, dim=0)  # [generated_len, V]

            response_ids = full_generated_ids[response_start_idx-1:-1]   # 和SFT部分的shift兼容
            response = tokenizer.decode(full_generated_ids[response_start_idx:], skip_special_tokens=True).strip()

            batch_valid_logits.append(response_logits.cpu())
            batch_valid_labels.append(response_ids.cpu())

            # ----------- Check whether the sampled response is correct or not
                # 先不管了，命中率太低。
            gen_solution = extract_solution(response)
            gt_solution = sample['ground_truth']
            # import pdb
            # breakpoint()

    import os
    if not os.path.exists(SCRATCH_PATH):
        os.makedirs(SCRATCH_PATH)

    torch.save({
        'rl_logits': batch_valid_logits,
        'rl_labels': batch_valid_labels
    }, f'{SCRATCH_PATH}{EXP_NAME}_logits.pt')




#         # outputs = model(**inputs, output_hidden_states=False, output_attentions=False)
#         # loss = outputs.loss
#         # logits = outputs.logits     #[B, L, V]

#         B, L, V = logits.shape
#         labels = inputs["labels"]   #[B, L]
#         valid_mask = ((labels!=-100)*inputs["attention_mask"]).bool()
#         for bb in range(logits.shape[0]):
#             batch_mask = valid_mask[bb]
#             batch_logits = logits[:, batch_mask, :].squeeze(0)
#             batch_valid_logits.append(batch_logits.cpu())

#             batch_label = labels[bb, batch_mask]
#             batch_valid_labels.append(batch_label.cpu())
# torch.save({
#     'rl_logits': batch_valid_logits,
#     'rl_labels': batch_valid_labels
# }, 'rl_logits.pt')


# # ----- Get logits
# i=0

# with torch.no_grad():
#     outputs = model(**inputs, output_hidden_states=False, output_attentions=False)
#     all_logits = outputs.logits  # [batch_size, seq_len, vocab_size]

# import pdb
# breakpoint()