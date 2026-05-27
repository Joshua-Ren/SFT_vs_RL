import os
from transformers import AutoConfig, AutoTokenizer

# 1. 这里的 Token 必须是从 https://huggingface.co/settings/tokens 获取的 "Read" 或 "Write" 类型
my_token = "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" 

model_id = "meta-llama/Llama-3.2-3B-Instruct"

try:
    print(f"正在尝试使用 Token 访问 {model_id}...")
    # 强制不使用缓存，直接握手服务器
    config = AutoConfig.from_pretrained(
        model_id, 
        token=my_token, 
        force_download=True
    )
    print("✅ 验证成功！Config 已成功获取。")
    
    # 验证你最担心的 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=my_token)
    print(f"✅ Tokenizer 加载成功！词表大小: {len(tokenizer)}")
    
except Exception as e:
    print(f"❌ 依然报错。具体原因：\n{e}")