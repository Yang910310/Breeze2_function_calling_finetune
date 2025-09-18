import os
import json
import math
import random
import time
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model

import matplotlib.pyplot as plt


# -------------------------
# 可修改的參數
# -------------------------
model_name = 'PenutChen/Llama-Breeze2-3B-Instruct-Text'
train_jsonl_path = r"D:\research_information\github\Breeze2_function_calling_finetune\en_final.jsonl"
output_dir = r"D:\research_information\github\Breeze2_FC_finetune\lora_models\lora_03"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 1
batch_size = 1
learning_rate = 2e-5

seed = 42

# -------------------------
# 設定隨機種子（可重現）
# -------------------------
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

set_seed(seed)

# -------------------------
# 載入模型與 tokenizer（改動點 C）
# -------------------------
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
print("tokenizer type:", type(tokenizer))
# 設 pad_token，右側 padding
tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map='cuda',   
)

# 訓練期省顯存/避免多餘輸出
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False
if hasattr(model.config, "output_hidden_states"):
    model.config.output_hidden_states = False
if hasattr(model.config, "output_attentions"):
    model.config.output_attentions = False
# # 若硬體允許，可啟用flash_attention_2
# try:
#     model.config.attn_implementation = "flash_attention_2"
# except Exception:
#     pass

# # 開啟 Gradient Checkpointing
# if hasattr(model, "gradient_checkpointing_enable"):
#     model.gradient_checkpointing_enable()

model.train()

# -------------------------
# 讀取並格式化訓練資料
# -------------------------
def format_messages_as_text(messages):
    parts = []
    for m in messages:
        role = m.get("role", "").strip()
        content = m.get("content", "").strip()
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        parts.append(f"<|{role}|>\n{content}\n<|end|>")
    return "\n".join(parts)

print("Loading training examples from jsonl...")
training_texts = []
with open(train_jsonl_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception as e:
            print(f"Warning: json load failed on line {i}: {e}")
            continue

        messages = None
        if isinstance(obj, dict) and "messages" in obj and isinstance(obj["messages"], list):
            messages = obj["messages"]
        elif isinstance(obj, list):
            messages = obj
        else:
            if isinstance(obj, dict) and "instruction" in obj and "response" in obj:
                messages = [
                    {"role": "system", "content": obj.get("system", "")},
                    {"role": "user", "content": obj.get("instruction", "")},
                    {"role": "assistant", "content": obj.get("response", "")},
                ]
            else:
                print(f"Skipping unrecognized format at line {i}")
                continue

        text = format_messages_as_text(messages)
        training_texts.append(text)

print(f"Loaded {len(training_texts)} training examples.")

# -------------------------
# 將訓練資料進行 tokenization，方便後續處理
# -------------------------
def tokenize_text(text, tokenizer):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) == 0:
        return None
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": ids.copy(),
    }

print("Tokenizing samples...")
all_samples = []
for t in training_texts:
    sample = tokenize_text(t, tokenizer)
    if sample:
        all_samples.append(sample)

if len(all_samples) == 0:
    raise ValueError("No samples produced from training data.")

train_dataset = Dataset.from_list(all_samples)
print(f"Total samples: {len(train_dataset)}")

# -------------------------
# 動態補齊（padding）本 batch 內所有樣本，使其長度一致，
# 並將 padding 部分的 label 設為 -100，避免計算 loss。
# -------------------------
def dynamic_collate(batch):
    """
    以本 batch 內最長序列為基準動態 padding，
    並把 labels 在 pad 位置設為 -100，避免計算 loss。
    """
    # 找出本批最長長度
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids, attention_masks, labels = [], [], []
    for x in batch:
        ids = x["input_ids"]
        mask = x["attention_mask"]
        lbs = x["labels"]

        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids = ids + [tokenizer.pad_token_id] * pad_len
            mask = mask + [0] * pad_len
            lbs = lbs + [tokenizer.pad_token_id] * pad_len

        # 將 pad 的 label 標為 -100
        lbs = [(-100 if m == 0 else t) for t, m in zip(lbs, mask)]

        input_ids.append(ids)
        attention_masks.append(mask)
        labels.append(lbs)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=dynamic_collate
)

# -------------------------
# LoRA 設定
# -------------------------
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj"]
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# -------------------------
# Optimizer & Scheduler
# -------------------------
optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=learning_rate
)
num_training_steps = len(train_dataloader) * epochs
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# 將模型移動到指定設備(GPU)
model.to(device)


# -------------------------
# 訓練迴圈
# -------------------------
batch_loss_history = []
batch_losses = []

print("Start training...")
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Training epoch {epoch+1}")
    for step, batch in enumerate(progress_bar):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        step_loss = float(loss.item())
        total_loss += step_loss
        batch_losses.append(step_loss)
        progress_bar.set_postfix({"loss": f"{step_loss:.4f}"})

        # 每10個batch記錄一次平均loss
        if (step + 1) % 10 == 0:
            avg_batch_loss = sum(batch_losses) / len(batch_losses)
            batch_loss_history.append(avg_batch_loss)
            batch_losses = []

    # epoch結束後，若還有未記錄的batch loss
    if batch_losses:
        avg_batch_loss = sum(batch_losses) / len(batch_losses)
        batch_loss_history.append(avg_batch_loss)
        batch_losses = []

    avg_loss = total_loss / len(train_dataloader)
    epoch_loss_history.append(avg_loss)
    print(f"Epoch {epoch+1} training loss: {avg_loss:.6f}")

# -------------------------
# 儲存模型與 tokenizer
# -------------------------
os.makedirs(output_dir, exist_ok=True)
print(f"Saving model & tokenizer to {output_dir} ...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# -------------------------
# 繪製 Loss 曲線
# -------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(batch_loss_history) + 1), batch_loss_history, marker='o')
plt.xlabel("Batch (每10個batch紀錄一次)")
plt.ylabel("Training Loss")
plt.title("Training Loss per 10 Batches")
plt.grid(True)
plt.tight_layout()
plt_path = os.path.join(output_dir, "training_loss.png")
plt.savefig(plt_path)
print(f"Saved loss plot to: {plt_path}")

print("Training finished.")
