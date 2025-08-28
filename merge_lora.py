import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 設定路徑
base_model_name = "PenutChen/Llama-Breeze2-3B-Instruct-Text"
adapter_dir = r"d:\research_information\Breeze2_FC_finetune\finetuned_models"
output_dir = r"d:\research_information\Breeze2_FC_finetune\merged_model"

os.makedirs(output_dir, exist_ok=True)

# 載入原始模型與 LoRA adapter
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype="auto",
    trust_remote_code=True,
    device_map="cuda"
)
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_dir)

# Merge LoRA weights
print("Merging LoRA adapter into base model...")
merged_model = model.merge_and_unload()

# 儲存合併後的模型
print(f"Saving merged model to {output_dir} ...")
merged_model.save_pretrained(output_dir, safe_serialization=True)  # 會產生 .safetensors 和 config.json

# 儲存 tokenizer
print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
tokenizer.save_pretrained(output_dir)

print("Done! 合併後的模型與 tokenizer 已儲存。")