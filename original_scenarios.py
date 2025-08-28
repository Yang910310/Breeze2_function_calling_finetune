import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from datasets import load_dataset

'''
從英文開源資料集中，利用AI讀取真實互動情境
'''

# 載入 .env 檔案中的環境變數
load_dotenv()

# 從環境變數中讀取 API 金鑰
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# 載入資料集
ds = load_dataset("glaiveai/glaive-function-calling-v2")
print("Dataset loaded successfully.")
print(ds)

# 讀取已存在的 JSON 檔案（如果有）
file_path = r"D:\research_information\Breeze2_FC_finetune\data\english\training\scenario_training_02.json"
if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        all_outputs = json.load(json_file)
else:
    all_outputs = []

# 取得目前的最大 index
start_index = all_outputs[-1]["index"] + 1 if all_outputs else 0

# 逐筆處理1000筆資料
for i, entry in enumerate(ds["train"].select(range(80000,81000))): 
    try:
        # 確保資料結構正確
        if isinstance(entry, dict) and "chat" in entry:
            conversation = entry["chat"]
            print(f"Processing entry {i}: {conversation}")

            # 定義輸入 prompt
            input_prompt = (
                "Please analyze the conversation below between a user and an "
                "assistant bot and identify the general life scenario it "
                "represents. Provide a concise overview of the scenario type, "
                "such as ’booking flights’ or ’ordering meals’. Avoid "
                "mentioning specific details like numbers or items. Your "
                "response should be a description of the scenario without "
                "additional commentary, and should not exceed 10 words.\n"
                f"Conversation:\n{conversation}\n"
                "Concise Overview of the Scenario:"
            )

            # 呼叫 OpenAI 模型
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=input_prompt,
                max_output_tokens=50
            )

            # 將每筆資料的輸出儲存到列表中
            output_data = {
                "index": start_index,
                "model_output": response.output_text
            }
            all_outputs.append(output_data)
            start_index += 1  # 遞增 index
        else:
            print(f"Skipping entry {i}: Invalid structure or missing 'chat' field.")
    except Exception as e:
        print(f"Error processing entry {i}: {e}")

# 確保有資料才寫入 JSON 檔案
if all_outputs:
    with open(file_path, "a", encoding="utf-8") as json_file:
        json.dump(all_outputs, json_file, ensure_ascii=False, indent=4)
    print("所有模型輸出已儲存至 scenario_training.json")
else:
    print("沒有資料被處理，未生成 scenario_training.json")