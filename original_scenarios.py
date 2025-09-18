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

# 設定要處理的資料數量(不等於最終訓練資料集的筆數，因為有些資料會被過濾掉)
NUM_SAMPLES = 1000

# 載入公開資料集
ds = load_dataset("glaiveai/glaive-function-calling-v2")
print("Dataset loaded successfully.")
print(ds)

# 隨機選取 NUM_SAMPLES 筆資料
sampled_data = ds["train"].shuffle(seed=42).select(range(NUM_SAMPLES))

# 讀取已存在的 JSON 檔案（如果有）
file_path = r"D:\research_information\github\Breeze2_function_calling_finetune\scenarios.jsonl"
if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        all_outputs = json.load(json_file)
else:
    all_outputs = []


# 逐筆處理
for i, entry in enumerate(sampled_data):  
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
                "index":i,
                "model_output": response.output_text
            }
            all_outputs.append(output_data)
            # start_index += 1  # 遞增 index
        else:
            print(f"Skipping entry {i}: Invalid structure or missing 'chat' field.")
    except Exception as e:
        print(f"Error processing entry {i}: {e}")

# 確保有資料才寫入 JSON 檔案
if all_outputs:
    with open(file_path, "a", encoding="utf-8") as json_file:
        json.dump(all_outputs, json_file, ensure_ascii=False, indent=4)
    print("所有模型輸出已儲存至 scenarios.jsonl")
else:
    print("沒有資料被處理，未生成 scenarios.jsonl")