import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from datasets import load_dataset

# 載入 .env 檔案中的環境變數
load_dotenv()

# 從環境變數中讀取 API 金鑰
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# 讀取已存在的 JSON 檔案（如果有）
file_path = "computer_scenarios_training.json"
if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        all_outputs = json.load(json_file)
else:
    all_outputs = []

# 取得目前的最大 index
start_index = all_outputs[-1]["index"] + 1 if all_outputs else 0

# 要執行的生成次數與每次生成的筆數
generate_times = 2
batch_size = 250

# 定義 prompt（固定）
input_prompt = (
    "你正在訓練一個模型，這個模型可以接收使用者的任務描述或查詢，以及可用的函數作為輸入，然後產生一連串的函數呼叫來完成該任務。"
    "目前，你正在產生簡單任務。"
    f"請生成{batch_size}筆使用者在日常生活中操作電腦的簡短任務，需符合以下條件：\n"
    "1.任務應為真實世界會發生，不接受虛構或誇張的情境任務，且能在2步驟內可完成。\n"
    "2.如果在任務中提到某些資訊、標準或限制，請提供這些資訊、標準或限制的詳細內容。不要假設模型能存取您的個人資訊或先驗知識，也不要假設模型有機會向您進一步澄清。\n"
    "3.請提供足夠的細節，並使任務描述盡可能具體，以便模型能以確定性的函數調用和參數進行操作。不要包含任何模糊或不明確的資訊。\n"
    "4.在任務描述中不要提及具體的工具或功能，也不要提出解決方案、提示或項目結果。\n"
    "5.為避免資料單一，請讓任務類型多元化。不要過度集中在特定類型的任務（如資料夾操作、檔案複製、刪除等），這類任務請控制在50筆以內。請包含不同面向的電腦使用情境，如瀏覽網頁、操作設定、啟動應用程式、搜尋資訊等。\n"
    "5.將任務描述限制在30個字內，避免使用形容詞和模糊的詞。\n"
    "6.任務中若需提及檔案或資料夾名稱，請使用多樣化的名稱，不要反覆使用相同的詞彙或範例名稱。\n"
    "6.每個任務用換行區分，開頭不要產生編號。"
)

# 逐次生成並追加資料
for round_num in range(generate_times):
    print(f"正在生成第 {round_num + 1} 次資料...")

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=input_prompt,
        max_output_tokens=15000
    )

    # 分行轉成清單
    scenarios = response.output_text.strip().split("\n")

    for scenario in scenarios:
        cleaned = scenario.strip()
        if cleaned:
            all_outputs.append({
                "index": start_index,
                "scenario": cleaned
            })
            start_index += 1  # 遞增 index

# 寫回 JSON 檔案
if all_outputs:
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(all_outputs, json_file, ensure_ascii=False, indent=4)
        print("所有模型輸出已儲存至 computer_scenarios_training.json")
else:
    print("沒有資料被處理，未生成 computer_scenarios_training.json")