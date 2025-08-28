import json
from openai import OpenAI
from dotenv import load_dotenv
import os

# 載入 .env 檔案中的環境變數
load_dotenv()

# 從環境變數中讀取 API 金鑰
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# 讀取已存在的 JSON 檔案（如果有）
file_path = r"D:\research_information\Breeze2_FC_finetune\data\english\training\scenario_training_02.json"
output_path = r"D:\research_information\Breeze2_FC_finetune\data\english\training\atomic_tasks.json"

# 初始化輸出列表
original_scenarios = []
all_outputs = []

# 如果 JSON 檔案存在，讀取其內容
if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        original_scenarios = json.load(json_file)
else:
    print("scenario_training_02.json中沒有資料。")
    exit()

# 取得目前的最大 index
start_index = 1000

# 處理 JSON 中的每一筆 "model_output" 資料
for i, entry in enumerate(original_scenarios[500:2000]):
    try:
        # 確保資料中有 'model_output' 欄位
        if isinstance(entry, dict) and "model_output" in entry:
            scenario = entry["model_output"]
            print(f"Processing entry {i}: {scenario}")

            input_prompt = (
                "You are training a model that can take a user’s task description"
                "or query, and available functions as input, and generate a"
                "sequence of function calls to accomplish the task. Currently,"
                "you are generating basic atom tasks. Given a general life"
                "scenario as the context, please generate a basic atom task"
                "that can be accomplished in one steps.\n"
                "Requirements of the task:\n"
                "1. The task should be a reasonable real life task based on the"
                "given scenario, and can be accomplished in one step.\n"
                "2. If you mention some information, criteria or constraints in"
                "the task, please give the details of these information,"
                "criteria or constraints. Do not assume the model has access"
                "to your personal information or prior knowledge, and it does"
                "not have chance to ask you for clarification.\n"
                "3. Please give enough details and make the task description as"
                "specific as possible, so the model can make deterministic"
                "function calls with deterministic arguments. Do not include"
                "any ambiguous or vague information.\n"
                "4. Do not mention specific tools or functions in the task"
                "description, and do not propose solutions, hints, or project"
                "outcomes.\n"
                "5. Limit the task description to 30 words, and avoid using"
                "adjectives and ambiguous words."
                "Given Scenario:\n"
                f"{scenario}\n"
                "Please give your response in one line directly, without any"
                "extra notation or format:"
            )

            # 呼叫 OpenAI 模型（範例，需替換為實際 API 呼叫方式）
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=input_prompt,
                max_output_tokens=50
            )

            # 輸出結果可依需求儲存或更新 all_outputs
            output_data = {
                "index": start_index,
                "atomic_task": response.output_text
            }
            all_outputs.append(output_data)
            start_index += 1
            
            print(f"atomic task for entry {i}: {response.output_text}")

        else:
            print(f"Skipping entry {i}: missing 'model_output' field.")
    except Exception as e:
        print(f"Error processing entry {i}: {e}")


# 將所有生成結果儲存為 JSON 檔案
with open(output_path, "a", encoding="utf-8") as json_file:
    json.dump(all_outputs, json_file, ensure_ascii=False, indent=4)

print("所有生成結果已儲存至 atomic_tasks.json")