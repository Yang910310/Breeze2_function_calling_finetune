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



# 讀取 JSON 檔
data_path = r"D:\research_information\Breeze2_FC_finetune\data\english\training\atomic_tasks.json"
output_path = r"D:\research_information\Breeze2_FC_finetune\data\english\training\en_functions.json"

# 初始化輸出列表
original_tasks = []
all_outputs = []

# 如果 JSON 檔案存在，讀取其內容
if os.path.exists(data_path):
    with open(data_path, "r", encoding="utf-8") as json_file:
        original_tasks = json.load(json_file)
else:
    print("atomic_task.json中沒有資料。")
    exit()

# 取得目前的最大 index
start_index = 1499


for i, entry in enumerate(original_tasks[1500:3000]):
    try:
        # 確保資料中有 "atomic_task" 欄位
        if isinstance(entry, dict) and "atomic_task" in entry:
            task = entry["atomic_task"]
            print(f"Processing entry {i}: {task}")

            # 定義輸入 prompt，將多筆資料合併
            en_input_prompt = (
                "You are training a model that can take a user’s task description\n"
                "or query, and available functions as input, and generate a\n"
                "sequence of function calls to accomplish the task. Currently,\n"
                "you are generating the training data for this model.\n"
                "Given a task , please generate corresponding aviliable functions that can be used\n"
                "to accomplish the task, and finally the \n"
                "task can be accomplished by calling these functions\n"
                "sequentially.\n"
                "## Requirements for the functions:\n"
                "1. The functions must possess a succinct, comprehensible name\n"
                "and description.\n"
                "2. The functions should not tailored for a current task, are to\n"
                "be used for other future tasks as well, hence the design of\n"
                "APIs should be sufficiently generalized.\n"
                "3. Avoid the recurrence of the task or its components in the\n"
                "function description and name, offering a generic perspective\n"
                "that can be employed across different contexts.\n"
                "4. Make every function sufficiently granular and independent,\n"
                "avoiding the conflation of multiple tasks within a single\n"
                "function and avert creating monolithic APIs.\n"
                "5. Consistency in terms of parameters and returns from each\n"
                "function is critical. For instance, if two functions are\n"
                "called sequentially, the output of the first should either\n"
                "align with or constitute a part of the input for the second\n"
                "function, irrespective of varying parameter terminologies.\n"
                "## Requirements for the number of functions:\n"
                "1. The task may need zero, 0~5 functions to\n"
                "complete it.\n"
                "2. If the task is about logic, comparision, set operation or\n"
                "calculation, which can be solved by large language models,\n"
                "then no function is needed for this task, just leave the\n"
                "func_list of this task empty.\n"
                "## task:\n"
                f"{task}\n"
                "## Response format:\n"
                "'''json\n"
                "[\n"
                "  {\n"
                "    \"task\": \"a task from the input\",\n"
                "    \"func_list\": [\n"
                "      {\n"
                "        \"name\": \"<function name>\",\n"
                "        \"description\": \"<function usage description>\",\n"
                "        \"parameters\": {\n"
                "          \"<param1>\": {\n"
                "            \"type\": \"<string | number | boolean | object | array | enum | anyOf>\",\n"
                "            \"description\": \"<param1 description>\"\n"
                "          }\n"
                "          ... <more parameters if needed>\n"
                "        },\n"
                "        \"required\": [\"<array of required parameters>\"],\n"
                "        \"responses\": {\n"
                "          \"<res1>\": {\n"
                "            \"type\": \"<value1 type>\",\n"
                "            \"description\": \"<value1 description>\"\n"
                "          },\n"
                "          \"<res2>\": {\n"
                "            \"type\": \"<value2 type>\",\n"
                "            \"description\": \"<value2 description>\"\n"
                "          }\n"
                "        }\n"
                "      }\n"
                "      ... <more functions if needed>\n"
                "    ]\n"
                "  }\n"
                "]\n"
                "'''\n"
                "## Please respond following the format above:\n"

            )

            # 呼叫 OpenAI 模型（範例，需替換為實際 API 呼叫方式）
            response = client.responses.create(
                model="gpt-4.1",
                input=en_input_prompt,
                max_output_tokens=5000
            )

            # 輸出結果可依需求儲存或更新 all_outputs
            output_data = {
                "index": start_index,
                "atomic_task": response.output_text
            }
            all_outputs.append(output_data)
            start_index += 1  # 遞增 index
            
            print(f"atomic task for entry {i}: {response.output_text}")

        else:
            print(f"Skipping entry {i}: missing 'atomic_task' field.")
    except Exception as e:
        print(f"Error processing entry {i}: {e}")

# 將所有生成結果儲存為 JSON 檔案
with open(output_path, "a", encoding="utf-8") as json_file:
    json.dump(all_outputs, json_file, ensure_ascii=False, indent=4)

print("所有生成結果已儲存至 en_functions.json")