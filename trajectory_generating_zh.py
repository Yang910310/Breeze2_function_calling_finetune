import json
import re
import openai
import os
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==== 路徑設定 ====
TASK_PATH = Path(r"D:\research_information\Breeze2_FC_finetune\data\english\training\atomic_tasks.json")
FUNC_PATH = Path(r"D:\research_information\Breeze2_FC_finetune\data\english\training\en_functions.json")
OUTPUT_PATH = Path("en_final_dataset.jsonl")

# ==== 載入資料 ====
with open(TASK_PATH, encoding="utf-8") as f:
    atomic_tasks = json.load(f)

with open(FUNC_PATH, encoding="utf-8") as f:
    function_mappings = json.load(f)

# 轉換 function 映射為 dict
function_dict = {}

for item in function_mappings:
    raw = item.get("atomic_task", "")
    if not raw.strip():
        continue  # 跳過空值

    try:
        task_entries = json.loads(raw)  # 嘗試反序列化
        for entry in task_entries:
            task = entry.get("task")
            funcs = entry.get("func_list", [])
            if task and funcs:
                function_dict[task] = funcs
    except Exception as e:
        print(f"⚠️ 無法解析 atomic_task (index={item.get('index')}): {e}")
        continue


# ==== 共用 OpenAI chat 呼叫 ====
def chat(system_prompt: str, messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "system", "content": system_prompt}] + messages
    )
    return response.choices[0].message.content


# ==== 主程序 ====
# all_records = []
early_stopped_tasks = []  # 紀錄被中止的任務

for task_item in atomic_tasks[385:1000]:  # 只處理前 50 筆
    # if task_item["index"] != 0:
    #     continue
    all_records = []
    task = task_item["atomic_task"]
    functions = function_dict.get(task, [])
    if not functions:
        continue

    print(f"\n[+] Simulating: {task[:60]}...")


    # -- Prompts --
    user_system_prompt = (
        "假設你是一位與 AI 助理互動的人類。\n"
        "你需要進行一場有意義的對話，同時始終記得展現出人類般的行為。\n"
        "避免詢問 AI 助理是否需要幫助，因為這與你扮演人類角色的設定相矛盾。\n"
        "你的主要目標是像一般使用者那樣持續進行對話。\n"
        "目前，你的目標是完成一項預先定義的任務，\n"
        "並為此目的尋求 AI 助理的協助。\n"
        "**任務**\n"
        f"{task}\n"
        "在這場對話中，你應該扮演積極主動的角色，\n"
        "並透過一系列函數（工具）呼叫，探索 AI 助理解決 **任務** 中問題的能力。\n"
        "你應遵守以下指引：\n"
        "1. 你的任務涉及一項需要多個步驟才能完成的複雜任務。\n"
        "   在你給 AI 助理的初始提問中，應提供對任務的詳細說明，\n"
        "   包括可能需要用來解決問題的資訊（例如潛在資料）。\n"
        "   然而，你應該避免提供具體的解決步驟（例如不要使用「首先」、「其次」這類詞），\n"
        "   也不應指示 AI 應使用哪些函數（工具）──這應由 AI 自行決定。\n"
        "2. 請記住，在這段多輪對話中，你正在扮演一位人類使用者。\n"
        "   你的提問與回應應該反映出人類的特質。\n"
        "   所有你的輸出都應該被包含在 \"<human>...</human>\" 標籤中。\n"
    )

    assistant_system_prompt = (
        "You are simulating the role of an expert in using functions (i.e., tools) to solve users’ tasks. "
        "You already possess knowledge on how to decompose the task into subtasks and understand which tools to use.\n"
        "**Task**\n"
        f"{task}\n"
        "**Available Functions for the task**\n"
        f"{json.dumps(functions, indent=2)}\n"
        "Please use the tools provided above to answer the question posed\n"
        "by \"<human>\". You must try as much as possible to use these\n"
        "tools, instead of directly answering the question using your\n"
        "prior knowledge.\n"
        "Your response must obey the following format:\n"
        "**Observation**: Carefully observe the user \"<human>\"’s question as\n"
        "well as the output of the function call (often enclosed\n"
        "within the \"<func_return>\" tag). Be sure to check for any\n"
        "errors in previous outputs, as they may not always be\n"
        "accurate. Enclose your observation within the \"<observation>\"\n"
        "tag.\n"
        "**Thought**: After observing and combining the previously listed\n"
        "steps, give detailed and clear thoughts, reasonings, or\n"
        "reflections, and according to the plan decide the next step.\n"
        "Enclose your thoughts within the \"<thought>\" tag.\n"
        "**Function Call**: Name and arguments of the function call. The\n"
        "function name must be same as its name in above function list,\n"
        "and the arguments must obey the format required by the function.\n"
        "Enclose the function call within the \"<func_call>\" tag.\n"
        "If possible, you can call multiple functions in parallel,\n"
        "be sure the functions called parallelly are independent of each other.\n"
        "Prefix function calls with:\n"
        "- <|use_tool|> if calling normal functions\n"
        "\n"
        "---\n"
        "### Example 1 (regular function call):\n"
        "<observation> User has provided two numbers - 15 and 25. </observation>\n"
        "<thought> Based on user’s request, we need to find the greatest\n"
        "common divisor of these two numbers. We can use the function\n"
        "’find_greatest_common_divisor’ to solve this problem. </thought>\n"
        "<|use_tool|><func_call>[\n"
        "{\n"
        "  \"name\": \"find_greatest_common_divisor\",\n"
        "  \"arguments\": {\"num1\": 15, \"num2\": 25}\n"
        "}\n"
        "]</func_call>\n"
        "\n"
        "---\n"
        "### Example 2 (parallel function call):\n"
        "<observation> User wants to know the weather in two cities - New York and London. </observation>\n"
        "<thought> We can use the function ’get_weather’ to find the weather in both cities. </thought>\n"
        "<|use_tool|><func_call>[\n"
        "{\n"
        "  \"name\": \"get_weather\",\n"
        "  \"arguments\": {\"city\": \"New York\"}\n"
        "},\n"
        "{\n"
        "  \"name\": \"get_weather\",\n"
        "  \"arguments\": {\"city\": \"London\"}\n"
        "}\n"
        "]</func_call>\n"
        "\n"
        "---\n"
        "### Example 3 (final answer without calling any function):\n"
        "<observation> find_greatest_common_divisor returns the result \"5\". </observation>\n"
        "<thought> The result is sufficient to answer the user’s question, so we present the final answer. </thought>\n"
        "<|answer|><final_answer>The requested operation was successfully completed. Result: 5.</final_answer>\n"
        "\n"
        "---\n"
        "When you believe the task is completed and no further function calls are needed,\n"
        "you must directly return the final response to the user using the following format:\n"
        "\n"
        "<|answer|><final_answer>Your final answer here...</final_answer>\n"
        "\n"
        "This final answer should be self-contained, clear, and directly answer the original task.\n"
        "Avoid calling any function named \"final_answer\".\n"
        "\n"
        "Furthermore, when the user \"<human>\" raises a question, you need\n"
        "to provide a structured plan to solve the question. The\n"
        "contents of the plan can be placed in the first round\n"
        "response’s <thought>, and try to follow this plan in\n"
        "every subsequent function call. If needed, revise the plan\n"
        "based on function call outcomes.\n"
)

    # -- 開始模擬對話 --

    messages = []
    user_msg = chat(user_system_prompt, [])
    messages.append({"role": "user", "content": user_msg})
    full_text = f"System:\n{assistant_system_prompt}\nUser:\n{user_msg}\n"

    assistant_turn_count = 0  # 初始化 assistant 出場次數

    while True:
        assistant_msg = chat(assistant_system_prompt, messages)
        messages.append({"role": "assistant", "content": assistant_msg})
        full_text += f"Assistant:\n{assistant_msg}\n"
        assistant_turn_count += 1

        # 若 assistant 已啟動 4 次（含），提前終止該任務模擬
        if assistant_turn_count >= 4:
            print("⚠️ Assistant 回合超過限制，提前終止任務模擬。")
            early_stopped_tasks.append(task)
            break

        if "<|answer|>" in assistant_msg:
            print("✅ 任務已完成，Assistant 回傳 final_answer。")
            break

        match = re.search(r"<func_call>(.*?)</func_call>", assistant_msg, re.DOTALL)
        if not match:
            print("❌ 無效 <func_call> 區塊")
            break

        try:
            func_calls = json.loads(match.group(1))
            print("Agent 呼叫的 functions: ", func_calls)
        except Exception as e:
            print("❌ JSON 格式錯誤：", e)
            break

        tool_responses = []
        for func in func_calls:
            fdef = next(f for f in functions if f["name"] == func["name"])
            tool_system_prompt = (
                "You are simulating a computer system with powerful computational capabilities and a complete setup. "
                "You possess ample external prior knowledge, allowing you to run any arbitrary function and execute calls to produce results, and you never make errors.\n"
                "**Function**\n"
                f"{json.dumps(fdef, indent=2)}\n"
                "Given a function call, you should execute the function and provide the results in JSON format. "
                "Your response should directly provide the results in JSON format, should not contain irrelevant information, and must enclose within <func_return> tag.\n"
                "### Example of function return:\n"
                "<func_call>\n"
                "{\n"
                "\"name\": \"get_weather\",\n"
                "\"arguments\": {\"city\": \"New York\"}\n"
                "}\n"
                "<func_return>\n"
                "{\n"
                "\"temperature\": \"25C\"\n"
                "}\n"
                "</func_return>\n"
            )
            tool_input = f"<func_call>\n{json.dumps(func, indent=2)}\n</func_call>"

            tool_output = chat(tool_system_prompt, [{"role": "user", "content": tool_input}])
            match_result = re.search(r"<func_return>\s*(\{.*?\})\s*</func_return>", tool_output, re.DOTALL)
            if not match_result:
                print("❌ Tool 回傳缺少 <func_return>")
                break

            try:
                parsed = json.loads(match_result.group(1))
            except json.JSONDecodeError as e:
                print("任務:", task[:30], "\n❌ JSON 解析失敗！內容如下：")
                print(match_result.group(1))
                print(f"錯誤細節: {e}")
                break 

            parsed = json.loads(match_result.group(1))
            tool_responses.append({
                "name": func["name"],
                "arguments": func["arguments"],
                "results": parsed
            })

        tool_block = f"Tool:\n{json.dumps(tool_responses, indent=2)}\n"
        messages.append({"role": "user", "content": tool_block})
        full_text += tool_block

    # 若不是因 assistant 過多回合中止，才加入正式記錄
    if 2 <= assistant_turn_count < 4:
        all_records.append({"text": full_text.strip()})
    # ==== 輸出 JSONL ====
    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# # ==== 輸出 JSONL ====
# with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
#     for record in all_records:
#         f.write(json.dumps(record, ensure_ascii=False) + "\n")


if early_stopped_tasks:
    print("\n⚠️ 以下任務因超過回合數而被中止模擬：")
    for t in early_stopped_tasks:
        print(f"- {t[:80]}...")

print(f"\n✅ 完成！共產出 {len(all_records)} 筆資料 -> {OUTPUT_PATH}")
