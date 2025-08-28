import json
import re
import os
import tiktoken
from transformers import AutoTokenizer


file_path = r"D:\research_information\Breeze2_FC_finetune\en_final_dataset.jsonl"
# if os.path.exists(file_path):
#     with open(file_path, "r", encoding="utf-8") as json_file:
#         all_outputs = json.load(json_file)

# # 處理每個 atomic_task 欄位
# for entry in all_outputs:
#     task = entry["atomic_task"]
#     # 移除前後的 ```json 和 ```
#     task = re.sub(r"^```json\s*", "", task)
#     task = re.sub(r"\s*```$", "", task)
#     # 移除 \n
#     task = task.replace("\\n", "")
#     # 移除多餘空格（保留結構性縮排的空白）
#     task = re.sub(r"\s{2,}", " ", task)
#     entry["atomic_task"] = task.strip()

# # 寫回 JSON 檔案
# if all_outputs:
#     with open(file_path, "w", encoding="utf-8") as json_file:
#         json.dump(all_outputs, json_file, ensure_ascii=False, indent=4)






# # 移除 atomic_task 中包含 "\"func_list\": []" 的資料
# def remove_func_list_empty_entries(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     # 過濾掉 atomic_task 中包含 "\"func_list\": []" 的資料
#     filtered_data = [
#         item for item in data
#         if '"func_list": []' not in item.get('atomic_task', '')
#     ]

#     with open(file_path, "w", encoding="utf-8") as json_file:
#         json.dump(filtered_data, json_file, ensure_ascii=False, indent=4)

#     print(f"已完成過濾，原始共 {len(data)} 筆，保留 {len(filtered_data)} 筆。")

# # 執行清除動作
# remove_func_list_empty_entries(file_path)







# # 清理 func_call 標籤
# # 將 <func_call> 替換為 <|python_tag|>，並刪除 </func_call>
# def clean_func_call_tags(json_data):
#     if isinstance(json_data, dict):
#         return {k: clean_func_call_tags(v) for k, v in json_data.items()}
#     elif isinstance(json_data, list):
#         return [clean_func_call_tags(item) for item in json_data]
#     elif isinstance(json_data, str):
#         return json_data.replace("<func_call>", "<|python_tag|>").replace("</func_call>", "")
#     else:
#         return json_data

# # 讀取 JSON 檔案
# data = []
# with open(file_path, "r", encoding="utf-8") as f:
#     for line in f:
#         line = line.strip()
#         if line:
#             data.append(json.loads(line))


# # 處理資料
# updated_data = clean_func_call_tags(data)

# # 儲存結果
# with open("output.jsonl", "w", encoding="utf-8") as f:
#     json.dump(updated_data, f, ensure_ascii=False, indent=2)

# print("處理完成：<func_call> 已替換，</func_call> 已刪除。結果儲存在 output.json")












# def convert_raw_text_to_breeze2_format(text: str) -> dict:
#     messages = []

#     # 提取所有標記段落（避免 Tool 吃進 Assistant）
#     pattern = r"(System:|User:|Assistant:|Tool:)\n(.*?)(?=(?:System:|User:|Assistant:|Tool:)|\Z)"
#     blocks = re.findall(pattern, text, re.DOTALL)

#     for role_tag, content in blocks:
#         role_tag = role_tag.strip()
#         content = content.strip()

#         if role_tag == "System:":
#             messages.append({"role": "system", "content": content})
#         elif role_tag == "User:":
#             # 抽出 <human> 裡的內容
#             match = re.search(r"<human>(.*?)</human>", content, re.DOTALL)
#             content = match.group(1).strip() if match else content
#             messages.append({"role": "user", "content": content})
#         elif role_tag == "Assistant:":
#             messages.append({"role": "assistant", "content": content})
#         elif role_tag == "Tool:":
#             messages.append({"role": "ipython", "content": content})

#     return {"messages": messages}


# if __name__ == "__main__":
#     input_path = r"D:\research_information\Breeze2_FC_finetune\output.jsonl"
#     output_path = "en_final.jsonl"

#     with open(input_path, "r", encoding="utf-8") as f:
#         raw = json.load(f)

#     with open(output_path, "w", encoding="utf-8") as f_out:
#         for item in raw:
#             try:
#                 chatml_data = convert_raw_text_to_breeze2_format(item["text"])
#                 json.dump(chatml_data, f_out, ensure_ascii=False)
#                 f_out.write("\n")
#             except Exception as e:
#                 print(f"❌ Error processing item: {e}")

#     print(f"✅ 已轉換完成，輸出至：{output_path}")







# 使用 tiktoken 計算每筆資料 token 數量

tokenizer = AutoTokenizer.from_pretrained("PenutChen/Llama-Breeze2-3B-Instruct-Text")

input_file = r"D:\research_information\Breeze2_FC_finetune\data\english\training\en_final.jsonl"
output_file = r"D:\research_information\Breeze2_FC_finetune\data\english\training\en_final_filtered.jsonl"
max_tokens = 2048

def count_tokens(text):
    return len(tokenizer.encode(text))

def count_jsonl_tokens(obj):
    # 假設每筆資料有 "messages" 欄位
    if "messages" in obj:
        total = 0
        for msg in obj["messages"]:
            if "content" in msg:
                total += count_tokens(msg["content"])
        return total
    else:
        return count_tokens(json.dumps(obj, ensure_ascii=False))

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:
    kept, dropped = 0, 0
    for line in fin:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception as e:
            print("JSON parse error:", e)
            continue
        tokens = count_jsonl_tokens(obj)
        if tokens <= max_tokens:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1
        else:
            dropped += 1
    print(f"Kept: {kept}, Dropped: {dropped}")