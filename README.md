# Breeze2_FC_finetune

本專案的目標是針對 **[PenutChen/Llama-Breeze2-3B-Instruct-Text](https://huggingface.co/PenutChen/Llama-Breeze2-3B-Instruct-Text)** 模型進行 **LoRA 微調 (Fine-tuning)**，並使用 **GTA (General Tool-augmented benchmark for function calling)** 進行評估。  

---

## 📌 專案流程

整體分為三大階段：

1. **生成與清理訓練資料**  
   - `original_scenarios.py`：準備原始情境資料  
   - `atomic_task_construct.py`：將情境拆解為原子任務  
   - `function_generating.py`：為原子任務產生對應函數  
   - `trajectory_generating.py`：產生互動軌跡資料  
   - `data_cleaning.py`：資料清理與格式化，輸出 JSONL 格式（每筆資料含有 `messages` 欄位）

2. **微調模型**  
   - `fine_tune.py`：針對 Llama-Breeze2-3B-Instruct-Text 進行 LoRA 微調  
   - 輸出結果會儲存在 `lora_models/`

3. **模型評估 (GTA)**  
   - 使用 `GTA/` 中的工具與資料，進行 function calling 能力的評估  

---

## 📂 專案結構

```
Breeze2_FC_finetune/
│
├── Breeze2_FC_finetune/
│   ├── .env
│   ├── atomic_task_construct.py      # 原子任務生成
│   ├── computer_use_data.py          # 特殊資料處理
│   ├── data_cleaning.py              # 訓練資料清理與格式化
│   ├── fine_tune.py                  # LoRA 微調主程式
│   ├── function_generating.py        # 函式產生器
│   ├── merge_lora.py                 # 合併 LoRA 權重
│   ├── original_scenarios.py         # 原始情境資料
│   ├── trajectory_generating.py      # 產生訓練軌跡
│   ├── trajectory_generating_zh.py   # 中文版本的訓練軌跡生成
│   │
│   ├── data/                         # 訓練資料 (JSONL 格式, 含 messages 欄位)
│   ├── lora_models/                  # 微調後模型輸出
│   │
│   └── GTA/                          # GTA 基準測試工具
│       ├── LICENSE.txt
│       └── README.md
```

---

## ⚙️ 環境需求

- Python ≥ 3.10
- GPU: NVIDIA RTX 系列（11GB VRAM 以上建議）
- CUDA ≥ 11.8
- 主要套件：
  - [PyTorch](https://pytorch.org/)
  - [Transformers](https://github.com/huggingface/transformers)
  - [PEFT (LoRA)](https://github.com/huggingface/peft)
  - [Datasets](https://github.com/huggingface/datasets)
  - [Accelerate](https://github.com/huggingface/accelerate)
  - 其他需求可參考 `requirements.txt` 或自行安裝


### 📦 安裝環境

建議使用虛擬環境 (venv 或 conda)，並安裝相依套件：

```bash
conda create -n breeze2_fc python=3.10
conda activate breeze2_fc

git clone https://github.com/Yang910310/Breeze2_function_calling_finetune.git

pip install -r requirements.txt
```

---

## 🚀 使用方式

### 1. 建立資料集
首先將您的 OpenAI API key 增加至 .env 檔案當中

接著依序執行以下程式產生 JSONL 訓練資料(請記得修改程式中的檔案路徑及相關參數)：
```bash
python original_scenarios.py
python atomic_task_construct.py
python function_generating.py
python trajectory_generating.py
python data_cleaning.py
```

### 2. 執行微調
```bash
python fine_tune.py
```

### 3. 合併 LoRA 權重（可選）
若需要將 LoRA 權重與原始模型合併：
```bash
python merge_lora.py
```

### 4. 使用 GTA 評估
```bash
# 依照 GTA README.md 的教學進行模型評估
```

---

## 📊 訓練資料格式

輸入/輸出的資料採用 **JSONL** 格式，每筆資料包含 `messages` 欄位，格式如下：

```json
{
  "messages": [
    {"role": "system", "content": "You are ..."},
    {"role": "user", "content": "請介紹 LoRA 微調的用途"},
    {"role": "assistant", "content": "LoRA 可以..."}
  ]
}
```

---

## 📌 注意事項
- 本專案尚未提供 **推論 (inference) script**，僅包含 **資料生成 → 微調 → 評估** 流程。
- GTA 子模組提供的 README 與 LICENSE 檔可參考官方文件。
