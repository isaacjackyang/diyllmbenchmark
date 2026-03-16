# diyllmbenchmark

`diyllmbenchmark` 是一個針對本地 OpenAI-compatible LLM 服務的互動式 benchmark 工具，目前支援：

- `Ollama`
- `llama.cpp` 的 `llama-server`

它會把你選的模型與參數組合逐一送測，量測串流輸出表現，最後產生報告、圖表、原始輸出與最佳設定摘要，適合用來比較不同模型或不同推論參數的實際效果。

## 這個工具能做什麼

- 比較多個模型在相同 prompt 下的表現
- 比較不同參數組合對速度與輸出型態的影響
- 支援兩種 benchmark 模式：
  - `chat`：一般文字回覆 benchmark
  - `tools`：檢查模型是否真的會輸出 `tool_calls`
- 自動產生 HTML 報告、PNG 圖表、JSONL 原始結果
- 若有 NVIDIA GPU 且系統可執行 `nvidia-smi`，會額外記錄 VRAM 使用量
- 若後端有暴露 reasoning / thinking 類型欄位，會一併保留在報告中

## 安裝需求

先準備好：

- 可執行 `python` 的環境
- 已啟動的本地 LLM 服務
  - `Ollama` 預設使用 `http://localhost:11434/v1`
  - `llama.cpp` 預設使用 `http://localhost:8080/v1`
- 可選：`nvidia-smi`
  - 若可用，報告會顯示 VRAM 指標
  - 若不可用，VRAM 欄位會顯示 `N/A`

安裝 Python 套件時，請使用專案內的 `.venv`，不要直接裝到系統全域環境。下方若以 Windows 為例，其他平台只要把 `.\.venv\Scripts\python.exe` 換成 `./.venv/bin/python` 即可。

Windows 建議直接執行：

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

這個腳本會自動：

- 找出可用的 `py` 或 `python`
- 在專案根目錄建立或重用 `.venv`
- 升級 `.venv` 內的 `pip`
- 依照 `requirements.txt` 安裝依賴到 `.venv`
- 驗證 `ollama_expert_bench.py` 需要的關鍵套件是否真的能 import

如果你想手動安裝，也請先建立虛擬環境：

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS / Linux 則可用：

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

如果你是搬到另一台電腦，建議不要雙擊 `.py`，而是在專案目錄開 PowerShell / CMD 後，使用 `.venv` 內的 Python 執行：

```powershell
.\.venv\Scripts\python.exe ollama_expert_bench.py
```

## 快速開始

1. 啟動你的本地 LLM 後端。

   Ollama 範例：

   ```bash
   ollama serve
   ```

   llama.cpp 範例：

   ```bash
   llama-server --port 8080
   ```

2. 執行 benchmark：

   ```powershell
   .\.venv\Scripts\python.exe ollama_expert_bench.py
   ```

3. 依照互動式選單完成設定。

這個工具目前是互動式操作，沒有命令列參數介面；所有設定都是在執行後逐步選擇。

## 目前正式版

目前正式入口檔案是：

```powershell
.\.venv\Scripts\python.exe ollama_expert_bench.py
```

這個單檔版本目前整合了：

- 原 V4 的全頁表格式參數設定 UI
- 原 V3 / V5 的 benchmark / 圖表 / HTML report / JSONL raw outputs / best config 輸出
- `Thinking TPS`、`Output TPS`、`Output/Thinking Ratio`
- `tools` 模式下各模型的 tool call 成功次數與成功率摘要
- 可把 `system prompt` 當成獨立測試維度，支援 `N/A`、固定數量或自訂數量，並用整頁編輯區直接貼上多段 prompt
- 產出的 HTML 報告會用中英對照顯示主要段落、欄位與指標說明
- HTML 報告中的結果卡片支援前端篩選，可勾選工具呼叫、thinking、有無 output、thinking mode 與狀態

互動式整頁 grid 的操作方式如下：

- `↑ / ↓`：移動參數列
- `← / →` 或 `Tab`：切換 `State / Values` 欄位
- `Space`：切換 `N/A` 與 `TEST`
- 在 `Values` 欄位直接輸入測試值
  - 數值參數可輸入數字、逗號、小數點、負號
  - `Thinking / Reasoning` 可輸入 `enable,disable` 或 `true,false`
- `Backspace`：刪除一個字元
- `d`：恢復該參數預設值
- `Enter` / `Ctrl+S`：確認設定並進入 review
- `Esc`：取消

這一頁會一次列出所有可調參數；若目前 backend 不支援，該列會顯示 `LOCK`。

在設定流程中，如果中途改變心意，也可以返回上一階段：

- `select / checkbox / confirm` 畫面：按 `Backspace`
- 一般 `text` 輸入框：當輸入框是空的時按 `Backspace`
- 參數 grid：在 `State` 欄按 `Backspace` 返回上一階段；在 `Values` 欄仍是刪字
- `system prompt` 編輯器：當編輯器是空的時按 `Backspace`

## 使用流程

執行後，程式會依序詢問：

1. 後端類型
   - `Ollama`
   - `llama.cpp (llama-server)`
2. Benchmark 模式
   - `chat`
   - `tools`
3. 模型
   - `Ollama` 會先嘗試從 `/api/tags` 自動抓模型清單
   - 抓不到時可手動輸入模型名稱
   - `llama.cpp` 需手動輸入 port 與模型名稱
4. 要測的參數群組
   - 可不選；不選時代表只比較模型預設值
5. 各參數的測試值
   - 以逗號分隔，例如 `0.1, 0.8`
   - `Thinking / Reasoning` 開關可輸入 `enable, disable`
6. 測試 prompt
7. `system prompt` 變體
   - 可選 `N/A`
   - 可選 `1 / 2 / 3` 種，或手動輸入自訂數量
   - 會開啟多行貼上編輯區，使用單獨一行的 `---` 分隔不同 system prompt

程式會把你輸入的所有參數值做笛卡兒積組合，所以總測試次數為：

`模型數量 x 參數組合數 x system prompt 變體數`

如果沒有選任何參數，且 `system prompt` 也選 `N/A`，就只會以各模型目前預設設定各跑一次。

## Benchmark 模式說明

### `chat` 模式

用來測一般文字輸出效能，主要觀察：

- `TPS`
- `TTFT`
- `Stream Duration`
- `Output Category = normal_content`

這個模式適合拿來看哪個模型或哪組參數「回得比較快」。

### `tools` 模式

這個模式會在 request 中附上一個測試用工具：

- `lookup_weather`

並要求模型先呼叫工具，而不是直接回答。這個模式主要看：

- 是否真的回傳 `tool_calls`
- `First Event (s)`
- `Stream Duration`

在 `tools` 模式中，成功結果不一定會輸出文字，所以 `TPS` 與 `TTFT` 可能是 `N/A`；這是正常的，判讀時請優先看：

- `Output Category = tool_call`
- `First Event (s)`

注意：這裡 benchmark 的是「模型有沒有發出 tool call」，不是實際去執行天氣查詢。

## 可調參數

程式內建以下參數，且會依後端自動過濾可用項目：

| 參數 | 支援後端 | 用途 |
| --- | --- | --- |
| `temperature` | `ollama`, `llama.cpp` | 控制創意與穩定度 |
| `num_ctx` | `ollama` | 上下文長度，常影響長文能力與 TTFT |
| `num_predict` | `ollama`, `llama.cpp` | 最大生成 token 數 |
| `top_p` | `ollama`, `llama.cpp` | 核心採樣，降低時通常更保守 |
| `min_p` | `ollama`, `llama.cpp` | 過濾低機率 token |
| `repeat_penalty` | `ollama`, `llama.cpp` | 降低重複輸出 |
| `enable_thinking` | `ollama`, `llama.cpp` | 測試 thinking / reasoning 開關對輸出與速度的影響 |
| `num_gpu` | `ollama` | GPU 卸載相關設定，常明顯影響速度 |

參數群組如下：

- `生成核心`：`temperature`、`num_ctx`、`num_predict`
- `採樣與懲罰`：`top_p`、`min_p`、`repeat_penalty`
- `Thinking / Reasoning`：`enable_thinking`
- `硬體與部署`：`num_gpu`

## 後端設定差異

### Ollama

- 會固定使用 `http://localhost:11434/v1`
- 會先嘗試自動列出本機模型
- 大多數參數會以 Ollama 對應名稱送進 `options`
- `enable_thinking` 會改送成頂層 `think=true/false`
- 若有成功結果，會額外輸出 `Ollama_Modelfile_Suggest`

### llama.cpp

- 需先自行啟動 `llama-server`
- 預設 port 是 `8080`
- 模型名稱只作為報告辨識用途，不會替你載入模型
- `num_predict` 會映射為 `n_predict`
- `enable_thinking` 會映射為 `chat_template_kwargs.enable_thinking=true/false`

## 產出檔案

Benchmark 完成後，程式會自動建立 `Report/` 資料夾，並把本次 benchmark 產物集中放在裡面：

- `Report/bench_{backend}_{capability}_{timestamp}.html`
  - 完整 HTML 報告
  - `Summary / 摘要` 區塊內含 Excel 下載按鈕
- `Report/bench_{backend}_{capability}_{timestamp}_summary.xlsx`
  - `Summary`、`Outcome Summary` 與 `tools` 模式下的工具成功率摘要 Excel
- `Report/bench_{backend}_{capability}_{timestamp}.png`
  - 圖表輸出
  - 若沒有可繪圖的成功結果，可能不會產生
- `Report/bench_{backend}_{capability}_{timestamp}_outputs.jsonl`
  - 每次 run 的原始結果，適合後續再分析
- `Report/best_config.json`
  - 本次 benchmark 選出的最佳配置
  - 若沒有成功結果，則不會產生
- `Report/Ollama_Modelfile_Suggest`
  - 僅在最佳結果來自 `Ollama` 時產生
- `ollama_expert_bench_crash.log`
  - 只有程式啟動或執行失敗時才會出現
  - 可用來排查搬機或環境差異造成的閃退問題

## `best_config.json` 怎麼選

工具會依模式自動選最佳結果：

- `chat` 模式
  - 優先選 `TPS` 最高者
  - 若沒有可用 `TPS`，則選 `TTFT` 最低者
- `tools` 模式
  - 優先選 `First Event (s)` 最低者
  - 若沒有可用 `First Event`，則選 `Stream Duration` 最短者

## 報告中的重要欄位

- `TPS (chunk/s)`
  - 以有文字內容的串流 chunk 估算吞吐量，適合做相對比較
- `Total Output (chars)`
  - 該次 run 最終保留的可見輸出總字數
- `Thinking Time (s)`
  - 從第一段 thinking 到第一段 output 的時間；若沒有 output，則到串流結束
- `Output Time (s)`
  - 從收到第一段 output 到串流結束的時間
- `Total Output Time (s)`
  - 從最早的 thinking 或 output 開始計時；若兩者都沒有，則退回第一個串流事件到串流結束
- `Thinking TPS (token/s)`
  - 以保留下來的 thinking 文字估算 token 數，再除以 `Thinking Time (s)` 估算吞吐量
- `Output TPS (token/s)`
  - 以最終 dialogue output 估算 token 數，再除以 `Output Time (s)` 估算吞吐量
- `Output/Thinking Ratio`
  - `output chars / thinking chars`，方便快速比較最終輸出相對於 thinking 的比例
- `TTFT (s)`
  - 從送出 request 到收到第一段文字的時間
- `First Event (s)`
  - 從送出 request 到收到第一個串流事件的時間，包含非文字事件
- `VRAM Peak (MiB)`
  - 測試期間觀察到的最高顯存占用
- `Efficiency Score`
  - `TPS / VRAM Peak (GiB)`，用來看單位顯存效率

## 輸出分類

報告裡常見的 `Output Category`：

- `normal_content`
  - 正常收到文字輸出
- `tool_call`
  - 在 `tools` 模式下，成功收到 `tool_calls`
- `text_reply_without_tool`
  - 模型直接回答了，但沒有呼叫工具
- `empty_reply`
  - 串流正常結束，但沒有文字內容
- `non_content_stream`
  - 串流只有非文字 payload
- `early_stop`
  - 串流提早中斷

## thinking / dialogue_output 保留機制

若後端串流資料中有額外 reasoning 或 thinking 欄位，工具會嘗試保留：

- `thinking`
  - 從 reasoning / thinking 類欄位整理出的文字
- `dialogue_output`
  - 一般文字回覆內容

這些內容會寫進：

- HTML 報告
- JSONL 原始輸出

如果某次 run 沒有這些內容，報告中會標示為 `none` 或留空。

## 測試

目前這個 repo 沒有隨附可直接執行的自動化測試檔；先前文件提到的 `test_stream_classification.py` 目前並不存在。

現階段較接近「驗證」的方式有兩種：

- 執行 `install.ps1`，確認 `.venv` 建立、依賴安裝與關鍵套件 import 檢查都能通過
- 實際跑一次 benchmark，確認互動流程、報告輸出與後端連線正常

如果之後補上正式測試檔，再把執行指令與覆蓋範圍補回這一節。

## 注意事項

- 這個專案目前以互動式操作為主，沒有 CLI 參數模式
- 所有 benchmark 都是透過 OpenAI-compatible `/v1/chat/completions` 介面發送
- `tools` 模式目前內建的工具只有 `lookup_weather`
- `Ollama` 的模型清單抓不到時，不代表後端不能用，只是需要手動輸入模型名
- `llama.cpp` 的模型名稱欄位只用來標記報告，不會幫你載入模型

