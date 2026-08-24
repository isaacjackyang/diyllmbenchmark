# diyllmbenchmark

`diyllmbenchmark` 是一個針對本地 OpenAI-compatible LLM 服務的互動式 benchmark 工具，目前支援：

- `Ollama`
- `llama.cpp` 的 `llama-server`

它會把你選的模型與參數組合逐一送測，量測串流輸出表現，最後產生報告、圖表、原始輸出與最佳設定摘要，適合用來比較不同模型或不同推論參數的實際效果。

## 這個工具能做什麼

- 比較多個模型在相同 prompt 下的表現
- 比較不同參數組合對速度與輸出型態的影響
- 支援四種 benchmark 模式：
  - `chat`：一般文字回覆 benchmark
  - `tools`：檢查模型是否真的會輸出 `tool_calls`
  - `suite-smoke-7`：以七道固定題目快速檢查數學、邏輯、推理、閱讀、翻譯、寫作與程式能力
  - `local-expert-battle-48`：以 PLC、工程計算、繁中語境與長文摘要共 48 題比較本地模型
- 自動產生 HTML 報告、內嵌長條圖、PNG 圖表、Excel 摘要與 JSONL 原始結果
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

Windows 安裝請執行：

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

這個腳本會自動：

- 優先找出穩定的 Python 3.13／3.12／3.11／3.10
- 在專案內建立 `.venv`
- 只在 `.venv` 內升級 `pip` 與安裝 `requirements.txt`
- 用 `.venv\Scripts\python.exe` 驗證所有關鍵套件

如果你是搬到另一台電腦，先執行上面的安裝指令。Windows 日常啟動建議直接雙擊：

```text
llm_expert_bench.cmd
```

這個啟動器永遠使用專案內的 `.venv`。若 `.venv` 尚未建立，或其中缺少 pandas 等套件，會自動執行 `install.ps1` 建立／修復後再啟動。你也可以在 PowerShell / CMD 執行：

```powershell
.\.venv\Scripts\python.exe .\llm_expert_bench.py
```

如果你想使用舊的終端機互動模式，可以改用：

```powershell
.\.venv\Scripts\python.exe .\llm_expert_bench.py --cli
```

執行後會啟動本機單檔 HTML UI，預設優先使用 [http://127.0.0.1:8765/](http://127.0.0.1:8765/)。

UI 的預設後端是 `llama.cpp`，Base URL 預設為 [http://localhost:8080/v1](http://localhost:8080/v1)。

如果 `8765` 已被占用，程式會自動往後尋找可用埠，並把實際網址寫到 `llm_expert_bench_ui_url.txt`。

如果瀏覽器沒有自動開啟，請直接打開 `llm_expert_bench_ui_url.txt`，或把裡面的 URL 貼到瀏覽器。

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

   ```text
   llm_expert_bench.cmd
   ```

   或使用 `.\.venv\Scripts\python.exe .\llm_expert_bench.py`。預設會打開本機 UI；如果沒有自動開啟，請查看 `llm_expert_bench_ui_url.txt`。

3. 在瀏覽器 UI 選擇模式、模型與參數後，按下 `Start Benchmark / 開始測試`。

一般使用以瀏覽器 UI 為主；需要舊式逐步選單時，可加上 `--cli`。

## 目前正式版

目前正式入口檔案是：

```powershell
.\.venv\Scripts\python.exe .\llm_expert_bench.py
```

這個單檔版本目前整合了：

- 原 V4 的全頁表格式參數設定 UI
- 原 V3 / V5 的 benchmark / 圖表 / HTML report / JSONL raw outputs / best config 輸出
- `Thinking TPS`、`Output TPS`、`Output/Thinking Ratio`
- `suite-smoke-7` 七題能力套裝，以及每題的思考時間、回答時間與 token 消耗
- `tools` 模式下各模型的 tool call 成功次數與成功率摘要
- 可把 `system prompt` 當成獨立測試維度，支援 `N/A`、固定數量或自訂數量，並用整頁編輯區直接貼上多段 prompt
- 產出的 HTML 報告會用中英對照顯示主要段落、欄位與指標說明

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

瀏覽器 UI 會依下列區塊完成設定；使用 `--cli` 時則會依序詢問相同內容：

1. 後端類型
   - `Ollama`
   - `llama.cpp (llama-server)`
2. Benchmark 模式
   - `chat`
   - `tools`
   - `suite-smoke-7`
   - `local-expert-battle-48`
3. 模型
   - `Ollama` 會先嘗試從 `/api/tags` 自動抓模型清單
   - 抓不到時可手動輸入模型名稱
   - `llama.cpp` 會顯示 `easy_llamacpp` GGUF catalog，可勾選一個或多個模型批次測試
   - `llama.cpp` 若不勾選模型並將 Models 欄位留空，會直接使用 Base URL 目前已載入的模型
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

`模型數量 x 參數組合數 x system prompt 變體數 x 每組題目數`

`chat` 與 `tools` 的每組題目數為 1；`suite-smoke-7` 為 7。如果沒有選任何參數，且 `system prompt` 也選 `N/A`，`chat` / `tools` 會以各模型預設設定各跑一次，`suite-smoke-7` 則會讓各模型跑七次。

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

### `suite-smoke-7` 模式

這是固定版本的快速能力套裝，每個模型與參數組合會各自執行七道獨立 request：

| 題目 ID | 類別 | 測驗內容 |
| --- | --- | --- |
| `smoke7-math-01` | `math` | 折扣與稅額計算 |
| `smoke7-logic-01` | `logic` | 誠實者邏輯題 |
| `smoke7-reasoning-01` | `reasoning` | 權限鏈推理 |
| `smoke7-reading-01` | `reading` | 短文理解 |
| `smoke7-translation-01` | `translation` | 技術內容中翻英 |
| `smoke7-writing-01` | `writing` | 簡潔專業寫作 |
| `smoke7-coding-01` | `coding` | 保留順序去重程式題 |

題庫版本目前為 `1.0.0`。每題都遵循相同 schema：

```json
{
  "id": "smoke7-math-01",
  "category": "math",
  "title": "中英短標題",
  "prompt": "送給模型的完整題目",
  "expected_output": "參考答案或預期輸出形式",
  "evaluation_guide": "人工或未來自動評分準則"
}
```

套裝題目由程式固定管理，因此 UI 選到此模式時不開放修改一般 Benchmark Prompt。HTML 報告會增加 `Suite Questions / 套裝題庫` 與 `Question Statistics / 題目統計`，Excel 會增加 `Suite Questions` 與 `Question Stats` 工作表。

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
- 預設 Base URL 是 `http://localhost:8080/v1`
- Models 留空時會讀取 `/v1/models`，直接測試目前 server 已載入的模型，不會執行換模
- 從 `easy_llamacpp` catalog 勾選模型並啟用批次換模時，會透過 `Start_LCPP.ps1` 依序載入 GGUF
- 手動填入模型名稱但停用批次換模時，模型名稱作為 request 與報告辨識用途，不會替你載入模型
- `num_predict` 會映射為 `n_predict`
- `enable_thinking` 會映射為 `chat_template_kwargs.enable_thinking=true/false`

## 產出檔案

Benchmark 完成後，程式會自動建立 `Report/` 資料夾，並把本次 benchmark 產物集中放在裡面：

- `Report/bench_{backend}_{capability}_{timestamp}.html`
  - 完整 HTML 報告
  - 內嵌模型長條圖，直接在 Report Viewer 比較 Output TPS、TTFT、VRAM 峰值與效率分數
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
- `llm_expert_bench_crash.log`
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
- `Prompt Tokens`
  - 後端有提供時，顯示本次請求的 prompt token 數
- `Thinking Tokens`
  - 依後端保留下來的 thinking 文字估算；多數 OpenAI-compatible 後端不會單獨回報 reasoning token
- `Answer Tokens`
  - 依最終可見回答文字估算
- `Completion Tokens`
  - 優先採用後端 usage 回報；沒有 usage 時以 `Thinking Tokens + Answer Tokens` 估算
- `Total Tokens`
  - 每題 request 的總消耗，優先採用後端 usage；沒有 usage 時以 `Prompt Tokens + Completion Tokens` 估算
- `Token Count Source`
  - `backend_usage` 表示總量來自後端，`mixed` 表示部分來自後端，`estimated` 表示全部由文字估算
- `Prefill TPS (tok/s)`
  - 預填充速度；優先使用 Ollama 的 `prompt_eval_duration`，否則回退為 `prompt_tokens / TTFT`
- `Total Output (chars)`
  - 該次 run 最終保留的可見輸出總字數
- `Total Output Time (s)`
  - 從收到第一段 output 到串流結束的時間
- `Thinking Time (s)`
  - 從 request 開始到第一段可見回答；包含 prefill，以及第一段回答前的 reasoning / thinking 時間
- `Answer Time (s)`
  - 從第一段可見回答到串流結束；沒有可見回答時顯示 `N/A`
- `Thinking TPS (tok/s)`
  - 以保留下來的 thinking 文字估算 token 吞吐量
- `Output TPS (tok/s)`
  - 以最終 dialogue output 估算 token 吞吐量
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

一般 HTML 報告的 `Model Comparison / 模型長條圖對比` 會依模型彙整：

- `Output TPS`：越高越好
- `TTFT`：越低越好
- `VRAM Peak`：越低越好
- `Efficiency Score`：越高越好

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

執行完整單元測試：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

完整測試涵蓋：

- 正常文字串流分類
- 空回覆與中斷情境
- `tools` 模式下的 `tool_call` 判定
- reasoning / thinking 欄位萃取
- 報告摘要對 `N/A` 欄位的輸出

## Local Expert Battle 48：本地模型對戰

在本機 UI 的 Benchmark Mode 選擇：

```text
在地工程專家對戰 48 題（Local Expert Battle 48）
```

此模式可直接使用 `llama.cpp` 的 `llama-server` 與本地 GGUF 模型；模型欄位可用逗號填入多個模型名稱，會以相同 48 題逐一對戰。題庫定義放在 `local_expert_battle.json`，每個領域固定 12 題：

| 領域 | 題數 | 評分方式 |
| --- | ---: | --- |
| PLC 程式 | 12 | 人工覆核：ST／梯形邏輯、配方、故障診斷、通訊 |
| 工程計算推理 | 12 | 數值容差自動比對，可再人工覆核 |
| 繁中語境 | 12 | 必要語意詞群自動檢查，可再人工覆核 |
| 長文摘要 | 12 | 人工覆核：準確性、重點、是否有幻覺 |

### 可以自動評分的題目

48 題中共有 24 題具備自動評分規則，其餘 24 題需要人工覆核：

| 類型 | 題數 | 自動評分方式 | 建議 |
| --- | ---: | --- | --- |
| 工程計算推理 | 12 | 擷取回答中的數字，依標準答案及各題容許誤差計分 | 可自動計分，建議抽查單位與計算過程 |
| 繁中語境 | 12 | 將必要語意拆成詞群，每個詞群命中任一同義詞即得該組分數 | 自動初評後建議覆核語意與工單品質 |
| PLC 程式 | 12 | 無自動規則 | 人工判斷程式、安全優先順序與診斷品質 |
| 長文摘要 | 12 | 無自動規則 | 人工判斷忠實度、重點與是否產生幻覺 |

工程計算的 12 題與標準答案如下：

| 題號 | 題目 | 標準答案 | 容許誤差 |
| --- | --- | ---: | ---: |
| `eng-01` | 公分換算 | 13200 cm² | ±0.5 |
| `eng-02` | 馬力換算 | 10.06 hp | ±0.02 |
| `eng-03` | 扭矩估算 | 98.8 N·m | ±0.2 |
| `eng-04` | 流量單位 | 40.0 L/min | ±0.1 |
| `eng-05` | Darcy 壓損 | 13500 Pa | ±10 |
| `eng-06` | 圓管流速 | 0.85 m/s | ±0.02 |
| `eng-07` | 鋼板重量 | 94.2 kg | ±0.2 |
| `eng-08` | 混凝土用量 | 0.69 m³ | ±0.01 |
| `eng-09` | 電流估算 | 21.8 A | ±0.2 |
| `eng-10` | 熱負荷 | 6.28 kWh | ±0.02 |
| `eng-11` | 齒輪轉速 | 500 rpm | ±0.5 |
| `eng-12` | 泵浦水力功率 | 3.30 kW | ±0.02 |

繁中語境的 12 題為：乾／幹工單改寫、製／制用字校正、公分與厘米、台灣現場口語理解、簡繁混用校正、卡料事件摘要、跳電歧義釐清、鎖螺絲工作指示、台灣日期格式、「對一下」工作展開、外來語設備名稱及繁中安全告示。

自動分數範圍為 0–1。若一題包含多個檢查項目，分數為命中項目比例；目前 `0.70` 以上視為該題勝利。模型請求失敗時，自動分數為 0。人工覆核檔若提供同一執行或同一模型／題號的分數，人工分數會覆蓋自動分數。工程計算目前主要比對數值，尚未嚴格驗證單位，因此錯誤單位仍可能需要人工發現。

長文摘要題會在執行時從 `~/wiki/` 擷取約 2.4K 字的真實頁面；預設為 Windows 的 `C:\Users\<使用者>\wiki`。若 wiki 在其他位置，先設定：

```powershell
$env:DIY_LLM_WIKI_DIR = 'D:\your\wiki'
.\.venv\Scripts\python.exe .\llm_expert_bench.py
```

Benchmark 完成時會自動建立 Battle 結果 HTML 與人工覆核 JSON 範本。`report.py` 會輸出每模型的各領域／總體勝率與平均回應時間；總體勝率只使用「已自動評分或已人工覆核」的題目，避免未評題目被當成失敗。Battle HTML 另有總體勝率、平均分數、平均回應時間與四個領域勝率的長條圖對比。

```powershell
# 重跑或套用人工評分後重新輸出結果
.\.venv\Scripts\python.exe .\report.py `
  --input .\Report\bench_llama.cpp_local-expert-battle-48_YYYYMMDD_HHMMSS_outputs.jsonl `
  --review .\Report\bench_llama.cpp_local-expert-battle-48_YYYYMMDD_HHMMSS_manual_review.json `
  --output .\Report\battle_result.html
```

人工覆核 JSON 的每筆 `score` 為 0–1，`0.70` 以上算該題勝利。PLC 與摘要題請依題內 rubric 評分；工程與繁中題的自動分數可保留或改以人工分數覆蓋。

### easy_llamacpp 模型目錄

選擇 `llama.cpp` backend 時，UI 的模型清單會讀取同層專案 `../easy_llamacpp/json/model-index.json`，也就是 `easy_llamacpp` 每次掃描 GGUF 後更新的目錄。按下「Refresh Model List」即可重新讀取，點選模型可加入 `Models / 模型` 欄位；`mmproj` 視覺 sidecar 會自動排除。

模型選擇有兩種工作方式：

1. **未勾選任何模型**：Models 欄位保持空白，工具會直接連線 `http://localhost:8080/v1`，從 `/v1/models` 取得目前載入的模型名稱後開始測試。即使批次換模勾選框仍為開啟，也會自動忽略，不會重啟 server。
2. **勾選一個或多個 GGUF**：若批次換模開啟，工具會由 `easy_llamacpp` 依序載入每個 GGUF，並用同一組題目與參數測試。

若你的 launcher 不在同層，啟動前設定：

```powershell
$env:DIY_LLAMACPP_ROOT = 'D:\tools\easy_llamacpp'
.\.venv\Scripts\python.exe .\llm_expert_bench.py
```

注意：模型清單是方便選擇 GGUF 的 launcher catalog；同一個 llama-server 預設只會載入一個模型。下列批次切換功能會在同一個 port 順序載入並測試；若要平行測試，仍需自行啟動多個 llama-server 並使用不同 port。

UI 的 `Batch switch selected GGUF models / 自動依序切換所選 GGUF 並測試` 預設為開啟。勾選多個模型、設定好參數後按 Start，工具會自動：

1. 以 `easy_llamacpp/PS1/Start_LCPP.ps1` 停止舊 server 並載入下一個 GGUF。
2. 等待該模型在 `http://127.0.0.1:8080/v1/models` ready。
3. 用完全相同的題目、參數與 system prompt 跑完該模型，再切換下一個。

此功能只管理本機 `localhost`／`127.0.0.1` 的 llama-server，且已選模型必須是 `model-index.json` 中存在、磁碟上可找到的 GGUF；否則會在開始前顯示明確錯誤。若你自行管理 server 或要測遠端端點，可取消批次換模並手動填入模型名稱；若要直接使用目前 8080 server，將模型全部取消勾選並保持 Models 空白即可。

## 注意事項

- 這個專案以瀏覽器 UI 為主要操作介面，`--cli` 提供舊式互動選單；目前沒有用命令列旗標直接提交完整 benchmark 設定的模式
- 所有 benchmark 都是透過 OpenAI-compatible `/v1/chat/completions` 介面發送
- `tools` 模式目前內建的工具只有 `lookup_weather`
- `Ollama` 的模型清單抓不到時，不代表後端不能用，只是需要手動輸入模型名
- `llama.cpp` 只有在勾選 catalog 模型且啟用批次換模時才會自動載入 GGUF；Models 留空則直接使用目前 server

