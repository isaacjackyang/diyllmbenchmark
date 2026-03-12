import json
import subprocess
import sys
import threading
import time
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import questionary
import requests
from openai import OpenAI
from questionary import Choice

try:
    import msvcrt
except ImportError:
    msvcrt = None


NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,name,memory.used,memory.total",
    "--format=csv,noheader,nounits",
]


PARAM_INFO = {
    "temperature": {
        "label": "溫度 (Temperature)",
        "range": "0.0 - 2.0",
        "desc": "調高更有創意，調低更穩定；Qwen 類模型常用 0.1 或 1.0。",
        "default": "0.1, 0.8",
        "backends": ["ollama", "llama.cpp"],
        "backend_keys": {"ollama": "temperature", "llama.cpp": "temperature"},
    },
    "num_ctx": {
        "label": "上下文長度 (Num_Ctx)",
        "range": "128 - 262144",
        "desc": "可測長上下文與 TTFT 影響；此項為 Ollama 請求級參數。",
        "default": "4096, 8192",
        "backends": ["ollama"],
        "backend_keys": {"ollama": "num_ctx"},
    },
    "num_predict": {
        "label": "最大生成 Token",
        "range": "-1, 1 - 4096",
        "desc": "控制回覆長度；llama.cpp 會映射為 n_predict。",
        "default": "256, 512",
        "backends": ["ollama", "llama.cpp"],
        "backend_keys": {"ollama": "num_predict", "llama.cpp": "n_predict"},
    },
    "top_p": {
        "label": "核心採樣 (Top_P)",
        "range": "0.0 - 1.0",
        "desc": "限制候選詞機率總和，數值越低越保守。",
        "default": "0.8, 0.95",
        "backends": ["ollama", "llama.cpp"],
        "backend_keys": {"ollama": "top_p", "llama.cpp": "top_p"},
    },
    "min_p": {
        "label": "最小概率 (Min_P)",
        "range": "0.0 - 1.0",
        "desc": "過濾低機率雜訊詞；常見平衡點在 0.05 左右。",
        "default": "0.02, 0.05",
        "backends": ["ollama", "llama.cpp"],
        "backend_keys": {"ollama": "min_p", "llama.cpp": "min_p"},
    },
    "repeat_penalty": {
        "label": "重複懲罰 (Repeat Penalty)",
        "range": "1.0 - 2.0",
        "desc": "降低重複句與繞圈輸出。",
        "default": "1.05, 1.15",
        "backends": ["ollama", "llama.cpp"],
        "backend_keys": {"ollama": "repeat_penalty", "llama.cpp": "repeat_penalty"},
    },
    "num_gpu": {
        "label": "GPU 層數 / 顯存卸載",
        "range": "0 - 100",
        "desc": "對 Ollama TPS 影響很大；llama.cpp 通常在 server 啟動時設定。",
        "default": "25, 50",
        "backends": ["ollama"],
        "backend_keys": {"ollama": "num_gpu"},
    },
}

PARAM_GROUPS = {
    "🔥 生成核心": ["temperature", "num_ctx", "num_predict"],
    "⚖️ 採樣與懲罰": ["top_p", "min_p", "repeat_penalty"],
    "🖥️ 硬體與部署": ["num_gpu"],
}


CAPABILITY_OPTIONS = {
    "chat": {
        "label": "聊天能力",
        "description": "一般對話輸出，沿用目前的文字串流 benchmark。",
        "default_prompt": (
            "解釋一下 3D 列印使用 PETG 時，長期受力下的潛變（creep）風險，"
            "以及有哪些實際的改善方式。"
        ),
    },
    "tools": {
        "label": "Tools 調用能力",
        "description": "要求模型先呼叫工具，檢查是否真的輸出 tool_calls。",
        "default_prompt": (
            "請查詢台北今天的天氣。若你支援 tools 或 function calling，"
            "請先呼叫 `lookup_weather` 工具，不要直接回答。"
        ),
    },
}

TOOL_BENCHMARK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Look up the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name in Chinese or English.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Preferred temperature unit.",
                    },
                },
                "required": ["city"],
            },
        },
    }
]


def get_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [model["name"] for model in models if model.get("name")]
    except requests.RequestException:
        return []


def parse_csv_values(raw_text):
    values = []
    for item in raw_text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            if "." in item or "e" in item.lower():
                value = float(item)
                values.append(int(value) if value.is_integer() else value)
            else:
                values.append(int(item))
        except ValueError as exc:
            raise ValueError(f"無法解析數值：{item}") from exc

    if not values:
        raise ValueError("至少需要一個測試值。")

    return values


def ask_param_values(param_key):
    info = PARAM_INFO[param_key]
    while True:
        raw_value = questionary.text(
            f"輸入 {info['label']} 測試值 (逗號隔開):",
            default=info["default"],
        ).ask()
        if raw_value is None:
            return None
        try:
            return parse_csv_values(raw_value)
        except ValueError as exc:
            print(f"⚠️ {exc} 請重新輸入。")


def format_param_dict(params):
    if not params:
        return "預設參數"
    joined = ", ".join(f"{key}={value}" for key, value in params.items())
    return "{" + joined + "}"


def build_backend_options(backend, params):
    backend_options = {}
    for key, value in params.items():
        backend_key = PARAM_INFO[key]["backend_keys"].get(backend)
        if backend_key:
            backend_options[backend_key] = value
    return backend_options


def build_benchmark_messages(capability, prompt):
    if capability == "tools":
        return [
            {
                "role": "system",
                "content": (
                    "You are being benchmarked for tool calling. "
                    "If a suitable tool is provided, call the tool before answering."
                ),
            },
            {"role": "user", "content": prompt},
        ]
    return [{"role": "user", "content": prompt}]


def build_chat_request_payload(config, model, request_kwargs):
    capability = config.get("capability", "chat")
    payload = {
        "model": model,
        "messages": build_benchmark_messages(capability, config["prompt"]),
        "stream": True,
        **request_kwargs,
    }
    if capability == "tools":
        payload["tools"] = TOOL_BENCHMARK_TOOLS
        payload["tool_choice"] = "auto"
    return payload


def parse_non_content_types(value):
    if value is None:
        return set()
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    else:
        items = [str(item).strip() for item in value]
    return {item for item in items if item and item != "none"}


def adjust_classification_for_capability(classification, capability):
    if capability != "tools":
        return classification

    adjusted = classification.copy()
    non_content_types = parse_non_content_types(adjusted.get("Non_Content_Types"))
    if "tool_calls" in non_content_types:
        adjusted["Status"] = "ok"
        adjusted["Output_Category"] = "tool_call"
        finish_reason = adjusted.get("Finish_Reason") or "unknown"
        adjusted["Diagnosis"] = (
            "Received tool_calls payload during the tool benchmark "
            f"(finish_reason={finish_reason})."
        )
        return adjusted

    if adjusted["Status"] == "ok" and adjusted["Output_Category"] == "normal_content":
        adjusted["Status"] = "warning"
        adjusted["Output_Category"] = "text_reply_without_tool"
        adjusted["Diagnosis"] = (
            "Received textual content, but no tool_calls payload was emitted during the "
            "tool benchmark."
        )
    elif adjusted["Output_Category"] == "empty_reply":
        adjusted["Diagnosis"] = (
            "The tool benchmark finished without textual content or tool_calls payload."
        )
    elif adjusted["Output_Category"] == "non_content_stream":
        adjusted["Diagnosis"] = (
            "The tool benchmark returned non-content payloads, but none were tool_calls."
        )

    return adjusted


def query_nvidia_vram_snapshot():
    try:
        result = subprocess.run(
            NVIDIA_SMI_QUERY,
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    snapshot = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            continue
        try:
            snapshot.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mib": int(parts[2]),
                    "memory_total_mib": int(parts[3]),
                }
            )
        except ValueError:
            continue

    return snapshot or None


def empty_vram_metrics():
    return {
        "VRAM_Base_MiB": None,
        "VRAM_Peak_MiB": None,
        "VRAM_Delta_MiB": None,
        "VRAM_Detail": "N/A",
    }


def format_mib_value(value):
    return "N/A" if value is None or pd.isna(value) else f"{int(value)} MiB"


def format_numeric_value(value, digits):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def format_text_value(value, default="N/A"):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    text = str(value).strip()
    return text or default


def calculate_efficiency_score(tps, vram_peak_mib):
    if tps is None or pd.isna(tps):
        return None
    if vram_peak_mib is None or pd.isna(vram_peak_mib) or vram_peak_mib <= 0:
        return None
    score = tps / (vram_peak_mib / 1024)
    return round(score, 3)


def normalize_text_content(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return "".join(parts)
    return str(value)


def extract_delta_payload(delta):
    if delta is None:
        return {}

    if isinstance(delta, dict):
        raw_payload = delta
    elif hasattr(delta, "model_dump"):
        try:
            raw_payload = delta.model_dump(exclude_none=True)
        except TypeError:
            raw_payload = delta.model_dump()
    elif hasattr(delta, "dict"):
        try:
            raw_payload = delta.dict(exclude_none=True)
        except TypeError:
            raw_payload = delta.dict()
    else:
        raw_payload = vars(delta)

    payload = {}
    for key, value in raw_payload.items():
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        payload[key] = value

    return payload


def normalize_non_content_type(field_name):
    field_map = {
        "role": "role",
        "tool_calls": "tool_calls",
        "reasoning": "reasoning",
        "refusal": "refusal",
        "audio": "audio",
        "function_call": "tool_calls",
    }
    return field_map.get(field_name, "other")


def inspect_stream_chunk(chunk):
    chunk_info = {
        "content": "",
        "non_content_types": [],
        "finish_reason": None,
    }

    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return chunk_info

    choice = choices[0]
    delta_payload = extract_delta_payload(getattr(choice, "delta", None))
    chunk_info["content"] = normalize_text_content(delta_payload.pop("content", None))
    chunk_info["non_content_types"] = sorted(
        {normalize_non_content_type(field_name) for field_name in delta_payload}
    )
    chunk_info["finish_reason"] = getattr(choice, "finish_reason", None) or None
    return chunk_info


def classify_stream_result(
    chunk_records,
    start_time,
    end_time,
    first_event_time=None,
    first_content_time=None,
    error_message=None,
):
    total_chunks = len(chunk_records)
    content_chunks = 0
    non_content_chunks = 0
    non_content_types = set()
    finish_reason = None

    for record in chunk_records:
        if record["content"]:
            content_chunks += 1
        if record["non_content_types"]:
            non_content_chunks += 1
            non_content_types.update(record["non_content_types"])
        if record["finish_reason"]:
            finish_reason = record["finish_reason"]

    first_event_seconds = round(first_event_time - start_time, 3) if first_event_time is not None else None
    stream_duration_seconds = round(end_time - start_time, 3)

    if content_chunks > 0 and first_content_time is not None:
        ttft = round(first_content_time - start_time, 3)
        generation_time = end_time - first_content_time
        tps = round(content_chunks / generation_time, 2) if generation_time > 0 else 0.0
    else:
        ttft = None
        tps = None

    substantive_non_content_types = sorted(
        non_content_type for non_content_type in non_content_types if non_content_type != "role"
    )

    if content_chunks > 0:
        output_category = "normal_content"
        if error_message:
            status = "error"
            diagnosis = "Stream interrupted after textual content was received."
        elif not finish_reason:
            status = "warning"
            diagnosis = "Textual content was received, but the stream ended without a terminal finish_reason."
        else:
            status = "ok"
            diagnosis = f"Received textual content and completed with finish_reason={finish_reason}."
    elif error_message:
        status = "error"
        output_category = "early_stop"
        diagnosis = "Stream interrupted before any textual content was received."
    elif finish_reason:
        status = "warning"
        if substantive_non_content_types:
            output_category = "non_content_stream"
            diagnosis = (
                f"Completed with finish_reason={finish_reason} but only non-content payloads "
                f"were received: {', '.join(substantive_non_content_types)}."
            )
        else:
            output_category = "empty_reply"
            diagnosis = f"Completed with finish_reason={finish_reason} but no textual content was received."
    else:
        status = "warning"
        output_category = "early_stop"
        diagnosis = "Stream ended before any textual content or terminal finish_reason was received."

    return {
        "Status": status,
        "Output_Category": output_category,
        "Diagnosis": diagnosis,
        "Finish_Reason": finish_reason,
        "Total_Chunks": total_chunks,
        "Content_Chunks": content_chunks,
        "Non_Content_Chunks": non_content_chunks,
        "Non_Content_Types": ", ".join(sorted(non_content_types)) if non_content_types else "none",
        "First_Event_s": first_event_seconds,
        "Stream_Duration_s": stream_duration_seconds,
        "TTFT": ttft,
        "TPS": tps,
    }


def build_result_row(
    run_id,
    config,
    model,
    param_set,
    applied_params,
    display_params,
    classification,
    vram_metrics,
    output_text,
    error_message,
):
    efficiency_score = calculate_efficiency_score(classification["TPS"], vram_metrics["VRAM_Peak_MiB"])
    return {
        "Run_ID": run_id,
        "Status": classification["Status"],
        "Capability": config.get("capability", "chat"),
        "Output_Category": classification["Output_Category"],
        "Diagnosis": classification["Diagnosis"],
        "Finish_Reason": classification["Finish_Reason"],
        "Backend": config["backend"],
        "Model": model,
        "Params": param_set.copy(),
        "Applied_Params": applied_params.copy(),
        "Config_Str": display_params,
        "TPS": classification["TPS"],
        "TTFT": classification["TTFT"],
        "First_Event_s": classification["First_Event_s"],
        "Stream_Duration_s": classification["Stream_Duration_s"],
        "Total_Chunks": classification["Total_Chunks"],
        "Content_Chunks": classification["Content_Chunks"],
        "Non_Content_Chunks": classification["Non_Content_Chunks"],
        "Non_Content_Types": classification["Non_Content_Types"],
        "VRAM_Base_MiB": vram_metrics["VRAM_Base_MiB"],
        "VRAM_Peak_MiB": vram_metrics["VRAM_Peak_MiB"],
        "VRAM_Delta_MiB": vram_metrics["VRAM_Delta_MiB"],
        "VRAM_Detail": vram_metrics["VRAM_Detail"],
        "Efficiency_Score": efficiency_score,
        "Output_Chars": len(output_text),
        "Output_Text": output_text,
        "Error": error_message or "",
    }


def filter_eligible_results(df, capability="chat"):
    success_categories = {"tool_call"} if capability == "tools" else {"normal_content"}
    return df[(df["Status"] == "ok") & (df["Output_Category"].isin(success_categories))].copy()


def build_outcome_summary_dataframe(df):
    outcome_df = (
        df["Output_Category"]
        .fillna("unknown")
        .value_counts(dropna=False)
        .rename_axis("Output Category")
        .reset_index(name="Count")
    )
    return outcome_df


def summarize_vram_samples(samples):
    if not samples:
        return empty_vram_metrics()

    baseline_snapshot = samples[0]
    total_base_mib = sum(item["memory_used_mib"] for item in baseline_snapshot)
    total_peak_mib = max(sum(item["memory_used_mib"] for item in snapshot) for snapshot in samples)

    per_gpu = {}
    for item in baseline_snapshot:
        per_gpu[item["index"]] = {
            "index": item["index"],
            "name": item["name"],
            "memory_total_mib": item["memory_total_mib"],
            "base_mib": item["memory_used_mib"],
            "peak_mib": item["memory_used_mib"],
        }

    for snapshot in samples:
        for item in snapshot:
            state = per_gpu.setdefault(
                item["index"],
                {
                    "index": item["index"],
                    "name": item["name"],
                    "memory_total_mib": item["memory_total_mib"],
                    "base_mib": item["memory_used_mib"],
                    "peak_mib": item["memory_used_mib"],
                },
            )
            state["peak_mib"] = max(state["peak_mib"], item["memory_used_mib"])
            state["memory_total_mib"] = item["memory_total_mib"]
            state["name"] = item["name"]

    detail_parts = []
    for gpu_index in sorted(per_gpu):
        gpu = per_gpu[gpu_index]
        delta_mib = gpu["peak_mib"] - gpu["base_mib"]
        detail_parts.append(
            (
                f"GPU {gpu['index']} {gpu['name']}: "
                f"{gpu['base_mib']} -> {gpu['peak_mib']} / {gpu['memory_total_mib']} MiB "
                f"(+{delta_mib} MiB)"
            )
        )

    return {
        "VRAM_Base_MiB": total_base_mib,
        "VRAM_Peak_MiB": total_peak_mib,
        "VRAM_Delta_MiB": total_peak_mib - total_base_mib,
        "VRAM_Detail": " | ".join(detail_parts) if detail_parts else "N/A",
    }


class NvidiaVRAMMonitor:
    def __init__(self, interval_seconds=0.2):
        self.interval_seconds = interval_seconds
        self.samples = []
        self._stop_event = threading.Event()
        self._thread = None
        self.enabled = False

    def start(self):
        baseline_snapshot = query_nvidia_vram_snapshot()
        if not baseline_snapshot:
            return False

        self.samples = [baseline_snapshot]
        self.enabled = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return True

    def _poll_loop(self):
        while not self._stop_event.wait(self.interval_seconds):
            snapshot = query_nvidia_vram_snapshot()
            if snapshot:
                self.samples.append(snapshot)

    def stop(self):
        if not self.enabled:
            return empty_vram_metrics()

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1)

        final_snapshot = query_nvidia_vram_snapshot()
        if final_snapshot:
            self.samples.append(final_snapshot)

        return summarize_vram_samples(self.samples)


def build_summary_dataframe(df):
    summary_columns = [
        "Run_ID",
        "Status",
        "Output_Category",
        "Model",
        "Finish_Reason",
        "Config_Str",
        "TPS",
        "TTFT",
        "First_Event_s",
        "Content_Chunks",
        "Total_Chunks",
        "VRAM_Peak_MiB",
        "Efficiency_Score",
    ]
    if "Capability" in df.columns:
        summary_columns.insert(2, "Capability")

    summary_df = df[summary_columns].copy()
    summary_df["Finish_Reason"] = summary_df["Finish_Reason"].apply(format_text_value)
    summary_df["TPS"] = summary_df["TPS"].apply(lambda value: format_numeric_value(value, 2))
    summary_df["TTFT"] = summary_df["TTFT"].apply(lambda value: format_numeric_value(value, 3))
    summary_df["First_Event_s"] = summary_df["First_Event_s"].apply(
        lambda value: format_numeric_value(value, 3)
    )
    summary_df["VRAM_Peak_MiB"] = summary_df["VRAM_Peak_MiB"].apply(
        lambda value: "N/A" if pd.isna(value) else int(value)
    )
    summary_df["Efficiency_Score"] = summary_df["Efficiency_Score"].apply(
        lambda value: format_numeric_value(value, 3)
    )
    summary_df["Chunks (content/total)"] = summary_df.apply(
        lambda row: f"{int(row['Content_Chunks'])}/{int(row['Total_Chunks'])}",
        axis=1,
    )
    summary_df = summary_df.drop(columns=["Content_Chunks", "Total_Chunks"])
    return summary_df.rename(
        columns={
            "Run_ID": "Run",
            "Capability": "Capability",
            "Output_Category": "Output Category",
            "Finish_Reason": "Finish Reason",
            "Config_Str": "Config",
            "TPS": "TPS (chunk/s)",
            "TTFT": "TTFT (s)",
            "First_Event_s": "First Event (s)",
            "VRAM_Peak_MiB": "VRAM Peak (MiB)",
            "Efficiency_Score": "Efficiency Score (TPS/GiB Peak)",
        }
    )


def dataframe_to_text_table(df):
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def serialize_result_value(value):
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def save_raw_outputs(df, report_stem):
    output_path = Path(f"{report_stem}_outputs.jsonl")
    with output_path.open("w", encoding="utf-8") as file:
        for _, row in df.iterrows():
            payload = {column: serialize_result_value(value) for column, value in row.items()}
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return output_path


def select_plot_dataframe(df, capability="chat"):
    if capability == "tools":
        return df.copy(), False

    eligible_df = filter_eligible_results(df, capability=capability)
    if eligible_df.empty:
        return df.copy(), True
    return eligible_df.copy(), False


def normalize_output_text(text):
    normalized = (text or "").replace("\r\n", "\n").strip()
    return normalized or "[No text returned]"


def pause_before_exit():
    if not sys.stdin.isatty():
        return

    try:
        if msvcrt is not None:
            print("\n按任意鍵結束...", end="", flush=True)
            msvcrt.getch()
            print()
        else:
            input("\n按 Enter 結束...")
    except (EOFError, KeyboardInterrupt):
        pass


def select_models_and_url(backend):
    if backend == "ollama":
        detected_models = get_ollama_models()
        models = []
        if detected_models:
            models = questionary.checkbox(
                "選擇測試模型:",
                choices=detected_models,
            ).ask() or []

        if not models:
            manual_input = questionary.text(
                "請輸入 Ollama 模型名稱 (逗號隔開):",
                default="qwen3.5:latest",
            ).ask()
            models = [name.strip() for name in (manual_input or "").split(",") if name.strip()]

        return "http://localhost:11434/v1", models

    port = questionary.text(
        "請輸入 llama-server 端口:",
        default="8080",
    ).ask()
    model_names = questionary.text(
        "請輸入載入中的模型名稱 (逗號隔開，僅作識別用):",
        default="llama.cpp-model",
    ).ask()
    models = [name.strip() for name in (model_names or "").split(",") if name.strip()]
    return f"http://localhost:{port or '8080'}/v1", models


def interactive_config():
    print("\n" + "═" * 62)
    print("🏆 LLM 專家參數 Benchmark 工具 V3")
    print("═" * 62)

    backend = questionary.select(
        "請選擇測試後端:",
        choices=[
            Choice("🦙 Ollama", value="ollama"),
            Choice("🏗️ llama.cpp (需先啟動 llama-server)", value="llama.cpp"),
        ],
    ).ask()
    if not backend:
        return None

    url, models = select_models_and_url(backend)
    if not models:
        print("⚠️ 沒有可用模型，已取消。")
        return None

    available_groups = {
        group_name: [key for key in param_keys if backend in PARAM_INFO[key]["backends"]]
        for group_name, param_keys in PARAM_GROUPS.items()
    }
    available_groups = {name: keys for name, keys in available_groups.items() if keys}

    selected_groups = questionary.checkbox(
        "選擇想測試的參數類別 (可不選，代表只比模型預設值):",
        choices=[
            Choice(f"{group_name} ({len(param_keys)} 項)", value=group_name)
            for group_name, param_keys in available_groups.items()
        ],
    ).ask() or []

    final_params = {}
    for group_name in selected_groups:
        param_keys = available_groups[group_name]
        selected_params = questionary.checkbox(
            f"勾選 {group_name} 要測試的參數:",
            choices=[
                Choice(
                    title=(
                        f"{PARAM_INFO[key]['label']} | 範圍: {PARAM_INFO[key]['range']} | "
                        f"{PARAM_INFO[key]['desc']}"
                    ),
                    value=key,
                )
                for key in param_keys
            ],
        ).ask() or []

        for key in selected_params:
            values = ask_param_values(key)
            if values is None:
                return None
            final_params[key] = values

    prompt = questionary.text(
        "測試 Prompt:",
        default="詳細解釋 3D 列印中，PETG 材質發生蠕變 (Creep) 的溫度臨界點。",
    ).ask()
    if prompt is None:
        return None

    return {
        "backend": backend,
        "url": url,
        "models": models,
        "params": final_params,
        "prompt": prompt,
    }


def run_bench(config):
    client = OpenAI(base_url=config["url"], api_key="sk-no-key-needed")
    param_keys = list(config["params"].keys())
    param_values = [config["params"][key] for key in param_keys]
    combos = [dict(zip(param_keys, combo)) for combo in product(*param_values)] if param_keys else [{}]

    results = []
    total_runs = len(config["models"]) * len(combos)
    vram_monitoring_enabled = query_nvidia_vram_snapshot() is not None
    config["vram_monitoring"] = "nvidia-smi" if vram_monitoring_enabled else "unavailable"
    print(f"\n⚡ 啟動測試，共 {total_runs} 組配置。")
    print("📌 TPS 以串流回傳片段估算，適合做相對比較。")
    if vram_monitoring_enabled:
        print("🧠 顯存監控: 已啟用 nvidia-smi 取樣。")
    else:
        print("🧠 顯存監控: 未偵測到 nvidia-smi，報告將顯示 N/A。")

    run_index = 0
    for model in config["models"]:
        for param_set in combos:
            run_index += 1
            applied_params = build_backend_options(config["backend"], param_set)
            display_params = format_param_dict(param_set)
            print(f"[{run_index}/{total_runs}] {model} | {display_params}")

            request_kwargs = {}
            if applied_params:
                request_kwargs["extra_body"] = (
                    {"options": applied_params}
                    if config["backend"] == "ollama"
                    else applied_params
                )

            start_time = time.time()
            first_event_time = None
            first_content_time = None
            output_parts = []
            chunk_records = []
            vram_monitor = NvidiaVRAMMonitor() if vram_monitoring_enabled else None
            vram_metrics = empty_vram_metrics()
            if vram_monitor is not None and not vram_monitor.start():
                vram_monitor = None
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": config["prompt"]}],
                    stream=True,
                    **request_kwargs,
                )

                for chunk in stream:
                    event_time = time.time()
                    if first_event_time is None:
                        first_event_time = event_time

                    chunk_info = inspect_stream_chunk(chunk)
                    chunk_records.append(chunk_info)

                    if chunk_info["content"] and first_content_time is None:
                        first_content_time = event_time
                    if chunk_info["content"]:
                        output_parts.append(chunk_info["content"])

                end_time = time.time()
                if vram_monitor is not None:
                    vram_metrics = vram_monitor.stop()
                output_text = "".join(output_parts)
                classification = classify_stream_result(
                    chunk_records=chunk_records,
                    start_time=start_time,
                    end_time=end_time,
                    first_event_time=first_event_time,
                    first_content_time=first_content_time,
                    error_message=None,
                )
                results.append(
                    build_result_row(
                        run_id=run_index,
                        config=config,
                        model=model,
                        param_set=param_set,
                        applied_params=applied_params,
                        display_params=display_params,
                        classification=classification,
                        vram_metrics=vram_metrics,
                        output_text=output_text,
                        error_message=None,
                    )
                )
            except Exception as exc:
                end_time = time.time()
                if vram_monitor is not None:
                    vram_metrics = vram_monitor.stop()
                output_text = "".join(output_parts)
                classification = classify_stream_result(
                    chunk_records=chunk_records,
                    start_time=start_time,
                    end_time=end_time,
                    first_event_time=first_event_time,
                    first_content_time=first_content_time,
                    error_message=str(exc),
                )
                print(f"❌ 失敗: {exc}")
                results.append(
                    build_result_row(
                        run_id=run_index,
                        config=config,
                        model=model,
                        param_set=param_set,
                        applied_params=applied_params,
                        display_params=display_params,
                        classification=classification,
                        vram_metrics=vram_metrics,
                        output_text=output_text,
                        error_message=str(exc),
                    )
                )

    return pd.DataFrame(results)


def plot_results(df, output_path, capability="chat"):
    if df.empty:
        return None

    plot_df, used_fallback = select_plot_dataframe(df, capability=capability)
    plot_df = plot_df.copy()

    def build_label(row):
        label = f"{row['Model']} | {row['Config_Str']}"
        return label if len(label) <= 42 else label[:39] + "..."

    def build_run_color(row):
        if row["Output_Category"] == "tool_call":
            return "seagreen"
        if row["Status"] == "ok":
            return "skyblue"
        if row["Status"] == "warning":
            return "darkorange"
        return "indianred"

    plot_df["Plot_Label"] = plot_df.apply(build_label, axis=1)
    plot_df["Plot_Color"] = plot_df.apply(build_run_color, axis=1)

    if capability == "tools":
        first_event_values = plot_df["First_Event_s"].fillna(0)
        duration_values = plot_df["Stream_Duration_s"].fillna(0)
        outcome_df = build_outcome_summary_dataframe(plot_df)

        fig, axes = plt.subplots(3, 1, figsize=(14, 15))

        axes[0].bar(plot_df["Plot_Label"], first_event_values, color=plot_df["Plot_Color"])
        if plot_df["First_Event_s"].notna().any():
            axes[0].axhline(
                y=plot_df["First_Event_s"].min(),
                color="green",
                linestyle="--",
                alpha=0.3,
            )
        axes[0].set_title("Tools Benchmark First Event Comparison")
        axes[0].set_ylabel("First Event (s)")

        axes[1].bar(plot_df["Plot_Label"], duration_values, color=plot_df["Plot_Color"])
        if plot_df["Stream_Duration_s"].notna().any():
            axes[1].axhline(
                y=plot_df["Stream_Duration_s"].min(),
                color="green",
                linestyle="--",
                alpha=0.3,
            )
        axes[1].set_title("Tools Benchmark Stream Duration Comparison")
        axes[1].set_ylabel("Stream Duration (s)")

        axes[2].bar(outcome_df["Output Category"], outcome_df["Count"], color="steelblue")
        axes[2].set_title("Tools Outcome Category Counts")
        axes[2].set_ylabel("Runs")
        axes[2].set_xlabel("Output Category")

        axes[0].tick_params(axis="x", rotation=35)
        axes[1].tick_params(axis="x", rotation=35)
        axes[2].tick_params(axis="x", rotation=20)
    else:
        eligible_df = filter_eligible_results(plot_df, capability=capability)
        if not eligible_df.empty:
            max_tps = eligible_df["TPS"].max()
            min_ttft = eligible_df["TTFT"].min()
            colors = ["gold" if tps == max_tps else "skyblue" for tps in eligible_df["TPS"]]
            ttft_colors = [
                "lightgreen" if ttft == min_ttft else "salmon" for ttft in eligible_df["TTFT"]
            ]
            has_vram_data = eligible_df["VRAM_Peak_MiB"].notna().any()
            has_efficiency_data = eligible_df["Efficiency_Score"].notna().any()

            subplot_count = 4 if has_efficiency_data else 3 if has_vram_data else 2
            fig, axes = plt.subplots(subplot_count, 1, figsize=(13, 5 * subplot_count), sharex=True)
            if subplot_count == 1:
                axes = [axes]

            axes[0].bar(eligible_df["Plot_Label"], eligible_df["TPS"], color=colors)
            axes[0].axhline(y=max_tps, color="red", linestyle="--", alpha=0.3)
            axes[0].set_title("Throughput Comparison")
            axes[0].set_ylabel("TPS (chunk/s)")

            axes[1].bar(eligible_df["Plot_Label"], eligible_df["TTFT"], color=ttft_colors)
            axes[1].axhline(y=min_ttft, color="green", linestyle="--", alpha=0.3)
            axes[1].set_title("First Token Latency Comparison")
            axes[1].set_ylabel("TTFT (s)")

            if has_vram_data:
                min_vram_peak = eligible_df["VRAM_Peak_MiB"].min()
                vram_colors = [
                    "lightgreen" if peak == min_vram_peak else "mediumpurple"
                    for peak in eligible_df["VRAM_Peak_MiB"]
                ]
                axes[2].bar(eligible_df["Plot_Label"], eligible_df["VRAM_Peak_MiB"], color=vram_colors)
                axes[2].axhline(y=min_vram_peak, color="green", linestyle="--", alpha=0.3)
                axes[2].set_title("VRAM Peak Comparison")
                axes[2].set_ylabel("VRAM Peak (MiB)")
                if has_efficiency_data:
                    max_efficiency = eligible_df["Efficiency_Score"].max()
                    efficiency_colors = [
                        "gold" if score == max_efficiency else "steelblue"
                        for score in eligible_df["Efficiency_Score"]
                    ]
                    axes[3].bar(
                        eligible_df["Plot_Label"],
                        eligible_df["Efficiency_Score"],
                        color=efficiency_colors,
                    )
                    axes[3].axhline(y=max_efficiency, color="orange", linestyle="--", alpha=0.3)
                    axes[3].set_title("Efficiency Score Comparison")
                    axes[3].set_ylabel("TPS/GiB Peak")
                    axes[3].set_xlabel("Model | Config")
                else:
                    axes[2].set_xlabel("Model | Config")
            else:
                axes[1].set_xlabel("Model | Config")
        else:
            status_df = (
                plot_df["Status"]
                .fillna("unknown")
                .value_counts(dropna=False)
                .rename_axis("Status")
                .reset_index(name="Count")
            )
            outcome_df = build_outcome_summary_dataframe(plot_df)
            fig, axes = plt.subplots(2, 1, figsize=(13, 10))

            axes[0].bar(status_df["Status"], status_df["Count"], color="steelblue")
            axes[0].set_title("Run Status Counts")
            axes[0].set_ylabel("Runs")

            axes[1].bar(outcome_df["Output Category"], outcome_df["Count"], color="slategray")
            title = "Outcome Category Counts"
            if used_fallback:
                title += " (Fallback)"
            axes[1].set_title(title)
            axes[1].set_ylabel("Runs")
            axes[1].set_xlabel("Output Category")
            axes[0].tick_params(axis="x", rotation=20)
            axes[1].tick_params(axis="x", rotation=20)

    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return Path(output_path)


def save_markdown_report(df, config, report_stem):
    report_path = Path(f"{report_stem}.md")
    summary_df = build_summary_dataframe(df)
    outcome_summary_df = build_outcome_summary_dataframe(df)

    with report_path.open("w", encoding="utf-8") as file:
        file.write("# Benchmark Report\n\n")
        file.write(f"- Backend: {config['backend']}\n")
        file.write(f"- Base URL: {config['url']}\n")
        file.write(f"- Models: {', '.join(config['models'])}\n")
        file.write(f"- Prompt: {config['prompt']}\n")
        file.write("- Note: TPS is estimated from streaming content chunks for relative comparison.\n\n")
        file.write("## Environment Notes\n\n")
        file.write(f"- VRAM monitoring: {config.get('vram_monitoring', 'unavailable')}\n\n")
        file.write("## Metric Notes\n\n")
        file.write("- `TPS (chunk/s)`: 每秒收到的文字內容 chunk 數，代表輸出吞吐速度；數值越高通常越快。\n")
        file.write("- `TTFT (s)`: Time To First Token，從送出請求到收到第一段文字內容的秒數；數值越低通常越快。\n")
        file.write("- `First Event (s)`: 從送出請求到收到第一個串流事件的秒數，包含 role 或非文字事件。\n")
        file.write("- `VRAM Peak (MiB)`: 測試期間輪詢到的 NVIDIA GPU 總顯存最高占用，用來比較實際壓力。\n")
        file.write("- `Efficiency Score (TPS/GiB Peak)`: `TPS / (VRAM Peak in GiB)`，代表每 1 GiB 峰值顯存換到多少輸出速度；數值越高越划算。\n")
        file.write("- `TPS (chunk/s)` 與 `TTFT (s)` 在沒有任何文字內容輸出時會顯示 `N/A`。\n\n")
        file.write("## Output Diagnosis Notes\n\n")
        file.write("- `empty_reply`: 串流正常結束，但沒有任何文字內容。\n")
        file.write("- `non_content_stream`: 串流有事件，但只有非文字 payload，例如 `tool_calls` 或 `reasoning`。\n")
        file.write("- `early_stop`: 串流在產生任何文字前提早結束，或在前期就被異常中斷。\n\n")
        file.write("## Summary\n\n")
        file.write(summary_df.to_markdown(index=False))
        file.write("\n\n## Outcome Summary\n\n")
        file.write(outcome_summary_df.to_markdown(index=False))
        file.write("\n\n## Generated Outputs\n")

        for _, row in df.iterrows():
            params_json = json.dumps(row["Params"], ensure_ascii=False)
            applied_params_json = json.dumps(row["Applied_Params"], ensure_ascii=False)
            file.write(f"\n### Run {row['Run_ID']}\n\n")
            file.write(f"- Status: {row['Status']}\n")
            file.write(f"- Output Category: {row['Output_Category']}\n")
            file.write(f"- Diagnosis: {row['Diagnosis']}\n")
            file.write(f"- Finish Reason: {format_text_value(row['Finish_Reason'])}\n")
            file.write(f"- Backend: {row['Backend']}\n")
            file.write(f"- Model: {row['Model']}\n")
            file.write(f"- Params: `{params_json}`\n")
            file.write(f"- Applied Params: `{applied_params_json}`\n")
            file.write(f"- TPS: {format_numeric_value(row['TPS'], 2)} chunk/s\n")
            file.write(f"- TTFT: {format_numeric_value(row['TTFT'], 3)} s\n")
            file.write(f"- First Event: {format_numeric_value(row['First_Event_s'], 3)} s\n")
            file.write(f"- Stream Duration: {format_numeric_value(row['Stream_Duration_s'], 3)} s\n")
            file.write(f"- Total Chunks: {int(row['Total_Chunks'])}\n")
            file.write(f"- Content Chunks: {int(row['Content_Chunks'])}\n")
            file.write(f"- Non-Content Chunks: {int(row['Non_Content_Chunks'])}\n")
            file.write(f"- Non-Content Types: {row['Non_Content_Types']}\n")
            file.write(f"- VRAM Base: {format_mib_value(row['VRAM_Base_MiB'])}\n")
            file.write(f"- VRAM Peak: {format_mib_value(row['VRAM_Peak_MiB'])}\n")
            file.write(f"- VRAM Delta: {format_mib_value(row['VRAM_Delta_MiB'])}\n")
            file.write(f"- VRAM Detail: {row['VRAM_Detail']}\n")
            file.write(
                f"- Efficiency Score: "
                f"{format_numeric_value(row['Efficiency_Score'], 3)} TPS/GiB Peak\n"
            )
            file.write(f"- Output Chars: {row['Output_Chars']}\n")
            if row["Error"]:
                file.write(f"- Error: {row['Error']}\n")
            file.write("\n```text\n")
            file.write(normalize_output_text(row["Output_Text"]))
            file.write("\n```\n")

    return report_path


def export_best_config(df, config):
    eligible_df = filter_eligible_results(df)
    if eligible_df.empty:
        print("⚠️ 沒有正常文字輸出的結果可匯出最佳配置。")
        return None

    best_row = eligible_df.loc[eligible_df["TPS"].idxmax()]
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": best_row["Backend"],
        "model": best_row["Model"],
        "params": best_row["Params"],
        "applied_params": best_row["Applied_Params"],
        "status": best_row["Status"],
        "output_category": best_row["Output_Category"],
        "finish_reason": best_row["Finish_Reason"],
        "diagnosis": best_row["Diagnosis"],
        "tps": best_row["TPS"],
        "ttft": best_row["TTFT"],
        "first_event_s": best_row["First_Event_s"],
        "stream_duration_s": best_row["Stream_Duration_s"],
        "vram_base_mib": best_row["VRAM_Base_MiB"],
        "vram_peak_mib": best_row["VRAM_Peak_MiB"],
        "vram_delta_mib": best_row["VRAM_Delta_MiB"],
        "vram_detail": best_row["VRAM_Detail"],
        "efficiency_score_tps_per_gib_peak": best_row["Efficiency_Score"],
        "prompt": config["prompt"],
    }

    with open("best_config.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=4)

    print("\n" + "⭐" * 18)
    print(f"🏆 性能冠軍: {best_row['Model']}")
    print(f"🚀 最高速度: {format_numeric_value(best_row['TPS'], 2)} TPS")
    print(f"⚙️ 最佳配置: {best_row['Config_Str']}")
    print(f"🧠 顯存峰值: {format_mib_value(best_row['VRAM_Peak_MiB'])}")
    if pd.notna(best_row["Efficiency_Score"]):
        print(f"📊 效率分數: {format_numeric_value(best_row['Efficiency_Score'], 3)} TPS/GiB Peak")
    print("⭐" * 18)
    print("✅ 已保存 best_config.json")

    if best_row["Backend"] == "ollama":
        with open("Ollama_Modelfile_Suggest", "w", encoding="utf-8") as file:
            file.write(f"FROM {best_row['Model']}\n")
            for key, value in best_row["Applied_Params"].items():
                file.write(f"PARAMETER {key} {value}\n")
        print("✅ 已生成 Ollama_Modelfile_Suggest")
    else:
        print("ℹ️ 本次後端不是 Ollama，略過 Modelfile 建議。")

    return payload


def plot_results(df, output_path):
    eligible_df = filter_eligible_results(df)
    if eligible_df.empty:
        return None

    def build_label(row):
        label = f"{row['Model']} | {row['Config_Str']}"
        return label if len(label) <= 42 else label[:39] + "..."

    eligible_df["Plot_Label"] = eligible_df.apply(build_label, axis=1)
    max_tps = eligible_df["TPS"].max()
    min_ttft = eligible_df["TTFT"].min()
    colors = ["gold" if tps == max_tps else "skyblue" for tps in eligible_df["TPS"]]
    ttft_colors = ["lightgreen" if ttft == min_ttft else "salmon" for ttft in eligible_df["TTFT"]]
    has_vram_data = eligible_df["VRAM_Peak_MiB"].notna().any()
    has_efficiency_data = eligible_df["Efficiency_Score"].notna().any()

    subplot_count = 4 if has_efficiency_data else 3 if has_vram_data else 2
    fig, axes = plt.subplots(subplot_count, 1, figsize=(13, 5 * subplot_count), sharex=True)
    if subplot_count == 1:
        axes = [axes]

    axes[0].bar(eligible_df["Plot_Label"], eligible_df["TPS"], color=colors)
    axes[0].axhline(y=max_tps, color="red", linestyle="--", alpha=0.3)
    axes[0].set_title("Throughput Comparison")
    axes[0].set_ylabel("TPS (chunk/s)")

    axes[1].bar(eligible_df["Plot_Label"], eligible_df["TTFT"], color=ttft_colors)
    axes[1].axhline(y=min_ttft, color="green", linestyle="--", alpha=0.3)
    axes[1].set_title("First Token Latency Comparison")
    axes[1].set_ylabel("TTFT (s)")

    if has_vram_data:
        min_vram_peak = eligible_df["VRAM_Peak_MiB"].min()
        vram_colors = [
            "lightgreen" if peak == min_vram_peak else "mediumpurple"
            for peak in eligible_df["VRAM_Peak_MiB"]
        ]
        axes[2].bar(eligible_df["Plot_Label"], eligible_df["VRAM_Peak_MiB"], color=vram_colors)
        axes[2].axhline(y=min_vram_peak, color="green", linestyle="--", alpha=0.3)
        axes[2].set_title("VRAM Peak Comparison")
        axes[2].set_ylabel("VRAM Peak (MiB)")
        if has_efficiency_data:
            max_efficiency = eligible_df["Efficiency_Score"].max()
            efficiency_colors = [
                "gold" if score == max_efficiency else "steelblue"
                for score in eligible_df["Efficiency_Score"]
            ]
            axes[3].bar(
                eligible_df["Plot_Label"],
                eligible_df["Efficiency_Score"],
                color=efficiency_colors,
            )
            axes[3].axhline(y=max_efficiency, color="orange", linestyle="--", alpha=0.3)
            axes[3].set_title("Efficiency Score Comparison")
            axes[3].set_ylabel("TPS/GiB Peak")
            axes[3].set_xlabel("Model | Config")
        else:
            axes[2].set_xlabel("Model | Config")
    else:
        axes[1].set_xlabel("Model | Config")

    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return Path(output_path)


def main():
    config = interactive_config()
    if not config:
        return

    results_df = run_bench(config)
    if results_df.empty:
        print("⚠️ 沒有任何測試結果。")
        return

    ok_count = int((results_df["Status"] == "ok").sum())
    warning_count = int((results_df["Status"] == "warning").sum())
    error_count = int((results_df["Status"] == "error").sum())

    print("\n" + "═" * 62)
    console_df = build_summary_dataframe(results_df)[
        [
            "Run",
            "Status",
            "Output Category",
            "Finish Reason",
            "Model",
            "TPS (chunk/s)",
            "TTFT (s)",
            "First Event (s)",
            "Chunks (content/total)",
            "Config",
        ]
    ]
    print(console_df.to_markdown(index=False))
    print(f"\n📌 結果統計: ok={ok_count}, warning={warning_count}, error={error_count}")

    report_stem = f"bench_{config['backend']}_{time.strftime('%Y%m%d_%H%M%S')}"
    report_path = save_markdown_report(results_df, config, report_stem)
    chart_path = plot_results(results_df, f"{report_stem}.png")
    export_best_config(results_df, config)

    print(f"\n✅ 報告: {report_path}")
    if chart_path:
        print(f"📈 圖表: {chart_path}")
    else:
        print("ℹ️ 沒有可供繪圖的正常文字輸出結果。")


def interactive_config():
    print("\n" + "=" * 62)
    print("LLM Benchmark V3")
    print("=" * 62)

    backend = questionary.select(
        "請選擇測試後端:",
        choices=[
            Choice("Ollama", value="ollama"),
            Choice("llama.cpp（llama-server）", value="llama.cpp"),
        ],
    ).ask()
    if not backend:
        return None

    capability = questionary.select(
        "請選擇要測試的能力:",
        choices=[
            Choice(
                f"{info['label']} | {info['description']}",
                value=capability_key,
            )
            for capability_key, info in CAPABILITY_OPTIONS.items()
        ],
    ).ask()
    if not capability:
        return None

    url, models = select_models_and_url(backend)
    if not models:
        print("⚠️ 沒有可用模型，已取消。")
        return None

    available_groups = {
        group_name: [key for key in param_keys if backend in PARAM_INFO[key]["backends"]]
        for group_name, param_keys in PARAM_GROUPS.items()
    }
    available_groups = {name: keys for name, keys in available_groups.items() if keys}

    selected_groups = questionary.checkbox(
        "選擇想測試的參數類別（可不選，代表只比模型預設值）:",
        choices=[
            Choice(f"{group_name} ({len(param_keys)} 個)", value=group_name)
            for group_name, param_keys in available_groups.items()
        ],
    ).ask() or []

    final_params = {}
    for group_name in selected_groups:
        param_keys = available_groups[group_name]
        selected_params = questionary.checkbox(
            f"選擇 {group_name} 內要測試的參數:",
            choices=[
                Choice(
                    title=(
                        f"{PARAM_INFO[key]['label']} | 範圍: {PARAM_INFO[key]['range']} | "
                        f"{PARAM_INFO[key]['desc']}"
                    ),
                    value=key,
                )
                for key in param_keys
            ],
        ).ask() or []

        for key in selected_params:
            values = ask_param_values(key)
            if values is None:
                return None
            final_params[key] = values

    prompt = questionary.text(
        "測試 Prompt:",
        default=CAPABILITY_OPTIONS[capability]["default_prompt"],
    ).ask()
    if prompt is None:
        return None

    return {
        "backend": backend,
        "capability": capability,
        "url": url,
        "models": models,
        "params": final_params,
        "prompt": prompt,
    }


def run_bench(config):
    client = OpenAI(base_url=config["url"], api_key="sk-no-key-needed")
    capability = config.get("capability", "chat")
    param_keys = list(config["params"].keys())
    param_values = [config["params"][key] for key in param_keys]
    combos = [dict(zip(param_keys, combo)) for combo in product(*param_values)] if param_keys else [{}]

    results = []
    total_runs = len(config["models"]) * len(combos)
    vram_monitoring_enabled = query_nvidia_vram_snapshot() is not None
    config["vram_monitoring"] = "nvidia-smi" if vram_monitoring_enabled else "unavailable"

    capability_label = CAPABILITY_OPTIONS.get(capability, {}).get("label", capability)
    print(f"\n開始測試，共 {total_runs} 組，模式: {capability_label}")
    print("TPS 與 TTFT 適用於文字輸出；tools 模式主要看 First Event 與 tool_call 成功率。")
    if vram_monitoring_enabled:
        print("VRAM 監控: 已啟用 nvidia-smi")
    else:
        print("VRAM 監控: 未偵測到 nvidia-smi，相關欄位將顯示 N/A")

    run_index = 0
    for model in config["models"]:
        for param_set in combos:
            run_index += 1
            applied_params = build_backend_options(config["backend"], param_set)
            display_params = format_param_dict(param_set)
            print(f"[{run_index}/{total_runs}] {model} | {display_params}")

            request_kwargs = {}
            if applied_params:
                request_kwargs["extra_body"] = (
                    {"options": applied_params}
                    if config["backend"] == "ollama"
                    else applied_params
                )

            start_time = time.time()
            first_event_time = None
            first_content_time = None
            output_parts = []
            chunk_records = []
            vram_monitor = NvidiaVRAMMonitor() if vram_monitoring_enabled else None
            vram_metrics = empty_vram_metrics()
            if vram_monitor is not None and not vram_monitor.start():
                vram_monitor = None

            try:
                request_payload = build_chat_request_payload(config, model, request_kwargs)
                stream = client.chat.completions.create(**request_payload)

                for chunk in stream:
                    event_time = time.time()
                    if first_event_time is None:
                        first_event_time = event_time

                    chunk_info = inspect_stream_chunk(chunk)
                    chunk_records.append(chunk_info)

                    if chunk_info["content"] and first_content_time is None:
                        first_content_time = event_time
                    if chunk_info["content"]:
                        output_parts.append(chunk_info["content"])

                end_time = time.time()
                if vram_monitor is not None:
                    vram_metrics = vram_monitor.stop()
                output_text = "".join(output_parts)
                classification = classify_stream_result(
                    chunk_records=chunk_records,
                    start_time=start_time,
                    end_time=end_time,
                    first_event_time=first_event_time,
                    first_content_time=first_content_time,
                    error_message=None,
                )
                classification = adjust_classification_for_capability(classification, capability)
                results.append(
                    build_result_row(
                        run_id=run_index,
                        config=config,
                        model=model,
                        param_set=param_set,
                        applied_params=applied_params,
                        display_params=display_params,
                        classification=classification,
                        vram_metrics=vram_metrics,
                        output_text=output_text,
                        error_message=None,
                    )
                )
            except Exception as exc:
                end_time = time.time()
                if vram_monitor is not None:
                    vram_metrics = vram_monitor.stop()
                output_text = "".join(output_parts)
                classification = classify_stream_result(
                    chunk_records=chunk_records,
                    start_time=start_time,
                    end_time=end_time,
                    first_event_time=first_event_time,
                    first_content_time=first_content_time,
                    error_message=str(exc),
                )
                classification = adjust_classification_for_capability(classification, capability)
                print(f"錯誤: {exc}")
                results.append(
                    build_result_row(
                        run_id=run_index,
                        config=config,
                        model=model,
                        param_set=param_set,
                        applied_params=applied_params,
                        display_params=display_params,
                        classification=classification,
                        vram_metrics=vram_metrics,
                        output_text=output_text,
                        error_message=str(exc),
                    )
                )

    return pd.DataFrame(results)


def save_markdown_report(df, config, report_stem):
    report_path = Path(f"{report_stem}.md")
    summary_df = build_summary_dataframe(df)
    outcome_summary_df = build_outcome_summary_dataframe(df)
    capability = config.get("capability", "chat")

    with report_path.open("w", encoding="utf-8") as file:
        file.write("# Benchmark Report\n\n")
        file.write(f"- Backend: {config['backend']}\n")
        file.write(f"- Capability: {capability}\n")
        file.write(f"- Base URL: {config['url']}\n")
        file.write(f"- Models: {', '.join(config['models'])}\n")
        file.write(f"- Prompt: {config['prompt']}\n")
        file.write("- Note: TPS is estimated from streaming content chunks for relative comparison.\n")
        if capability == "tools":
            file.write(
                "- Tool mode note: successful tool-calling runs may not emit text tokens, "
                "so `TPS` and `TTFT` can be `N/A`; focus on `Output Category=tool_call` "
                "and `First Event (s)`.\n"
            )
        file.write("\n## Environment Notes\n\n")
        file.write(f"- VRAM monitoring: {config.get('vram_monitoring', 'unavailable')}\n\n")
        file.write("## Metric Notes\n\n")
        file.write("- `TPS (chunk/s)`: Estimated throughput from text-bearing streaming chunks.\n")
        file.write("- `TTFT (s)`: Time to first text chunk.\n")
        file.write("- `First Event (s)`: Time to the first streamed event of any kind.\n")
        file.write("- `VRAM Peak (MiB)`: Highest observed total NVIDIA GPU memory usage during a run.\n")
        file.write(
            "- `Efficiency Score (TPS/GiB Peak)`: `TPS / (VRAM Peak in GiB)` when both values are available.\n\n"
        )
        file.write("## Output Diagnosis Notes\n\n")
        file.write("- `normal_content`: Received text output as expected for chat benchmarking.\n")
        file.write("- `tool_call`: Received `tool_calls` payload as expected for tool benchmarking.\n")
        file.write("- `text_reply_without_tool`: Returned text, but did not emit any tool call in tool mode.\n")
        file.write("- `empty_reply`: Stream completed without text output.\n")
        file.write("- `non_content_stream`: Stream only carried non-text payloads.\n")
        file.write("- `early_stop`: Stream ended before a complete reply or tool call was received.\n\n")
        file.write("## Summary\n\n")
        file.write(summary_df.to_markdown(index=False))
        file.write("\n\n## Outcome Summary\n\n")
        file.write(outcome_summary_df.to_markdown(index=False))
        file.write("\n\n## Generated Outputs\n")

        for _, row in df.iterrows():
            params_json = json.dumps(row["Params"], ensure_ascii=False)
            applied_params_json = json.dumps(row["Applied_Params"], ensure_ascii=False)
            file.write(f"\n### Run {row['Run_ID']}\n\n")
            file.write(f"- Status: {row['Status']}\n")
            file.write(f"- Capability: {row.get('Capability', capability)}\n")
            file.write(f"- Output Category: {row['Output_Category']}\n")
            file.write(f"- Diagnosis: {row['Diagnosis']}\n")
            file.write(f"- Finish Reason: {format_text_value(row['Finish_Reason'])}\n")
            file.write(f"- Backend: {row['Backend']}\n")
            file.write(f"- Model: {row['Model']}\n")
            file.write(f"- Params: `{params_json}`\n")
            file.write(f"- Applied Params: `{applied_params_json}`\n")
            file.write(f"- TPS: {format_numeric_value(row['TPS'], 2)} chunk/s\n")
            file.write(f"- TTFT: {format_numeric_value(row['TTFT'], 3)} s\n")
            file.write(f"- First Event: {format_numeric_value(row['First_Event_s'], 3)} s\n")
            file.write(f"- Stream Duration: {format_numeric_value(row['Stream_Duration_s'], 3)} s\n")
            file.write(f"- Total Chunks: {int(row['Total_Chunks'])}\n")
            file.write(f"- Content Chunks: {int(row['Content_Chunks'])}\n")
            file.write(f"- Non-Content Chunks: {int(row['Non_Content_Chunks'])}\n")
            file.write(f"- Non-Content Types: {row['Non_Content_Types']}\n")
            file.write(f"- VRAM Base: {format_mib_value(row['VRAM_Base_MiB'])}\n")
            file.write(f"- VRAM Peak: {format_mib_value(row['VRAM_Peak_MiB'])}\n")
            file.write(f"- VRAM Delta: {format_mib_value(row['VRAM_Delta_MiB'])}\n")
            file.write(f"- VRAM Detail: {row['VRAM_Detail']}\n")
            file.write(
                f"- Efficiency Score: "
                f"{format_numeric_value(row['Efficiency_Score'], 3)} TPS/GiB Peak\n"
            )
            file.write(f"- Output Chars: {row['Output_Chars']}\n")
            if row["Error"]:
                file.write(f"- Error: {row['Error']}\n")
            file.write("\n```text\n")
            file.write(normalize_output_text(row["Output_Text"]))
            file.write("\n```\n")

    return report_path


def select_best_result(eligible_df, capability):
    if capability == "tools":
        if eligible_df["First_Event_s"].notna().any():
            best_row = eligible_df.loc[eligible_df["First_Event_s"].idxmin()]
            return best_row, "first_event_s"
        best_row = eligible_df.loc[eligible_df["Stream_Duration_s"].idxmin()]
        return best_row, "stream_duration_s"

    if eligible_df["TPS"].notna().any():
        best_row = eligible_df.loc[eligible_df["TPS"].idxmax()]
        return best_row, "tps"

    best_row = eligible_df.loc[eligible_df["TTFT"].idxmin()]
    return best_row, "ttft"


def export_best_config(df, config):
    capability = config.get("capability", "chat")
    eligible_df = filter_eligible_results(df, capability=capability)
    if eligible_df.empty:
        print("⚠️ 沒有符合本次測試模式的成功結果，因此不輸出 best_config.json。")
        return None

    best_row, selection_metric = select_best_result(eligible_df, capability)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": best_row["Backend"],
        "capability": capability,
        "selection_metric": selection_metric,
        "model": best_row["Model"],
        "params": best_row["Params"],
        "applied_params": best_row["Applied_Params"],
        "status": best_row["Status"],
        "output_category": best_row["Output_Category"],
        "finish_reason": best_row["Finish_Reason"],
        "diagnosis": best_row["Diagnosis"],
        "tps": best_row["TPS"],
        "ttft": best_row["TTFT"],
        "first_event_s": best_row["First_Event_s"],
        "stream_duration_s": best_row["Stream_Duration_s"],
        "vram_base_mib": best_row["VRAM_Base_MiB"],
        "vram_peak_mib": best_row["VRAM_Peak_MiB"],
        "vram_delta_mib": best_row["VRAM_Delta_MiB"],
        "vram_detail": best_row["VRAM_Detail"],
        "efficiency_score_tps_per_gib_peak": best_row["Efficiency_Score"],
        "prompt": config["prompt"],
    }

    with open("best_config.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=4)

    print("\n" + "=" * 18)
    print(f"最佳模型: {best_row['Model']}")
    if capability == "tools":
        print(f"最佳首事件延遲: {format_numeric_value(best_row['First_Event_s'], 3)} s")
        print(f"輸出類型: {best_row['Output_Category']}")
    else:
        print(f"最高 TPS: {format_numeric_value(best_row['TPS'], 2)} TPS")
    print(f"最佳設定: {best_row['Config_Str']}")
    print(f"VRAM Peak: {format_mib_value(best_row['VRAM_Peak_MiB'])}")
    if pd.notna(best_row["Efficiency_Score"]):
        print(f"效率分數: {format_numeric_value(best_row['Efficiency_Score'], 3)} TPS/GiB Peak")
    print("=" * 18)
    print("已保存 best_config.json")

    if best_row["Backend"] == "ollama":
        with open("Ollama_Modelfile_Suggest", "w", encoding="utf-8") as file:
            file.write(f"FROM {best_row['Model']}\n")
            for key, value in best_row["Applied_Params"].items():
                file.write(f"PARAMETER {key} {value}\n")
        print("已輸出 Ollama_Modelfile_Suggest")
    else:
        print("本次後端不是 Ollama，略過 Modelfile 建議。")

    return payload


def plot_results(df, output_path, capability="chat"):
    eligible_df = filter_eligible_results(df, capability=capability)
    if eligible_df.empty:
        return None

    eligible_df = eligible_df.copy()

    def build_label(row):
        label = f"{row['Model']} | {row['Config_Str']}"
        return label if len(label) <= 42 else label[:39] + "..."

    eligible_df["Plot_Label"] = eligible_df.apply(build_label, axis=1)

    if capability == "tools":
        min_first_event = eligible_df["First_Event_s"].min()
        min_stream_duration = eligible_df["Stream_Duration_s"].min()
        fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
        first_event_colors = [
            "lightgreen" if value == min_first_event else "steelblue"
            for value in eligible_df["First_Event_s"]
        ]
        duration_colors = [
            "lightgreen" if value == min_stream_duration else "slategray"
            for value in eligible_df["Stream_Duration_s"]
        ]

        axes[0].bar(eligible_df["Plot_Label"], eligible_df["First_Event_s"], color=first_event_colors)
        axes[0].axhline(y=min_first_event, color="green", linestyle="--", alpha=0.3)
        axes[0].set_title("Tool Call First Event Comparison")
        axes[0].set_ylabel("First Event (s)")

        axes[1].bar(
            eligible_df["Plot_Label"],
            eligible_df["Stream_Duration_s"],
            color=duration_colors,
        )
        axes[1].axhline(y=min_stream_duration, color="green", linestyle="--", alpha=0.3)
        axes[1].set_title("Tool Call Stream Duration Comparison")
        axes[1].set_ylabel("Stream Duration (s)")
        axes[1].set_xlabel("Model | Config")
    else:
        max_tps = eligible_df["TPS"].max()
        min_ttft = eligible_df["TTFT"].min()
        colors = ["gold" if tps == max_tps else "skyblue" for tps in eligible_df["TPS"]]
        ttft_colors = ["lightgreen" if ttft == min_ttft else "salmon" for ttft in eligible_df["TTFT"]]
        has_vram_data = eligible_df["VRAM_Peak_MiB"].notna().any()
        has_efficiency_data = eligible_df["Efficiency_Score"].notna().any()

        subplot_count = 4 if has_efficiency_data else 3 if has_vram_data else 2
        fig, axes = plt.subplots(subplot_count, 1, figsize=(13, 5 * subplot_count), sharex=True)
        if subplot_count == 1:
            axes = [axes]

        axes[0].bar(eligible_df["Plot_Label"], eligible_df["TPS"], color=colors)
        axes[0].axhline(y=max_tps, color="red", linestyle="--", alpha=0.3)
        axes[0].set_title("Throughput Comparison")
        axes[0].set_ylabel("TPS (chunk/s)")

        axes[1].bar(eligible_df["Plot_Label"], eligible_df["TTFT"], color=ttft_colors)
        axes[1].axhline(y=min_ttft, color="green", linestyle="--", alpha=0.3)
        axes[1].set_title("First Token Latency Comparison")
        axes[1].set_ylabel("TTFT (s)")

        if has_vram_data:
            min_vram_peak = eligible_df["VRAM_Peak_MiB"].min()
            vram_colors = [
                "lightgreen" if peak == min_vram_peak else "mediumpurple"
                for peak in eligible_df["VRAM_Peak_MiB"]
            ]
            axes[2].bar(eligible_df["Plot_Label"], eligible_df["VRAM_Peak_MiB"], color=vram_colors)
            axes[2].axhline(y=min_vram_peak, color="green", linestyle="--", alpha=0.3)
            axes[2].set_title("VRAM Peak Comparison")
            axes[2].set_ylabel("VRAM Peak (MiB)")
            if has_efficiency_data:
                max_efficiency = eligible_df["Efficiency_Score"].max()
                efficiency_colors = [
                    "gold" if score == max_efficiency else "steelblue"
                    for score in eligible_df["Efficiency_Score"]
                ]
                axes[3].bar(
                    eligible_df["Plot_Label"],
                    eligible_df["Efficiency_Score"],
                    color=efficiency_colors,
                )
                axes[3].axhline(y=max_efficiency, color="orange", linestyle="--", alpha=0.3)
                axes[3].set_title("Efficiency Score Comparison")
                axes[3].set_ylabel("TPS/GiB Peak")
                axes[3].set_xlabel("Model | Config")
            else:
                axes[2].set_xlabel("Model | Config")
        else:
            axes[1].set_xlabel("Model | Config")

    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return Path(output_path)


def main():
    config = interactive_config()
    if not config:
        return

    results_df = run_bench(config)
    if results_df.empty:
        print("⚠️ 沒有產生任何 benchmark 結果。")
        return

    ok_count = int((results_df["Status"] == "ok").sum())
    warning_count = int((results_df["Status"] == "warning").sum())
    error_count = int((results_df["Status"] == "error").sum())

    print("\n" + "=" * 62)
    summary_df = build_summary_dataframe(results_df)
    console_columns = ["Run", "Status"]
    if "Capability" in summary_df.columns:
        console_columns.append("Capability")
    console_columns.extend(
        [
            "Output Category",
            "Finish Reason",
            "Model",
            "TPS (chunk/s)",
            "TTFT (s)",
            "First Event (s)",
            "Chunks (content/total)",
            "Config",
        ]
    )
    console_df = summary_df[console_columns]
    print(console_df.to_markdown(index=False))
    print(f"\n結果統計: ok={ok_count}, warning={warning_count}, error={error_count}")

    capability = config.get("capability", "chat")
    report_stem = f"bench_{config['backend']}_{capability}_{time.strftime('%Y%m%d_%H%M%S')}"
    report_path = save_markdown_report(results_df, config, report_stem)
    chart_path = plot_results(results_df, f"{report_stem}.png", capability=capability)
    export_best_config(results_df, config)

    print(f"\n報告已保存: {report_path}")
    if chart_path:
        print(f"圖表已保存: {chart_path}")
    else:
        print("沒有可繪圖的成功結果，略過圖表輸出。")


def interactive_config():
    capability_defaults = {
        "chat": "Explain the long-term creep risk of PETG in 3D printing and how to reduce it.",
        "tools": (
            "Check today's weather in Taipei. If you support tools or function calling, "
            "call the `lookup_weather` tool first instead of answering directly."
        ),
    }

    print("\n" + "=" * 62)
    print("LLM Benchmark V3")
    print("=" * 62)

    backend = questionary.select(
        "Select backend:",
        choices=[
            Choice("Ollama", value="ollama"),
            Choice("llama.cpp (llama-server)", value="llama.cpp"),
        ],
    ).ask()
    if not backend:
        return None

    capability = questionary.select(
        "Select benchmark mode:",
        choices=[
            Choice("Chat | Standard chat response benchmark", value="chat"),
            Choice("Tools | Check whether the model emits tool_calls", value="tools"),
        ],
    ).ask()
    if not capability:
        return None

    url, models = select_models_and_url(backend)
    if not models:
        print("No models available. Cancelled.")
        return None

    available_groups = {
        group_name: [key for key in param_keys if backend in PARAM_INFO[key]["backends"]]
        for group_name, param_keys in PARAM_GROUPS.items()
    }
    available_groups = {name: keys for name, keys in available_groups.items() if keys}

    selected_groups = questionary.checkbox(
        "Select parameter groups to benchmark (optional):",
        choices=[
            Choice(f"{group_name} ({len(param_keys)} params)", value=group_name)
            for group_name, param_keys in available_groups.items()
        ],
    ).ask() or []

    final_params = {}
    for group_name in selected_groups:
        param_keys = available_groups[group_name]
        selected_params = questionary.checkbox(
            f"Select params from {group_name}:",
            choices=[
                Choice(
                    title=(
                        f"{PARAM_INFO[key]['label']} | Range: {PARAM_INFO[key]['range']} | "
                        f"{PARAM_INFO[key]['desc']}"
                    ),
                    value=key,
                )
                for key in param_keys
            ],
        ).ask() or []

        for key in selected_params:
            values = ask_param_values(key)
            if values is None:
                return None
            final_params[key] = values

    prompt = questionary.text(
        "Benchmark prompt:",
        default=capability_defaults[capability],
    ).ask()
    if prompt is None:
        return None

    return {
        "backend": backend,
        "capability": capability,
        "url": url,
        "models": models,
        "params": final_params,
        "prompt": prompt,
    }


def run_bench(config):
    client = OpenAI(base_url=config["url"], api_key="sk-no-key-needed")
    capability = config.get("capability", "chat")
    capability_label = {"chat": "chat", "tools": "tools"}.get(capability, capability)
    param_keys = list(config["params"].keys())
    param_values = [config["params"][key] for key in param_keys]
    combos = [dict(zip(param_keys, combo)) for combo in product(*param_values)] if param_keys else [{}]

    results = []
    total_runs = len(config["models"]) * len(combos)
    vram_monitoring_enabled = query_nvidia_vram_snapshot() is not None
    config["vram_monitoring"] = "nvidia-smi" if vram_monitoring_enabled else "unavailable"

    print(f"\nStarting benchmark with {total_runs} runs. Mode: {capability_label}")
    print("Chat mode focuses on TPS/TTFT. Tools mode focuses on tool_call success and first event latency.")
    if vram_monitoring_enabled:
        print("VRAM monitoring: enabled via nvidia-smi")
    else:
        print("VRAM monitoring: nvidia-smi not available, VRAM fields will be N/A")

    run_index = 0
    for model in config["models"]:
        for param_set in combos:
            run_index += 1
            applied_params = build_backend_options(config["backend"], param_set)
            display_params = format_param_dict(param_set)
            print(f"[{run_index}/{total_runs}] {model} | {display_params}")

            request_kwargs = {}
            if applied_params:
                request_kwargs["extra_body"] = (
                    {"options": applied_params}
                    if config["backend"] == "ollama"
                    else applied_params
                )

            start_time = time.time()
            first_event_time = None
            first_content_time = None
            output_parts = []
            chunk_records = []
            vram_monitor = NvidiaVRAMMonitor() if vram_monitoring_enabled else None
            vram_metrics = empty_vram_metrics()
            if vram_monitor is not None and not vram_monitor.start():
                vram_monitor = None

            try:
                request_payload = build_chat_request_payload(config, model, request_kwargs)
                stream = client.chat.completions.create(**request_payload)

                for chunk in stream:
                    event_time = time.time()
                    if first_event_time is None:
                        first_event_time = event_time

                    chunk_info = inspect_stream_chunk(chunk)
                    chunk_records.append(chunk_info)

                    if chunk_info["content"] and first_content_time is None:
                        first_content_time = event_time
                    if chunk_info["content"]:
                        output_parts.append(chunk_info["content"])

                end_time = time.time()
                if vram_monitor is not None:
                    vram_metrics = vram_monitor.stop()
                output_text = "".join(output_parts)
                classification = classify_stream_result(
                    chunk_records=chunk_records,
                    start_time=start_time,
                    end_time=end_time,
                    first_event_time=first_event_time,
                    first_content_time=first_content_time,
                    error_message=None,
                )
                classification = adjust_classification_for_capability(classification, capability)
                results.append(
                    build_result_row(
                        run_id=run_index,
                        config=config,
                        model=model,
                        param_set=param_set,
                        applied_params=applied_params,
                        display_params=display_params,
                        classification=classification,
                        vram_metrics=vram_metrics,
                        output_text=output_text,
                        error_message=None,
                    )
                )
            except Exception as exc:
                end_time = time.time()
                if vram_monitor is not None:
                    vram_metrics = vram_monitor.stop()
                output_text = "".join(output_parts)
                classification = classify_stream_result(
                    chunk_records=chunk_records,
                    start_time=start_time,
                    end_time=end_time,
                    first_event_time=first_event_time,
                    first_content_time=first_content_time,
                    error_message=str(exc),
                )
                classification = adjust_classification_for_capability(classification, capability)
                print(f"Error: {exc}")
                results.append(
                    build_result_row(
                        run_id=run_index,
                        config=config,
                        model=model,
                        param_set=param_set,
                        applied_params=applied_params,
                        display_params=display_params,
                        classification=classification,
                        vram_metrics=vram_metrics,
                        output_text=output_text,
                        error_message=str(exc),
                    )
                )

    return pd.DataFrame(results)


def save_markdown_report(df, config, report_stem):
    report_path = Path(f"{report_stem}.md")
    summary_df = build_summary_dataframe(df)
    outcome_summary_df = build_outcome_summary_dataframe(df)
    capability = config.get("capability", "chat")

    with report_path.open("w", encoding="utf-8") as file:
        file.write("# Benchmark Report\n\n")
        file.write(f"- Backend: {config['backend']}\n")
        file.write(f"- Capability: {capability}\n")
        file.write(f"- Base URL: {config['url']}\n")
        file.write(f"- Models: {', '.join(config['models'])}\n")
        file.write(f"- Prompt: {config['prompt']}\n")
        file.write("- Note: TPS is estimated from streaming content chunks for relative comparison.\n")
        if capability == "tools":
            file.write(
                "- Tool mode note: successful tool-calling runs may not emit text tokens, "
                "so `TPS` and `TTFT` can be `N/A`; focus on `Output Category=tool_call` "
                "and `First Event (s)`.\n"
            )
        file.write("\n## Environment Notes\n\n")
        file.write(f"- VRAM monitoring: {config.get('vram_monitoring', 'unavailable')}\n\n")
        file.write("## Metric Notes\n\n")
        file.write("- `TPS (chunk/s)`: Estimated throughput from text-bearing streaming chunks.\n")
        file.write("- `TTFT (s)`: Time to first text chunk.\n")
        file.write("- `First Event (s)`: Time to the first streamed event of any kind.\n")
        file.write("- `VRAM Peak (MiB)`: Highest observed total NVIDIA GPU memory usage during a run.\n")
        file.write(
            "- `Efficiency Score (TPS/GiB Peak)`: `TPS / (VRAM Peak in GiB)` when both values are available.\n\n"
        )
        file.write("## Output Diagnosis Notes\n\n")
        file.write("- `normal_content`: Received text output as expected for chat benchmarking.\n")
        file.write("- `tool_call`: Received `tool_calls` payload as expected for tool benchmarking.\n")
        file.write("- `text_reply_without_tool`: Returned text, but did not emit any tool call in tool mode.\n")
        file.write("- `empty_reply`: Stream completed without text output.\n")
        file.write("- `non_content_stream`: Stream only carried non-text payloads.\n")
        file.write("- `early_stop`: Stream ended before a complete reply or tool call was received.\n\n")
        file.write("## Summary\n\n")
        file.write(summary_df.to_markdown(index=False))
        file.write("\n\n## Outcome Summary\n\n")
        file.write(outcome_summary_df.to_markdown(index=False))
        file.write("\n\n## Generated Outputs\n")

        for _, row in df.iterrows():
            params_json = json.dumps(row["Params"], ensure_ascii=False)
            applied_params_json = json.dumps(row["Applied_Params"], ensure_ascii=False)
            file.write(f"\n### Run {row['Run_ID']}\n\n")
            file.write(f"- Status: {row['Status']}\n")
            file.write(f"- Capability: {row.get('Capability', capability)}\n")
            file.write(f"- Output Category: {row['Output_Category']}\n")
            file.write(f"- Diagnosis: {row['Diagnosis']}\n")
            file.write(f"- Finish Reason: {format_text_value(row['Finish_Reason'])}\n")
            file.write(f"- Backend: {row['Backend']}\n")
            file.write(f"- Model: {row['Model']}\n")
            file.write(f"- Params: `{params_json}`\n")
            file.write(f"- Applied Params: `{applied_params_json}`\n")
            file.write(f"- TPS: {format_numeric_value(row['TPS'], 2)} chunk/s\n")
            file.write(f"- TTFT: {format_numeric_value(row['TTFT'], 3)} s\n")
            file.write(f"- First Event: {format_numeric_value(row['First_Event_s'], 3)} s\n")
            file.write(f"- Stream Duration: {format_numeric_value(row['Stream_Duration_s'], 3)} s\n")
            file.write(f"- Total Chunks: {int(row['Total_Chunks'])}\n")
            file.write(f"- Content Chunks: {int(row['Content_Chunks'])}\n")
            file.write(f"- Non-Content Chunks: {int(row['Non_Content_Chunks'])}\n")
            file.write(f"- Non-Content Types: {row['Non_Content_Types']}\n")
            file.write(f"- VRAM Base: {format_mib_value(row['VRAM_Base_MiB'])}\n")
            file.write(f"- VRAM Peak: {format_mib_value(row['VRAM_Peak_MiB'])}\n")
            file.write(f"- VRAM Delta: {format_mib_value(row['VRAM_Delta_MiB'])}\n")
            file.write(f"- VRAM Detail: {row['VRAM_Detail']}\n")
            file.write(
                f"- Efficiency Score: "
                f"{format_numeric_value(row['Efficiency_Score'], 3)} TPS/GiB Peak\n"
            )
            file.write(f"- Output Chars: {row['Output_Chars']}\n")
            if row["Error"]:
                file.write(f"- Error: {row['Error']}\n")
            file.write("\n```text\n")
            file.write(normalize_output_text(row["Output_Text"]))
            file.write("\n```\n")

    return report_path


def select_best_result(eligible_df, capability):
    if capability == "tools":
        if eligible_df["First_Event_s"].notna().any():
            best_row = eligible_df.loc[eligible_df["First_Event_s"].idxmin()]
            return best_row, "first_event_s"
        best_row = eligible_df.loc[eligible_df["Stream_Duration_s"].idxmin()]
        return best_row, "stream_duration_s"

    if eligible_df["TPS"].notna().any():
        best_row = eligible_df.loc[eligible_df["TPS"].idxmax()]
        return best_row, "tps"

    best_row = eligible_df.loc[eligible_df["TTFT"].idxmin()]
    return best_row, "ttft"


def export_best_config(df, config):
    capability = config.get("capability", "chat")
    eligible_df = filter_eligible_results(df, capability=capability)
    if eligible_df.empty:
        print("No successful result for this benchmark mode, so best_config.json was not written.")
        return None

    best_row, selection_metric = select_best_result(eligible_df, capability)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": best_row["Backend"],
        "capability": capability,
        "selection_metric": selection_metric,
        "model": best_row["Model"],
        "params": best_row["Params"],
        "applied_params": best_row["Applied_Params"],
        "status": best_row["Status"],
        "output_category": best_row["Output_Category"],
        "finish_reason": best_row["Finish_Reason"],
        "diagnosis": best_row["Diagnosis"],
        "tps": best_row["TPS"],
        "ttft": best_row["TTFT"],
        "first_event_s": best_row["First_Event_s"],
        "stream_duration_s": best_row["Stream_Duration_s"],
        "vram_base_mib": best_row["VRAM_Base_MiB"],
        "vram_peak_mib": best_row["VRAM_Peak_MiB"],
        "vram_delta_mib": best_row["VRAM_Delta_MiB"],
        "vram_detail": best_row["VRAM_Detail"],
        "efficiency_score_tps_per_gib_peak": best_row["Efficiency_Score"],
        "prompt": config["prompt"],
    }

    with open("best_config.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=4)

    print("\n" + "=" * 18)
    print(f"Best model: {best_row['Model']}")
    if capability == "tools":
        print(f"Best first-event latency: {format_numeric_value(best_row['First_Event_s'], 3)} s")
        print(f"Output category: {best_row['Output_Category']}")
    else:
        print(f"Highest TPS: {format_numeric_value(best_row['TPS'], 2)} TPS")
    print(f"Best config: {best_row['Config_Str']}")
    print(f"VRAM Peak: {format_mib_value(best_row['VRAM_Peak_MiB'])}")
    if pd.notna(best_row["Efficiency_Score"]):
        print(f"Efficiency score: {format_numeric_value(best_row['Efficiency_Score'], 3)} TPS/GiB Peak")
    print("=" * 18)
    print("Saved best_config.json")

    if best_row["Backend"] == "ollama":
        with open("Ollama_Modelfile_Suggest", "w", encoding="utf-8") as file:
            file.write(f"FROM {best_row['Model']}\n")
            for key, value in best_row["Applied_Params"].items():
                file.write(f"PARAMETER {key} {value}\n")
        print("Saved Ollama_Modelfile_Suggest")
    else:
        print("Backend is not Ollama, so Modelfile output was skipped.")

    return payload


def plot_results(df, output_path, capability="chat"):
    eligible_df = filter_eligible_results(df, capability=capability)
    if eligible_df.empty:
        return None

    eligible_df = eligible_df.copy()

    def build_label(row):
        label = f"{row['Model']} | {row['Config_Str']}"
        return label if len(label) <= 42 else label[:39] + "..."

    eligible_df["Plot_Label"] = eligible_df.apply(build_label, axis=1)

    if capability == "tools":
        min_first_event = eligible_df["First_Event_s"].min()
        min_stream_duration = eligible_df["Stream_Duration_s"].min()
        fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
        first_event_colors = [
            "lightgreen" if value == min_first_event else "steelblue"
            for value in eligible_df["First_Event_s"]
        ]
        duration_colors = [
            "lightgreen" if value == min_stream_duration else "slategray"
            for value in eligible_df["Stream_Duration_s"]
        ]

        axes[0].bar(eligible_df["Plot_Label"], eligible_df["First_Event_s"], color=first_event_colors)
        axes[0].axhline(y=min_first_event, color="green", linestyle="--", alpha=0.3)
        axes[0].set_title("Tool Call First Event Comparison")
        axes[0].set_ylabel("First Event (s)")

        axes[1].bar(
            eligible_df["Plot_Label"],
            eligible_df["Stream_Duration_s"],
            color=duration_colors,
        )
        axes[1].axhline(y=min_stream_duration, color="green", linestyle="--", alpha=0.3)
        axes[1].set_title("Tool Call Stream Duration Comparison")
        axes[1].set_ylabel("Stream Duration (s)")
        axes[1].set_xlabel("Model | Config")
    else:
        max_tps = eligible_df["TPS"].max()
        min_ttft = eligible_df["TTFT"].min()
        colors = ["gold" if tps == max_tps else "skyblue" for tps in eligible_df["TPS"]]
        ttft_colors = ["lightgreen" if ttft == min_ttft else "salmon" for ttft in eligible_df["TTFT"]]
        has_vram_data = eligible_df["VRAM_Peak_MiB"].notna().any()
        has_efficiency_data = eligible_df["Efficiency_Score"].notna().any()

        subplot_count = 4 if has_efficiency_data else 3 if has_vram_data else 2
        fig, axes = plt.subplots(subplot_count, 1, figsize=(13, 5 * subplot_count), sharex=True)
        if subplot_count == 1:
            axes = [axes]

        axes[0].bar(eligible_df["Plot_Label"], eligible_df["TPS"], color=colors)
        axes[0].axhline(y=max_tps, color="red", linestyle="--", alpha=0.3)
        axes[0].set_title("Throughput Comparison")
        axes[0].set_ylabel("TPS (chunk/s)")

        axes[1].bar(eligible_df["Plot_Label"], eligible_df["TTFT"], color=ttft_colors)
        axes[1].axhline(y=min_ttft, color="green", linestyle="--", alpha=0.3)
        axes[1].set_title("First Token Latency Comparison")
        axes[1].set_ylabel("TTFT (s)")

        if has_vram_data:
            min_vram_peak = eligible_df["VRAM_Peak_MiB"].min()
            vram_colors = [
                "lightgreen" if peak == min_vram_peak else "mediumpurple"
                for peak in eligible_df["VRAM_Peak_MiB"]
            ]
            axes[2].bar(eligible_df["Plot_Label"], eligible_df["VRAM_Peak_MiB"], color=vram_colors)
            axes[2].axhline(y=min_vram_peak, color="green", linestyle="--", alpha=0.3)
            axes[2].set_title("VRAM Peak Comparison")
            axes[2].set_ylabel("VRAM Peak (MiB)")
            if has_efficiency_data:
                max_efficiency = eligible_df["Efficiency_Score"].max()
                efficiency_colors = [
                    "gold" if score == max_efficiency else "steelblue"
                    for score in eligible_df["Efficiency_Score"]
                ]
                axes[3].bar(
                    eligible_df["Plot_Label"],
                    eligible_df["Efficiency_Score"],
                    color=efficiency_colors,
                )
                axes[3].axhline(y=max_efficiency, color="orange", linestyle="--", alpha=0.3)
                axes[3].set_title("Efficiency Score Comparison")
                axes[3].set_ylabel("TPS/GiB Peak")
                axes[3].set_xlabel("Model | Config")
            else:
                axes[2].set_xlabel("Model | Config")
        else:
            axes[1].set_xlabel("Model | Config")

    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return Path(output_path)


def main():
    config = interactive_config()
    if not config:
        return

    results_df = run_bench(config)
    if results_df.empty:
        print("No benchmark rows were produced.")
        return

    ok_count = int((results_df["Status"] == "ok").sum())
    warning_count = int((results_df["Status"] == "warning").sum())
    error_count = int((results_df["Status"] == "error").sum())

    print("\n" + "=" * 62)
    summary_df = build_summary_dataframe(results_df)
    console_columns = ["Run", "Status"]
    if "Capability" in summary_df.columns:
        console_columns.append("Capability")
    console_columns.extend(
        [
            "Output Category",
            "Finish Reason",
            "Model",
            "TPS (chunk/s)",
            "TTFT (s)",
            "First Event (s)",
            "Chunks (content/total)",
            "Config",
        ]
    )
    console_df = summary_df[console_columns]
    print(console_df.to_markdown(index=False))
    print(f"\nResult counts: ok={ok_count}, warning={warning_count}, error={error_count}")

    capability = config.get("capability", "chat")
    report_stem = f"bench_{config['backend']}_{capability}_{time.strftime('%Y%m%d_%H%M%S')}"
    report_path = save_markdown_report(results_df, config, report_stem)
    chart_path = plot_results(results_df, f"{report_stem}.png", capability=capability)
    export_best_config(results_df, config)

    print(f"\nSaved report: {report_path}")
    if chart_path:
        print(f"Saved chart: {chart_path}")
    else:
        print("Skipped chart output because there were no eligible successful results.")


def save_markdown_report(df, config, report_stem):
    report_path = Path(f"{report_stem}.md")
    summary_df = build_summary_dataframe(df)
    outcome_summary_df = build_outcome_summary_dataframe(df)
    capability = config.get("capability", "chat")

    with report_path.open("w", encoding="utf-8") as file:
        file.write("# Benchmark Report\n\n")
        file.write(f"- Backend: {config['backend']}\n")
        file.write(f"- Capability: {capability}\n")
        file.write(f"- Base URL: {config['url']}\n")
        file.write(f"- Models: {', '.join(config['models'])}\n")
        file.write(f"- Prompt: {config['prompt']}\n")
        file.write("- Note: TPS is estimated from streaming content chunks for relative comparison.\n")
        if capability == "tools":
            file.write(
                "- Tool mode note: successful tool-calling runs may not emit text tokens, "
                "so `TPS` and `TTFT` can be `N/A`; focus on `Output Category=tool_call` "
                "and `First Event (s)`.\n"
            )
        file.write("\n## Environment Notes\n\n")
        file.write(f"- VRAM monitoring: {config.get('vram_monitoring', 'unavailable')}\n\n")
        file.write("## Metric Notes\n\n")
        file.write("- `TPS (chunk/s)`: Estimated throughput from text-bearing streaming chunks.\n")
        file.write("- `TTFT (s)`: Time to first text chunk.\n")
        file.write("- `First Event (s)`: Time to the first streamed event of any kind.\n")
        file.write("- `VRAM Peak (MiB)`: Highest observed total NVIDIA GPU memory usage during a run.\n")
        file.write(
            "- `Efficiency Score (TPS/GiB Peak)`: `TPS / (VRAM Peak in GiB)` when both values are available.\n\n"
        )
        file.write("## Output Diagnosis Notes\n\n")
        file.write("- `normal_content`: Received text output as expected for chat benchmarking.\n")
        file.write("- `tool_call`: Received `tool_calls` payload as expected for tool benchmarking.\n")
        file.write("- `text_reply_without_tool`: Returned text, but did not emit any tool call in tool mode.\n")
        file.write("- `empty_reply`: Stream completed without text output.\n")
        file.write("- `non_content_stream`: Stream only carried non-text payloads.\n")
        file.write("- `early_stop`: Stream ended before a complete reply or tool call was received.\n\n")
        file.write("## Summary\n\n")
        file.write(dataframe_to_text_table(summary_df))
        file.write("\n\n## Outcome Summary\n\n")
        file.write(dataframe_to_text_table(outcome_summary_df))
        file.write("\n\n## Generated Outputs\n")

        for _, row in df.iterrows():
            params_json = json.dumps(row["Params"], ensure_ascii=False)
            applied_params_json = json.dumps(row["Applied_Params"], ensure_ascii=False)
            file.write(f"\n### Run {row['Run_ID']}\n\n")
            file.write(f"- Status: {row['Status']}\n")
            file.write(f"- Capability: {row.get('Capability', capability)}\n")
            file.write(f"- Output Category: {row['Output_Category']}\n")
            file.write(f"- Diagnosis: {row['Diagnosis']}\n")
            file.write(f"- Finish Reason: {format_text_value(row['Finish_Reason'])}\n")
            file.write(f"- Backend: {row['Backend']}\n")
            file.write(f"- Model: {row['Model']}\n")
            file.write(f"- Params: `{params_json}`\n")
            file.write(f"- Applied Params: `{applied_params_json}`\n")
            file.write(f"- TPS: {format_numeric_value(row['TPS'], 2)} chunk/s\n")
            file.write(f"- TTFT: {format_numeric_value(row['TTFT'], 3)} s\n")
            file.write(f"- First Event: {format_numeric_value(row['First_Event_s'], 3)} s\n")
            file.write(f"- Stream Duration: {format_numeric_value(row['Stream_Duration_s'], 3)} s\n")
            file.write(f"- Total Chunks: {int(row['Total_Chunks'])}\n")
            file.write(f"- Content Chunks: {int(row['Content_Chunks'])}\n")
            file.write(f"- Non-Content Chunks: {int(row['Non_Content_Chunks'])}\n")
            file.write(f"- Non-Content Types: {row['Non_Content_Types']}\n")
            file.write(f"- VRAM Base: {format_mib_value(row['VRAM_Base_MiB'])}\n")
            file.write(f"- VRAM Peak: {format_mib_value(row['VRAM_Peak_MiB'])}\n")
            file.write(f"- VRAM Delta: {format_mib_value(row['VRAM_Delta_MiB'])}\n")
            file.write(f"- VRAM Detail: {row['VRAM_Detail']}\n")
            file.write(
                f"- Efficiency Score: "
                f"{format_numeric_value(row['Efficiency_Score'], 3)} TPS/GiB Peak\n"
            )
            file.write(f"- Output Chars: {row['Output_Chars']}\n")
            if row["Error"]:
                file.write(f"- Error: {row['Error']}\n")
            file.write("\n```text\n")
            file.write(normalize_output_text(row["Output_Text"]))
            file.write("\n```\n")

    return report_path


def main():
    config = interactive_config()
    if not config:
        return

    results_df = run_bench(config)
    if results_df.empty:
        print("No benchmark rows were produced.")
        return

    ok_count = int((results_df["Status"] == "ok").sum())
    warning_count = int((results_df["Status"] == "warning").sum())
    error_count = int((results_df["Status"] == "error").sum())
    capability = config.get("capability", "chat")
    report_stem = f"bench_{config['backend']}_{capability}_{time.strftime('%Y%m%d_%H%M%S')}"

    raw_outputs_path = save_raw_outputs(results_df, report_stem)

    print("\n" + "=" * 62)
    summary_df = build_summary_dataframe(results_df)
    console_columns = ["Run", "Status"]
    if "Capability" in summary_df.columns:
        console_columns.append("Capability")
    console_columns.extend(
        [
            "Output Category",
            "Finish Reason",
            "Model",
            "TPS (chunk/s)",
            "TTFT (s)",
            "First Event (s)",
            "Chunks (content/total)",
            "Config",
        ]
    )
    console_df = summary_df[console_columns]
    print(dataframe_to_text_table(console_df))
    print(f"\nResult counts: ok={ok_count}, warning={warning_count}, error={error_count}")

    report_path = None
    chart_path = None

    try:
        report_path = save_markdown_report(results_df, config, report_stem)
    except Exception as exc:
        print(f"Report generation failed: {exc}")

    try:
        chart_path = plot_results(results_df, f"{report_stem}.png", capability=capability)
    except Exception as exc:
        print(f"Chart generation failed: {exc}")

    try:
        export_best_config(results_df, config)
    except Exception as exc:
        print(f"best_config export failed: {exc}")

    if report_path:
        print(f"\nSaved report: {report_path}")
    else:
        print("\nMarkdown report was not saved.")

    print(f"Saved raw outputs: {raw_outputs_path}")
    if chart_path:
        print(f"Saved chart: {chart_path}")
    else:
        print("Skipped chart output because there were no eligible successful results.")


def plot_results(df, output_path, capability="chat"):
    if df.empty:
        return None

    plot_df, used_fallback = select_plot_dataframe(df, capability=capability)
    plot_df = plot_df.copy()

    def build_label(row):
        label = f"{row['Model']} | {row['Config_Str']}"
        return label if len(label) <= 42 else label[:39] + "..."

    def build_run_color(row):
        if row["Output_Category"] == "tool_call":
            return "seagreen"
        if row["Status"] == "ok":
            return "skyblue"
        if row["Status"] == "warning":
            return "darkorange"
        return "indianred"

    plot_df["Plot_Label"] = plot_df.apply(build_label, axis=1)
    plot_df["Plot_Color"] = plot_df.apply(build_run_color, axis=1)

    if capability == "tools":
        first_event_values = plot_df["First_Event_s"].fillna(0)
        duration_values = plot_df["Stream_Duration_s"].fillna(0)
        outcome_df = build_outcome_summary_dataframe(plot_df)

        fig, axes = plt.subplots(3, 1, figsize=(14, 15))

        axes[0].bar(plot_df["Plot_Label"], first_event_values, color=plot_df["Plot_Color"])
        if plot_df["First_Event_s"].notna().any():
            axes[0].axhline(
                y=plot_df["First_Event_s"].min(),
                color="green",
                linestyle="--",
                alpha=0.3,
            )
        axes[0].set_title("Tools Benchmark First Event Comparison")
        axes[0].set_ylabel("First Event (s)")

        axes[1].bar(plot_df["Plot_Label"], duration_values, color=plot_df["Plot_Color"])
        if plot_df["Stream_Duration_s"].notna().any():
            axes[1].axhline(
                y=plot_df["Stream_Duration_s"].min(),
                color="green",
                linestyle="--",
                alpha=0.3,
            )
        axes[1].set_title("Tools Benchmark Stream Duration Comparison")
        axes[1].set_ylabel("Stream Duration (s)")

        axes[2].bar(outcome_df["Output Category"], outcome_df["Count"], color="steelblue")
        axes[2].set_title("Tools Outcome Category Counts")
        axes[2].set_ylabel("Runs")
        axes[2].set_xlabel("Output Category")

        axes[0].tick_params(axis="x", rotation=35)
        axes[1].tick_params(axis="x", rotation=35)
        axes[2].tick_params(axis="x", rotation=20)
    else:
        eligible_df = filter_eligible_results(plot_df, capability=capability)
        if not eligible_df.empty:
            max_tps = eligible_df["TPS"].max()
            min_ttft = eligible_df["TTFT"].min()
            colors = ["gold" if tps == max_tps else "skyblue" for tps in eligible_df["TPS"]]
            ttft_colors = [
                "lightgreen" if ttft == min_ttft else "salmon" for ttft in eligible_df["TTFT"]
            ]
            has_vram_data = eligible_df["VRAM_Peak_MiB"].notna().any()
            has_efficiency_data = eligible_df["Efficiency_Score"].notna().any()

            subplot_count = 4 if has_efficiency_data else 3 if has_vram_data else 2
            fig, axes = plt.subplots(subplot_count, 1, figsize=(13, 5 * subplot_count), sharex=True)
            if subplot_count == 1:
                axes = [axes]

            axes[0].bar(eligible_df["Plot_Label"], eligible_df["TPS"], color=colors)
            axes[0].axhline(y=max_tps, color="red", linestyle="--", alpha=0.3)
            axes[0].set_title("Throughput Comparison")
            axes[0].set_ylabel("TPS (chunk/s)")

            axes[1].bar(eligible_df["Plot_Label"], eligible_df["TTFT"], color=ttft_colors)
            axes[1].axhline(y=min_ttft, color="green", linestyle="--", alpha=0.3)
            axes[1].set_title("First Token Latency Comparison")
            axes[1].set_ylabel("TTFT (s)")

            if has_vram_data:
                min_vram_peak = eligible_df["VRAM_Peak_MiB"].min()
                vram_colors = [
                    "lightgreen" if peak == min_vram_peak else "mediumpurple"
                    for peak in eligible_df["VRAM_Peak_MiB"]
                ]
                axes[2].bar(eligible_df["Plot_Label"], eligible_df["VRAM_Peak_MiB"], color=vram_colors)
                axes[2].axhline(y=min_vram_peak, color="green", linestyle="--", alpha=0.3)
                axes[2].set_title("VRAM Peak Comparison")
                axes[2].set_ylabel("VRAM Peak (MiB)")
                if has_efficiency_data:
                    max_efficiency = eligible_df["Efficiency_Score"].max()
                    efficiency_colors = [
                        "gold" if score == max_efficiency else "steelblue"
                        for score in eligible_df["Efficiency_Score"]
                    ]
                    axes[3].bar(
                        eligible_df["Plot_Label"],
                        eligible_df["Efficiency_Score"],
                        color=efficiency_colors,
                    )
                    axes[3].axhline(y=max_efficiency, color="orange", linestyle="--", alpha=0.3)
                    axes[3].set_title("Efficiency Score Comparison")
                    axes[3].set_ylabel("TPS/GiB Peak")
                    axes[3].set_xlabel("Model | Config")
                else:
                    axes[2].set_xlabel("Model | Config")
            else:
                axes[1].set_xlabel("Model | Config")
        else:
            status_df = (
                plot_df["Status"]
                .fillna("unknown")
                .value_counts(dropna=False)
                .rename_axis("Status")
                .reset_index(name="Count")
            )
            outcome_df = build_outcome_summary_dataframe(plot_df)
            fig, axes = plt.subplots(2, 1, figsize=(13, 10))

            axes[0].bar(status_df["Status"], status_df["Count"], color="steelblue")
            axes[0].set_title("Run Status Counts")
            axes[0].set_ylabel("Runs")

            axes[1].bar(outcome_df["Output Category"], outcome_df["Count"], color="slategray")
            title = "Outcome Category Counts"
            if used_fallback:
                title += " (Fallback)"
            axes[1].set_title(title)
            axes[1].set_ylabel("Runs")
            axes[1].set_xlabel("Output Category")
            axes[0].tick_params(axis="x", rotation=20)
            axes[1].tick_params(axis="x", rotation=20)

    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return Path(output_path)


def normalize_reasoning_content(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(normalize_reasoning_content(item) for item in value)
    if isinstance(value, dict):
        preferred_keys = (
            "text",
            "content",
            "reasoning",
            "reasoning_content",
            "thinking",
            "summary",
            "output_text",
        )
        parts = []
        for key in preferred_keys:
            if key in value:
                text = normalize_reasoning_content(value[key])
                if text:
                    parts.append(text)
        if parts:
            return "".join(parts)
        return "".join(normalize_reasoning_content(item) for item in value.values())
    return str(value)


def build_retained_sections(thinking_text, dialogue_output_text):
    retained_sections = []
    if (thinking_text or "").strip():
        retained_sections.append("thinking")
    if (dialogue_output_text or "").strip():
        retained_sections.append("dialogue_output")
    return retained_sections


def normalize_non_content_type(field_name):
    field_map = {
        "role": "role",
        "tool_calls": "tool_calls",
        "reasoning": "reasoning",
        "reasoning_content": "reasoning",
        "thinking": "reasoning",
        "refusal": "refusal",
        "audio": "audio",
        "function_call": "tool_calls",
    }
    return field_map.get(field_name, "other")


def inspect_stream_chunk(chunk):
    chunk_info = {
        "content": "",
        "thinking": "",
        "non_content_types": [],
        "finish_reason": None,
    }

    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return chunk_info

    choice = choices[0]
    delta_payload = extract_delta_payload(getattr(choice, "delta", None))
    chunk_info["content"] = normalize_text_content(delta_payload.pop("content", None))

    reasoning_parts = []
    for reasoning_key in ("reasoning", "reasoning_content", "thinking"):
        reasoning_value = delta_payload.pop(reasoning_key, None)
        if reasoning_value is not None:
            reasoning_parts.append(normalize_reasoning_content(reasoning_value))
    chunk_info["thinking"] = "".join(reasoning_parts)

    chunk_info["non_content_types"] = sorted(
        {normalize_non_content_type(field_name) for field_name in delta_payload}
        | ({"reasoning"} if chunk_info["thinking"] else set())
    )
    chunk_info["finish_reason"] = getattr(choice, "finish_reason", None) or None
    return chunk_info


def build_result_row(
    run_id,
    config,
    model,
    param_set,
    applied_params,
    display_params,
    classification,
    vram_metrics,
    dialogue_output_text,
    thinking_text,
    error_message,
):
    efficiency_score = calculate_efficiency_score(classification["TPS"], vram_metrics["VRAM_Peak_MiB"])
    retained_sections = build_retained_sections(thinking_text, dialogue_output_text)
    return {
        "Run_ID": run_id,
        "Status": classification["Status"],
        "Capability": config.get("capability", "chat"),
        "Output_Category": classification["Output_Category"],
        "Diagnosis": classification["Diagnosis"],
        "Finish_Reason": classification["Finish_Reason"],
        "Backend": config["backend"],
        "Model": model,
        "Params": param_set.copy(),
        "Applied_Params": applied_params.copy(),
        "Config_Str": display_params,
        "TPS": classification["TPS"],
        "TTFT": classification["TTFT"],
        "First_Event_s": classification["First_Event_s"],
        "Stream_Duration_s": classification["Stream_Duration_s"],
        "Total_Chunks": classification["Total_Chunks"],
        "Content_Chunks": classification["Content_Chunks"],
        "Non_Content_Chunks": classification["Non_Content_Chunks"],
        "Non_Content_Types": classification["Non_Content_Types"],
        "VRAM_Base_MiB": vram_metrics["VRAM_Base_MiB"],
        "VRAM_Peak_MiB": vram_metrics["VRAM_Peak_MiB"],
        "VRAM_Delta_MiB": vram_metrics["VRAM_Delta_MiB"],
        "VRAM_Detail": vram_metrics["VRAM_Detail"],
        "Efficiency_Score": efficiency_score,
        "Retained_Sections": retained_sections,
        "Thinking_Chars": len(thinking_text),
        "Thinking_Text": thinking_text,
        "Dialogue_Output_Chars": len(dialogue_output_text),
        "Dialogue_Output_Text": dialogue_output_text,
        "Output_Chars": len(dialogue_output_text),
        "Output_Text": dialogue_output_text,
        "Error": error_message or "",
    }


def run_bench(config):
    client = OpenAI(base_url=config["url"], api_key="sk-no-key-needed")
    capability = config.get("capability", "chat")
    capability_label = {"chat": "chat", "tools": "tools"}.get(capability, capability)
    param_keys = list(config["params"].keys())
    param_values = [config["params"][key] for key in param_keys]
    combos = [dict(zip(param_keys, combo)) for combo in product(*param_values)] if param_keys else [{}]

    results = []
    total_runs = len(config["models"]) * len(combos)
    vram_monitoring_enabled = query_nvidia_vram_snapshot() is not None
    config["vram_monitoring"] = "nvidia-smi" if vram_monitoring_enabled else "unavailable"

    print(f"\nStarting benchmark with {total_runs} runs. Mode: {capability_label}")
    print("Chat mode focuses on TPS/TTFT. Tools mode focuses on tool_call success and first event latency.")
    if vram_monitoring_enabled:
        print("VRAM monitoring: enabled via nvidia-smi")
    else:
        print("VRAM monitoring: nvidia-smi not available, VRAM fields will be N/A")

    run_index = 0
    for model in config["models"]:
        for param_set in combos:
            run_index += 1
            applied_params = build_backend_options(config["backend"], param_set)
            display_params = format_param_dict(param_set)
            print(f"[{run_index}/{total_runs}] {model} | {display_params}")

            request_kwargs = {}
            if applied_params:
                request_kwargs["extra_body"] = (
                    {"options": applied_params}
                    if config["backend"] == "ollama"
                    else applied_params
                )

            start_time = time.time()
            first_event_time = None
            first_content_time = None
            dialogue_output_parts = []
            thinking_parts = []
            chunk_records = []
            vram_monitor = NvidiaVRAMMonitor() if vram_monitoring_enabled else None
            vram_metrics = empty_vram_metrics()
            if vram_monitor is not None and not vram_monitor.start():
                vram_monitor = None

            try:
                request_payload = build_chat_request_payload(config, model, request_kwargs)
                stream = client.chat.completions.create(**request_payload)

                for chunk in stream:
                    event_time = time.time()
                    if first_event_time is None:
                        first_event_time = event_time

                    chunk_info = inspect_stream_chunk(chunk)
                    chunk_records.append(chunk_info)

                    if chunk_info["content"] and first_content_time is None:
                        first_content_time = event_time
                    if chunk_info["content"]:
                        dialogue_output_parts.append(chunk_info["content"])
                    if chunk_info["thinking"]:
                        thinking_parts.append(chunk_info["thinking"])

                end_time = time.time()
                if vram_monitor is not None:
                    vram_metrics = vram_monitor.stop()
                dialogue_output_text = "".join(dialogue_output_parts)
                thinking_text = "".join(thinking_parts)
                classification = classify_stream_result(
                    chunk_records=chunk_records,
                    start_time=start_time,
                    end_time=end_time,
                    first_event_time=first_event_time,
                    first_content_time=first_content_time,
                    error_message=None,
                )
                classification = adjust_classification_for_capability(classification, capability)
                results.append(
                    build_result_row(
                        run_id=run_index,
                        config=config,
                        model=model,
                        param_set=param_set,
                        applied_params=applied_params,
                        display_params=display_params,
                        classification=classification,
                        vram_metrics=vram_metrics,
                        dialogue_output_text=dialogue_output_text,
                        thinking_text=thinking_text,
                        error_message=None,
                    )
                )
            except Exception as exc:
                end_time = time.time()
                if vram_monitor is not None:
                    vram_metrics = vram_monitor.stop()
                dialogue_output_text = "".join(dialogue_output_parts)
                thinking_text = "".join(thinking_parts)
                classification = classify_stream_result(
                    chunk_records=chunk_records,
                    start_time=start_time,
                    end_time=end_time,
                    first_event_time=first_event_time,
                    first_content_time=first_content_time,
                    error_message=str(exc),
                )
                classification = adjust_classification_for_capability(classification, capability)
                print(f"Error: {exc}")
                results.append(
                    build_result_row(
                        run_id=run_index,
                        config=config,
                        model=model,
                        param_set=param_set,
                        applied_params=applied_params,
                        display_params=display_params,
                        classification=classification,
                        vram_metrics=vram_metrics,
                        dialogue_output_text=dialogue_output_text,
                        thinking_text=thinking_text,
                        error_message=str(exc),
                    )
                )

    return pd.DataFrame(results)


def save_markdown_report(df, config, report_stem):
    report_path = Path(f"{report_stem}.md")
    summary_df = build_summary_dataframe(df)
    outcome_summary_df = build_outcome_summary_dataframe(df)
    capability = config.get("capability", "chat")

    with report_path.open("w", encoding="utf-8") as file:
        file.write("# Benchmark Report\n\n")
        file.write(f"- Backend: {config['backend']}\n")
        file.write(f"- Capability: {capability}\n")
        file.write(f"- Base URL: {config['url']}\n")
        file.write(f"- Models: {', '.join(config['models'])}\n")
        file.write(f"- Prompt: {config['prompt']}\n")
        file.write("- Note: TPS is estimated from streaming content chunks for relative comparison.\n")
        if capability == "tools":
            file.write(
                "- Tool mode note: successful tool-calling runs may not emit text tokens, "
                "so `TPS` and `TTFT` can be `N/A`; focus on `Output Category=tool_call` "
                "and `First Event (s)`.\n"
            )
        file.write("\n## Environment Notes\n\n")
        file.write(f"- VRAM monitoring: {config.get('vram_monitoring', 'unavailable')}\n\n")
        file.write("## Metric Notes\n\n")
        file.write("- `TPS (chunk/s)`: Estimated throughput from text-bearing streaming chunks.\n")
        file.write("- `TTFT (s)`: Time to first text chunk.\n")
        file.write("- `First Event (s)`: Time to the first streamed event of any kind.\n")
        file.write("- `VRAM Peak (MiB)`: Highest observed total NVIDIA GPU memory usage during a run.\n")
        file.write(
            "- `Efficiency Score (TPS/GiB Peak)`: `TPS / (VRAM Peak in GiB)` when both values are available.\n\n"
        )
        file.write("## Retained Text Notes\n\n")
        file.write("- `thinking`: Reasoning/thinking text captured from non-content reasoning payloads when the backend exposed them.\n")
        file.write("- `dialogue_output`: Final conversational text emitted in normal content chunks.\n")
        file.write("- If neither exists for a run, the retained sections list will be `none`.\n\n")
        file.write("## Output Diagnosis Notes\n\n")
        file.write("- `normal_content`: Received text output as expected for chat benchmarking.\n")
        file.write("- `tool_call`: Received `tool_calls` payload as expected for tool benchmarking.\n")
        file.write("- `text_reply_without_tool`: Returned text, but did not emit any tool call in tool mode.\n")
        file.write("- `empty_reply`: Stream completed without text output.\n")
        file.write("- `non_content_stream`: Stream only carried non-text payloads.\n")
        file.write("- `early_stop`: Stream ended before a complete reply or tool call was received.\n\n")
        file.write("## Summary\n\n")
        file.write(dataframe_to_text_table(summary_df))
        file.write("\n\n## Outcome Summary\n\n")
        file.write(dataframe_to_text_table(outcome_summary_df))
        file.write("\n\n## Generated Outputs\n")

        for _, row in df.iterrows():
            params_json = json.dumps(row["Params"], ensure_ascii=False)
            applied_params_json = json.dumps(row["Applied_Params"], ensure_ascii=False)
            retained_sections = row.get("Retained_Sections", []) or []
            retained_sections_label = ", ".join(retained_sections) if retained_sections else "none"
            thinking_text = row.get("Thinking_Text", "")
            dialogue_output_text = row.get("Dialogue_Output_Text", row.get("Output_Text", ""))

            file.write(f"\n### Run {row['Run_ID']}\n\n")
            file.write(f"- Status: {row['Status']}\n")
            file.write(f"- Capability: {row.get('Capability', capability)}\n")
            file.write(f"- Output Category: {row['Output_Category']}\n")
            file.write(f"- Diagnosis: {row['Diagnosis']}\n")
            file.write(f"- Finish Reason: {format_text_value(row['Finish_Reason'])}\n")
            file.write(f"- Backend: {row['Backend']}\n")
            file.write(f"- Model: {row['Model']}\n")
            file.write(f"- Params: `{params_json}`\n")
            file.write(f"- Applied Params: `{applied_params_json}`\n")
            file.write(f"- Retained Sections: {retained_sections_label}\n")
            file.write(f"- Thinking Chars: {int(row.get('Thinking_Chars', len(thinking_text)))}\n")
            file.write(
                f"- Dialogue Output Chars: "
                f"{int(row.get('Dialogue_Output_Chars', len(dialogue_output_text)))}\n"
            )
            file.write(f"- TPS: {format_numeric_value(row['TPS'], 2)} chunk/s\n")
            file.write(f"- TTFT: {format_numeric_value(row['TTFT'], 3)} s\n")
            file.write(f"- First Event: {format_numeric_value(row['First_Event_s'], 3)} s\n")
            file.write(f"- Stream Duration: {format_numeric_value(row['Stream_Duration_s'], 3)} s\n")
            file.write(f"- Total Chunks: {int(row['Total_Chunks'])}\n")
            file.write(f"- Content Chunks: {int(row['Content_Chunks'])}\n")
            file.write(f"- Non-Content Chunks: {int(row['Non_Content_Chunks'])}\n")
            file.write(f"- Non-Content Types: {row['Non_Content_Types']}\n")
            file.write(f"- VRAM Base: {format_mib_value(row['VRAM_Base_MiB'])}\n")
            file.write(f"- VRAM Peak: {format_mib_value(row['VRAM_Peak_MiB'])}\n")
            file.write(f"- VRAM Delta: {format_mib_value(row['VRAM_Delta_MiB'])}\n")
            file.write(f"- VRAM Detail: {row['VRAM_Detail']}\n")
            file.write(
                f"- Efficiency Score: "
                f"{format_numeric_value(row['Efficiency_Score'], 3)} TPS/GiB Peak\n"
            )
            if row["Error"]:
                file.write(f"- Error: {row['Error']}\n")

            if thinking_text:
                file.write("\n#### thinking\n\n```text\n")
                file.write(normalize_output_text(thinking_text))
                file.write("\n```\n")

            if dialogue_output_text:
                file.write("\n#### dialogue_output\n\n```text\n")
                file.write(normalize_output_text(dialogue_output_text))
                file.write("\n```\n")

            if not thinking_text and not dialogue_output_text:
                file.write("\n_No retained thinking or dialogue output text for this run._\n")


if __name__ == "__main__":
    try:
        main()
    finally:
        pause_before_exit()
