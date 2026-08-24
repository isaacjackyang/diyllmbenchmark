import os
import unittest
from pathlib import Path
import tempfile
from unittest import mock

import pandas as pd
import llm_expert_bench as bench


class ReasoningToggleTests(unittest.TestCase):
    def test_parse_csv_values_supports_reasoning_toggle_aliases(self):
        self.assertEqual(
            bench.parse_csv_values("enable,disable,true,false,1,0", param_key="enable_thinking"),
            [True, False, True, False, True, False],
        )

    def test_build_param_rows_includes_reasoning_toggle_for_both_backends(self):
        ollama_rows = {row.key: row for row in bench.build_param_rows("ollama")}
        llamacpp_rows = {row.key: row for row in bench.build_param_rows("llama.cpp")}

        self.assertIn("enable_thinking", ollama_rows)
        self.assertIn("enable_thinking", llamacpp_rows)
        self.assertTrue(ollama_rows["enable_thinking"].supported)
        self.assertTrue(llamacpp_rows["enable_thinking"].supported)
        self.assertEqual(ollama_rows["enable_thinking"].default_value, "disable, enable")

    def test_build_backend_extra_body_routes_ollama_thinking_to_top_level(self):
        param_set = {"temperature": 0.1, "enable_thinking": True}

        self.assertEqual(
            bench.build_backend_options("ollama", param_set),
            {"temperature": 0.1, "think": True},
        )
        self.assertEqual(
            bench.build_backend_extra_body("ollama", param_set),
            {"think": True, "options": {"temperature": 0.1}},
        )

    def test_build_backend_extra_body_routes_llamacpp_thinking_to_body(self):
        param_set = {"temperature": 0.1, "enable_thinking": False}

        self.assertEqual(
            bench.build_backend_options("llama.cpp", param_set),
            {"temperature": 0.1, "enable_thinking": False},
        )
        self.assertEqual(
            bench.build_backend_extra_body("llama.cpp", param_set),
            {"temperature": 0.1, "chat_template_kwargs": {"enable_thinking": False}},
        )

    def test_build_ollama_modelfile_params_skips_non_modelfile_think_flag(self):
        param_set = {"temperature": 0.1, "enable_thinking": True, "num_predict": 256}

        self.assertEqual(
            bench.build_ollama_modelfile_params(param_set),
            {"temperature": 0.1, "num_predict": 256},
        )

    def test_format_param_dict_uses_enable_disable_labels(self):
        self.assertEqual(
            bench.format_param_dict({"enable_thinking": True, "temperature": 0.1}),
            "{enable_thinking=enable, temperature=0.1}",
        )

    def test_suite_smoke_7_schema_has_seven_unique_questions(self):
        suite = bench.get_suite_definition("suite-smoke-7")
        questions = suite["questions"]

        self.assertEqual(suite["id"], "suite-smoke-7")
        self.assertEqual(suite["version"], "1.0.0")
        self.assertEqual(len(questions), 7)
        self.assertEqual(len({question["id"] for question in questions}), 7)
        self.assertEqual(
            {question["category"] for question in questions},
            {"math", "logic", "reasoning", "reading", "translation", "writing", "coding"},
        )
        for question in questions:
            self.assertTrue(
                set(bench.SUITE_QUESTION_REQUIRED_FIELDS).issubset(question),
                question["id"],
            )

    def test_suite_smoke_7_ui_config_uses_fixed_prompt_and_run_count(self):
        config = bench.normalize_web_ui_config(
            {
                "backend": "ollama",
                "capability": "suite-smoke-7",
                "url": "http://localhost:11434/v1",
                "models": "model-a, model-b",
                "prompt": "This custom prompt must be ignored.",
                "params": {},
            }
        )
        summary = bench.summarize_config_for_ui(config)

        self.assertEqual(config["prompt"], bench.CAPABILITY_DEFAULTS["suite-smoke-7"])
        self.assertEqual(summary["question_count"], 7)
        self.assertEqual(summary["estimated_run_count"], 14)

    def test_build_summary_dataframe_includes_thinking_mode_column(self):
        df = pd.DataFrame(
            [
                {
                    "Run_ID": 1,
                    "Status": "ok",
                    "Capability": "chat",
                    "Output_Category": "normal_content",
                    "Model": "demo-model",
                    "System_Prompt_Label": "N/A",
                    "Thinking_Mode": True,
                    "Finish_Reason": "stop",
                    "Config_Str": "{enable_thinking=enable}",
                    "Output_Chars": 12,
                    "Output_Time_s": 0.8,
                    "TPS": 2.0,
                    "Thinking_TPS": 4.0,
                    "Output_TPS": 5.0,
                    "Output_Thinking_Ratio": 1.25,
                    "TTFT": 0.1,
                    "First_Event_s": 0.05,
                    "Content_Chunks": 2,
                    "Total_Chunks": 3,
                    "VRAM_Peak_MiB": 1024,
                    "Efficiency_Score": 2.0,
                }
            ]
        )

        summary_df = bench.build_summary_dataframe(df)
        localized_summary_df = bench.localize_report_dataframe(summary_df)

        self.assertIn("Thinking Mode", summary_df.columns)
        self.assertEqual(summary_df.loc[0, "Thinking Mode"], "enable")
        self.assertIn("Thinking Mode<br>思考模式", localized_summary_df.columns)
        self.assertEqual(
            localized_summary_df.loc[0, "Thinking Mode<br>思考模式"],
            "enable / 啟用",
        )

    def test_save_markdown_report_marks_run_thinking_mode(self):
        classification = {
            "Status": "ok",
            "Output_Category": "normal_content",
            "Diagnosis": "ok",
            "Finish_Reason": "stop",
            "TPS": 2.0,
            "TTFT": 0.1,
            "First_Event_s": 0.05,
            "Stream_Duration_s": 1.2,
            "Total_Chunks": 3,
            "Content_Chunks": 2,
            "Non_Content_Chunks": 1,
            "Non_Content_Types": "reasoning",
            "Thinking_TPS": 4.0,
            "Output_TPS": 5.0,
            "Output_Thinking_Ratio": 1.25,
            "Output_Time_s": 0.9,
            "Thinking_Chars": 8,
            "Output_Chars": 10,
        }
        vram_metrics = {
            "VRAM_Base_MiB": 1000,
            "VRAM_Peak_MiB": 1200,
            "VRAM_Delta_MiB": 200,
            "VRAM_Detail": "demo",
        }
        row = bench.build_result_row(
            run_id=1,
            config={"backend": "ollama", "capability": "chat"},
            model="demo-model",
            param_set={"enable_thinking": True, "temperature": 0.1},
            applied_params={"temperature": 0.1, "think": True},
            display_params="{enable_thinking=enable, temperature=0.1}",
            classification=classification,
            vram_metrics=vram_metrics,
            dialogue_output_text="demo output",
            thinking_text="demo think",
            error_message="",
            system_prompt_label="N/A",
            system_prompt_text="",
        )
        df = pd.DataFrame([row])
        config = {
            "backend": "ollama",
            "capability": "chat",
            "url": "http://localhost:11434/v1",
            "models": ["demo-model"],
            "prompt": "demo prompt",
            "vram_monitoring": "nvidia-smi",
            "system_prompts": [],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = bench.save_markdown_report(df, config, str(Path(temp_dir) / "report"))
            report_text = Path(report_path).read_text(encoding="utf-8")

        self.assertIn("Thinking Mode", report_text)
        self.assertIn("思考模式", report_text)
        self.assertIn("enable / 啟用", report_text)
        self.assertIn("Model Comparison", report_text)
        self.assertIn('class="metric-bar-fill"', report_text)

    def test_normalize_web_ui_config_parses_payload(self):
        payload = {
            "backend": "ollama",
            "capability": "tools",
            "url": "http://localhost:11434/v1",
            "models": "qwen3.5:latest, qwen3.5:tools",
            "prompt": "Check weather in Taipei",
            "system_prompts": "first system prompt\n---\nsecond system prompt",
            "params": {
                "temperature": {"enabled": True, "raw_value": "0.1, 0.8"},
                "enable_thinking": {"enabled": True, "raw_value": "disable"},
                "num_ctx": {"enabled": False, "raw_value": "4096"},
            },
        }

        config = bench.normalize_web_ui_config(payload)

        self.assertEqual(config["backend"], "ollama")
        self.assertEqual(config["capability"], "tools")
        self.assertEqual(config["models"], ["qwen3.5:latest", "qwen3.5:tools"])
        self.assertEqual(config["params"]["temperature"], [0.1, 0.8])
        self.assertEqual(config["params"]["enable_thinking"], [False])
        self.assertEqual(config["system_prompts"], ["first system prompt", "second system prompt"])

    def test_list_report_entries_collects_related_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_dir = Path(temp_dir)
            report_path = report_dir / "bench_ollama_chat_20260717_120000.html"
            report_path.write_text("<html>demo</html>", encoding="utf-8")
            (report_dir / "bench_ollama_chat_20260717_120000.png").write_bytes(b"png")
            (report_dir / "bench_ollama_chat_20260717_120000_summary.xlsx").write_bytes(b"xlsx")
            (report_dir / "bench_ollama_chat_20260717_120000_outputs.jsonl").write_text("{}", encoding="utf-8")
            (report_dir / "best_config.json").write_text("{}", encoding="utf-8")
            (report_dir / "Ollama_Modelfile_Suggest").write_text("FROM demo", encoding="utf-8")

            entries = bench.list_report_entries(report_dir=report_dir)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["id"], "bench_ollama_chat_20260717_120000")
        self.assertEqual(entries[0]["html_url"], "/report-files/bench_ollama_chat_20260717_120000.html")
        artifact_labels = [item["label"] for item in entries[0]["artifact_links"]]
        self.assertIn("Chart", artifact_labels)
        self.assertIn("Summary Excel", artifact_labels)
        self.assertIn("Raw Outputs", artifact_labels)
        self.assertIn("Best Config", artifact_labels)

    def test_persist_ui_launch_hint_writes_url_and_report_dir(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            try:
                hint_path = bench.persist_ui_launch_hint(
                    "http://127.0.0.1:8765/",
                    Path(temp_dir) / "Report",
                )
                hint_text = hint_path.read_text(encoding="utf-8")
            finally:
                os.chdir(original_cwd)

        self.assertEqual(hint_path.name, bench.UI_LAUNCH_HINT_FILENAME)
        self.assertIn("http://127.0.0.1:8765/", hint_text)
        self.assertIn("Report directory:", hint_text)

    def test_build_ui_launch_message_includes_manual_open_guidance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            launch_hint_path = Path(temp_dir) / bench.UI_LAUNCH_HINT_FILENAME
            launch_hint_path.write_text("demo", encoding="utf-8")
            message = bench.build_ui_launch_message(
                "http://127.0.0.1:8765/",
                Path(temp_dir) / "Report",
                launch_hint_path,
                browser_opened=False,
            )

        self.assertIn("http://127.0.0.1:8765/", message)
        self.assertIn("Launch hint file:", message)
        self.assertIn("Please open the URL manually.", message)

    def test_try_open_browser_falls_back_to_windows_startfile(self):
        with (
            mock.patch.object(bench.webbrowser, "open", return_value=False) as mock_browser_open,
            mock.patch.object(bench.os, "startfile", return_value=None, create=True) as mock_startfile,
            mock.patch.object(bench.sys, "platform", "win32"),
        ):
            opened, error = bench.try_open_browser("http://127.0.0.1:8765/")

        self.assertTrue(opened)
        self.assertIsNone(error)
        mock_browser_open.assert_called_once_with("http://127.0.0.1:8765/", new=2)
        mock_startfile.assert_called_once_with("http://127.0.0.1:8765/")

    def test_save_ui_defaults_and_normalize_web_ui_config_use_saved_values(self):
        original_cwd = Path.cwd()
        custom_defaults = {
            "default_backend": "llama.cpp",
            "default_capability": "tools",
            "backend_defaults": {
                "ollama": {
                    "url": "http://demo-ollama:11434/v1",
                    "models": "qwen3.5:latest, qwen3.5:tools",
                    "params": {
                        "temperature": {"enabled": True, "raw_value": "0.2"},
                    },
                },
                "llama.cpp": {
                    "url": "http://demo-llama:8080/v1",
                    "models": "demo-llama",
                    "params": {
                        "top_p": {"enabled": True, "raw_value": "0.85, 0.95"},
                    },
                },
            },
            "capability_defaults": {
                "chat": {
                    "prompt": "custom chat prompt",
                    "system_prompts": ["chat system"],
                },
                "tools": {
                    "prompt": "custom tools prompt",
                    "system_prompts": ["tool system 1", "tool system 2"],
                },
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            try:
                saved_defaults = bench.save_ui_defaults(custom_defaults)
                loaded_defaults = bench.load_ui_defaults()
                config = bench.normalize_web_ui_config(
                    {
                        "backend": "llama.cpp",
                        "capability": "tools",
                        "models": "demo-llama",
                        "params": {},
                    }
                )
            finally:
                os.chdir(original_cwd)

        self.assertEqual(saved_defaults["default_backend"], "llama.cpp")
        self.assertEqual(loaded_defaults["default_capability"], "tools")
        self.assertEqual(
            loaded_defaults["backend_defaults"]["llama.cpp"]["url"],
            "http://demo-llama:8080/v1",
        )
        self.assertEqual(
            loaded_defaults["capability_defaults"]["tools"]["system_prompts"],
            ["tool system 1", "tool system 2"],
        )
        self.assertEqual(config["url"], "http://demo-llama:8080/v1")
        self.assertEqual(config["prompt"], "custom tools prompt")
        self.assertEqual(config["system_prompts"], ["tool system 1", "tool system 2"])

    def test_build_web_ui_bootstrap_payload_uses_saved_default_backend(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            try:
                bench.save_ui_defaults(
                    {
                        "default_backend": "llama.cpp",
                        "default_capability": "chat",
                        "backend_defaults": {
                            "llama.cpp": {
                                "url": "http://saved-llama:8080/v1",
                                "models": "saved-llama",
                                "params": {},
                            }
                        },
                    }
                )
                payload = bench.build_web_ui_bootstrap_payload(bench.BenchmarkWebUiState())
            finally:
                os.chdir(original_cwd)

        self.assertEqual(payload["default_backend"], "llama.cpp")
        self.assertEqual(payload["backend_state"]["backend"], "llama.cpp")
        self.assertEqual(payload["backend_state"]["default_url"], "http://saved-llama:8080/v1")
        self.assertEqual(payload["backend_state"]["default_models_text"], "saved-llama")
        self.assertTrue(payload["ui_defaults_path"].endswith(bench.UI_DEFAULTS_FILENAME))

    def test_build_web_ui_bootstrap_payload_uses_bilingual_labels(self):
        payload = bench.build_web_ui_bootstrap_payload(bench.BenchmarkWebUiState())

        capability_labels = [item["label"] for item in payload["capabilities"]]
        self.assertIn("Chat / 對話", capability_labels)
        self.assertIn("Tools / 工具呼叫", capability_labels)
        self.assertIn("Suite Smoke 7 / 七項能力套裝", capability_labels)
        self.assertEqual(payload["app_title"], "DIY LLM Benchmark / DIY LLM Benchmark 控制台")

    def test_single_file_html_contains_bilingual_titles_and_buttons(self):
        html = bench.build_single_file_benchmark_ui_html()

        self.assertIn("DIY LLM Benchmark Control Room / DIY LLM Benchmark 控制台", html)
        self.assertIn("Benchmark Setup / 測試設定", html)
        self.assertIn("Start Benchmark / 開始測試", html)
        self.assertIn("Refresh Reports / 重新整理報告", html)
        self.assertIn("Close Local UI / 關閉本機 UI", html)
        self.assertIn("Maintenance Page / 維護頁面", html)
        self.assertIn("Show Maintenance / 顯示維護設定", html)
        self.assertIn("Hide Maintenance / 隱藏維護設定", html)
        self.assertIn('aria-controls="maintenance-panel"', html)
        self.assertIn('id="maintenance-panel" class="panel stack maintenance-panel" hidden', html)
        self.assertIn(r'replace(/\r\n/g, "\n")', html)
        self.assertIn(r'items.join("\n---\n")', html)
        self.assertIn("Save UI Defaults / 儲存 UI 預設", html)
        self.assertIn("Reset Built-in Defaults / 還原內建預設", html)
        self.assertIn('id="prompt-hint"', html)
        self.assertIn("Built-in suite-smoke-7 runs seven fixed questions", html)


if __name__ == "__main__":
    unittest.main()
