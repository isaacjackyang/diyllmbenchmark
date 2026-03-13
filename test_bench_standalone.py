import unittest
from pathlib import Path
import tempfile

import pandas as pd
import ollama_expert_bench as bench


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


if __name__ == "__main__":
    unittest.main()
