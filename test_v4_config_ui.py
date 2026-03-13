import importlib.util
import unittest
from pathlib import Path

import pandas as pd


MODULE_CANDIDATES = [
    Path(__file__).with_name("ollama_expert_bench.py"),
    Path(__file__).with_name("ollama_expert_bench_V5.py"),
    Path(__file__).with_name("ollama_expert_bench_V4.py"),
]
MODULE_PATH = next((path for path in MODULE_CANDIDATES if path.exists()), MODULE_CANDIDATES[0])
MODULE_SPEC = importlib.util.spec_from_file_location("bench_v4", MODULE_PATH)
v4 = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(v4)


class V4ConfigUiTests(unittest.TestCase):
    def test_build_param_rows_keeps_all_params_visible(self):
        rows = v4.build_param_rows("llama.cpp")
        row_by_key = {row.key: row for row in rows}
        param_info = v4.PARAM_INFO if hasattr(v4, "PARAM_INFO") else v4.v3.PARAM_INFO

        self.assertEqual(set(row_by_key), set(param_info))
        self.assertTrue(row_by_key["temperature"].supported)
        self.assertFalse(row_by_key["num_ctx"].supported)

    def test_validate_param_rows_parses_enabled_rows(self):
        rows = v4.build_param_rows("ollama")
        row_by_key = {row.key: row for row in rows}
        row_by_key["temperature"].enabled = True
        row_by_key["temperature"].raw_value = "0.1, 0.8"
        row_by_key["top_p"].enabled = True
        row_by_key["top_p"].raw_value = "0.8, 0.95"

        params, error_row_index, error_message = v4.validate_param_rows(rows)

        self.assertIsNone(error_row_index)
        self.assertIsNone(error_message)
        self.assertEqual(params["temperature"], [0.1, 0.8])
        self.assertEqual(params["top_p"], [0.8, 0.95])
        self.assertEqual(v4.estimate_combo_count(params), 4)

    def test_validate_param_rows_reports_first_invalid_entry(self):
        rows = v4.build_param_rows("ollama")
        row_by_key = {row.key: row for row in rows}
        row_by_key["temperature"].enabled = True
        row_by_key["temperature"].raw_value = "abc"

        params, error_row_index, error_message = v4.validate_param_rows(rows)

        self.assertIsNone(params)
        self.assertIsNotNone(error_row_index)
        self.assertIn("溫度", error_message)

    def test_build_console_summary_dataframe_includes_new_metrics(self):
        df = pd.DataFrame(
            [
                {
                    "Run_ID": 1,
                    "Status": "ok",
                    "Capability": "chat",
                    "Output_Category": "normal_content",
                    "Diagnosis": "ok",
                    "Finish_Reason": "stop",
                    "Backend": "ollama",
                    "Model": "demo-model",
                    "Params": {"temperature": 0.1},
                    "Applied_Params": {"temperature": 0.1},
                    "Config_Str": "{temperature=0.1}",
                    "TPS": 5.0,
                    "Thinking_TPS": 12.5,
                    "Output_TPS": 8.5,
                    "Output_Thinking_Ratio": 0.68,
                    "TTFT": 0.2,
                    "First_Event_s": 0.1,
                    "Stream_Duration_s": 1.0,
                    "Total_Chunks": 3,
                    "Content_Chunks": 2,
                    "Thinking_Chunks": 1,
                    "Non_Content_Chunks": 1,
                    "Non_Content_Types": "reasoning",
                    "VRAM_Base_MiB": 1000,
                    "VRAM_Peak_MiB": 1200,
                    "VRAM_Delta_MiB": 200,
                    "VRAM_Detail": "demo",
                    "Efficiency_Score": 4.267,
                    "Retained_Sections": ["thinking", "dialogue_output"],
                    "Thinking_Chars": 40,
                    "Thinking_Text": "thinking text",
                    "Dialogue_Output_Chars": 27,
                    "Dialogue_Output_Text": "output text",
                    "Output_Chars": 27,
                    "Output_Text": "output text",
                    "Error": "",
                }
            ]
        )

        console_df = v4.build_console_summary_dataframe(df)

        self.assertIn("Thinking TPS (char/s)", console_df.columns)
        self.assertIn("Output TPS (char/s)", console_df.columns)
        self.assertIn("Output/Thinking Ratio", console_df.columns)
        self.assertEqual(console_df.loc[0, "Thinking TPS (char/s)"], "12.50")
        self.assertEqual(console_df.loc[0, "Output TPS (char/s)"], "8.50")
        self.assertEqual(console_df.loc[0, "Output/Thinking Ratio"], "0.680")


if __name__ == "__main__":
    unittest.main()
