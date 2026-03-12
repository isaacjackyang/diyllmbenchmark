import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


MODULE_PATH = Path(__file__).with_name("ollama_expert_bench_V3.py")
MODULE_SPEC = importlib.util.spec_from_file_location("bench_v3", MODULE_PATH)
bench_v3 = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(bench_v3)


def make_chunk(
    *,
    content=None,
    finish_reason=None,
    role=None,
    tool_calls=None,
    reasoning=None,
    refusal=None,
    audio=None,
    extra_fields=None,
    choices=True,
):
    if not choices:
        return SimpleNamespace(choices=[])

    delta_fields = {}
    if content is not None:
        delta_fields["content"] = content
    if role is not None:
        delta_fields["role"] = role
    if tool_calls is not None:
        delta_fields["tool_calls"] = tool_calls
    if reasoning is not None:
        delta_fields["reasoning"] = reasoning
    if refusal is not None:
        delta_fields["refusal"] = refusal
    if audio is not None:
        delta_fields["audio"] = audio
    if extra_fields:
        delta_fields.update(extra_fields)

    delta = SimpleNamespace(**delta_fields)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def classify_chunks(chunks, error_message=None):
    start_time = 100.0
    first_event_time = None
    first_content_time = None
    chunk_records = []

    for index, chunk in enumerate(chunks, start=1):
        event_time = start_time + (index * 0.2)
        if first_event_time is None:
            first_event_time = event_time

        chunk_info = bench_v3.inspect_stream_chunk(chunk)
        chunk_records.append(chunk_info)

        if chunk_info["content"] and first_content_time is None:
            first_content_time = event_time

    end_time = start_time + ((len(chunks) + 1) * 0.2 if chunks else 0.5)
    return bench_v3.classify_stream_result(
        chunk_records=chunk_records,
        start_time=start_time,
        end_time=end_time,
        first_event_time=first_event_time,
        first_content_time=first_content_time,
        error_message=error_message,
    )


class StreamClassificationTests(unittest.TestCase):
    def test_normal_content_stream(self):
        result = classify_chunks(
            [
                make_chunk(role="assistant"),
                make_chunk(content="hello", finish_reason="stop"),
            ]
        )
        self.assertEqual(result["Status"], "ok")
        self.assertEqual(result["Output_Category"], "normal_content")
        self.assertEqual(result["Finish_Reason"], "stop")
        self.assertEqual(result["Content_Chunks"], 1)
        self.assertIsNotNone(result["TTFT"])
        self.assertIsNotNone(result["TPS"])

    def test_empty_reply_stream(self):
        result = classify_chunks(
            [
                make_chunk(role="assistant"),
                make_chunk(finish_reason="stop"),
            ]
        )
        self.assertEqual(result["Status"], "warning")
        self.assertEqual(result["Output_Category"], "empty_reply")
        self.assertEqual(result["Finish_Reason"], "stop")
        self.assertEqual(result["Non_Content_Types"], "role")
        self.assertIsNone(result["TTFT"])
        self.assertIsNone(result["TPS"])

    def test_non_content_stream(self):
        result = classify_chunks(
            [
                make_chunk(tool_calls=[{"name": "lookup"}]),
                make_chunk(finish_reason="stop"),
            ]
        )
        self.assertEqual(result["Status"], "warning")
        self.assertEqual(result["Output_Category"], "non_content_stream")
        self.assertEqual(result["Finish_Reason"], "stop")
        self.assertIn("tool_calls", result["Non_Content_Types"])

    def test_inspect_stream_chunk_extracts_thinking_text(self):
        chunk_info = bench_v3.inspect_stream_chunk(
            make_chunk(
                reasoning=[
                    {"text": "First I inspect the question. "},
                    {"text": "Then I decide the next step."},
                ]
            )
        )
        self.assertEqual(
            chunk_info["thinking"],
            "First I inspect the question. Then I decide the next step.",
        )
        self.assertIn("reasoning", chunk_info["non_content_types"])

    def test_tool_mode_marks_tool_calls_as_success(self):
        base_result = classify_chunks(
            [
                make_chunk(tool_calls=[{"name": "lookup_weather"}]),
                make_chunk(finish_reason="tool_calls"),
            ]
        )
        result = bench_v3.adjust_classification_for_capability(base_result, "tools")
        self.assertEqual(result["Status"], "ok")
        self.assertEqual(result["Output_Category"], "tool_call")
        self.assertIn("tool_calls", result["Diagnosis"])

    def test_tool_mode_flags_plain_text_without_tool_call(self):
        base_result = classify_chunks(
            [
                make_chunk(role="assistant"),
                make_chunk(content="It is sunny today.", finish_reason="stop"),
            ]
        )
        result = bench_v3.adjust_classification_for_capability(base_result, "tools")
        self.assertEqual(result["Status"], "warning")
        self.assertEqual(result["Output_Category"], "text_reply_without_tool")

    def test_early_stop_without_exception(self):
        result = classify_chunks([make_chunk(choices=False)])
        self.assertEqual(result["Status"], "warning")
        self.assertEqual(result["Output_Category"], "early_stop")
        self.assertIsNone(result["Finish_Reason"])
        self.assertIsNone(result["TTFT"])
        self.assertIsNone(result["TPS"])

    def test_early_stop_with_exception(self):
        result = classify_chunks([make_chunk(role="assistant")], error_message="socket closed")
        self.assertEqual(result["Status"], "error")
        self.assertEqual(result["Output_Category"], "early_stop")
        self.assertIn("before any textual content", result["Diagnosis"])

    def test_exception_after_content_keeps_normal_content_category(self):
        result = classify_chunks([make_chunk(content="hello")], error_message="timeout")
        self.assertEqual(result["Status"], "error")
        self.assertEqual(result["Output_Category"], "normal_content")
        self.assertEqual(result["Content_Chunks"], 1)
        self.assertIsNotNone(result["TTFT"])
        self.assertIsNotNone(result["TPS"])

    def test_summary_and_report_show_na_for_zero_output(self):
        df = pd.DataFrame(
            [
                {
                    "Run_ID": 1,
                    "Status": "warning",
                    "Output_Category": "empty_reply",
                    "Diagnosis": "Completed with finish_reason=stop but no textual content was received.",
                    "Finish_Reason": "stop",
                    "Backend": "ollama",
                    "Model": "demo-model",
                    "Params": {"temperature": 1},
                    "Applied_Params": {"temperature": 1},
                    "Config_Str": "{temperature=1}",
                    "TPS": None,
                    "TTFT": None,
                    "First_Event_s": 0.2,
                    "Stream_Duration_s": 0.6,
                    "Total_Chunks": 2,
                    "Content_Chunks": 0,
                    "Non_Content_Chunks": 1,
                    "Non_Content_Types": "role",
                    "VRAM_Base_MiB": 1000,
                    "VRAM_Peak_MiB": 1000,
                    "VRAM_Delta_MiB": 0,
                    "VRAM_Detail": "GPU 0 demo: 1000 -> 1000 / 16384 MiB (+0 MiB)",
                    "Efficiency_Score": None,
                    "Output_Chars": 0,
                    "Output_Text": "",
                    "Error": "",
                }
            ]
        )

        summary_df = bench_v3.build_summary_dataframe(df)
        self.assertEqual(summary_df.loc[0, "TPS (chunk/s)"], "N/A")
        self.assertEqual(summary_df.loc[0, "TTFT (s)"], "N/A")
        self.assertEqual(summary_df.loc[0, "Output Category"], "empty_reply")

        config = {
            "backend": "ollama",
            "url": "http://localhost:11434/v1",
            "models": ["demo-model"],
            "prompt": "demo prompt",
            "vram_monitoring": "nvidia-smi",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = bench_v3.save_markdown_report(df, config, str(Path(temp_dir) / "report"))
            report_text = Path(report_path).read_text(encoding="utf-8")

        self.assertIn("Output Category: empty_reply", report_text)
        self.assertIn("Diagnosis:", report_text)
        self.assertIn("TPS: N/A chunk/s", report_text)
        self.assertIn("TTFT: N/A s", report_text)

    def test_filter_eligible_results_excludes_warning_and_error_rows(self):
        df = pd.DataFrame(
            [
                {
                    "Status": "ok",
                    "Output_Category": "normal_content",
                    "TPS": 10.0,
                    "TTFT": 0.5,
                    "VRAM_Peak_MiB": 1024,
                    "Efficiency_Score": 10.0,
                },
                {
                    "Status": "warning",
                    "Output_Category": "empty_reply",
                    "TPS": None,
                    "TTFT": None,
                    "VRAM_Peak_MiB": 1024,
                    "Efficiency_Score": None,
                },
                {
                    "Status": "error",
                    "Output_Category": "normal_content",
                    "TPS": 8.0,
                    "TTFT": 0.7,
                    "VRAM_Peak_MiB": 1024,
                    "Efficiency_Score": 8.0,
                },
            ]
        )

        eligible_df = bench_v3.filter_eligible_results(df)
        self.assertEqual(len(eligible_df), 1)
        self.assertEqual(eligible_df.iloc[0]["Status"], "ok")
        self.assertEqual(eligible_df.iloc[0]["Output_Category"], "normal_content")

    def test_filter_eligible_results_for_tools_mode(self):
        df = pd.DataFrame(
            [
                {
                    "Status": "ok",
                    "Output_Category": "tool_call",
                    "First_Event_s": 0.2,
                },
                {
                    "Status": "warning",
                    "Output_Category": "text_reply_without_tool",
                    "First_Event_s": 0.1,
                },
            ]
        )

        eligible_df = bench_v3.filter_eligible_results(df, capability="tools")
        self.assertEqual(len(eligible_df), 1)
        self.assertEqual(eligible_df.iloc[0]["Output_Category"], "tool_call")

    def test_tool_mode_request_payload_includes_tools(self):
        config = {
            "backend": "ollama",
            "capability": "tools",
            "prompt": "use tool please",
        }

        payload = bench_v3.build_chat_request_payload(config, "demo-model", {})
        self.assertEqual(payload["model"], "demo-model")
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertIn("tools", payload)
        self.assertEqual(payload["messages"][0]["role"], "system")

    def test_save_raw_outputs_keeps_output_text_for_tools_runs(self):
        df = pd.DataFrame(
            [
                {
                    "Run_ID": 1,
                    "Status": "ok",
                    "Capability": "tools",
                    "Output_Category": "tool_call",
                    "Diagnosis": "Received tool_calls payload.",
                    "Finish_Reason": "tool_calls",
                    "Backend": "ollama",
                    "Model": "demo-model",
                    "Params": {},
                    "Applied_Params": {},
                    "Config_Str": "{}",
                    "TPS": None,
                    "TTFT": None,
                    "First_Event_s": 0.2,
                    "Stream_Duration_s": 0.4,
                    "Total_Chunks": 2,
                    "Content_Chunks": 0,
                    "Non_Content_Chunks": 1,
                    "Non_Content_Types": "tool_calls",
                    "VRAM_Base_MiB": None,
                    "VRAM_Peak_MiB": None,
                    "VRAM_Delta_MiB": None,
                    "VRAM_Detail": "N/A",
                    "Efficiency_Score": None,
                    "Output_Chars": 0,
                    "Output_Text": "",
                    "Error": "",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = bench_v3.save_raw_outputs(df, str(Path(temp_dir) / "report"))
            output_text = Path(output_path).read_text(encoding="utf-8")

        self.assertIn('"Output_Category": "tool_call"', output_text)
        self.assertIn('"Capability": "tools"', output_text)

    def test_select_plot_dataframe_keeps_all_tools_runs(self):
        df = pd.DataFrame(
            [
                {
                    "Status": "warning",
                    "Output_Category": "text_reply_without_tool",
                    "Model": "demo-model",
                    "Config_Str": "{}",
                },
                {
                    "Status": "error",
                    "Output_Category": "early_stop",
                    "Model": "demo-model",
                    "Config_Str": "{temperature=0.1}",
                },
            ]
        )

        plot_df, used_fallback = bench_v3.select_plot_dataframe(df, capability="tools")
        self.assertEqual(len(plot_df), 2)
        self.assertFalse(used_fallback)

    def test_build_result_row_marks_retained_sections(self):
        classification = {
            "Status": "ok",
            "Output_Category": "normal_content",
            "Diagnosis": "ok",
            "Finish_Reason": "stop",
            "TPS": 10.0,
            "TTFT": 0.5,
            "First_Event_s": 0.2,
            "Stream_Duration_s": 1.0,
            "Total_Chunks": 4,
            "Content_Chunks": 2,
            "Non_Content_Chunks": 1,
            "Non_Content_Types": "reasoning",
        }
        vram_metrics = {
            "VRAM_Base_MiB": 1000,
            "VRAM_Peak_MiB": 2000,
            "VRAM_Delta_MiB": 1000,
            "VRAM_Detail": "demo",
        }

        row = bench_v3.build_result_row(
            run_id=1,
            config={"backend": "ollama", "capability": "chat"},
            model="demo-model",
            param_set={},
            applied_params={},
            display_params="{}",
            classification=classification,
            vram_metrics=vram_metrics,
            dialogue_output_text="Final answer",
            thinking_text="Hidden reasoning",
            error_message=None,
        )

        self.assertEqual(row["Retained_Sections"], ["thinking", "dialogue_output"])
        self.assertEqual(row["Thinking_Text"], "Hidden reasoning")
        self.assertEqual(row["Dialogue_Output_Text"], "Final answer")

    def test_report_labels_thinking_and_dialogue_output(self):
        df = pd.DataFrame(
            [
                {
                    "Run_ID": 1,
                    "Status": "ok",
                    "Capability": "chat",
                    "Output_Category": "normal_content",
                    "Diagnosis": "Completed with finish_reason=stop.",
                    "Finish_Reason": "stop",
                    "Backend": "ollama",
                    "Model": "demo-model",
                    "Params": {},
                    "Applied_Params": {},
                    "Config_Str": "{}",
                    "TPS": 10.0,
                    "TTFT": 0.2,
                    "First_Event_s": 0.1,
                    "Stream_Duration_s": 0.8,
                    "Total_Chunks": 3,
                    "Content_Chunks": 1,
                    "Non_Content_Chunks": 1,
                    "Non_Content_Types": "reasoning",
                    "VRAM_Base_MiB": None,
                    "VRAM_Peak_MiB": None,
                    "VRAM_Delta_MiB": None,
                    "VRAM_Detail": "N/A",
                    "Efficiency_Score": None,
                    "Retained_Sections": ["thinking", "dialogue_output"],
                    "Thinking_Chars": 8,
                    "Thinking_Text": "Thoughts",
                    "Dialogue_Output_Chars": 12,
                    "Dialogue_Output_Text": "Final answer",
                    "Output_Chars": 12,
                    "Output_Text": "Final answer",
                    "Error": "",
                }
            ]
        )

        config = {
            "backend": "ollama",
            "url": "http://localhost:11434/v1",
            "models": ["demo-model"],
            "prompt": "demo prompt",
            "vram_monitoring": "nvidia-smi",
            "capability": "chat",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = bench_v3.save_markdown_report(df, config, str(Path(temp_dir) / "report"))
            report_text = Path(report_path).read_text(encoding="utf-8")

        self.assertIn("Retained Sections: thinking, dialogue_output", report_text)
        self.assertIn("#### thinking", report_text)
        self.assertIn("#### dialogue_output", report_text)


if __name__ == "__main__":
    unittest.main()
