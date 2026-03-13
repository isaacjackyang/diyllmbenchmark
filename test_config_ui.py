import unittest

import ollama_expert_bench as bench


def fragments_to_text(fragments):
    parts = []
    for fragment in fragments:
        if len(fragment) >= 2:
            parts.append(fragment[1])
    return "".join(parts)


class ConfigUiTests(unittest.TestCase):
    def build_rows(self, backend="ollama"):
        return bench.build_param_rows(backend)

    def row_by_key(self, rows, key):
        return next(row for row in rows if row.key == key)

    def test_build_param_rows_marks_supported_and_locked_params(self):
        ollama_rows = self.build_rows("ollama")
        llama_rows = self.build_rows("llama.cpp")

        self.assertEqual(
            [row.key for row in ollama_rows],
            bench.ordered_param_keys(),
        )
        self.assertEqual(
            set(row.key for row in ollama_rows),
            set(bench.PARAM_INFO),
        )
        self.assertTrue(self.row_by_key(ollama_rows, "num_ctx").supported)
        self.assertFalse(self.row_by_key(llama_rows, "num_ctx").supported)
        self.assertEqual(
            self.row_by_key(ollama_rows, "temperature").raw_value,
            str(bench.PARAM_INFO["temperature"]["default"]),
        )

    def test_validate_param_rows_returns_enabled_values_only(self):
        rows = self.build_rows("ollama")

        temperature = self.row_by_key(rows, "temperature")
        top_p = self.row_by_key(rows, "top_p")
        temperature.enabled = True
        temperature.raw_value = "0.1, 0.8"
        top_p.enabled = True
        top_p.raw_value = "0.9, 0.95"

        final_params, error_row_index, error_message = bench.validate_param_rows(rows)

        self.assertEqual(
            final_params,
            {
                "temperature": [0.1, 0.8],
                "top_p": [0.9, 0.95],
            },
        )
        self.assertIsNone(error_row_index)
        self.assertIsNone(error_message)
        self.assertEqual(bench.estimate_combo_count(final_params), 4)

    def test_validate_param_rows_reports_first_invalid_row(self):
        rows = self.build_rows("ollama")

        temperature = self.row_by_key(rows, "temperature")
        temperature.enabled = True
        temperature.raw_value = "abc"

        final_params, error_row_index, error_message = bench.validate_param_rows(rows)

        self.assertIsNone(final_params)
        self.assertEqual(error_row_index, rows.index(temperature))
        self.assertIn("溫度", error_message)
        self.assertIn("無法解析數值：abc", error_message)

    def test_row_value_count_returns_lock_dash_count_and_err(self):
        unsupported_rows = self.build_rows("llama.cpp")
        locked_row = self.row_by_key(unsupported_rows, "num_ctx")
        self.assertEqual(bench.row_value_count(locked_row), "LOCK")

        rows = self.build_rows("ollama")
        temperature = self.row_by_key(rows, "temperature")
        top_p = self.row_by_key(rows, "top_p")
        repetition_penalty = self.row_by_key(rows, "repeat_penalty")

        self.assertEqual(bench.row_value_count(temperature), "-")

        top_p.enabled = True
        top_p.raw_value = "0.8, 0.9, 0.95"
        self.assertEqual(bench.row_value_count(top_p), "3")

        repetition_penalty.enabled = True
        repetition_penalty.raw_value = "oops"
        self.assertEqual(bench.row_value_count(repetition_penalty), "ERR")

    def test_build_grid_fragments_shows_preview_summary_and_hint(self):
        rows = self.build_rows("ollama")
        temperature = self.row_by_key(rows, "temperature")
        top_p = self.row_by_key(rows, "top_p")
        temperature.enabled = True
        temperature.raw_value = "0.1, 0.8"
        top_p.enabled = True
        top_p.raw_value = "0.9, 0.95"

        fragments = bench.build_grid_fragments(
            rows=rows,
            selected_row_index=rows.index(temperature),
            selected_column_index=1,
            message="custom hint",
            message_style="class:hint",
        )
        rendered_text = fragments_to_text(fragments)
        fragment_styles = [fragment[0] for fragment in fragments if fragment]

        self.assertIn("LLM Benchmark | Full-Page Parameter Grid", rendered_text)
        self.assertIn("Config Preview", rendered_text)
        self.assertIn("Selected params: 2", rendered_text)
        self.assertIn("Combination count: 4", rendered_text)
        self.assertIn("Validation: OK", rendered_text)
        self.assertIn("custom hint", rendered_text)
        self.assertIn("class:table.cell.current", fragment_styles)

    def test_build_grid_fragments_shows_validation_error(self):
        rows = self.build_rows("ollama")
        temperature = self.row_by_key(rows, "temperature")
        temperature.enabled = True
        temperature.raw_value = "bad"

        fragments = bench.build_grid_fragments(
            rows=rows,
            selected_row_index=rows.index(temperature),
            selected_column_index=0,
            message="",
            message_style="class:hint",
        )
        rendered_text = fragments_to_text(fragments)

        self.assertIn("Selected params: 1", rendered_text)
        self.assertIn("Combination count: ERR", rendered_text)
        self.assertIn("Validation: 溫度 (Temperature): 無法解析數值：bad", rendered_text)

    def test_parse_system_prompt_blocks_accepts_expected_count(self):
        prompts = bench.parse_system_prompt_blocks("first prompt\n---\nsecond prompt", 2)

        self.assertEqual(prompts, ["first prompt", "second prompt"])

    def test_parse_system_prompt_blocks_rejects_wrong_count(self):
        with self.assertRaises(ValueError):
            bench.parse_system_prompt_blocks("only one prompt", 2)


if __name__ == "__main__":
    unittest.main()
