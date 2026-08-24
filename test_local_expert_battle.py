import tempfile
import unittest
import json
from collections import Counter
from pathlib import Path
from unittest import mock

import llm_expert_bench as bench
import report


class LocalExpertBattleTests(unittest.TestCase):
    def test_openai_compatible_model_list_reads_served_ids(self):
        response = mock.Mock()
        response.json.return_value = {
            "data": [
                {"id": "loaded-model"},
                {"id": "loaded-model"},
                {"id": "second-model"},
                {"id": ""},
            ]
        }
        with mock.patch.object(bench.requests, "get", return_value=response) as request:
            models = bench.get_openai_compatible_models("http://localhost:8080/v1/")

        request.assert_called_once_with("http://localhost:8080/v1/models", timeout=3)
        response.raise_for_status.assert_called_once_with()
        self.assertEqual(models, ["loaded-model", "second-model"])

    def test_testset_has_twelve_questions_per_domain(self):
        suite = bench.get_local_expert_battle_definition()
        counts = Counter(question["category"] for question in suite["questions"])

        self.assertEqual(suite["id"], bench.LOCAL_EXPERT_BATTLE_SUITE_ID)
        self.assertEqual(counts, {
            "plc": 12,
            "engineering_calculation": 12,
            "traditional_chinese": 12,
            "long_summary": 12,
        })
        self.assertEqual(len({question["id"] for question in suite["questions"]}), 48)

    def test_suite_hydrates_wiki_summary_prompts_at_runtime(self):
        with mock.patch.object(
            bench,
            "load_local_expert_wiki_excerpt",
            return_value=("A" * 2400, "C:/demo/wiki.md", 2400),
        ):
            questions = bench.resolve_benchmark_questions(
                {"capability": bench.LOCAL_EXPERT_BATTLE_SUITE_ID}
            )

        summary_question = next(question for question in questions if question["id"] == "sum-01")
        self.assertEqual(len(questions), 48)
        self.assertIn("A" * 120, summary_question["prompt"])
        self.assertEqual(summary_question["wiki_source_path"], "C:/demo/wiki.md")
        self.assertEqual(summary_question["wiki_excerpt_chars"], 2400)

    def test_report_scores_objective_rows_and_marks_manual_rows_pending(self):
        rows = [
            {
                "Run_ID": 1,
                "Model": "model-a",
                "Status": "ok",
                "Question_Category": "engineering_calculation",
                "Question_ID": "eng-01",
                "Question_Title": "area",
                "Question_Auto_Checks": {"numeric": [{"value": 13200, "tolerance": 0.5}]},
                "Dialogue_Output_Text": "13200 cm²",
                "Answer_Time_s": 1.5,
            },
            {
                "Run_ID": 2,
                "Model": "model-a",
                "Status": "ok",
                "Question_Category": "plc",
                "Question_ID": "plc-01",
                "Question_Title": "edge trigger",
                "Dialogue_Output_Text": "Use R_TRIG.",
                "Answer_Time_s": 2.0,
            },
        ]

        scored = report.score_rows(rows, reviews=[])
        engineering = next(item for item in scored if item["question_id"] == "eng-01")
        plc = next(item for item in scored if item["question_id"] == "plc-01")

        self.assertEqual(engineering["score"], 1.0)
        self.assertIsNone(plc["score"])
        self.assertEqual(plc["evaluation_state"], "待人工覆核")

        reviewed = report.score_rows(rows, reviews=[{"model": "model-a", "question_id": "plc-01", "score": 0.8}])
        self.assertEqual(next(item for item in reviewed if item["question_id"] == "plc-01")["score"], 0.8)

    def test_review_template_and_html_are_written(self):
        scored = [
            {
                "run_id": "1",
                "model": "model-a",
                "domain": "plc",
                "question_id": "plc-01",
                "question_title": "edge trigger",
                "status": "ok",
                "score": None,
                "auto_score": None,
                "review_score": None,
                "evaluation_state": "待人工覆核",
                "response_time_s": 1.0,
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            review_path = temp_path / "review.json"
            output_path = temp_path / "battle.html"
            report.make_review_template(scored, review_path)
            html = report.render_html(report.build_summary(scored), scored, temp_path / "outputs.jsonl", review_path)
            output_path.write_text(html, encoding="utf-8")

            self.assertIn('"question_id": "plc-01"', review_path.read_text(encoding="utf-8"))
            self.assertIn("EXPERT", output_path.read_text(encoding="utf-8"))
            self.assertIn("模型長條圖對比", output_path.read_text(encoding="utf-8"))
            self.assertIn('class="bar-fill"', output_path.read_text(encoding="utf-8"))

    def test_llama_cpp_catalog_reads_easy_llamacpp_index_and_skips_mmproj(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index_path = root / "json" / "model-index.json"
            index_path.parent.mkdir()
            index_path.write_text(
                json.dumps(
                    {
                        "default_model_id": "chat-model",
                        "models": [
                            {"id": "chat-model", "name": "Chat Model", "path": "D:/Models/chat.gguf"},
                            {"id": "vision", "name": "mmproj-chat", "path": "D:/Models/mmproj-chat.gguf"},
                        ],
                    }
                ),
                encoding="utf-8-sig",
            )
            with mock.patch.object(bench, "get_llama_cpp_launcher_root", return_value=root):
                models = bench.get_llama_cpp_models()

        self.assertEqual(models[0]["name"], "Chat Model")
        self.assertTrue(models[0]["is_default"])
        self.assertEqual(len(models), 1)

    def test_web_config_resolves_selected_ggufs_for_batch_switching(self):
        catalog = [
            {"name": "Model A", "path": "D:/Models/a.gguf", "available": True, "id": "a"},
            {"name": "Model B", "path": "D:/Models/b.gguf", "available": True, "id": "b"},
        ]
        with mock.patch.object(bench, "get_llama_cpp_models", return_value=catalog):
            config = bench.normalize_web_ui_config(
                {
                    "backend": "llama.cpp",
                    "capability": "chat",
                    "url": "http://127.0.0.1:8080/v1",
                    "models": "Model A, Model B",
                    "llama_cpp_auto_switch": True,
                    "params": {},
                }
            )

        self.assertTrue(config["llama_cpp_auto_switch"])
        self.assertEqual([item["path"] for item in config["llama_cpp_model_entries"]], [
            "D:/Models/a.gguf",
            "D:/Models/b.gguf",
        ])

    def test_web_config_uses_current_llama_cpp_server_when_no_model_is_selected(self):
        with mock.patch.object(
            bench,
            "get_openai_compatible_models",
            return_value=["currently-loaded-model"],
        ) as get_served_models:
            config = bench.normalize_web_ui_config(
                {
                    "backend": "llama.cpp",
                    "capability": "chat",
                    "url": "http://localhost:8080/v1",
                    "models": "",
                    "llama_cpp_auto_switch": True,
                    "params": {},
                }
            )

        get_served_models.assert_called_once_with("http://localhost:8080/v1")
        self.assertEqual(config["models"], ["currently-loaded-model"])
        self.assertTrue(config["use_current_llama_cpp_model"])
        self.assertFalse(config["llama_cpp_auto_switch"])

    def test_web_config_has_safe_label_when_current_llama_cpp_model_cannot_be_listed(self):
        with mock.patch.object(bench, "get_openai_compatible_models", return_value=[]):
            config = bench.normalize_web_ui_config(
                {
                    "backend": "llama.cpp",
                    "capability": "chat",
                    "models": "",
                    "params": {},
                }
            )

        self.assertEqual(config["url"], "http://localhost:8080/v1")
        self.assertEqual(config["models"], [bench.CURRENT_LLAMA_CPP_MODEL_FALLBACK])
        self.assertFalse(config["llama_cpp_auto_switch"])


if __name__ == "__main__":
    unittest.main()
