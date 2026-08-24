#!/usr/bin/env python3
"""Score and render a Local Expert Battle JSONL run.

Objective questions are scored from the saved benchmark output.  PLC and
long-summary questions intentionally remain pending until a reviewer writes a
0..1 score into the generated review template.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


DOMAIN_ORDER = ("plc", "engineering_calculation", "traditional_chinese", "long_summary")
DOMAIN_LABELS = {
    "plc": "PLC 程式",
    "engineering_calculation": "工程計算推理",
    "traditional_chinese": "繁中語境",
    "long_summary": "長文摘要",
}
PASS_SCORE = 0.70
NUMBER_PATTERN = re.compile(r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if isinstance(item, dict):
            rows.append(item)
    if not rows:
        raise ValueError(f"No JSONL rows found in {path}")
    return rows


def load_reviews(path: Path | None) -> list[dict]:
    if not path or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("reviews", [])
    if not isinstance(payload, list):
        raise ValueError("Review file must be a JSON list or {\"reviews\": [...]}.")
    return [item for item in payload if isinstance(item, dict)]


def find_review(reviews: list[dict], row: dict) -> float | None:
    run_id = str(row.get("Run_ID", ""))
    model = str(row.get("Model", ""))
    question_id = str(row.get("Question_ID", ""))
    for review in reviews:
        if str(review.get("run_id", "")) == run_id and run_id:
            return coerce_score(review.get("score", review.get("review_score")))
    for review in reviews:
        if str(review.get("model", "")) == model and str(review.get("question_id", "")) == question_id:
            return coerce_score(review.get("score", review.get("review_score")))
    return None


def coerce_score(value) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, score)) if math.isfinite(score) else None


def parse_auto_checks(row: dict) -> dict:
    checks = row.get("Question_Auto_Checks") or {}
    if isinstance(checks, str):
        try:
            checks = json.loads(checks)
        except json.JSONDecodeError:
            return {}
    return checks if isinstance(checks, dict) else {}


def extract_numbers(text: str) -> list[float]:
    values = []
    for token in NUMBER_PATTERN.findall(text or ""):
        try:
            values.append(float(token.replace(",", "")))
        except ValueError:
            pass
    return values


def automatic_score(row: dict) -> tuple[float | None, str]:
    checks = parse_auto_checks(row)
    if not checks:
        return None, "待人工覆核"
    if str(row.get("Status", "")) != "ok":
        return 0.0, "請求失敗"

    answer = str(row.get("Dialogue_Output_Text") or row.get("Output_Text") or "")
    components = []
    numeric_checks = checks.get("numeric") or []
    if numeric_checks:
        actual_values = extract_numbers(answer)
        numeric_hits = 0
        for expected in numeric_checks:
            target = float(expected["value"])
            tolerance = float(expected.get("tolerance", 0.001))
            if any(abs(value - target) <= tolerance for value in actual_values):
                numeric_hits += 1
        components.append(numeric_hits / len(numeric_checks))

    required_groups = checks.get("required_terms") or []
    if required_groups:
        answer_lower = answer.lower()
        term_hits = sum(any(str(term).lower() in answer_lower for term in group) for group in required_groups)
        components.append(term_hits / len(required_groups))

    if not components:
        return None, "無可執行自動規則"
    return sum(components) / len(components), "自動評分"


def response_time(row: dict) -> float | None:
    for field in ("Answer_Time_s", "Stream_Duration_s", "Output_Time_s"):
        try:
            value = float(row.get(field))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0:
            return value
    return None


def score_rows(rows: list[dict], reviews: list[dict]) -> list[dict]:
    scored = []
    for row in rows:
        category = str(row.get("Question_Category", ""))
        if category not in DOMAIN_ORDER:
            continue
        auto_score, evaluation_state = automatic_score(row)
        review_score = find_review(reviews, row)
        score = review_score if review_score is not None else auto_score
        state = "人工覆核" if review_score is not None else evaluation_state
        scored.append(
            {
                "run_id": str(row.get("Run_ID", "")),
                "model": str(row.get("Model", "unknown")),
                "domain": category,
                "question_id": str(row.get("Question_ID", "")),
                "question_title": str(row.get("Question_Title", "")),
                "status": str(row.get("Status", "")),
                "score": score,
                "auto_score": auto_score,
                "review_score": review_score,
                "evaluation_state": state,
                "response_time_s": response_time(row),
            }
        )
    if not scored:
        raise ValueError("This JSONL does not contain Local Expert Battle question metadata.")
    return scored


def build_summary(scored: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in scored:
        grouped[(item["model"], item["domain"])].append(item)

    models = sorted({item["model"] for item in scored})
    summary = []
    for model in models:
        for domain in (*DOMAIN_ORDER, "overall"):
            items = (
                [item for item in scored if item["model"] == model]
                if domain == "overall"
                else grouped[(model, domain)]
            )
            eligible = [item for item in items if item["score"] is not None]
            wins = sum(item["score"] >= PASS_SCORE for item in eligible)
            times = [item["response_time_s"] for item in items if item["response_time_s"] is not None]
            summary.append(
                {
                    "model": model,
                    "domain": domain,
                    "runs": len(items),
                    "scored": len(eligible),
                    "pending": len(items) - len(eligible),
                    "wins": wins,
                    "win_rate": wins / len(eligible) if eligible else None,
                    "average_response_s": mean(times) if times else None,
                    "average_score": mean([item["score"] for item in eligible]) if eligible else None,
                }
            )
    return summary


def format_percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def format_number(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def make_review_template(scored: list[dict], path: Path) -> None:
    existing = load_reviews(path) if path.exists() else []
    existing_keys = {(str(item.get("model", "")), str(item.get("question_id", ""))) for item in existing}
    template = list(existing)
    for item in scored:
        if item["auto_score"] is not None or (item["model"], item["question_id"]) in existing_keys:
            continue
        template.append(
            {
                "model": item["model"],
                "question_id": item["question_id"],
                "score": None,
                "note": f"{DOMAIN_LABELS[item['domain']]}｜{item['question_title']}；0–1，0.70 以上算勝。",
            }
        )
    path.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_table(summary: list[dict]) -> str:
    rows = []
    for item in summary:
        label = "總體" if item["domain"] == "overall" else DOMAIN_LABELS[item["domain"]]
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['model'])}</td>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{item['wins']} / {item['scored']}</td>"
            f"<td class=\"rate\">{format_percent(item['win_rate'])}</td>"
            f"<td>{item['pending']}</td>"
            f"<td>{format_number(item['average_score'])}</td>"
            f"<td>{format_number(item['average_response_s'])} s</td>"
            "</tr>"
        )
    return "".join(rows)


def render_bar_charts(summary: list[dict]) -> str:
    overall = [item for item in summary if item["domain"] == "overall"]
    palette = ("#ef7148", "#b8db5e", "#60a5a8", "#d8a657", "#b79ad8", "#e58aa0")
    metrics = (
        ("win_rate", "總體勝率", True, lambda value: format_percent(value)),
        ("average_score", "平均分數", True, lambda value: format_number(value)),
        ("average_response_s", "平均回應時間", False, lambda value: f"{format_number(value)} s"),
    )
    cards = []
    for key, title, higher_is_better, formatter in metrics:
        available = [item[key] for item in overall if item[key] is not None]
        if not available:
            continue
        scale_max = max(available) or 1.0
        rows = []
        for index, item in enumerate(overall):
            value = item[key]
            width = 0.0 if value is None else max(1.5, min(100.0, value / scale_max * 100.0))
            rows.append(
                '<div class="bar-row">'
                f'<div class="bar-label" title="{html.escape(item["model"])}">{html.escape(item["model"])}</div>'
                f'<div class="bar-track"><div class="bar-fill" style="width:{width:.2f}%;background:{palette[index % len(palette)]}"></div></div>'
                f'<div class="bar-value">{formatter(value)}</div>'
                '</div>'
            )
        direction = "越高越好" if higher_is_better else "越低越好"
        cards.append(
            '<article class="chart-card">'
            f'<h3>{html.escape(title)}</h3><p>{direction}</p>{"".join(rows)}'
            '</article>'
        )

    domain_cards = []
    for domain in DOMAIN_ORDER:
        items = [item for item in summary if item["domain"] == domain]
        if not any(item["win_rate"] is not None for item in items):
            continue
        rows = []
        for index, item in enumerate(items):
            value = item["win_rate"]
            width = 0.0 if value is None else max(1.5, value * 100.0)
            rows.append(
                '<div class="bar-row">'
                f'<div class="bar-label" title="{html.escape(item["model"])}">{html.escape(item["model"])}</div>'
                f'<div class="bar-track"><div class="bar-fill" style="width:{width:.2f}%;background:{palette[index % len(palette)]}"></div></div>'
                f'<div class="bar-value">{format_percent(value)}</div>'
                '</div>'
            )
        domain_cards.append(
            '<article class="chart-card">'
            f'<h3>{html.escape(DOMAIN_LABELS[domain])}</h3><p>領域勝率 · 越高越好</p>{"".join(rows)}'
            '</article>'
        )

    if not cards and not domain_cards:
        return '<p class="note">尚無已評分資料可繪製長條圖。</p>'
    return '<div class="chart-grid">' + "".join(cards + domain_cards) + '</div>'


def render_html(summary: list[dict], scored: list[dict], source: Path, review_path: Path) -> str:
    overall = [item for item in summary if item["domain"] == "overall"]
    leader = max((item for item in overall if item["win_rate"] is not None), key=lambda item: item["win_rate"], default=None)
    pending = sum(item["pending"] for item in overall)
    leader_text = "尚無可自動比較的結果" if leader is None else f"{leader['model']} · {format_percent(leader['win_rate'])}"
    detail_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['model'])}</td><td>{html.escape(DOMAIN_LABELS[item['domain']])}</td>"
        f"<td>{html.escape(item['question_id'])}</td><td>{html.escape(item['question_title'])}</td>"
        f"<td>{format_number(item['score'])}</td><td>{html.escape(item['evaluation_state'])}</td>"
        f"<td>{format_number(item['response_time_s'])} s</td></tr>"
        for item in scored
    )
    return f"""<!doctype html>
<html lang=\"zh-Hant\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>Local Expert Battle · 結果</title>
<style>
  :root {{ color-scheme: dark; --ink:#11110f; --paper:#f1eadb; --line:#3b3931; --muted:#b9b09f; --hot:#ef7148; --good:#b8db5e; --panel:#1b1b17; }}
  * {{ box-sizing:border-box }} body {{ margin:0; background:var(--ink); color:var(--paper); font-family:ui-monospace,Consolas,"Noto Sans TC",monospace; }}
  main {{ max-width:1440px; margin:auto; padding:34px 28px 64px; }}
  .eyebrow {{ color:var(--hot); letter-spacing:.18em; font-weight:700; font-size:.76rem; }} h1 {{ font-family:Georgia,"Noto Serif TC",serif; font-size:clamp(2.6rem,7vw,6.6rem); line-height:.86; margin:14px 0 24px; letter-spacing:-.06em; }}
  .hero {{ display:grid; grid-template-columns:1.35fr .65fr; gap:24px; border-bottom:1px solid var(--line); padding-bottom:28px; }} .lead {{ max-width:66ch; color:var(--muted); line-height:1.7; }}
  .leader {{ background:var(--hot); color:#19100d; padding:24px; align-self:end; }} .leader small {{ display:block; font-weight:700; letter-spacing:.12em; }} .leader strong {{ display:block; font-size:1.55rem; margin-top:10px; overflow-wrap:anywhere; }}
  .cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:24px 0; }} .card {{ border:1px solid var(--line); padding:16px; min-height:112px; background:var(--panel); }} .card span {{ display:block; color:var(--muted); font-size:.78rem; }} .card strong {{ display:block; color:var(--good); margin-top:14px; font-size:1.45rem; }}
  section {{ margin-top:34px; }} h2 {{ font-family:Georgia,"Noto Serif TC",serif; font-size:2rem; margin:0 0 12px; }} .note {{ color:var(--muted); line-height:1.65; }}
  .table-wrap {{ overflow:auto; border:1px solid var(--line); }} table {{ width:100%; border-collapse:collapse; min-width:780px; }} th,td {{ text-align:left; padding:12px 14px; border-bottom:1px solid var(--line); vertical-align:top; }} th {{ color:var(--muted); font-size:.75rem; letter-spacing:.08em; text-transform:uppercase; }} td.rate {{ color:var(--good); font-weight:800; }} tr:last-child td {{ border-bottom:0; }}
  .chart-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }} .chart-card {{ border-top:3px solid var(--hot); padding:16px 0 4px; }} .chart-card h3 {{ margin:0; font-family:Georgia,"Noto Serif TC",serif; font-size:1.2rem; }} .chart-card p {{ color:var(--muted); font-size:.78rem; margin:5px 0 14px; }}
  .bar-row {{ display:grid; grid-template-columns:minmax(100px,.8fr) minmax(130px,2fr) 70px; gap:10px; align-items:center; min-height:34px; }} .bar-label {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:.82rem; }} .bar-track {{ height:13px; background:#302f29; overflow:hidden; }} .bar-fill {{ height:100%; }} .bar-value {{ text-align:right; font-variant-numeric:tabular-nums; font-size:.8rem; color:var(--paper); }}
  details {{ margin-top:30px; }} summary {{ cursor:pointer; color:var(--hot); font-weight:700; padding:12px 0; }} code {{ color:var(--good); }} @media(max-width:760px) {{ main {{ padding:24px 16px }} .hero {{ grid-template-columns:1fr }} .cards {{ grid-template-columns:repeat(2,1fr) }} .chart-grid {{ grid-template-columns:1fr }} }}
</style></head><body><main>
<div class=\"hero\"><div><div class=\"eyebrow\">LOCAL · LLAMA.CPP · MODEL ARENA</div><h1>EXPERT<br>BATTLE</h1><p class=\"lead\">PLC、工程計算、繁中語境與本機 wiki 長文摘要的在地模型對戰。勝率只以已完成自動評分或人工覆核的題目計算，避免把「未評」誤當失敗。</p></div><aside class=\"leader\"><small>目前領先</small><strong>{html.escape(leader_text)}</strong></aside></div>
<div class=\"cards\"><div class=\"card\"><span>模型數</span><strong>{len(overall)}</strong></div><div class=\"card\"><span>已執行題數</span><strong>{len(scored)}</strong></div><div class=\"card\"><span>待人工覆核</span><strong>{pending}</strong></div><div class=\"card\"><span>勝利門檻</span><strong>{PASS_SCORE:.0%}</strong></div></div>
<section><h2>模型長條圖對比</h2><p class=\"note\">以實際已評分結果呈現；沒有分數的待覆核題目不會被當成零分。</p>{render_bar_charts(summary)}</section>
<section><h2>各領域與總體勝率</h2><p class=\"note\">「勝」= 分數 ≥ 0.70。工程計算採數值容差比對；繁中語境採必要語意詞群；PLC 與長文摘要預設待人工覆核。</p><div class=\"table-wrap\"><table><thead><tr><th>模型</th><th>領域</th><th>勝 / 已評</th><th>勝率</th><th>待評</th><th>平均分數</th><th>平均回應時間</th></tr></thead><tbody>{render_table(summary)}</tbody></table></div></section>
<section><h2>人工覆核</h2><p class=\"note\">請填寫 <code>{html.escape(review_path.name)}</code> 的 score（0–1），再執行：<code>python report.py --input {html.escape(source.name)} --review {html.escape(review_path.name)}</code>。PLC 與摘要的重點是可執行性、來源忠實與無幻覺。</p></section>
<details><summary>逐題結果（{len(scored)}）</summary><div class=\"table-wrap\"><table><thead><tr><th>模型</th><th>領域</th><th>題號</th><th>題目</th><th>分數</th><th>評分狀態</th><th>回應時間</th></tr></thead><tbody>{detail_rows}</tbody></table></div></details>
</main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Local Expert Battle scores from benchmark JSONL.")
    parser.add_argument("--input", required=True, type=Path, help="*_outputs.jsonl produced by llm_expert_bench.py")
    parser.add_argument("--review", type=Path, help="Optional manual review JSON file")
    parser.add_argument("--output", type=Path, help="Output HTML path")
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = (args.output or input_path.with_name(f"{input_path.stem}_battle.html")).resolve()
    review_path = (args.review or input_path.with_name(f"{input_path.stem}_manual_review.json")).resolve()
    scored = score_rows(load_jsonl(input_path), load_reviews(args.review))
    make_review_template(scored, review_path)
    summary = build_summary(scored)
    output_path.write_text(render_html(summary, scored, input_path, review_path), encoding="utf-8")
    print(f"Battle report: {output_path}")
    print(f"Manual review template: {review_path}")
    for item in summary:
        if item["domain"] == "overall":
            print(f"{item['model']}: {item['wins']}/{item['scored']} · {format_percent(item['win_rate'])} · {format_number(item['average_response_s'])} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
