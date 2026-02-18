"""
Build HTML comparison report: baseline greedy vs checkpoint.

Supports two checkpoint formats:
  - Greedy JSONL (same format as baseline — has full response text)
  - Pass@16 JSONL (aggregate stats only — n_correct/n_total)

Auto-detects format based on whether records have a "response" field.

Usage:
    python src/build_comparison_report.py [baseline_path] [checkpoint_path]
"""

import json
import html
import os
import re
import sys


# ── Inline entity extraction (avoids importing datasets library) ─────────

DAYS = {
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday",
}
MONTHS = {
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
}
COMMON_NON_NAMES = {
    "If", "How", "What", "When", "Where", "Who", "Why", "Which",
    "The", "He", "She", "They", "His", "Her", "It", "Its", "Their",
    "Each", "Every", "Some", "All", "One", "Two", "Three", "Four",
    "Five", "Six", "Seven", "Eight", "Nine", "Ten",
    "This", "That", "These", "Those", "There", "Here",
    "After", "Before", "During", "Since", "While", "Until",
    "In", "On", "At", "For", "From", "By", "To", "Of", "With",
    "But", "And", "Or", "So", "Yet", "Then", "Than", "Also",
    "An", "A", "As", "Be", "Do", "Is", "Was", "Are", "Were",
    "Has", "Had", "Have", "Can", "Could", "Would", "Should",
    "Not", "No", "Yes", "Both", "Many", "Most", "Only",
    "Step", "Answer", "Total", "Calculate", "Find", "Determine",
    "Given", "Let", "Solve", "Note", "Problem", "Solution",
    "Half", "Twice", "Double", "Triple", "Quarter",
    "UV", "Art", "TV", "DVD", "GPS", "ATM", "GPA", "SAT", "ACT",
    "NFL", "NBA", "MLB", "USA", "US", "UK", "PM", "AM",
    "North", "South", "East", "West", "Central",
    "City", "Town", "Street", "Avenue", "Road", "Park", "Drive",
    "Lake", "River", "Mountain", "Island", "Valley", "Beach", "Bay",
    "York", "Angeles", "Francisco", "Diego",
    "America", "American", "European", "African", "Asian",
    "German", "French", "Chinese", "Japanese", "Italian", "Spanish",
    "English", "British", "Mexican", "Canadian",
    "Facebook", "Instagram", "Twitter", "YouTube", "Amazon", "Google",
    "Metropolitan", "Museum", "Library", "University", "College",
    "Hospital", "Church", "Airport", "Station",
    "National", "International", "Local", "Royal",
    "Walmart", "Target", "Costco",
    "Student", "Students", "Teacher", "Teachers", "Farmer", "Class",
    "Children", "People", "Workers", "Company", "Store", "School",
    "Family", "Friends", "Team", "Group", "Member", "Members",
    "Boy", "Girl", "Boys", "Girls", "Man", "Woman", "Men", "Women",
    "Brother", "Sister", "Mother", "Father", "Uncle", "Aunt",
    "Son", "Daughter", "Husband", "Wife", "Baby",
    "Mom", "Dad", "Grandma", "Grandpa", "Grandad",
    "Grandmother", "Grandfather", "Cousin", "Neighbor",
    "Day", "Week", "Month", "Year", "Hour", "Minute",
    "Morning", "Afternoon", "Evening", "Night",
    "Western", "Eastern", "Northern", "Southern",
    "Lego", "Legos", "Scrabble", "Pokemon",
    "Junior", "Senior", "Little", "Big", "Great",
    "Christmas", "Easter", "Halloween", "Thanksgiving",
    "First", "Second", "Third", "Fourth", "Fifth", "Last", "Next",
    "New", "Old", "Once", "Twice", "Per", "Plus",
    "Currently", "Originally", "Recently", "Finally",
    "However", "Therefore", "Meanwhile", "Although",
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sir",
}
ALL_EXCLUSIONS = DAYS | MONTHS | COMMON_NON_NAMES


def _strip_possessive(name: str) -> str:
    if name.endswith("'s") and len(name) > 3:
        return name[:-2]
    if (name.endswith("s") and not name.endswith("ss") and len(name) > 3):
        return name[:-1]
    return name


def extract_entity_names(question: str) -> list[str]:
    sentences = re.split(r'[.?!\n]+', question)
    candidates = []
    for sent in sentences:
        words = sent.strip().split()
        if not words:
            continue
        for word in words:
            clean = re.sub(r"[^a-zA-Z']", "", word)
            if not clean or len(clean) < 2:
                continue
            if not clean[0].isupper():
                continue
            if clean.isupper() and len(clean) > 2:
                continue
            base = _strip_possessive(clean)
            if base in ALL_EXCLUSIONS or clean in ALL_EXCLUSIONS:
                continue
            if len(base) < 2:
                continue
            candidates.append(base)
    seen = set()
    unique = []
    for name in candidates:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            unique.append(name)
    return unique


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def key_by_idx(records: list[dict]) -> dict[int, dict]:
    return {r["idx"]: r for r in records}


def pass_rate_to_est_pass1(rate: float) -> float:
    return rate


def render_math(text: str) -> str:
    """Convert LaTeX-like notation into readable format for HTML."""
    t = html.escape(text)
    t = t.replace("\\(", "<i>").replace("\\)", "</i>")
    t = t.replace("\\[", "<div class='math'>").replace("\\]", "</div>")
    t = t.replace("\\boxed{", "<span class='boxed'>").replace("}", "}")
    t = t.replace("\n", "<br>")
    return t


def build_html(baseline_records, checkpoint_records):
    bg = key_by_idx(baseline_records)
    cp = key_by_idx(checkpoint_records)
    common = sorted(set(bg) & set(cp))

    # Auto-detect checkpoint format: greedy (has "response") vs pass@16 (has "n_correct")
    sample = checkpoint_records[0] if checkpoint_records else {}
    cp_is_greedy = "response" in sample

    rows = []
    for idx in common:
        b = bg[idx]
        c = cp[idx]
        entities = extract_entity_names(b["question"])
        b_correct = b["is_correct"]

        if cp_is_greedy:
            c_correct_bool = c["is_correct"]
            c_rate = 1.0 if c_correct_bool else 0.0
            if not b_correct and c_correct_bool:
                cat = "fixed"
            elif b_correct and not c_correct_bool:
                cat = "broken"
            elif b_correct and c_correct_bool:
                cat = "still_right"
            elif not b_correct and not c_correct_bool:
                cat = "still_wrong"
            else:
                cat = "unchanged"
        else:
            c_rate = c["n_correct"] / c["n_total"]
            if not b_correct and c_rate > 0.5:
                cat = "fixed"
            elif b_correct and c_rate < 0.5:
                cat = "broken"
            elif not b_correct and c_rate <= 0.0625:
                cat = "still_wrong"
            elif b_correct and c_rate >= 0.9:
                cat = "still_right"
            elif c_rate > (1.0 if b_correct else 0.0):
                cat = "improved"
            elif c_rate < (1.0 if b_correct else 0.0):
                cat = "regressed"
            else:
                cat = "unchanged"

        row = {
            "idx": idx,
            "question": b["question"],
            "gold_answer": b["gold_answer"],
            "gold_label": b["gold_label"],
            "entities": entities,
            "n_entities": len(entities),
            "b_correct": b_correct,
            "b_pred": b["predicted_answer"],
            "b_method": b["extraction_method"],
            "b_response": b["response"],
            "b_error": b.get("error_type", ""),
            "b_tokens": b.get("n_tokens", 0),
            "c_rate": c_rate,
            "category": cat,
            "cp_is_greedy": cp_is_greedy,
        }

        if cp_is_greedy:
            row.update({
                "c_correct_bool": c["is_correct"],
                "c_pred": c.get("predicted_answer", ""),
                "c_method": c.get("extraction_method", ""),
                "c_response": c.get("response", ""),
                "c_error": c.get("error_type", ""),
                "c_tokens": c.get("n_tokens", 0),
            })
        else:
            row.update({
                "c_correct": c["n_correct"],
                "c_total": c["n_total"],
                "c_methods": c.get("methods", {}),
            })

        rows.append(row)

    cat_order = {"fixed": 0, "improved": 1, "unchanged": 2, "still_right": 3,
                 "regressed": 4, "broken": 5, "still_wrong": 6}
    rows.sort(key=lambda r: (cat_order.get(r["category"], 9), -r["c_rate"]))

    n_baseline_correct = sum(1 for r in rows if r["b_correct"])
    if cp_is_greedy:
        n_cp_correct = sum(1 for r in rows if r["c_correct_bool"])
        cp_metric_label = "Checkpoint Pass@1"
        cp_metric_value = f"{n_cp_correct}%"
    else:
        avg_cp_rate = sum(r["c_rate"] for r in rows) / len(rows) if rows else 0
        cp_metric_label = "Checkpoint Avg Pass Rate"
        cp_metric_value = f"{avg_cp_rate*100:.1f}%"

    cat_counts = {}
    for r in rows:
        cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1

    cat_colors = {
        "fixed": "#22c55e",
        "improved": "#86efac",
        "unchanged": "#9ca3af",
        "still_right": "#60a5fa",
        "regressed": "#fbbf24",
        "broken": "#ef4444",
        "still_wrong": "#6b7280",
    }
    cat_labels_greedy = {
        "fixed": "Fixed (was wrong, now correct)",
        "still_right": "Still correct",
        "broken": "Broken (was correct, now wrong)",
        "still_wrong": "Still wrong",
        "unchanged": "Unchanged",
    }
    cat_labels_p16 = {
        "fixed": "Fixed (was wrong, now >50%)",
        "improved": "Improved",
        "unchanged": "Unchanged",
        "still_right": "Still right (was right, still >90%)",
        "regressed": "Regressed",
        "broken": "Broken (was right, now <50%)",
        "still_wrong": "Still wrong (was wrong, still ~0%)",
    }
    cat_labels = cat_labels_greedy if cp_is_greedy else cat_labels_p16

    mode_desc = "greedy (temp=0) pass@1" if cp_is_greedy else "16 samples (temp=0.7) pass rate"

    parts = []
    parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Baseline vs Checkpoint Comparison</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f172a; color: #e2e8f0; padding: 20px; line-height: 1.6; }
  .container { max-width: 1400px; margin: 0 auto; }
  h1 { font-size: 1.8em; margin-bottom: 8px; color: #f1f5f9; }
  .subtitle { color: #94a3b8; margin-bottom: 24px; }

  .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                   gap: 12px; margin-bottom: 24px; }
  .stat-card { background: #1e293b; border-radius: 10px; padding: 16px; text-align: center; }
  .stat-card .value { font-size: 2em; font-weight: 700; }
  .stat-card .label { font-size: 0.85em; color: #94a3b8; margin-top: 4px; }

  .cat-bar { display: flex; height: 32px; border-radius: 8px; overflow: hidden; margin-bottom: 24px; }
  .cat-segment { display: flex; align-items: center; justify-content: center;
                  font-size: 0.75em; font-weight: 600; color: #0f172a; min-width: 30px; }

  .filter-bar { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px; }
  .filter-btn { background: #1e293b; border: 1px solid #334155; color: #94a3b8;
                 padding: 6px 14px; border-radius: 20px; cursor: pointer; font-size: 0.85em;
                 transition: all 0.2s; }
  .filter-btn:hover { border-color: #60a5fa; color: #60a5fa; }
  .filter-btn.active { background: #1e40af; border-color: #3b82f6; color: #fff; }

  .problem { background: #1e293b; border-radius: 12px; margin-bottom: 16px;
              border-left: 4px solid #334155; overflow: hidden; }
  .problem.cat-fixed { border-left-color: #22c55e; }
  .problem.cat-improved { border-left-color: #86efac; }
  .problem.cat-broken { border-left-color: #ef4444; }
  .problem.cat-regressed { border-left-color: #fbbf24; }
  .problem.cat-still_right { border-left-color: #60a5fa; }
  .problem.cat-still_wrong { border-left-color: #6b7280; }
  .problem.cat-unchanged { border-left-color: #9ca3af; }

  .problem-header { padding: 14px 18px; cursor: pointer; display: flex;
                     justify-content: space-between; align-items: center; }
  .problem-header:hover { background: #253348; }
  .problem-header .left { display: flex; align-items: center; gap: 12px; flex: 1; min-width:0; }
  .problem-header .idx { font-weight: 700; color: #60a5fa; min-width: 60px; flex-shrink:0; }
  .problem-header .question-preview { color: #cbd5e1; font-size: 0.9em;
                                        overflow: hidden; text-overflow: ellipsis;
                                        white-space: nowrap; }
  .problem-header .badges { display: flex; gap: 8px; align-items: center; flex-shrink: 0; }
  .badge { padding: 2px 10px; border-radius: 12px; font-size: 0.78em; font-weight: 600; white-space:nowrap; }
  .badge-correct { background: #166534; color: #86efac; }
  .badge-wrong { background: #7f1d1d; color: #fca5a5; }
  .badge-rate { background: #1e3a5f; color: #93c5fd; }
  .badge-entities { background: #312e81; color: #a5b4fc; }
  .badge-cat { padding: 2px 10px; border-radius: 12px; font-size: 0.78em; font-weight: 600; }

  .problem-body { display: none; padding: 0 18px 18px; }
  .problem.open .problem-body { display: block; }
  .problem.open .arrow { transform: rotate(90deg); }
  .arrow { color: #64748b; transition: transform 0.2s; font-size: 1.2em; }

  .section-label { font-size: 0.8em; font-weight: 600; color: #64748b;
                    text-transform: uppercase; letter-spacing: 0.05em; margin: 12px 0 6px; }
  .question-text { background: #0f172a; padding: 14px; border-radius: 8px;
                    font-size: 0.92em; line-height: 1.7; color: #e2e8f0; }
  .gold-box { background: #14532d; padding: 10px 14px; border-radius: 8px;
               font-size: 0.88em; margin-top: 8px; }
  .gold-box .gold-answer { font-size: 1.3em; font-weight: 700; color: #4ade80; }
  .gold-steps { color: #86efac; font-size: 0.85em; margin-top: 4px; white-space: pre-wrap; }

  .comparison { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; }
  @media (max-width: 900px) { .comparison { grid-template-columns: 1fr; } }

  .panel { background: #0f172a; border-radius: 8px; padding: 14px; }
  .panel h3 { font-size: 0.9em; margin-bottom: 8px; }
  .panel.baseline h3 { color: #94a3b8; }
  .panel.checkpoint h3 { color: #60a5fa; }

  .response-text { font-size: 0.85em; line-height: 1.65; color: #cbd5e1;
                    max-height: 500px; overflow-y: auto; white-space: pre-wrap;
                    word-break: break-word; }
  .meta { display: flex; gap: 16px; margin-top: 8px; font-size: 0.8em; color: #64748b; }

  .entity-tags { display: flex; gap: 4px; flex-wrap: wrap; margin-top: 4px; }
  .entity-tag { background: #312e81; color: #c4b5fd; padding: 1px 8px;
                 border-radius: 10px; font-size: 0.78em; }

  .rate-bar { height: 8px; background: #334155; border-radius: 4px; margin-top: 8px; overflow: hidden; }
  .rate-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }

  .toolbar { display: flex; gap: 12px; margin-bottom: 16px; }
  .toolbar button { background: #1e293b; border: 1px solid #334155; color: #94a3b8;
                     padding: 8px 16px; border-radius: 8px; cursor: pointer; }
  .toolbar button:hover { border-color: #60a5fa; color: #60a5fa; }
</style>
</head>
<body>
<div class="container">
""")

    parts.append(f"""
<h1>Baseline vs Checkpoint: {len(common)} GSM8K Problems</h1>
<p class="subtitle">
  Baseline = greedy (temp=0) pass@1 &nbsp;|&nbsp;
  Checkpoint = {mode_desc} &nbsp;|&nbsp; No system prompt
</p>

<div class="summary-grid">
  <div class="stat-card">
    <div class="value">{n_baseline_correct}%</div>
    <div class="label">Baseline Pass@1</div>
  </div>
  <div class="stat-card">
    <div class="value" style="color:#60a5fa">{cp_metric_value}</div>
    <div class="label">{cp_metric_label}</div>
  </div>
  <div class="stat-card">
    <div class="value" style="color:#22c55e">{cat_counts.get('fixed', 0)}</div>
    <div class="label">Fixed (wrong &rarr; right)</div>
  </div>
  <div class="stat-card">
    <div class="value" style="color:#ef4444">{cat_counts.get('broken', 0)}</div>
    <div class="label">Broken (right &rarr; wrong)</div>
  </div>
  <div class="stat-card">
    <div class="value">{len(rows)}</div>
    <div class="label">Total Problems</div>
  </div>
</div>
""")

    # Category bar
    parts.append('<div class="cat-bar">')
    for cat in ["fixed", "improved", "still_right", "unchanged", "regressed", "broken", "still_wrong"]:
        n = cat_counts.get(cat, 0)
        if n == 0:
            continue
        pct = n / len(rows) * 100
        parts.append(f'<div class="cat-segment" style="width:{pct}%;background:{cat_colors[cat]}" '
                      f'title="{cat_labels.get(cat, cat)}: {n}">{n}</div>')
    parts.append('</div>')

    # Filter buttons
    parts.append('<div class="filter-bar">')
    parts.append('<button class="filter-btn active" onclick="filterProblems(\'all\')">All</button>')
    for cat in ["fixed", "improved", "still_right", "unchanged", "regressed", "broken", "still_wrong"]:
        n = cat_counts.get(cat, 0)
        if n == 0:
            continue
        parts.append(f'<button class="filter-btn" onclick="filterProblems(\'{cat}\')">'
                      f'{cat_labels.get(cat, cat)} ({n})</button>')
    parts.append('</div>')

    parts.append('<div class="toolbar">')
    parts.append('<button onclick="toggleAll()">Expand / Collapse All</button>')
    parts.append('</div>')

    # Problem cards
    for rank, r in enumerate(rows, 1):
        q_preview = r["question"][:120] + ("..." if len(r["question"]) > 120 else "")
        b_badge = "badge-correct" if r["b_correct"] else "badge-wrong"
        b_label = "Correct" if r["b_correct"] else "Wrong"

        cat_color = cat_colors.get(r["category"], "#9ca3af")
        escaped_q = html.escape(r["question"])
        escaped_b_response = html.escape(r["b_response"])
        escaped_gold = html.escape(r["gold_label"])
        entity_tags = "".join(f'<span class="entity-tag">{html.escape(e)}</span>' for e in r["entities"])

        if cp_is_greedy:
            c_badge = "badge-correct" if r["c_correct_bool"] else "badge-wrong"
            c_label = "Correct" if r["c_correct_bool"] else "Wrong"
            cp_header_badge = (f'<span class="badge {c_badge}">Checkpoint: {c_label}</span>')
        else:
            c_rate_pct = r["c_rate"] * 100
            rate_color = "#22c55e" if c_rate_pct >= 75 else "#60a5fa" if c_rate_pct >= 50 else "#fbbf24" if c_rate_pct >= 25 else "#ef4444"
            cp_header_badge = (
                f'<span class="badge badge-rate" style="background:{rate_color}30;color:{rate_color}">'
                f'Checkpoint: {r["c_correct"]}/{r["c_total"]}</span>'
            )

        parts.append(f"""
<div class="problem cat-{r['category']}" data-cat="{r['category']}">
  <div class="problem-header" onclick="toggleProblem(this)">
    <div class="left">
      <span class="idx">#{r['idx']}</span>
      <span class="question-preview">{html.escape(q_preview)}</span>
    </div>
    <div class="badges">
      <span class="badge badge-entities">{r['n_entities']} ent</span>
      <span class="badge {b_badge}">Base: {b_label}</span>
      {cp_header_badge}
      <span class="badge-cat" style="background:{cat_color}30;color:{cat_color}">
        {r['category'].replace('_', ' ').title()}
      </span>
      <span class="arrow">&#9654;</span>
    </div>
  </div>
  <div class="problem-body">
    <div class="section-label">Question</div>
    <div class="question-text">{escaped_q}</div>

    <div class="entity-tags" style="margin-top:6px">{entity_tags if entity_tags else '<span style="color:#64748b;font-size:0.85em">No named entities detected</span>'}</div>

    <div class="gold-box">
      <div>Gold answer: <span class="gold-answer">{html.escape(r['gold_answer'])}</span></div>
      <div class="gold-steps">{escaped_gold}</div>
    </div>

    <div class="comparison">
      <div class="panel baseline">
        <h3>Baseline (Greedy, temp=0)</h3>
        <div>Predicted: <b>{html.escape(str(r['b_pred']))}</b>
          <span class="badge {b_badge}" style="margin-left:8px">{b_label}</span>
          <span style="color:#64748b;font-size:0.8em;margin-left:8px">via {r['b_method']}</span>
        </div>
        <div class="response-text">{escaped_b_response}</div>
        <div class="meta">
          <span>{r['b_tokens']} tokens</span>
          <span>{r['b_error']}</span>
        </div>
      </div>
""")

        if cp_is_greedy:
            escaped_c_response = html.escape(r["c_response"])
            c_badge_inner = "badge-correct" if r["c_correct_bool"] else "badge-wrong"
            c_label_inner = "Correct" if r["c_correct_bool"] else "Wrong"
            parts.append(f"""
      <div class="panel checkpoint">
        <h3>Checkpoint (Greedy, temp=0)</h3>
        <div>Predicted: <b>{html.escape(str(r['c_pred']))}</b>
          <span class="badge {c_badge_inner}" style="margin-left:8px">{c_label_inner}</span>
          <span style="color:#64748b;font-size:0.8em;margin-left:8px">via {r['c_method']}</span>
        </div>
        <div class="response-text">{escaped_c_response}</div>
        <div class="meta">
          <span>{r['c_tokens']} tokens</span>
          <span>{r['c_error']}</span>
        </div>
      </div>
""")
        else:
            c_rate_pct = r["c_rate"] * 100
            rate_color = "#22c55e" if c_rate_pct >= 75 else "#60a5fa" if c_rate_pct >= 50 else "#fbbf24" if c_rate_pct >= 25 else "#ef4444"
            c_methods_str = ", ".join(f"{m}: {c}" for m, c in sorted(r["c_methods"].items(), key=lambda x: -x[1]))
            parts.append(f"""
      <div class="panel checkpoint">
        <h3>Checkpoint (16 samples, temp=0.7)</h3>
        <div>Pass rate: <b>{r['c_correct']}/{r['c_total']}</b> ({c_rate_pct:.0f}%)</div>
        <div class="rate-bar">
          <div class="rate-fill" style="width:{c_rate_pct}%;background:{rate_color}"></div>
        </div>
        <div class="meta">
          <span>Methods: {c_methods_str}</span>
        </div>
        <div style="margin-top:12px;font-size:0.85em;color:#94a3b8;">
          <i>Individual responses not saved. {r['c_correct']}/{r['c_total']} of 16 samples correct.</i>
        </div>
      </div>
""")

        parts.append("""
    </div>
  </div>
</div>
""")

    parts.append("""
</div>
<script>
function toggleProblem(header) {
  header.parentElement.classList.toggle('open');
}
function toggleAll() {
  const problems = document.querySelectorAll('.problem');
  const anyOpen = [...problems].some(p => p.classList.contains('open') && p.style.display !== 'none');
  problems.forEach(p => {
    if (anyOpen) p.classList.remove('open');
    else p.classList.add('open');
  });
}
function filterProblems(cat) {
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('.problem').forEach(p => {
    if (cat === 'all' || p.dataset.cat === cat) {
      p.style.display = '';
    } else {
      p.style.display = 'none';
    }
  });
}
</script>
</body>
</html>
""")

    return "\n".join(parts)


def main():
    baseline_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/mxu/Downloads/baseline_eval.jsonl"
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else "/Users/mxu/Downloads/eval_checkpoint_no_system_prompt.jsonl"

    print(f"Loading baseline:   {baseline_path}")
    baseline = load_jsonl(baseline_path)
    print(f"  {len(baseline)} records")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = load_jsonl(checkpoint_path)
    print(f"  {len(checkpoint)} records")

    html_content = build_html(baseline, checkpoint)

    os.makedirs("analysis", exist_ok=True)
    output_path = "analysis/baseline_vs_checkpoint.html"
    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"\nReport saved to {output_path}")
    print(f"Open in browser: file://{os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
