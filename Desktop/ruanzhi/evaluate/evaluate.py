"""
成员4负责: Unified evaluation — compute and print the final comparison table.

This script reads results from all three attack experiments and produces
a single table for the report. Run after all attacks are done.

Run:
    python evaluate/evaluate.py

Inputs (must exist first):
    ./results/baseline/textfooler_results.csv
    ./results/baseline/bertattack_results.csv
    ./results/improved/improved_results.json

Output:
    ./results/final_comparison.csv  -- the table for your report
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from configs.config import RESULTS_DIR


def parse_textattack_csv(csv_path, method_name):
    """Parse a TextAttack result CSV and return a metrics dict."""
    if not os.path.exists(csv_path):
        print(f"  [SKIP] {csv_path} not found. Run baseline_attack.py first.")
        return None

    df = pd.read_csv(csv_path)
    total = len(df)
    successful = df[df["result_type"] == "Successful"]
    failed = df[df["result_type"] == "Failed"]
    skipped = df[df["result_type"] == "Skipped"]

    asr = len(successful) / (total - len(skipped)) * 100 if (total - len(skipped)) > 0 else 0

    avg_words_changed = (
        successful["num_words_changed"].mean()
        if "num_words_changed" in successful.columns and len(successful) > 0
        else float("nan")
    )
    avg_queries = (
        df["num_queries"].mean()
        if "num_queries" in df.columns
        else float("nan")
    )

    return {
        "Method": method_name,
        "Total": total,
        "Successful": len(successful),
        "Failed": len(failed),
        "Skipped": len(skipped),
        "ASR (%)": round(asr, 1),
        "Avg Words Changed": round(avg_words_changed, 2),
        "Avg Queries": round(avg_queries, 1),
    }


def parse_improved_json(json_path):
    """Parse results from improved_attack.py and return rows for AWIR and WIR baseline."""
    if not os.path.exists(json_path):
        print(f"  [SKIP] {json_path} not found. Run improved_attack.py first.")
        return []

    with open(json_path) as f:
        data = json.load(f)

    rows = []
    name_map = {
        "WIR_baseline": "WIR (control, ours)",
        "AWIR_improved": "AWIR (ours, improved)",
    }
    for key, r in data.items():
        rows.append({
            "Method": name_map.get(key, key),
            "Total": r["total"],
            "Successful": r["success"],
            "Failed": r["total"] - r["success"],
            "Skipped": 0,
            "ASR (%)": round(r["asr"], 1),
            "Avg Words Changed": round(r.get("avg_perturb_rate", float("nan")), 3),
            "Avg Queries": round(r["avg_queries"], 1),
        })
    return rows


def main():
    print("="*65)
    print("FINAL COMPARISON TABLE")
    print("="*65)

    rows = []

    # Baseline results
    for name, fname in [
        ("TextFooler (Jin 2020)", "baseline/textfooler_results.csv"),
        ("BERT-Attack (Li 2020)", "baseline/bertattack_results.csv"),
        ("HotFlip (Ebrahimi 2018)", "baseline/hotflip_results.csv"),
    ]:
        r = parse_textattack_csv(os.path.join(RESULTS_DIR, fname), name)
        if r:
            rows.append(r)

    # Improved attack results
    improved_path = os.path.join(RESULTS_DIR, "improved", "improved_results.json")
    rows.extend(parse_improved_json(improved_path))

    if not rows:
        print("No results found. Please run the attack scripts first.")
        return

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    output_path = os.path.join(RESULTS_DIR, "final_comparison.csv")
    df.to_csv(output_path, index=False)
    print(f"\nTable saved to {output_path}")

    # Also compute defense comparison if robust model results exist
    robust_path = os.path.join(RESULTS_DIR, "defense", "textfooler_robust_results.csv")
    if os.path.exists(robust_path):
        print("\n" + "="*65)
        print("DEFENSE COMPARISON (Clean Model vs Robust Model)")
        print("="*65)
        clean_r = parse_textattack_csv(
            os.path.join(RESULTS_DIR, "baseline", "textfooler_results.csv"),
            "Clean BERT + TextFooler"
        )
        robust_r = parse_textattack_csv(robust_path, "Robust BERT + TextFooler")
        if clean_r and robust_r:
            defense_df = pd.DataFrame([clean_r, robust_r])
            print(defense_df[["Method", "ASR (%)", "Avg Queries"]].to_string(index=False))


if __name__ == "__main__":
    main()
