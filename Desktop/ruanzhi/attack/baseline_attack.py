"""
基线攻击复现: TextFooler 和 BERT-Attack（均为黑盒攻击）

Run:
    python attack/baseline_attack.py --attack textfooler
    python attack/baseline_attack.py --attack bertattack
    python attack/baseline_attack.py  # 两个都跑

Outputs (in ./results/baseline/):
    textfooler_results.csv   -- Jin et al. AAAI 2020   (black-box)
    bertattack_results.csv   -- Li et al. EMNLP 2020   (black-box)
    baseline_summary.txt     -- printed comparison table

Paper targets to reproduce (IMDB, BERT victim):
    TextFooler: Attack Success Rate ~87%, Avg perturbed words ~6
    BERT-Attack: Attack Success Rate ~90%+
"""
import argparse
import os
import sys

from textattack import AttackArgs, Attacker
from textattack.attack_recipes import BERTAttackLi2020, TextFoolerJin2019
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import BertForSequenceClassification, BertTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from configs.config import ATTACK_NUM_EXAMPLES, DATASET, PRETRAINED_MODEL_DIR, RESULTS_DIR


def run_single_attack(recipe_class, model_wrapper, dataset, num_examples, output_csv):
    attack = recipe_class.build(model_wrapper)
    attack_args = AttackArgs(
        num_examples=num_examples,
        log_to_csv=output_csv,
        csv_coloring_style="plain",
        disable_stdout=False,
    )
    attacker = Attacker(attack, dataset, attack_args)
    results = attacker.attack_dataset()
    return results


def summarize_results(csv_path, method_name):
    import pandas as pd
    df = pd.read_csv(csv_path)
    total = len(df)
    success = (df["result_type"] == "Successful").sum()
    asr = success / total * 100 if total > 0 else 0.0

    if "num_words_changed" in df.columns:
        avg_words_changed = df.loc[df["result_type"] == "Successful", "num_words_changed"].mean()
    else:
        avg_words_changed = float("nan")

    print(f"\n[{method_name}]")
    print(f"  Total examples:       {total}")
    print(f"  Successful attacks:   {success}")
    print(f"  Attack Success Rate:  {asr:.1f}%")
    print(f"  Avg words changed:    {avg_words_changed:.1f}")
    return {"method": method_name, "total": total, "success": success,
            "asr": asr, "avg_words_changed": avg_words_changed}


def main(args):
    out_dir = os.path.join(args.results_dir, "baseline")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading model from {args.model_dir}")
    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    model = BertForSequenceClassification.from_pretrained(args.model_dir)
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    print(f"Loading dataset: {args.dataset} (test split, {args.num_examples} examples)")
    dataset = HuggingFaceDataset(args.dataset, "plain_text", split="test")

    summaries = []

    run = args.attack  # "textfooler" | "bertattack" | "all"

    # --- TextFooler (Jin et al., AAAI 2020) ---
    if run in ("textfooler", "all"):
        print("\n" + "="*50)
        print("Running TextFooler (Jin et al., AAAI 2020)...")
        print("="*50)
        tf_csv = os.path.join(out_dir, "textfooler_results.csv")
        run_single_attack(TextFoolerJin2019, model_wrapper, dataset, args.num_examples, tf_csv)
        summaries.append(summarize_results(tf_csv, "TextFooler"))

    # --- BERT-Attack (Li et al., EMNLP 2020) ---
    if run in ("bertattack", "all"):
        print("\n" + "="*50)
        print("Running BERT-Attack (Li et al., EMNLP 2020)...")
        print("="*50)
        ba_csv = os.path.join(out_dir, "bertattack_results.csv")
        run_single_attack(BERTAttackLi2020, model_wrapper, dataset, args.num_examples, ba_csv)
        summaries.append(summarize_results(ba_csv, "BERT-Attack"))

    # --- Print comparison table ---
    print("\n" + "="*50)
    print("BASELINE SUMMARY (compare with paper tables)")
    print("="*50)
    print(f"{'Method':<15} {'ASR':>8} {'Avg Words Changed':>20}")
    print("-"*45)
    for s in summaries:
        print(f"{s['method']:<15} {s['asr']:>7.1f}% {s['avg_words_changed']:>20.1f}")

    summary_path = os.path.join(out_dir, "baseline_summary.txt")
    with open(summary_path, "w") as f:
        for s in summaries:
            f.write(f"{s}\n")
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=PRETRAINED_MODEL_DIR)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--num_examples", type=int, default=ATTACK_NUM_EXAMPLES)
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument(
        "--attack",
        choices=["textfooler", "bertattack", "all"],
        default="all",
        help="Which attack to run. Default: both.",
    )
    args = parser.parse_args()
    main(args)
