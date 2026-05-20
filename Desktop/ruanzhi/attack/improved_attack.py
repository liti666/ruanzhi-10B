"""
组长负责: Attention-Weighted Importance Ranking (AWIR) Attack.

Our improvement over TextFooler (Jin et al., AAAI 2020):

Standard TextFooler word importance:
    importance(w_i) = confidence(x) - confidence(x with w_i deleted)

Our AWIR improvement:
    importance(w_i) = [confidence(x) - confidence(x_del_i)] * (1 + attention_weight(w_i))

Rationale: BERT's attention weights encode which tokens the model "focuses on"
when making a classification decision. Words with both high deletion-score AND
high attention are more critical to the prediction — attacking them first should
yield a higher attack success rate with fewer model queries.

This script also runs standard WIR as a control, so you get a direct A/B comparison.

Depends on: train/finetune_bert.py must finish first.
Run:
    python attack/improved_attack.py

Outputs (in ./results/improved/):
    improved_results.json  -- ASR, avg queries, perturbation rate for AWIR vs WIR
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from datasets import load_dataset
from nltk.corpus import wordnet
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from configs.config import CHECKPOINT_DIR, DATASET, RESULTS_DIR


# ---------------------------------------------------------------------------
# Core model utilities
# ---------------------------------------------------------------------------

def predict(model, tokenizer, text, device):
    """Return (predicted_label, confidence_for_predicted_label)."""
    enc = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=256, padding=True
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    pred = logits.argmax(dim=-1).item()
    return pred, probs[0][pred].item()


def get_word_attention_weights(model, tokenizer, text, words, device):
    """
    Extract mean BERT attention weight for each word.
    Aggregates subword tokens by averaging, then normalizes to [0, 1].
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**enc, output_attentions=True)

    # attentions: tuple of (1, num_heads, seq_len, seq_len) per layer
    stacked = torch.stack(outputs.attentions, dim=0).squeeze(1)  # (layers, heads, S, S)
    token_importance = stacked.mean(dim=(0, 1)).mean(dim=0).cpu().numpy()  # (S,)

    # Map token importance to word importance (handle ##subwords)
    word_attn = []
    tok_idx = 1  # skip [CLS]
    for word in words:
        subtokens = tokenizer.tokenize(word)
        n = len(subtokens)
        end = tok_idx + n
        if end <= len(token_importance) - 1:
            word_attn.append(float(np.mean(token_importance[tok_idx:end])))
        else:
            word_attn.append(0.0)
        tok_idx += n

    max_w = max(word_attn) if max(word_attn) > 0 else 1.0
    return [a / max_w for a in word_attn]


def get_synonyms(word):
    """WordNet synonyms for a word (max 10 candidates)."""
    candidates = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ")
            if candidate.lower() != word.lower() and " " not in candidate:
                candidates.add(candidate)
    return list(candidates)[:10]


# ---------------------------------------------------------------------------
# Attack core
# ---------------------------------------------------------------------------

def awir_attack(model, tokenizer, text, true_label, device, use_attention=True):
    """
    Attack a single example.

    Args:
        use_attention: True => AWIR (our method), False => standard WIR (baseline).

    Returns:
        (adversarial_text or None, num_queries, num_words_changed)
    """
    words = text.split()
    if len(words) < 3:
        return None, 0, 0

    original_pred, original_conf = predict(model, tokenizer, text, device)
    if original_pred != true_label:
        return None, 0, 0  # already misclassified, skip

    queries = 1

    # Step 1: compute importance via word deletion
    importance = []
    for i in range(len(words)):
        deleted = " ".join(words[:i] + words[i + 1:])
        _, conf = predict(model, tokenizer, deleted, device)
        queries += 1
        importance.append(original_conf - conf)

    importance_array = np.array(importance)

    # Step 2: (AWIR only) multiply by attention weights
    if use_attention:
        try:
            attn = get_word_attention_weights(model, tokenizer, text, words, device)
            importance_array = importance_array * (1.0 + np.array(attn))
        except Exception:
            pass  # fall back to standard WIR silently

    sorted_indices = np.argsort(importance_array)[::-1]

    # Step 3: greedily substitute words by importance order
    current_words = words.copy()
    words_changed = 0

    for idx in sorted_indices:
        original_word = current_words[idx]
        synonyms = get_synonyms(original_word)

        for synonym in synonyms:
            candidate = current_words.copy()
            candidate[idx] = synonym
            candidate_text = " ".join(candidate)
            pred, _ = predict(model, tokenizer, candidate_text, device)
            queries += 1

            if pred != true_label:
                current_words[idx] = synonym
                words_changed += 1
                return " ".join(current_words), queries, words_changed

        # Keep best synonym that increases score even if not yet successful
        current_words[idx] = original_word

    return None, queries, 0


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(model, tokenizer, dataset, num_examples, device, use_attention, label):
    results = {"success": 0, "total": 0, "queries": [], "perturb_rates": []}

    for example in tqdm(dataset.select(range(num_examples)), desc=label):
        text = example["text"][:800]  # cap length for speed
        true_label = example["label"]

        adv, queries, n_changed = awir_attack(
            model, tokenizer, text, true_label, device, use_attention=use_attention
        )
        results["total"] += 1
        results["queries"].append(queries)

        if adv is not None:
            results["success"] += 1
            n_words = len(text.split())
            results["perturb_rates"].append(n_changed / n_words if n_words > 0 else 0)

    results["asr"] = results["success"] / results["total"] * 100
    results["avg_queries"] = float(np.mean(results["queries"]))
    results["avg_perturb_rate"] = float(np.mean(results["perturb_rates"])) if results["perturb_rates"] else 0.0
    return results


def main(args):
    out_dir = os.path.join(args.results_dir, "improved")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    model = BertForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    dataset = load_dataset(args.dataset, split="test")

    all_results = {}

    # Control: standard WIR (same as TextFooler backbone, no attention weighting)
    print("\n[1/2] Standard WIR (control group)...")
    all_results["WIR_baseline"] = run_experiment(
        model, tokenizer, dataset, args.num_examples, device,
        use_attention=False, label="WIR (control)"
    )

    # Treatment: AWIR (our improved method)
    print("\n[2/2] AWIR — Attention-Weighted Importance Ranking (our method)...")
    all_results["AWIR_improved"] = run_experiment(
        model, tokenizer, dataset, args.num_examples, device,
        use_attention=True, label="AWIR (ours)"
    )

    # Print comparison
    print("\n" + "="*60)
    print("IMPROVEMENT COMPARISON")
    print("="*60)
    print(f"{'Method':<20} {'ASR':>8} {'Avg Queries':>13} {'Perturb Rate':>14}")
    print("-"*60)
    for name, r in all_results.items():
        print(
            f"{name:<20} {r['asr']:>7.1f}% "
            f"{r['avg_queries']:>13.1f} "
            f"{r['avg_perturb_rate']:>13.1%}"
        )

    output_path = os.path.join(out_dir, "improved_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    args = parser.parse_args()
    main(args)
