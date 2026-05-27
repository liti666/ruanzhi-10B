"""
AWIR Attack — Google Colab standalone version.
Upload this file to Colab and run: !python improved_attack_colab.py

Output: results/improved/improved_results.json
Download that file back to your local results/improved/ folder.
"""
import json
import os

import numpy as np
import torch
from datasets import load_dataset
from nltk.corpus import wordnet
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

# ---------------------------------------------------------------------------
# Config (inline to avoid local path dependency)
# ---------------------------------------------------------------------------
PRETRAINED_MODEL_DIR = "textattack/bert-base-uncased-imdb"
DATASET = "imdb"
NUM_EXAMPLES = 100


# ---------------------------------------------------------------------------
# Semantic similarity
# ---------------------------------------------------------------------------
def compute_semantic_similarity(orig_texts, pert_texts):
    if not orig_texts:
        return float("nan")
    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer("all-MiniLM-L6-v2")
        orig_embs = model.encode(orig_texts, convert_to_tensor=True, show_progress_bar=False)
        pert_embs = model.encode(pert_texts, convert_to_tensor=True, show_progress_bar=False)
        sims = util.cos_sim(orig_embs, pert_embs).diagonal()
        return round(float(sims.mean()), 4)
    except ImportError:
        print("[WARNING] sentence-transformers not installed")
        return float("nan")


# ---------------------------------------------------------------------------
# Core model utilities
# ---------------------------------------------------------------------------
def predict_batch(model, tokenizer, texts, device):
    enc = tokenizer(
        texts, return_tensors="pt", truncation=True, max_length=256, padding=True
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    preds = logits.argmax(dim=-1).tolist()
    confs = probs[range(len(texts)), preds].tolist()
    return preds, confs


def predict(model, tokenizer, text, device):
    preds, confs = predict_batch(model, tokenizer, [text], device)
    return preds[0], confs[0]


def get_word_attention_weights(model, tokenizer, text, words, device):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**enc, output_attentions=True)
    stacked = torch.stack(outputs.attentions, dim=0).squeeze(1)
    token_importance = stacked.mean(dim=(0, 1)).mean(dim=0).cpu().numpy()
    word_attn = []
    tok_idx = 1
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
    words = text.split()
    n_words = len(words)
    if n_words < 3:
        return None, 0, 0

    original_pred, original_conf = predict(model, tokenizer, text, device)
    if original_pred != true_label:
        return None, 0, 0

    # Step 1: batch word deletion importance
    deletion_texts = [" ".join(words[:i] + words[i + 1:]) for i in range(n_words)]
    _, confs = predict_batch(model, tokenizer, deletion_texts, device)
    queries = 1 + n_words
    importance_array = np.array([original_conf - c for c in confs])

    # Step 2: AWIR attention weighting
    if use_attention:
        try:
            attn = get_word_attention_weights(model, tokenizer, text, words, device)
            importance_array = importance_array * (1.0 + np.array(attn))
        except Exception:
            pass

    sorted_indices = np.argsort(importance_array)[::-1]

    # Step 3: greedy synonym substitution (batched per position)
    current_words = words.copy()
    words_changed = 0

    for idx in sorted_indices:
        original_word = current_words[idx]
        synonyms = get_synonyms(original_word)
        if not synonyms:
            continue

        candidates_texts = []
        for synonym in synonyms:
            candidate = current_words.copy()
            candidate[idx] = synonym
            candidates_texts.append(" ".join(candidate))

        preds, _ = predict_batch(model, tokenizer, candidates_texts, device)
        queries += len(synonyms)

        for j, pred in enumerate(preds):
            if pred != true_label:
                current_words[idx] = synonyms[j]
                words_changed += 1
                return " ".join(current_words), queries, words_changed

        current_words[idx] = original_word

    return None, queries, 0


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
def run_experiment(model, tokenizer, dataset, num_examples, device, use_attention, label):
    results = {"success": 0, "total": 0, "queries": [], "perturb_rates": []}
    orig_texts = []
    pert_texts = []

    for example in tqdm(dataset.select(range(num_examples)), desc=label):
        text = example["text"][:800]
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
            orig_texts.append(text)
            pert_texts.append(adv)

    results["asr"] = results["success"] / results["total"] * 100
    results["avg_queries"] = float(np.mean(results["queries"]))
    results["avg_perturb_rate"] = float(np.mean(results["perturb_rates"])) if results["perturb_rates"] else 0.0
    results["sem_sim"] = compute_semantic_similarity(orig_texts, pert_texts)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)

    out_dir = "./results/improved"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model from {PRETRAINED_MODEL_DIR}...")
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_DIR).to(device)
    model.eval()

    print(f"Loading dataset: {DATASET} (test split)")
    dataset = load_dataset(DATASET, split="test")

    all_results = {}

    print("\n[1/2] Standard WIR (control group)...")
    all_results["WIR_baseline"] = run_experiment(
        model, tokenizer, dataset, NUM_EXAMPLES, device,
        use_attention=False, label="WIR (control)"
    )

    print("\n[2/2] AWIR — Attention-Weighted Importance Ranking (our method)...")
    all_results["AWIR_improved"] = run_experiment(
        model, tokenizer, dataset, NUM_EXAMPLES, device,
        use_attention=True, label="AWIR (ours)"
    )

    print("\n" + "=" * 80)
    print("IMPROVEMENT COMPARISON")
    print("=" * 80)
    print(f"{'Method':<20} {'ASR':>8} {'Avg Queries':>13} {'Perturb Rate':>14} {'Sem. Similarity':>17}")
    print("-" * 80)
    for name, r in all_results.items():
        print(
            f"{name:<20} {r['asr']:>7.1f}% "
            f"{r['avg_queries']:>13.1f} "
            f"{r['avg_perturb_rate']:>13.1%} "
            f"{r['sem_sim']:>17.4f}"
        )

    output_path = os.path.join(out_dir, "improved_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
