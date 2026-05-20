"""
成员1负责: Fine-tune BERT on IMDB, save checkpoint, report clean accuracy.

Run:
    python train/finetune_bert.py

Expected output:
    Epoch 3 eval accuracy: ~93%
    Model saved to ./checkpoints/bert-imdb/

Time estimate: ~2h on GPU, ~10h+ on CPU (run overnight if no GPU).
"""
import argparse
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from configs.config import (
    CHECKPOINT_DIR, DATASET, LEARNING_RATE, MAX_LENGTH,
    MODEL_NAME, NUM_LABELS, TRAIN_BATCH_SIZE, TRAIN_EPOCHS,
)


def tokenize_batch(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    raw = load_dataset(args.dataset)

    print(f"Loading model: {args.model_name}")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(
        args.model_name, num_labels=args.num_labels
    )

    tokenized = raw.map(
        lambda ex: tokenize_batch(ex, tokenizer),
        batched=True,
        remove_columns=["text"],
    )
    tokenized.set_format("torch")

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=os.path.join(args.save_dir, "logs"),
        logging_steps=100,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print(f"\nSaving model to {args.save_dir}")
    trainer.save_model(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)

    print("\n--- Final Evaluation ---")
    results = trainer.evaluate()
    print(f"Clean Accuracy: {results['eval_accuracy']:.4f}")
    print(f"Eval Loss:      {results['eval_loss']:.4f}")
    print("\nModel ready. Other members can now load it from:", args.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--num_labels", type=int, default=NUM_LABELS)
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--save_dir", default=CHECKPOINT_DIR)
    args = parser.parse_args()
    main(args)
