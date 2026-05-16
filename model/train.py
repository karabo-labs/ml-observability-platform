"""
Train a DistilBERT sentiment classifier on IMDB.
Outputs model + reference statistics for drift detection.
"""

import json
import os
import sys
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = Path("model_output")
HUB_MODEL_ID = os.environ.get("HF_MODEL_ID", "DynamicKarabo/sentiment-distilbert")
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def compute_reference_stats(dataset, trainer):
    """Run predictions on a reference set and save distribution stats for drift detection."""
    preds = trainer.predict(dataset)
    logits = preds.predictions
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

    # Per-class confidence distribution
    stats = {
        "n_samples": len(probs),
        "class_proportions": {
            "negative": float(np.mean(np.argmax(probs, axis=1) == 0)),
            "positive": float(np.mean(np.argmax(probs, axis=1) == 1)),
        },
        "confidence_mean": float(np.mean(np.max(probs, axis=1))),
        "confidence_std": float(np.std(np.max(probs, axis=1))),
        "confidence_histogram": {
            "bins": np.linspace(0, 1, 11).tolist(),
            "counts": np.histogram(np.max(probs, axis=1), bins=10, range=(0, 1))[0].tolist(),
        },
        "prediction_histogram": {
            # Joint distribution: negative confidence bins, positive confidence bins
            "neg_bins": np.linspace(0, 1, 11).tolist(),
            "neg_counts": np.histogram(probs[:, 0], bins=10, range=(0, 1))[0].tolist(),
            "pos_bins": np.linspace(0, 1, 11).tolist(),
            "pos_counts": np.histogram(probs[:, 1], bins=10, range=(0, 1))[0].tolist(),
        },
    }
    return stats


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1.compute(predictions=predictions, references=labels)["f1"],
    }


def main():
    print("=" * 60)
    print("ML Observability — Training Pipeline")
    print("=" * 60)

    # Load tokenizer and model
    print(f"\n[1/5] Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # Load dataset
    print("\n[2/5] Loading IMDB dataset")
    dataset = load_dataset("imdb")

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.remove_columns(["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    # Split into train/eval
    train_dataset = tokenized["train"].shuffle(seed=42).select(range(20000))  # subset for speed
    eval_dataset = tokenized["test"].shuffle(seed=42).select(range(5000))

    # Training args
    print("\n[3/5] Configuring training")
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\n[4/5] Training...")
    trainer.train()

    # Evaluate
    eval_results = trainer.evaluate()
    print(f"\n✅ Evaluation: accuracy={eval_results['eval_accuracy']:.4f}, f1={eval_results['eval_f1']:.4f}")

    # Compute reference stats for drift detection
    print("\n[5/5] Computing reference statistics for drift detection")
    ref_dataset = tokenized["test"].shuffle(seed=42).select(range(2000))
    ref_stats = compute_reference_stats(ref_dataset, trainer)

    # Save model + reference stats
    print(f"\nSaving model to {OUTPUT_DIR}/")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    with open(OUTPUT_DIR / "reference_stats.json", "w") as f:
        json.dump(ref_stats, f, indent=2)

    # Save metrics metadata
    metrics = {
        "accuracy": eval_results["eval_accuracy"],
        "f1": eval_results["eval_f1"],
        "loss": eval_results["eval_loss"],
        "n_train": len(train_dataset),
        "n_eval": len(eval_dataset),
        "model_name": MODEL_NAME,
    }
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"✅ Training complete!")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   F1 Score:  {metrics['f1']:.4f}")
    print(f"   Reference stats saved to {OUTPUT_DIR / 'reference_stats.json'}")
    print(f"{'=' * 60}")

    # Return metrics for CI/CD
    return metrics


if __name__ == "__main__":
    main()
