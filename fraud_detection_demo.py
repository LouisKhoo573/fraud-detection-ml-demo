"""Educational payment-fraud classification using fully synthetic transactions.

No real customer data, bank systems, or proprietary models are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

FEATURES = [
    "amount_usd",
    "hour_of_day",
    "new_device",
    "cross_border",
    "transactions_last_24h",
    "account_age_days",
    "merchant_risk_score",
    "is_weekend",
]


def make_synthetic_transactions(n: int = 15000, seed: int = 42) -> pd.DataFrame:
    """Generate illustrative transactions and synthetic fraud labels.

    The labels are drawn from a probability model that depends on transaction
    characteristics. They do not describe real-world fraud behavior or rates.
    """
    if n < 200:
        raise ValueError("Use at least 200 synthetic transactions.")
    rng = np.random.default_rng(seed)
    amount = np.clip(rng.lognormal(mean=3.9, sigma=1.15, size=n), 1, 15000)
    hour = rng.integers(0, 24, size=n)
    new_device = rng.binomial(1, 0.13, size=n)
    cross_border = rng.binomial(1, 0.09, size=n)
    velocity = rng.poisson(lam=3.1, size=n)
    age_days = np.clip(rng.exponential(scale=440, size=n), 1, 3650)
    merchant_risk = rng.beta(2.0, 6.0, size=n)
    weekend = rng.binomial(1, 2 / 7, size=n)

    # A deliberately simple, invented ground-truth mechanism for teaching.
    night = ((hour < 5) | (hour > 22)).astype(int)
    logits = (
        -7.7
        + 0.90 * np.log1p(amount / 80)
        + 2.10 * new_device
        + 1.75 * cross_border
        + 0.38 * velocity
        + 1.30 * (age_days < 30)
        + 2.80 * merchant_risk
        + 0.80 * night
    )
    fraud_probability = 1.0 / (1.0 + np.exp(-logits))
    is_fraud = rng.binomial(1, fraud_probability)
    return pd.DataFrame({
        "amount_usd": np.round(amount, 2),
        "hour_of_day": hour,
        "new_device": new_device,
        "cross_border": cross_border,
        "transactions_last_24h": velocity,
        "account_age_days": np.round(age_days).astype(int),
        "merchant_risk_score": np.round(merchant_risk, 4),
        "is_weekend": weekend,
        "is_fraud": is_fraud,
    })


def best_f1_threshold(y_true: pd.Series, scores: np.ndarray) -> float:
    """Choose a classification threshold on validation data, not test data."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    return float(thresholds[int(np.argmax(f1))])


def run_demo(n: int, seed: int, out_dir: Path) -> dict:
    df = make_synthetic_transactions(n=n, seed=seed)
    X, y = df[FEATURES], df["is_fraud"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )

    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=9,
        min_samples_leaf=6,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    threshold = best_f1_threshold(y_val, model.predict_proba(X_val)[:, 1])
    scores = model.predict_proba(X_test)[:, 1]
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()

    metrics = {
        "data_type": "fully synthetic — demonstration only",
        "seed": seed,
        "transaction_count": n,
        "synthetic_fraud_rate": round(float(y.mean()), 4),
        "test_records": int(len(y_test)),
        "validation_selected_threshold": round(threshold, 4),
        "test_precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "test_f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "test_average_precision": round(float(average_precision_score(y_test, scores)), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, scores)), 4),
        "test_confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    df.head(100).to_csv(out_dir / "synthetic_transactions_sample.csv", index=False)

    prec, rec, _ = precision_recall_curve(y_test, scores)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(rec, prec, label="Random forest (synthetic data)")
    ax.axhline(float(y_test.mean()), linestyle="--", label="Fraud prevalence baseline")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall Curve (Synthetic Data)", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "precision_recall_curve.png", dpi=160)
    plt.close(fig)

    print("SYNTHETIC FRAUD DETECTION DEMO — NOT A REAL BANK MODEL")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"\nGenerated results in: {out_dir.resolve()}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Payment fraud ML demo with synthetic data")
    parser.add_argument("--n-transactions", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    run_demo(args.n_transactions, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
