# Payment Fraud Detection — Machine Learning Demo

[![Python ML Tests](https://github.com/LouisKhoo573/fraud-detection-ml-demo/actions/workflows/python-tests.yml/badge.svg)](https://github.com/LouisKhoo573/fraud-detection-ml-demo/actions/workflows/python-tests.yml)

An **educational Python project** showing how to classify simulated payment transactions, measure fraud-detection performance, and examine the precision–recall trade-off.

> **Personal demonstration project.** All transactions and fraud labels are generated synthetically. This is **not** a Capital One project, company model, production fraud detector, or a report on real fraud rates. No bank/customer/private data are included.

## What it does

1. Generates simulated transactions with amount, transaction time, new-device flag, cross-border flag, recent transaction count, account age, and a synthetic merchant-risk signal.
2. Creates synthetic fraud labels using a deliberately invented probability formula.
3. Splits records into **70% training / 15% validation / 15% test**, stratified by label.
4. Trains a class-weighted **Random Forest** classifier.
5. Picks a decision threshold using **only validation data**, then evaluates on unseen test data.
6. Saves **precision, recall, F1, average precision (PR-AUC summary), ROC-AUC, and a confusion matrix**, along with a precision–recall chart.

## Run locally

Install Python 3.10 or newer, then in this folder run:

```bash
python -m pip install -r requirements.txt
python fraud_detection_demo.py
```

Optional:

```bash
python fraud_detection_demo.py --n-transactions 20000 --seed 17
python -m unittest discover -s tests -v
```

Generated files appear in `outputs/`:

- `metrics.json` — test-set results and selected threshold.
- `precision_recall_curve.png` — precision–recall trade-off on synthetic test transactions.
- `synthetic_transactions_sample.csv` — first 100 simulated transactions for inspection.

## Key concepts

- **Class imbalance:** Fraud is rare, so accuracy alone is misleading.
- **Precision:** Of the transactions flagged, how many were labeled fraudulent?
- **Recall:** Of the fraudulent transactions, how many were flagged?
- **Threshold selection:** Choose on validation data; evaluate once on test data.
- **Average precision:** Summarizes precision–recall across thresholds and is useful for imbalanced classification.

## Limitations and responsible use

These invented labels, synthetic features, and model results **cannot establish real-world accuracy**. Real payment systems require validated labels, temporal splits, careful leakage checks, privacy protections, fairness assessments, human review, and ongoing monitoring. Do not use this code to approve, decline, or block real transactions.

## Profile

[Louis Khoo on GitHub](https://github.com/LouisKhoo573) · [LinkedIn](https://www.linkedin.com/in/louis-khoo-1493a2232/) · [Crunchbase](https://www.crunchbase.com/person/louis-khoo-2ae4)

